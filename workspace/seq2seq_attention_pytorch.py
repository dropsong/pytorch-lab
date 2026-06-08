"""
PyTorch version of the TensorFlow 2.2 seq2seq + Bahdanau attention example.

Default direction follows the original notebook:
Spanish input -> English output.

Quick smoke test:
    python seq2seq_attention_pytorch.py --num-examples 512 --epochs 1 --batch-size 32

Longer training:
    python seq2seq_attention_pytorch.py --num-examples 30000 --epochs 10
"""

from __future__ import annotations

import argparse
import random
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"


def unicode_to_ascii(text: str) -> str:
    """Remove accents such as 'á' -> 'a' while keeping plain ASCII letters."""
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def preprocess_sentence(sentence: str) -> str:
    # Keep the text processing intentionally close to the TensorFlow notebook:
    # normalize words first, then make punctuation become separate tokens.
    sentence = unicode_to_ascii(sentence.lower().strip())
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.strip()
    return f"{START_TOKEN} {sentence} {END_TOKEN}"


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def build(cls, sentences: list[str]) -> "Vocabulary":
        # The four special tokens occupy stable ids:
        # 0 is padding, 1/2 mark sequence boundaries, 3 handles unknown words.
        counter: Counter[str] = Counter()
        for sentence in sentences:
            counter.update(sentence.split())

        id_to_token = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]
        for token, _ in counter.most_common():
            if token not in id_to_token:
                id_to_token.append(token)
        token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def start_id(self) -> int:
        return self.token_to_id[START_TOKEN]

    @property
    def end_id(self) -> int:
        return self.token_to_id[END_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, sentence: str) -> list[int]:
        # Convert "<start> hola . <end>" into integer ids for nn.Embedding.
        return [self.token_to_id.get(token, self.unk_id) for token in sentence.split()]

    def decode(self, ids: list[int]) -> str:
        # Turn predicted ids back into words, stopping at <end> like inference code usually does.
        tokens = []
        for idx in ids:
            token = self.id_to_token[idx]
            if token == END_TOKEN:
                break
            if token not in {PAD_TOKEN, START_TOKEN}:
                tokens.append(token)
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.id_to_token)


class TranslationDataset(Dataset):
    def __init__(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
    ) -> None:
        # Store variable-length 1D tensors. collate_batch pads them only inside a batch,
        # which avoids padding every sentence to the global maximum length.
        self.source = [torch.tensor(source_vocab.encode(s), dtype=torch.long) for s in source_sentences]
        self.target = [torch.tensor(target_vocab.encode(s), dtype=torch.long) for s in target_sentences]

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.source[index], self.target[index]


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    # DataLoader calls this to turn a list of samples into two dense tensors:
    # source: (B, T_in_of_this_batch), target: (B, T_out_of_this_batch).
    source_batch, target_batch = zip(*batch)
    source = pad_sequence(source_batch, batch_first=True, padding_value=0)
    target = pad_sequence(target_batch, batch_first=True, padding_value=0)
    return source, target


def load_sentence_pairs(
    path: Path,
    num_examples: int | None,
    reverse: bool,
) -> tuple[list[str], list[str]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if num_examples is not None:
        lines = lines[:num_examples]

    source_sentences: list[str] = []
    target_sentences: list[str] = []
    for line in lines:
        english, spanish = line.split("\t")[:2]
        # reverse=True keeps the original notebook direction: Spanish -> English.
        if reverse:
            source, target = spanish, english
        else:
            source, target = english, spanish
        source_sentences.append(preprocess_sentence(source))
        target_sentences.append(preprocess_sentence(target))
    return source_sentences, target_sentences


def split_pairs(
    source: list[str],
    target: list[str],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    indices = list(range(len(source)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(len(indices) * (1.0 - validation_fraction))
    train_idx, val_idx = indices[:split], indices[split:]
    return (
        [source[i] for i in train_idx],
        [target[i] for i in train_idx],
        [source[i] for i in val_idx],
        [target[i] for i in val_idx],
    )


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)

    def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # source stores token ids: (B, T_in). Embedding turns every id into an E-dim vector.
        embedded = self.embedding(source)
        # outputs keeps the hidden state for every input position.
        # hidden is only the final hidden state; the decoder uses it as its initial state.
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(
        self,
        query: torch.Tensor,
        values: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query is the decoder hidden state for the current output step.
        # PyTorch GRU stores it as (num_layers, B, H), here num_layers=1.
        query = query.permute(1, 0, 2)
        # values are encoder outputs: one H-dim vector per source token.
        # W2(query) has length dimension 1 and broadcasts across T_in.
        score = self.V(torch.tanh(self.W1(values) + self.W2(query)))
        # score has one scalar per source position. Padding positions must not receive attention.
        score = score.masked_fill(~source_mask.unsqueeze(-1), float("-inf"))
        attention_weights = F.softmax(score, dim=1)
        # Weighted sum over source positions. Because values are vectors, context is also a vector.
        context = torch.sum(attention_weights * values, dim=1)
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(embedding_dim + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        current_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # current_token is either the previous target token during training,
        # or the previous prediction during inference.
        embedded = self.embedding(current_token)
        # context summarizes the source sentence for this particular decoder step.
        context, attention_weights = self.attention(hidden, encoder_outputs, source_mask)
        # The GRU receives both "what was the previous word" and "what source info matters now".
        final_input = torch.cat([context.unsqueeze(1), embedded], dim=-1)
        output, hidden = self.gru(final_input, hidden)
        # logits are raw scores over the target vocabulary; cross_entropy applies softmax internally.
        logits = self.fc(output.squeeze(1))
        return logits, hidden, attention_weights


class Seq2SeqAttention(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(source_vocab_size, embedding_dim, hidden_size)
        self.decoder = Decoder(target_vocab_size, embedding_dim, hidden_size)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        batch_size, target_length = target.shape
        # True means a real source token; False means <pad>. Attention uses this mask.
        source_mask = source != 0
        encoder_outputs, hidden = self.encoder(source)
        # First decoder input is <start> for every sample.
        decoder_input = target[:, 0].unsqueeze(1)
        logits_by_step = []

        for t in range(1, target_length):
            logits, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, source_mask)
            logits_by_step.append(logits)
            # Teacher forcing feeds the real previous token; otherwise we feed the model's guess.
            use_teacher = random.random() < teacher_forcing_ratio
            next_token = target[:, t] if use_teacher else logits.argmax(dim=-1)
            decoder_input = next_token.unsqueeze(1)

        # Return all per-step predictions as (B, T_out - 1, target_vocab_size).
        return torch.stack(logits_by_step, dim=1)


def masked_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # logits predicts target[:, 1:], because target[:, 0] is <start>.
    # ignore_index=0 removes <pad> from the loss average.
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target[:, 1:].reshape(-1),
        ignore_index=0,
    )


def train_one_epoch(
    model: Seq2SeqAttention,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    teacher_forcing_ratio: float,
    print_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    for step, (source, target) in enumerate(loader, start=1):
        source, target = source.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        # Forward returns predictions for every target position except <start>.
        logits = model(source, target, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = masked_loss(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if step % print_every == 0:
            print(f"  batch {step:4d}/{len(loader)} loss {loss.item():.4f}")
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_loss(model: Seq2SeqAttention, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    for source, target in loader:
        source, target = source.to(DEVICE), target.to(DEVICE)
        logits = model(source, target, teacher_forcing_ratio=0.0)
        total_loss += masked_loss(logits, target).item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def translate(
    model: Seq2SeqAttention,
    sentence: str,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    max_target_length: int,
) -> tuple[str, list[str], list[str], torch.Tensor]:
    model.eval()
    processed = preprocess_sentence(sentence)
    # Inference uses batch size 1, but the model still expects a batch dimension.
    source_ids = torch.tensor([source_vocab.encode(processed)], dtype=torch.long, device=DEVICE)
    source_mask = source_ids != 0
    encoder_outputs, hidden = model.encoder(source_ids)
    decoder_input = torch.tensor([[target_vocab.start_id]], dtype=torch.long, device=DEVICE)

    predicted_ids: list[int] = []
    attention_rows: list[torch.Tensor] = []
    for _ in range(max_target_length):
        # Unlike training, the next decoder input comes from this step's prediction.
        logits, hidden, attention_weights = model.decoder(
            decoder_input,
            hidden,
            encoder_outputs,
            source_mask,
        )
        predicted_id = int(logits.argmax(dim=-1).item())
        predicted_ids.append(predicted_id)
        # Store one attention distribution per generated target token for plotting later.
        attention_rows.append(attention_weights.squeeze(0).squeeze(-1).cpu())
        if predicted_id == target_vocab.end_id:
            break
        decoder_input = torch.tensor([[predicted_id]], dtype=torch.long, device=DEVICE)

    source_tokens = processed.split()
    predicted_tokens = target_vocab.decode(predicted_ids).split()
    if attention_rows:
        attention = torch.stack(attention_rows, dim=0)[:, : len(source_tokens)]
    else:
        attention = torch.empty(0, len(source_tokens))
    return " ".join(predicted_tokens), source_tokens, predicted_tokens, attention


def strip_boundary_tokens(sentence: str) -> str:
    return " ".join(
        token for token in sentence.split() if token not in {START_TOKEN, END_TOKEN, PAD_TOKEN}
    )


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    overlap = 0
    for token in pred_tokens:
        if ref_counts[token] > 0:
            overlap += 1
            ref_counts[token] -= 1
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def evaluate_translations(
    model: Seq2SeqAttention,
    source_sentences: list[str],
    target_sentences: list[str],
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    max_target_length: int,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for source_sentence, target_sentence in zip(source_sentences[:limit], target_sentences[:limit]):
        source_text = strip_boundary_tokens(source_sentence)
        reference = strip_boundary_tokens(target_sentence)
        prediction, _, _, _ = translate(
            model,
            source_text,
            source_vocab,
            target_vocab,
            max_target_length=max_target_length,
        )
        rows.append(
            {
                "source": source_text,
                "reference": reference,
                "prediction": prediction,
                "exact": float(prediction == reference),
                "token_f1": token_f1(prediction, reference),
            }
        )

    metrics = {
        "exact_match": sum(row["exact"] for row in rows) / max(len(rows), 1),
        "token_f1": sum(row["token_f1"] for row in rows) / max(len(rows), 1),
    }
    return rows, metrics


def write_eval_report(
    path: Path,
    args: argparse.Namespace,
    train_losses: list[float],
    val_losses: list[float],
    metrics: dict[str, float],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Seq2Seq Attention 训练评估报告",
        "",
        "## 配置",
        "",
        f"- 数据量：`{args.num_examples}`",
        f"- epochs：`{args.epochs}`",
        f"- batch size：`{args.batch_size}`",
        f"- embedding dim：`{args.embedding_dim}`",
        f"- hidden size：`{args.hidden_size}`",
        f"- learning rate：`{args.learning_rate}`",
        f"- teacher forcing ratio：`{args.teacher_forcing_ratio}`",
        f"- 方向：`{'English -> Spanish' if args.english_to_spanish else 'Spanish -> English'}`",
        "",
        "## Loss",
        "",
        "验证 loss 使用 `teacher_forcing_ratio=0.0` 计算，条件更接近推理时的自回归生成；它会比 teacher-forced loss 更严格。",
        "",
        "| Epoch | Train Loss | Val Loss |",
        "| --- | ---: | ---: |",
    ]
    for idx, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
        lines.append(f"| {idx} | {train_loss:.4f} | {val_loss:.4f} |")

    lines.extend(
        [
            "",
            "## 推理指标",
            "",
            f"- 样例数：`{len(rows)}`",
            f"- Exact match：`{metrics['exact_match']:.4f}`",
            f"- Token F1：`{metrics['token_f1']:.4f}`",
            "",
            "## 推理样例",
            "",
            "| # | Source | Reference | Prediction | Token F1 |",
            "| ---: | --- | --- | --- | ---: |",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        source = row["source"].replace("|", "\\|")
        reference = row["reference"].replace("|", "\\|")
        prediction = row["prediction"].replace("|", "\\|")
        lines.append(
            f"| {idx} | {source} | {reference} | {prediction} | {row['token_f1']:.4f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_attention(
    attention: torch.Tensor,
    source_tokens: list[str],
    predicted_tokens: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.matshow(attention.numpy(), cmap="viridis")
    ax.set_xticks(range(len(source_tokens)))
    ax.set_yticks(range(len(predicted_tokens)))
    ax.set_xticklabels(source_tokens, rotation=90)
    ax.set_yticklabels(predicted_tokens)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def print_shape_demo() -> None:
    batch_size, input_length, hidden_size, embedding_dim = 2, 5, 8, 6
    print("Shape demo for one decoder step")
    print(f"source tokens                    (B, T_in)       = ({batch_size}, {input_length})")
    print(f"encoder_outputs / EO             (B, T_in, H)    = ({batch_size}, {input_length}, {hidden_size})")
    print(f"decoder hidden / H_t             (1, B, H)       = (1, {batch_size}, {hidden_size})")
    print(f"query after permute              (B, 1, H)       = ({batch_size}, 1, {hidden_size})")
    print(f"W1(EO) + W2(query) broadcasts to (B, T_in, H)    = ({batch_size}, {input_length}, {hidden_size})")
    print(f"V(tanh(...)) score               (B, T_in, 1)    = ({batch_size}, {input_length}, 1)")
    print(f"softmax(score, dim=1)            (B, T_in, 1)    = ({batch_size}, {input_length}, 1)")
    print(f"sum(attn * EO, dim=1) context    (B, H)          = ({batch_size}, {hidden_size})")
    print(f"target token embedding           (B, 1, E)       = ({batch_size}, 1, {embedding_dim})")
    print(f"concat(context, embedding)       (B, 1, H + E)   = ({batch_size}, 1, {hidden_size + embedding_dim})")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data_spa_en/spa.txt"))
    parser.add_argument("--num-examples", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=Path("seq2seq_attention_pytorch.pt"))
    parser.add_argument("--attention-plot", type=Path, default=Path("attention.png"))
    parser.add_argument("--eval-output", type=Path, default=Path("seq2seq_attention_eval.md"))
    parser.add_argument("--eval-examples", type=int, default=12)
    parser.add_argument("--translate", type=str, default="hace mucho frio aqui.")
    parser.add_argument("--english-to-spanish", action="store_true")
    parser.add_argument("--shape-demo", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.shape_demo:
        print_shape_demo()
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = args.data_path
    if not data_path.exists():
        data_path = Path(__file__).resolve().parent / args.data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {args.data_path}")

    source, target = load_sentence_pairs(
        data_path,
        args.num_examples,
        reverse=not args.english_to_spanish,
    )
    train_source, train_target, val_source, val_target = split_pairs(source, target, seed=args.seed)
    source_vocab = Vocabulary.build(train_source)
    target_vocab = Vocabulary.build(train_target)
    train_dataset = TranslationDataset(train_source, train_target, source_vocab, target_vocab)
    val_dataset = TranslationDataset(val_source, val_target, source_vocab, target_vocab)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        drop_last=False,
    )

    max_source_length = max(len(s.split()) for s in source)
    max_target_length = max(len(s.split()) for s in target)
    print(f"device: {DEVICE}")
    print(f"train/val examples: {len(train_dataset)}/{len(val_dataset)}")
    print(f"source vocab: {len(source_vocab)}, target vocab: {len(target_vocab)}")
    print(f"max source length: {max_source_length}, max target length: {max_target_length}")

    model = Seq2SeqAttention(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            print_every=args.print_every,
        )
        val_loss = evaluate_loss(model, val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"epoch {epoch:2d} train_loss {train_loss:.4f} "
            f"val_loss {val_loss:.4f} time {time.time() - start:.1f}s"
        )

    checkpoint = {
        "model": model.state_dict(),
        "source_vocab": source_vocab,
        "target_vocab": target_vocab,
        "args": vars(args),
    }
    torch.save(checkpoint, args.checkpoint)
    print(f"saved checkpoint: {args.checkpoint}")

    result, source_tokens, predicted_tokens, attention = translate(
        model,
        args.translate,
        source_vocab,
        target_vocab,
        max_target_length=max_target_length,
    )
    print(f"input: {args.translate}")
    print(f"predicted: {result}")
    if len(predicted_tokens) > 0 and attention.numel() > 0:
        plot_attention(attention, source_tokens, predicted_tokens, args.attention_plot)
        print(f"saved attention plot: {args.attention_plot}")

    eval_rows, eval_metrics = evaluate_translations(
        model,
        val_source,
        val_target,
        source_vocab,
        target_vocab,
        max_target_length=max_target_length,
        limit=args.eval_examples,
    )
    write_eval_report(args.eval_output, args, train_losses, val_losses, eval_metrics, eval_rows)
    print(f"eval exact_match: {eval_metrics['exact_match']:.4f}")
    print(f"eval token_f1: {eval_metrics['token_f1']:.4f}")
    print(f"saved eval report: {args.eval_output}")


if __name__ == "__main__":
    main()
