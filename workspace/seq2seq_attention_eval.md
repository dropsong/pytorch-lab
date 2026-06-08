# Seq2Seq Attention 训练评估报告

## 配置

- 数据量：`30000`
- epochs：`10`
- batch size：`64`
- embedding dim：`256`
- hidden size：`1024`
- learning rate：`0.001`
- teacher forcing ratio：`1.0`
- 方向：`Spanish -> English`

## Loss

验证 loss 使用 `teacher_forcing_ratio=0.0` 计算，条件更接近推理时的自回归生成；它会比 teacher-forced loss 更严格。

| Epoch | Train Loss | Val Loss |
| --- | ---: | ---: |
| 1 | 2.5672 | 3.3508 |
| 2 | 1.1438 | 2.8969 |
| 3 | 0.5454 | 2.8470 |
| 4 | 0.2825 | 2.8568 |
| 5 | 0.1808 | 2.9333 |
| 6 | 0.1377 | 3.0318 |
| 7 | 0.1206 | 3.2111 |
| 8 | 0.1156 | 3.1156 |
| 9 | 0.1119 | 3.3226 |
| 10 | 0.1145 | 3.3696 |

## 推理指标

- 样例数：`20`
- Exact match：`0.2000`
- Token F1：`0.7337`

## 推理样例

| # | Source | Reference | Prediction | Token F1 |
| ---: | --- | --- | --- | ---: |
| 1 | ella cocina muy bien . | she cooks very well . | she cooks very well . | 1.0000 |
| 2 | aprendi a cocinar . | i ve learned to cook . | i ll cook to cook . | 0.6667 |
| 3 | ¿ era cierta su historia ? | was her story true ? | was his story true ? | 0.8000 |
| 4 | tom tomo un gran riesgo . | tom took a big risk . | tom took a big man . | 0.8333 |
| 5 | yo se lo que es aquello . | i know what that is . | i know what s that . | 0.8333 |
| 6 | vamos a divertirnos mucho . | we will have much fun . | we ll go very much . | 0.5000 |
| 7 | tom es intolerante . | tom is intolerant . | tom is a slight . | 0.6667 |
| 8 | te lo dejo a ti . | i m leaving it to you . | i ll thank you . | 0.5000 |
| 9 | lo envidio . | i envy him . | i envy him . | 1.0000 |
| 10 | la noche estaba fresca . | the night was cool . | the night was cold . | 0.8000 |
| 11 | se quedo ciega . | she went blind . | he went blind . | 0.7500 |
| 12 | este libro es muy nuevo . | this book is very new . | this book is very new . | 1.0000 |
| 13 | a ella le gusta mucho la tarta . | she really likes cake . | she likes him a cake . | 0.7273 |
| 14 | es mi trabajo . | it s my job . | it s work . | 0.6667 |
| 15 | necesito encontrar mi lapicera . | i need to find my pen . | i need to find my pen . | 1.0000 |
| 16 | quiero a mi mama . | i want my mommy . | i want my mom . | 0.8000 |
| 17 | ¿ que has encontrado ? | what have you found ? | what ve you ve ? | 0.6000 |
| 18 | esperad un segundo . | wait one second . | hold on a second . | 0.4444 |
| 19 | mira con atencion . | watch closely . | look at us . | 0.2857 |
| 20 | el cafe me mantiene despierta . | coffee keeps me awake . | coffee keeps me mad . | 0.8000 |
