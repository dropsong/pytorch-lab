.PHONY: build up logs shell test-gpu down

COMPOSE := docker compose
SERVICE := dl-lab

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d --build

logs:
	$(COMPOSE) logs -f $(SERVICE)

shell:
	$(COMPOSE) exec $(SERVICE) bash

test-gpu:
	$(COMPOSE) exec $(SERVICE) python test_gpu.py

down:
	$(COMPOSE) down
