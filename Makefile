SHELL := /bin/bash

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: help env setup install install-pdf index reindex worker warmup dev up test docker-up docker-index

help:
	@printf '%s\n' \
		'Common commands:' \
		'  make setup        Create .venv, install dependencies, create .env if missing' \
		'  make up           One-command local startup: setup, auto-index, serve' \
		'  make dev          Start the API/UI without rebuilding the index' \
		'  make index        Rebuild the local RAG index' \
		'  make worker       Process one queued async ingestion job' \
		'  make warmup       Warm up embedding and reranker models' \
		'  make install-pdf  Install optional PDF table extraction dependencies' \
		'  make test         Run Python and JS tests' \
		'  make docker-up    Start the Docker Compose API service' \
		'' \
		'Examples:' \
		'  make up' \
		'  PORT=8001 make dev'

env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		printf '%s\n' 'Created .env from .env.example. Edit OPENAI_API_KEY before running the app.'; \
	else \
		printf '%s\n' '.env already exists'; \
	fi

setup: env install

install:
	@if [ ! -x "$(PYTHON)" ]; then \
		printf '%s\n' 'Creating virtual environment in $(VENV)'; \
		python3 -m venv "$(VENV)"; \
	fi
	@$(PIP) install --upgrade pip
	@$(PIP) install -e ".[dev]"

install-pdf: install
	@$(PIP) install -e ".[pdf-tables]"

index: setup
	@$(PYTHON) -m scripts.ingest

reindex: index

worker: setup
	@$(PYTHON) -m scripts.worker --once

warmup: setup
	@$(PYTHON) -m scripts.warmup

dev:
	@HOST="$(HOST)" PORT="$(PORT)" ./scripts/dev.sh --skip-index

up:
	@HOST="$(HOST)" PORT="$(PORT)" ./scripts/dev.sh

test: setup
	@$(VENV)/bin/pytest -q
	@node --test tests/js/ui-utils.test.mjs

docker-up:
	@docker compose up --build api

docker-index:
	@docker compose run --rm api python -m scripts.ingest
