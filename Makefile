.PHONY: install dev up down test test-unit test-integration test-smoke test-layer1 test-layer2 test-layer3 test-rag test-dspy test-cov lint format seed-neo4j seed-chroma setup-dbs setup-and-test optimize-dspy serve

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

up:
	docker compose up -d

down:
	docker compose down

# Testing
test:
	pytest tests/ -v --cov=graphocr

test-unit:
	pytest tests/ -v -m unit

test-integration:
	pytest tests/ -v -m integration

test-smoke:
	pytest tests/ -v -m smoke

test-layer1:
	pytest tests/ -v -m layer1

test-layer2:
	pytest tests/ -v -m layer2

test-layer3:
	pytest tests/ -v -m layer3

test-rag:
	pytest tests/ -v -m rag

test-dspy:
	pytest tests/ -v -m dspy

test-cov:
	pytest tests/ -v --cov=graphocr --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

seed-neo4j:
	python scripts/seed_neo4j.py

seed-chroma:
	python scripts/ingest_policies.py --policy-dir tests/fixtures/sample_policies/

setup-dbs:
	python scripts/setup_dbs.py

setup-and-test: setup-dbs test

optimize-dspy:
	python scripts/optimize_dspy.py

serve:
	uvicorn graphocr.app:app --reload --host 0.0.0.0 --port 8080
