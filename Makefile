.PHONY: install lint format test download financials parse index data

install:
	pip install -e ".[dev]"

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

test:
	pytest -q

download:
	finrag-download

financials:
	finrag-financials

parse:
	finrag-parse

index:
	finrag-index

data: download financials parse index
