.PHONY: install lint format test download financials parse data

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

data: download financials parse
