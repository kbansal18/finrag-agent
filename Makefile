.PHONY: install lint format test download financials data

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

data: download financials
