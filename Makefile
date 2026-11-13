.PHONY: install lint format test download

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
