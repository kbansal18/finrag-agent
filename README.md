# finrag-agent

Agentic RAG over SEC 10-K filings, with a reproducible evaluation harness.

> Work in progress. A full README with architecture, eval results, and demo link is coming.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env        # then fill in GROQ_API_KEY and SEC_USER_AGENT
make data                   # 10-K filings + XBRL financials table (config/corpus.yaml)
make test
```
