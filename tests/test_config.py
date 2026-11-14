import pytest

from finrag.config import PROJECT_ROOT, Settings, load_corpus_config


def test_repo_corpus_config_is_valid():
    cfg = load_corpus_config(PROJECT_ROOT / "config" / "corpus.yaml")
    assert cfg.tickers and cfg.fiscal_years


def test_tickers_normalized_deduped_and_financials_default(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "fiscal_years: [2024]\ncompanies:\n"
        "  - {ticker: aapl, sector: Tech}\n  - {ticker: ' msft ', sector: Tech}\n"
    )
    cfg = load_corpus_config(p)
    assert cfg.tickers == ["AAPL", "MSFT"]
    assert cfg.financials_years == [2024]
    p.write_text(
        "fiscal_years: [2024]\ncompanies:\n"
        "  - {ticker: aapl, sector: Tech}\n  - {ticker: AAPL, sector: Tech}\n"
    )
    with pytest.raises(ValueError):
        load_corpus_config(p)


def test_secret_is_masked():
    s = Settings(_env_file=None, groq_api_key="gsk_secret")
    assert "gsk_secret" not in repr(s)


@pytest.mark.parametrize("ua", [None, "", "Jane Doe", "Jane jane@example.com"])
def test_sec_user_agent_requires_real_email(ua):
    with pytest.raises(ValueError):
        Settings(_env_file=None, sec_user_agent=ua).require_sec_user_agent()
