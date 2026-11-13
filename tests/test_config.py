import pytest

from finrag.config import PROJECT_ROOT, Settings, load_corpus_config


def test_repo_corpus_config_is_valid():
    cfg = load_corpus_config(PROJECT_ROOT / "config" / "corpus.yaml")
    assert cfg.tickers and cfg.fiscal_years


def test_tickers_normalized_and_deduped(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("fiscal_years: [2024]\ntickers: [aapl, ' msft ']\n")
    assert load_corpus_config(p).tickers == ["AAPL", "MSFT"]
    p.write_text("fiscal_years: [2024]\ntickers: [aapl, AAPL]\n")
    with pytest.raises(ValueError):
        load_corpus_config(p)


def test_secret_is_masked():
    s = Settings(_env_file=None, groq_api_key="gsk_secret")
    assert "gsk_secret" not in repr(s)


@pytest.mark.parametrize("ua", [None, "", "Jane Doe", "Jane jane@example.com"])
def test_sec_user_agent_requires_real_email(ua):
    with pytest.raises(ValueError):
        Settings(_env_file=None, sec_user_agent=ua).require_sec_user_agent()
