from pathlib import Path

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


def test_project_root_editable_install_regular_install_and_override(tmp_path):
    from finrag.config import find_project_root

    repo = tmp_path / "repo"
    (repo / "src" / "finrag").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("")
    editable = repo / "src" / "finrag" / "config.py"
    assert find_project_root(editable, tmp_path / "elsewhere", None) == repo

    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "finrag" / "config.py"
    site.parent.mkdir(parents=True)
    assert find_project_root(site, repo, None) == repo  # falls back to the app's cwd

    assert find_project_root(site, repo, "/custom") == Path("/custom")
