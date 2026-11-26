"""Typed configuration.

Two kinds of config, kept separate on purpose:

* ``Settings``: environment-specific values and secrets. Loaded from environment
  variables, falling back to a git-ignored ``.env`` file. Secrets are ``SecretStr`` so
  they are masked in logs, reprs and tracebacks.
* ``CorpusConfig``: *what* we index. A committed YAML file, because it is part of the
  experiment definition and must be reproducible from the repo alone.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class Settings(BaseSettings):
    """Runtime settings. Env var names are the upper-cased field names."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: SecretStr | None = None
    # Groq retired its Llama 3.x models in Aug 2026; gpt-oss-120b is the recommended
    # replacement. Kept in config so the model can be swapped without code changes.
    llm_model: str = "openai/gpt-oss-120b"
    sec_user_agent: str | None = None
    data_dir: Path = PROJECT_ROOT / "data"
    # Where a fresh deployment downloads the prebuilt index (see finrag.data_bundle).
    # Not secret: it is a public GitHub Release asset. The checksum pins the exact file.
    data_bundle_url: str | None = None
    data_bundle_sha256: str | None = None
    corpus_config_path: Path = PROJECT_ROOT / "config" / "corpus.yaml"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def financials_db_path(self) -> Path:
        return self.data_dir / "financials.db"

    def require_sec_user_agent(self) -> str:
        """SEC blocks requests without a contact email in the User-Agent; fail early."""
        ua = (self.sec_user_agent or "").strip()
        if not _EMAIL_RE.search(ua) or "example.com" in ua:
            raise ValueError(
                "Set SEC_USER_AGENT in .env to 'Your Name your.email@domain' "
                "(SEC requires a real contact email)."
            )
        return ua


class Company(BaseModel):
    ticker: str
    sector: str

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class CorpusConfig(BaseModel):
    """Which companies and years make up the knowledge base."""

    form_type: str = "10-K"
    fiscal_years: list[int] = Field(min_length=1)
    financials_years: list[int] | None = None  # defaults to fiscal_years
    companies: list[Company] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> CorpusConfig:
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("duplicate tickers in corpus config")
        if self.financials_years is None:
            self.financials_years = list(self.fiscal_years)
        return self

    @property
    def tickers(self) -> list[str]:
        return [c.ticker for c in self.companies]


def load_corpus_config(path: Path) -> CorpusConfig:
    with path.open(encoding="utf-8") as f:
        return CorpusConfig.model_validate(yaml.safe_load(f))


@lru_cache
def get_settings() -> Settings:
    """Cached so every module shares one Settings instance."""
    return Settings()
