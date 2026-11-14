"""Thin, polite HTTP client for SEC endpoints, shared by every ingestion step."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


class SecClient:
    """Throttled below SEC's 10 req/s limit, with retry and backoff on 429/5xx."""

    MIN_INTERVAL_S = 0.15  # about 7 req/s

    def __init__(self, user_agent: str, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        retry = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))
        self._last_request = 0.0

    def get(self, url: str) -> requests.Response:
        wait = self.MIN_INTERVAL_S - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        resp = self._session.get(url, timeout=60)
        resp.raise_for_status()
        return resp

    def get_json(self, url: str) -> Any:
        return self.get(url).json()

    def download(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(self.get(url).content)
        tmp.replace(dest)  # atomic: a crash never leaves a half-written file behind

    def company_index(self) -> dict[str, tuple[int, str]]:
        """ticker -> (CIK, registrant name), from SEC's official ticker map."""
        data = self.get_json(TICKER_MAP_URL)
        return {row["ticker"].upper(): (int(row["cik_str"]), row["title"]) for row in data.values()}

    def ticker_to_cik(self) -> dict[str, int]:
        return {t: cik for t, (cik, _) in self.company_index().items()}
