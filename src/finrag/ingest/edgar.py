"""Download 10-K filings from SEC EDGAR.

Flow: ticker -> CIK (SEC's ticker map) -> company filing history (submissions API)
-> pick one 10-K per requested fiscal year -> download the primary HTML document.

Design notes:
* Pure functions (``parse_filing_table``, ``select_annual_reports``) hold the logic, so
  it is unit-tested without network access. ``SecClient`` only does I/O.
* Idempotent: files already on disk are skipped, so re-runs are cheap and safe.
* A ``manifest.json`` records the provenance of every document (CIK, accession number,
  dates, source URL). Later stages read metadata from it, and answers can cite it.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finrag.config import CorpusConfig, get_settings, load_corpus_config
from finrag.ingest.sec_client import SecClient

logger = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: int
    form: str
    accession_number: str
    filing_date: str  # YYYY-MM-DD
    report_date: str  # YYYY-MM-DD, end of the fiscal period covered
    primary_document: str

    @property
    def fiscal_year(self) -> int:
        """Calendar year in which the fiscal period ends."""
        return int(self.report_date[:4])

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik,
            accession=self.accession_number.replace("-", ""),
            document=self.primary_document,
        )

    def local_path(self, raw_dir: Path) -> Path:
        suffix = Path(self.primary_document).suffix or ".htm"
        return raw_dir / self.ticker / f"{self.ticker}_FY{self.fiscal_year}_{self.form}{suffix}"


# --------------------------------------------------------------------------- pure logic


def parse_filing_table(table: dict[str, list[Any]], ticker: str, cik: int) -> list[Filing]:
    """Turn EDGAR's column-oriented filing table into ``Filing`` rows.

    EDGAR returns parallel arrays (``form[i]``, ``accessionNumber[i]``, ...), not a list
    of records.
    """
    filings = []
    for form, acc, filed, report, doc in zip(
        table.get("form", []),
        table.get("accessionNumber", []),
        table.get("filingDate", []),
        table.get("reportDate", []),
        table.get("primaryDocument", []),
        strict=False,
    ):
        if not (report and doc):  # some filing types have no period or document
            continue
        filings.append(Filing(ticker, cik, form, acc, filed, report, doc))
    return filings


def select_annual_reports(
    filings: list[Filing], fiscal_years: list[int], form_type: str = "10-K"
) -> dict[int, Filing]:
    """Pick one original filing per fiscal year.

    Exact form match excludes amendments (``10-K/A``), which usually contain only a few
    changed items and would add near-duplicate text to the index. If a year somehow has
    several originals, keep the most recently filed.
    """
    wanted = set(fiscal_years)
    chosen: dict[int, Filing] = {}
    for f in filings:
        if f.form != form_type or f.fiscal_year not in wanted:
            continue
        current = chosen.get(f.fiscal_year)
        if current is None or f.filing_date > current.filing_date:
            chosen[f.fiscal_year] = f
    return chosen


# --------------------------------------------------------------------------------- I/O


def iter_filing_tables(client: SecClient, cik: int) -> Iterator[dict[str, list[Any]]]:
    """Yield the recent-filings table, then older paginated tables, lazily.

    "recent" holds only about the last 1,000 filings. Banks file thousands of prospectus
    supplements a year, so their older 10-Ks are only in the paginated files. Laziness
    means we stop fetching once we have what we need.
    """
    data = client.get_json(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
    yield data["filings"]["recent"]
    for page in data["filings"].get("files", []):
        yield client.get_json(SUBMISSIONS_URL.format(name=page["name"]))


def find_annual_reports(
    client: SecClient, ticker: str, cik: int, corpus: CorpusConfig
) -> dict[int, Filing]:
    found: dict[int, Filing] = {}
    for table in iter_filing_tables(client, cik):
        filings = parse_filing_table(table, ticker, cik)
        for year, filing in select_annual_reports(
            filings, corpus.fiscal_years, corpus.form_type
        ).items():
            found.setdefault(year, filing)  # newer tables come first
        if len(found) == len(corpus.fiscal_years):
            break
    return found


def download_corpus(corpus: CorpusConfig, raw_dir: Path, user_agent: str) -> list[dict]:
    client = SecClient(user_agent)
    ciks = client.ticker_to_cik()
    manifest: list[dict] = []

    for ticker in corpus.tickers:
        cik = ciks.get(ticker)
        if cik is None:
            logger.error("%s: ticker not found in SEC ticker map, skipping", ticker)
            continue

        reports = find_annual_reports(client, ticker, cik, corpus)
        missing = sorted(set(corpus.fiscal_years) - reports.keys())
        if missing:
            logger.warning("%s: no %s found for fiscal years %s", ticker, corpus.form_type, missing)

        for year in sorted(reports):
            filing = reports[year]
            path = filing.local_path(raw_dir)
            if path.exists():
                logger.info("%s FY%d: already downloaded", ticker, year)
            else:
                logger.info("%s FY%d: downloading %s", ticker, year, filing.url)
                client.download(filing.url, path)
            manifest.append(
                {
                    **asdict(filing),
                    "fiscal_year": year,
                    "url": filing.url,
                    "path": str(path.relative_to(raw_dir)),
                }
            )

    manifest_path = raw_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %d filings to %s", len(manifest), manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Download 10-K filings from SEC EDGAR.")
    parser.add_argument("--config", type=Path, default=settings.corpus_config_path)
    parser.add_argument("--out", type=Path, default=settings.raw_dir)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_corpus(load_corpus_config(args.config), args.out, settings.require_sec_user_agent())


if __name__ == "__main__":
    main()
