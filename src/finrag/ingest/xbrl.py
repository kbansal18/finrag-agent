"""Build a SQLite table of annual financial metrics from SEC XBRL "company facts".

Why a table as well as the 10-K text? Exact numbers ("What was NVDA's FY2025 revenue?")
are where RAG over prose is weakest: the figure may sit in a table that chunked badly,
and the LLM may copy the wrong column. A typed table answers those exactly. The agent
gets both tools and must decide which one fits the question.

Three XBRL pitfalls this module handles, each covered by a test:

1. Tag heterogeneity. Companies tag the same idea differently (Apple's revenue is
   ``RevenueFromContractWithCustomerExcludingAssessedTax``, banks often use ``Revenues``).
   Each metric lists candidate concepts in priority order; the concept actually used is
   stored, so every number is traceable.
2. Comparatives. A 10-K repeats prior years' figures, and the ``fy`` field on a fact is
   the fiscal year of the *filing*, not the period. We key facts on the period's end date
   instead, and when several filings report the same period we keep the latest filed
   (which picks up restatements).
3. Mixed durations. 10-Ks sometimes include quarterly facts. Flow metrics (revenue,
   cash flow) must span roughly one year; balance-sheet metrics must sit on a fiscal
   year-end date.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from finrag.config import CorpusConfig, get_settings, load_corpus_config
from finrag.ingest.sec_client import SecClient

logger = logging.getLogger(__name__)

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ANNUAL_FORMS = {"10-K", "10-K/A"}
ANNUAL_DAYS = range(350, 381)  # 52/53-week fiscal years land in here


@dataclass(frozen=True)
class MetricSpec:
    description: str
    concepts: tuple[str, ...]  # us-gaap concepts, highest priority first
    unit: str = "USD"
    kind: Literal["duration", "instant"] = "duration"


METRICS: dict[str, MetricSpec] = {
    "revenue": MetricSpec(
        "Total revenue / net sales for the fiscal year",
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenuesNetOfInterestExpense",
            "SalesRevenueNet",
        ),
    ),
    "cost_of_revenue": MetricSpec(
        "Cost of revenue / cost of sales",
        ("CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"),
    ),
    "gross_profit": MetricSpec("Gross profit", ("GrossProfit",)),
    "operating_income": MetricSpec("Operating income (loss)", ("OperatingIncomeLoss",)),
    "net_income": MetricSpec("Net income attributable to the company", ("NetIncomeLoss",)),
    "eps_diluted": MetricSpec(
        "Diluted earnings per share", ("EarningsPerShareDiluted",), unit="USD/shares"
    ),
    "rnd_expense": MetricSpec(
        "Research and development expense", ("ResearchAndDevelopmentExpense",)
    ),
    "sga_expense": MetricSpec(
        "Selling, general and administrative expense",
        ("SellingGeneralAndAdministrativeExpense",),
    ),
    "operating_cash_flow": MetricSpec(
        "Net cash provided by operating activities",
        ("NetCashProvidedByUsedInOperatingActivities",),
    ),
    "capex": MetricSpec(
        "Purchases of property, plant and equipment",
        ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
    ),
    "total_assets": MetricSpec("Total assets at fiscal year-end", ("Assets",), kind="instant"),
    "total_liabilities": MetricSpec(
        "Total liabilities at fiscal year-end", ("Liabilities",), kind="instant"
    ),
    "stockholders_equity": MetricSpec(
        "Total stockholders' equity at fiscal year-end", ("StockholdersEquity",), kind="instant"
    ),
    "cash": MetricSpec(
        "Cash and cash equivalents at fiscal year-end",
        ("CashAndCashEquivalentsAtCarryingValue",),
        kind="instant",
    ),
    "long_term_debt": MetricSpec(
        "Long-term debt (non-current) at fiscal year-end",
        ("LongTermDebtNoncurrent", "LongTermDebt"),
        kind="instant",
    ),
}


@dataclass(frozen=True)
class FactRow:
    ticker: str
    fiscal_year: int
    period_end: str
    metric: str
    value: float
    unit: str
    xbrl_concept: str
    accession: str
    filed: str


# --------------------------------------------------------------------------- pure logic


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _annual_entries(us_gaap: dict, spec: MetricSpec, concept: str) -> list[dict]:
    entries = us_gaap.get(concept, {}).get("units", {}).get(spec.unit, [])
    out = []
    for e in entries:
        if e.get("form") not in ANNUAL_FORMS or "end" not in e or "val" not in e:
            continue
        if spec.kind == "duration":
            if "start" not in e or _days(e["start"], e["end"]) not in ANNUAL_DAYS:
                continue
        elif "start" in e:
            continue
        out.append(e)
    return out


def extract_annual_facts(companyfacts: dict, ticker: str, years: list[int]) -> list[FactRow]:
    """Reduce a company-facts document to one value per (metric, fiscal year)."""
    us_gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    wanted = set(years)

    # Fiscal year-end dates, learned from annual duration facts. Balance-sheet values are
    # only accepted on these dates, which filters out odd interim instants.
    year_ends = {
        e["end"]
        for name, spec in METRICS.items()
        if spec.kind == "duration"
        for concept in spec.concepts
        for e in _annual_entries(us_gaap, spec, concept)
    }

    rows: list[FactRow] = []
    for metric, spec in METRICS.items():
        # period_end -> (concept priority, filed, entry, concept)
        best: dict[str, tuple[int, str, dict, str]] = {}
        for priority, concept in enumerate(spec.concepts):
            for e in _annual_entries(us_gaap, spec, concept):
                if spec.kind == "instant" and e["end"] not in year_ends:
                    continue
                cur = best.get(e["end"])
                better = (
                    cur is None
                    or priority < cur[0]
                    or (priority == cur[0] and e.get("filed", "") > cur[1])
                )
                if better:
                    best[e["end"]] = (priority, e.get("filed", ""), e, concept)

        # Fiscal year = calendar year of period end (same rule as the 10-K corpus).
        by_year: dict[int, tuple[str, dict, str]] = {}
        for end, (_, _, e, concept) in best.items():
            fy = int(end[:4])
            if fy in wanted and (fy not in by_year or end > by_year[fy][0]):
                by_year[fy] = (end, e, concept)

        for fy, (end, e, concept) in sorted(by_year.items()):
            rows.append(
                FactRow(
                    ticker=ticker,
                    fiscal_year=fy,
                    period_end=end,
                    metric=metric,
                    value=float(e["val"]),
                    unit=spec.unit,
                    xbrl_concept=concept,
                    accession=e.get("accn", ""),
                    filed=e.get("filed", ""),
                )
            )
    return rows


# ------------------------------------------------------------------------------ storage

SCHEMA = """
CREATE TABLE companies (
    ticker  TEXT PRIMARY KEY,
    cik     INTEGER NOT NULL,
    name    TEXT NOT NULL,
    sector  TEXT NOT NULL
);
CREATE TABLE metrics (
    metric       TEXT PRIMARY KEY,
    description  TEXT NOT NULL,
    unit         TEXT NOT NULL,
    kind         TEXT NOT NULL  -- 'duration' (over the year) or 'instant' (at year-end)
);
CREATE TABLE financials (
    ticker        TEXT    NOT NULL REFERENCES companies(ticker),
    fiscal_year   INTEGER NOT NULL,
    period_end    TEXT    NOT NULL,
    metric        TEXT    NOT NULL REFERENCES metrics(metric),
    value         REAL    NOT NULL,
    unit          TEXT    NOT NULL,
    xbrl_concept  TEXT    NOT NULL,
    accession     TEXT    NOT NULL,
    filed         TEXT    NOT NULL,
    PRIMARY KEY (ticker, fiscal_year, metric)
);
"""


def build_database(db_path: Path, companies: list[dict[str, Any]], facts: list[FactRow]) -> None:
    """Rebuild the database from scratch, atomically.

    Writing to a temp file and renaming means a reader (the running app) never sees a
    half-built database, and a rebuild never mixes old and new rows.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(tmp) as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO companies VALUES (:ticker, :cik, :name, :sector)", companies)
        conn.executemany(
            "INSERT INTO metrics VALUES (?, ?, ?, ?)",
            [(m, s.description, s.unit, s.kind) for m, s in METRICS.items()],
        )
        conn.executemany(
            "INSERT INTO financials VALUES "
            "(:ticker, :fiscal_year, :period_end, :metric, :value, :unit, "
            ":xbrl_concept, :accession, :filed)",
            [f.__dict__ for f in facts],
        )
    tmp.replace(db_path)


# --------------------------------------------------------------------------------- I/O


def build_financials(
    corpus: CorpusConfig, raw_dir: Path, db_path: Path, user_agent: str, refresh: bool = False
) -> int:
    client = SecClient(user_agent)
    ciks = client.ticker_to_cik()
    cache_dir = raw_dir / "xbrl"
    companies: list[dict[str, Any]] = []
    facts: list[FactRow] = []

    for company in corpus.companies:
        cik = ciks.get(company.ticker)
        if cik is None:
            logger.error("%s: ticker not found in SEC ticker map, skipping", company.ticker)
            continue

        # Raw JSON is cached so the table can be rebuilt (e.g. after adding a metric)
        # without hitting SEC again.
        cache = cache_dir / f"{company.ticker}.json"
        if refresh or not cache.exists():
            logger.info("%s: downloading company facts", company.ticker)
            client.download(COMPANYFACTS_URL.format(cik=cik), cache)
        doc = json.loads(cache.read_text(encoding="utf-8"))

        rows = extract_annual_facts(doc, company.ticker, corpus.financials_years or [])
        companies.append(
            {
                "ticker": company.ticker,
                "cik": cik,
                "name": doc.get("entityName", company.ticker),
                "sector": company.sector,
            }
        )
        facts.extend(rows)

        found = {r.metric for r in rows}
        missing = sorted(set(METRICS) - found)
        logger.info(
            "%s: %d facts; metrics not reported: %s",
            company.ticker,
            len(rows),
            ", ".join(missing) or "none",
        )

    build_database(db_path, companies, facts)
    logger.info("Wrote %d facts for %d companies to %s", len(facts), len(companies), db_path)
    return len(facts)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the XBRL financials SQLite table.")
    parser.add_argument("--config", type=Path, default=settings.corpus_config_path)
    parser.add_argument("--db", type=Path, default=settings.financials_db_path)
    parser.add_argument("--refresh", action="store_true", help="re-download cached JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_financials(
        load_corpus_config(args.config),
        settings.raw_dir,
        args.db,
        settings.require_sec_user_agent(),
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
