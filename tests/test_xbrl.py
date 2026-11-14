import sqlite3

from finrag.ingest.xbrl import METRICS, build_database, extract_annual_facts


def fact(val, end, start=None, form="10-K", filed="2025-02-01", accn="A"):
    e = {"val": val, "end": end, "form": form, "filed": filed, "accn": accn}
    if start:
        e["start"] = start
    return e


def doc(**concepts):
    """Build a minimal company-facts document: concept=[entries] (USD unit)."""
    return {"facts": {"us-gaap": {c: {"units": {"USD": es}} for c, es in concepts.items()}}}


def by_key(rows):
    return {(r.metric, r.fiscal_year): r for r in rows}


def test_comparatives_keyed_by_period_and_latest_filing_wins():
    # FY2024 revenue appears in the FY2024 10-K and again (restated) in the FY2025 10-K.
    d = doc(
        Revenues=[
            fact(100, "2024-12-31", "2024-01-01", filed="2025-02-01"),
            fact(105, "2024-12-31", "2024-01-01", filed="2026-02-01"),
            fact(120, "2025-12-31", "2025-01-01", filed="2026-02-01"),
        ]
    )
    rows = by_key(extract_annual_facts(d, "X", [2024, 2025]))
    assert rows[("revenue", 2024)].value == 105
    assert rows[("revenue", 2025)].value == 120


def test_quarterly_and_non_annual_forms_are_excluded():
    d = doc(
        Revenues=[
            fact(30, "2024-12-31", "2024-10-01"),  # a quarter reported inside a 10-K
            fact(25, "2024-09-30", "2024-01-01", form="10-Q"),
            fact(100, "2024-12-31", "2024-01-01"),
        ]
    )
    rows = extract_annual_facts(d, "X", [2024])
    assert [(r.metric, r.value) for r in rows] == [("revenue", 100)]


def test_concept_priority_and_provenance():
    d = doc(
        Revenues=[fact(100, "2024-12-31", "2024-01-01")],
        RevenueFromContractWithCustomerExcludingAssessedTax=[fact(90, "2024-12-31", "2024-01-01")],
    )
    row = by_key(extract_annual_facts(d, "X", [2024]))[("revenue", 2024)]
    assert row.value == 100 and row.xbrl_concept == "Revenues"


def test_instants_only_on_fiscal_year_ends():
    d = doc(
        Revenues=[fact(100, "2024-12-31", "2024-01-01")],
        Assets=[fact(500, "2024-12-31"), fact(480, "2024-06-30")],
    )
    rows = extract_annual_facts(d, "X", [2024])
    assets = [r for r in rows if r.metric == "total_assets"]
    assert [(a.period_end, a.value) for a in assets] == [("2024-12-31", 500)]


def test_years_outside_range_dropped():
    d = doc(Revenues=[fact(1, "2019-12-31", "2019-01-01"), fact(2, "2024-12-31", "2024-01-01")])
    assert {r.fiscal_year for r in extract_annual_facts(d, "X", [2024])} == {2024}


def test_build_database_roundtrip(tmp_path):
    d = doc(Revenues=[fact(100, "2024-12-31", "2024-01-01")])
    rows = extract_annual_facts(d, "X", [2024])
    db = tmp_path / "f.db"
    build_database(db, [{"ticker": "X", "cik": 1, "name": "X Corp", "sector": "Tech"}], rows)
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT value FROM financials WHERE ticker='X' AND metric='revenue'"
        ).fetchone() == (100.0,)
        assert conn.execute("SELECT COUNT(*) FROM metrics").fetchone() == (len(METRICS),)
