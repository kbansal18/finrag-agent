from finrag.ingest.edgar import Filing, parse_filing_table, select_annual_reports

TABLE = {
    "form": ["10-K", "8-K", "10-K/A", "10-K", "424B2"],
    "accessionNumber": ["0001-24-1", "0001-24-2", "0001-24-3", "0001-23-1", "0001-23-2"],
    "filingDate": ["2024-11-01", "2024-08-01", "2024-12-15", "2023-11-03", "2023-05-01"],
    "reportDate": ["2024-09-28", "2024-08-01", "2024-09-28", "2023-09-30", ""],
    "primaryDocument": ["aapl-2024.htm", "x.htm", "a.htm", "aapl-2023.htm", "p.htm"],
}


def test_parse_skips_rows_without_report_date():
    filings = parse_filing_table(TABLE, "AAPL", 320193)
    assert len(filings) == 4
    assert all(f.report_date for f in filings)


def test_select_excludes_amendments_and_other_forms():
    filings = parse_filing_table(TABLE, "AAPL", 320193)
    chosen = select_annual_reports(filings, [2023, 2024])
    assert {y: f.accession_number for y, f in chosen.items()} == {
        2024: "0001-24-1",
        2023: "0001-23-1",
    }


def test_select_keeps_latest_filed_original():
    a = Filing("X", 1, "10-K", "a", "2024-01-01", "2023-12-31", "a.htm")
    b = Filing("X", 1, "10-K", "b", "2024-02-01", "2023-12-31", "b.htm")
    assert select_annual_reports([a, b], [2023])[2023].accession_number == "b"


def test_url_and_local_path(tmp_path):
    f = Filing(
        "AAPL", 320193, "10-K", "0000320193-24-000123", "2024-11-01", "2024-09-28", "aapl.htm"
    )
    assert f.url.endswith("/320193/000032019324000123/aapl.htm")
    assert f.local_path(tmp_path) == tmp_path / "AAPL" / "AAPL_FY2024_10-K.htm"
