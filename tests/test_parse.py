from finrag.ingest.parse import PREAMBLE, html_to_text, split_sections


def items(sections):
    return [s.item for s in sections]


def test_tables_are_rows_with_currency_and_negatives_glued():
    html = """<table>
      <tr><td>Net sales</td><td>$</td><td>416,161</td><td></td></tr>
      <tr><td>Other income</td><td>(</td><td>321</td><td>)</td></tr>
      <tr><td>Margin</td><td>46.9</td><td>%</td></tr>
    </table>"""
    text = html_to_text(html)
    assert "Net sales | $416,161" in text
    assert "Margin | 46.9%" in text
    assert "( 321 )" not in text


def test_inline_tags_do_not_split_lines_and_hidden_xbrl_dropped():
    html = """<div style="display:none"><ix:header>dei:Hidden 123</ix:header></div>
      <p>See <a href="#x">Item 8</a> of this <span>report</span>.</p>
      <div>Apple Inc. | 2025 Form 10-K | 29</div><div>42</div>"""
    assert html_to_text(html) == "See Item 8 of this report."


def _doc(lines: list[str]) -> str:
    return "\n".join(lines)


BODY = "x " * 400  # enough text that a real section dwarfs a TOC entry


def test_toc_skipped_and_body_headings_found():
    toc = [
        f"Item {i} | Title | {n}"
        for n, i in enumerate(
            ["1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "15"]
        )
    ]
    body = []
    for i in ["1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "15"]:
        body += [f"Item {i}. Heading", BODY]
    secs = split_sections(_doc(["Cover page", *toc, *body]))
    assert items(secs)[0] == PREAMBLE
    assert "Item 1 | Title" in secs[0].text  # TOC stays in the preamble
    assert secs[1].item == "1" and secs[1].text.startswith("Item 1. Heading")


def test_stray_cross_reference_and_page_headers_ignored():
    text = _doc(
        [
            "Item 1. Business", BODY,
            "Item 1A. Risk Factors", BODY,
            "Item 1A", BODY,  # running page header
            "Item 8", "of this report has more detail.",  # stray reference
            "Item 7. MD&A", BODY,
            "Item 8. Financial Statements", BODY,
        ]
    )  # fmt: skip
    secs = split_sections(text)
    assert items(secs) == ["0", "1", "1A", "7", "8"] or items(secs) == ["1", "1A", "7", "8"]
    risk = next(s for s in secs if s.item == "1A")
    assert risk.text.count("Item 1A") == 2  # page header kept inside, not a new section


def test_spaced_item_letter():
    secs = split_sections(_doc(["Item 1. Business", BODY, "Item 1 C . Cybersecurity", BODY]))
    assert "1C" in items(secs)


def test_part_iii_one_liners_survive():
    lines = ["Item 9A. Controls", BODY]
    for i in ["10", "11", "12", "13", "14"]:
        lines.append(f"Item {i}. Incorporated by reference to the proxy statement.")
    lines += ["Item 15. Exhibits", BODY]
    assert items(split_sections(_doc(lines)))[-6:] == ["10", "11", "12", "13", "14", "15"]


def test_cross_referenced_items_relocated():
    text = _doc(
        [
            "Item 1. Business", BODY,
            "Item 7. MD&A", "See pages 50-150.",
            "Item 8. Financial Statements", "See pages 151-300.",
            "Item 15. Exhibits", "Exhibit list",
            "Management's discussion and analysis", "Revenue grew. " * 50,
            "Report of Independent Registered Public Accounting Firm", "We audited. " * 50,
        ]
    )  # fmt: skip
    secs = split_sections(text)
    mdna = [s for s in secs if s.item == "7"]
    assert any(s.text.startswith("Management's discussion") for s in mdna)
    fs = [s for s in secs if s.item == "8"]
    assert any(s.text.startswith("Report of Independent") for s in fs)
    exhibits = next(s for s in secs if s.item == "15")
    assert "Revenue grew" not in exhibits.text
