"""Parse 10-K HTML into clean text, split into the standard Item sections.

Pipeline: HTML -> text blocks (tables kept as pipe-delimited rows) -> find Item headings
-> sections -> one JSON file per filing in ``data/processed``.

Design notes:
* Text extraction is block-aware: newlines only at block elements (div, p, tr, ...).
  BeautifulSoup's default ``get_text("\\n")`` breaks at *every* tag, which splits
  "see Item 8 of this report" across lines and creates false headings.
* Tables are rendered row by row as ``cell | cell | cell``. Financial statements are
  tables, and flattening them into a stream of numbers would destroy the row/column
  association the LLM needs to read a value correctly.
* Finding the real Item headings is a sequence problem. Every 10-K lists the Items in
  its table of contents, some repeat them as page headers, and cross-references put
  stray "Item 8" lines in the body. We (1) drop table-of-contents clusters, then
  (2) keep the longest subsequence of candidates in strictly increasing Item order.
  Stray references break the order, so they fall out of the longest sequence. Ties are
  broken by how much text follows each heading before the next different Item: a TOC
  entry is followed by ~30 characters, the real "Item 1. Business" by ~150,000. This also
  catches TOCs too spread out for step (1).
* Some filers (JPM; NVIDIA for financial statements) leave Item 7 or 8 as a one-line
  pointer and put the content later, after Item 15. Those sections are cut out of the
  host section at their real heading and labelled correctly, so citations and section
  filters stay right.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

from finrag.config import get_settings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Standard 10-K items, in document order, with canonical titles.
ITEMS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity and Related Stockholder Matters",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
ITEM_ORDER = {k: i for i, k in enumerate(ITEMS)}
PREAMBLE = "0"  # cover page and table of contents, before Item 1

BLOCK_TAGS = {
    "address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "ol", "p", "section", "ul",
}  # fmt: skip

HEADING_RE = re.compile(r"^item\s*(\d{1,2})\s?([a-c])?\b\s*[.:\-–—]?", re.IGNORECASE)
MAX_HEADING_LEN = 250
TOC_GAP_CHARS = 300  # TOC entries sit closer together than this
TOC_MIN_RUN = 10  # a real TOC lists ~20 Items; Part III's 5 one-liners must survive
STUB_MAX_CHARS = 2000  # an Item shorter than this may just point elsewhere

# Where cross-referenced content really starts (first matching short line).
RELOCATION_HEADINGS = {
    "7": re.compile(r"management['’]s discussion and analysis", re.IGNORECASE),
    "8": re.compile(r"report of independent registered public accounting firm", re.IGNORECASE),
}


@dataclass
class Section:
    item: str
    title: str
    text: str


@dataclass
class ParsedFiling:
    ticker: str
    fiscal_year: int
    form: str
    period_end: str
    filing_date: str
    url: str
    sections: list[Section] = field(default_factory=list)


# ------------------------------------------------------------------------ HTML -> text


def _cell_text(cell: Tag) -> str:
    return " ".join(cell.get_text(" ").split())


def _table_to_text(table: Tag) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [c for c in (_cell_text(td) for td in tr.find_all(["td", "th"])) if c]
        if cells:
            rows.append(cells)
    # Layout tables (one column, used for positioning text) are rendered as plain lines.
    # In data tables, one-cell rows ("Risk-based capital metrics:") get a trailing " |"
    # so they stay part of the table instead of ending it.
    is_data_table = any(len(r) > 1 for r in rows)
    out = []
    for cells in rows:
        row = " | ".join(cells)
        # 10-K tables put "$", ")", "%" and footnote markers like "(b)" in their own
        # cells. Glue them back on, or every later value shifts one column right.
        row = re.sub(r"\$ \| ", "$", row)
        row = re.sub(r" \| (\)|%)", r"\1", row)
        row = re.sub(r" \| ((?:\([a-z]\))+)(?= \||$)", r" \1", row)
        row = re.sub(r"\(\s+([\d.,]+)\s+\)", r"(\1)", row)  # "( 321 )" -> "(321)"
        if is_data_table and len(cells) == 1:
            row += " |"
        out.append(row)
    return "\n".join(out)


def html_to_text(html: bytes | str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()

    for table in soup.find_all("table"):
        if table.find_parent("table") is None:  # outermost tables only
            table.replace_with(NavigableString(f"\n{_table_to_text(table)}\n"))
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = soup.get_text("").replace("\xa0", " ")
    lines = []
    for raw in text.split("\n"):
        line = " ".join(raw.split())
        # Drop page furniture: bare page numbers, "Table of Contents" back-links, and
        # running footers like "Apple Inc. | 2025 Form 10-K | 29".
        if not line or re.fullmatch(r"\d{1,3}|table of contents", line, re.IGNORECASE):
            continue
        if len(line) < 80 and re.search(r"form 10-k\s*\|\s*\d{1,3}$", line, re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------- sectioning


def _heading_candidates(lines: list[str]) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(lines):
        if len(line) > MAX_HEADING_LEN:
            continue
        m = HEADING_RE.match(line)
        if m:
            item = m.group(1) + (m.group(2) or "").upper()
            if item in ITEM_ORDER:
                out.append((i, item))
    return out


def _drop_toc_clusters(cands: list[tuple[int, str]], offsets: list[int]) -> list[tuple[int, str]]:
    """Remove table-of-contents runs.

    A run is a sequence of candidates in increasing Item order, each within TOC_GAP_CHARS
    of the previous one. The order condition matters: the body's "Item 1" often follows
    the TOC immediately, and the reset from "Item 16" back to "Item 1" ends the run.
    """
    keep = [True] * len(cands)
    run_start = 0
    for i in range(1, len(cands) + 1):
        end_of_run = (
            i == len(cands)
            or offsets[cands[i][0]] - offsets[cands[i - 1][0]] > TOC_GAP_CHARS
            or ITEM_ORDER[cands[i][1]] <= ITEM_ORDER[cands[i - 1][1]]
        )
        if end_of_run:
            if i - run_start >= TOC_MIN_RUN:
                for j in range(run_start, i):
                    keep[j] = False
            run_start = i
    return [c for c, k in zip(cands, keep, strict=True) if k]


def _longest_increasing(
    cands: list[tuple[int, str]], offsets: list[int], total_chars: int
) -> list[tuple[int, str]]:
    """Longest strictly-increasing-Item subsequence; ties go to the most text covered.

    O(n^2) dynamic programme; n is at most a few hundred.
    """
    n = len(cands)
    if n == 0:
        return []
    # Text between a candidate and the next candidate with a *different* item. Using the
    # next different item means repeated page headers ("Item 1A" on every page) give the
    # first occurrence the whole section, so it wins over later repeats.
    weight = []
    for i, (line, item) in enumerate(cands):
        nxt = next((cands[j][0] for j in range(i + 1, n) if cands[j][1] != item), None)
        weight.append((offsets[nxt] if nxt is not None else total_chars) - offsets[line])

    score = [(1, weight[i]) for i in range(n)]
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if ITEM_ORDER[cands[j][1]] < ITEM_ORDER[cands[i][1]]:
                cand = (score[j][0] + 1, score[j][1] + weight[i])
                if cand > score[i]:
                    score[i], prev[i] = cand, j
    i = max(range(n), key=lambda k: score[k])
    seq = []
    while i != -1:
        seq.append(cands[i])
        i = prev[i]
    return seq[::-1]


def split_sections(text: str) -> list[Section]:
    lines = text.split("\n")
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    candidates = _drop_toc_clusters(_heading_candidates(lines), offsets)
    heads = _longest_increasing(candidates, offsets, len(text))
    if not heads:
        return [Section(PREAMBLE, "Full document", text)]

    sections = [Section(PREAMBLE, "Cover page and table of contents",
                        "\n".join(lines[: heads[0][0]]))]  # fmt: skip
    for k, (start, item) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        sections.append(Section(item, ITEMS[item], "\n".join(lines[start:end])))
    sections = _relocate_cross_referenced(sections)
    return [s for s in sections if s.text.strip()]


def _relocate_cross_referenced(sections: list[Section]) -> list[Section]:
    """Move Item 7/8 content that a stub points to back under its own Item label."""
    stubs = {
        s.item for s in sections if s.item in RELOCATION_HEADINGS and len(s.text) < STUB_MAX_CHARS
    }
    # host section index -> [(line number, item)]
    cuts: dict[int, list[tuple[int, str]]] = {}
    for item in stubs:
        stub_idx = next(i for i, s in enumerate(sections) if s.item == item)
        pattern = RELOCATION_HEADINGS[item]
        for host in range(stub_idx + 1, len(sections)):
            lines = sections[host].text.split("\n")
            line_no = next(
                (n for n, ln in enumerate(lines) if n > 0 and pattern.fullmatch(ln.rstrip(" .:"))),
                None,
            )
            if line_no is not None:
                cuts.setdefault(host, []).append((line_no, item))
                break

    out: list[Section] = []
    for i, sec in enumerate(sections):
        if i not in cuts:
            out.append(sec)
            continue
        lines = sec.text.split("\n")
        bounds = sorted(cuts[i])
        out.append(Section(sec.item, sec.title, "\n".join(lines[: bounds[0][0]])))
        for k, (start, item) in enumerate(bounds):
            end = bounds[k + 1][0] if k + 1 < len(bounds) else len(lines)
            out.append(Section(item, ITEMS[item], "\n".join(lines[start:end])))
    return out


# --------------------------------------------------------------------------------- I/O


def parse_corpus(raw_dir: Path, out_dir: Path) -> list[ParsedFiling]:
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = []
    for entry in manifest:
        html = (raw_dir / entry["path"]).read_bytes()
        filing = ParsedFiling(
            ticker=entry["ticker"],
            fiscal_year=entry["fiscal_year"],
            form=entry["form"],
            period_end=entry["report_date"],
            filing_date=entry["filing_date"],
            url=entry["url"],
            sections=split_sections(html_to_text(html)),
        )
        out = out_dir / f"{filing.ticker}_FY{filing.fiscal_year}.json"
        out.write_text(json.dumps(asdict(filing), indent=1), encoding="utf-8")
        found = [s.item for s in filing.sections if s.item != PREAMBLE]
        missing = [i for i in ("1", "1A", "7", "7A", "8") if i not in found]
        logger.info(
            "%s FY%d: %d sections, %d chars%s",
            filing.ticker,
            filing.fiscal_year,
            len(found),
            sum(len(s.text) for s in filing.sections),
            f"; MISSING key items {missing}" if missing else "",
        )
        parsed.append(filing)
    return parsed


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Parse downloaded 10-Ks into sections.")
    parser.add_argument("--raw", type=Path, default=settings.raw_dir)
    parser.add_argument("--out", type=Path, default=settings.data_dir / "processed")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parse_corpus(args.raw, args.out)


if __name__ == "__main__":
    main()
