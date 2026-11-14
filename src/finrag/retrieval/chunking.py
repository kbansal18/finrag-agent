"""Two chunking strategies, compared head to head in the evaluation.

``fixed`` (baseline): slide a window of N tokens with overlap over the whole filing,
ignoring structure. This is what most RAG tutorials do.

``section`` (structure-aware):
* Never crosses an Item boundary, so a chunk is never half Risk Factors, half Properties.
* Packs whole paragraphs and table rows; it only cuts mid-sentence when one sentence is
  longer than a chunk.
* When a table must be split, its first row (usually the column headers, e.g. fiscal
  years) is repeated in every piece, so "391,035" still says which year it belongs to.
* Starts a new chunk at a sub-heading once the current chunk is half full.
* Prepends a contextual header to the text that gets embedded and keyword-indexed:
  ``Apple Inc. (AAPL) | FY2025 Form 10-K | Item 8: Financial Statements | CONSOLIDATED
  STATEMENTS OF OPERATIONS``. Without it, "Revenue increased 8%" in ten companies' MD&A
  look nearly identical to the retriever.

The contextual header is a separate flag, so its effect can be measured on its own.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from finrag.retrieval.tokenizer import Tokenizer

PREAMBLE = "0"  # cover page + table of contents: excluded from the index


@dataclass(frozen=True)
class ChunkingConfig:
    name: str
    strategy: Literal["fixed", "section"]
    max_tokens: int = 384  # + header stays under bge's 512-token input limit
    overlap_tokens: int = 64  # fixed strategy only
    contextual_header: bool = False


CHUNKING_CONFIGS: dict[str, ChunkingConfig] = {
    "fixed": ChunkingConfig("fixed", "fixed"),
    "section": ChunkingConfig("section", "section", contextual_header=True),
}


@dataclass
class Chunk:
    chunk_id: str
    ticker: str
    company: str
    fiscal_year: int
    item: str
    item_title: str
    subheading: str
    text: str  # what the LLM reads and the UI shows
    embed_text: str  # what is embedded and keyword-indexed (text, maybe with header)
    n_tokens: int
    url: str


# ------------------------------------------------------------------------------ helpers


def _is_table_line(line: str) -> bool:
    return " | " in line or line.endswith(" |")


# A value cell is a bare number: "$391,035", "(321)", "15.7%", "8". Years ("2025") and
# dates ("December 31, 2025") are not values; they are what header rows are made of.
_NUMBER_CELL = re.compile(r"[$(]?-?\$?[\d,]*\.?\d+%?\)?(?: \([a-z]\))*")
_YEAR_CELL = re.compile(r"(19|20)\d{2}")


def _is_value_cell(cell: str) -> bool:
    cell = cell.strip()
    return bool(_NUMBER_CELL.fullmatch(cell)) and not _YEAR_CELL.fullmatch(cell)


def _table_header(rows: list[str], max_rows: int = 3) -> list[str]:
    """Column-header rows: the multi-cell rows before the first row with values.

    Financial tables often have two header rows ("Standardized | Advanced" above
    "December 31, 2025 | December 31, 2024"); both are needed to read a value.
    """
    header = []
    for row in rows:
        cells = row.split(" | ")
        if any(_is_value_cell(c) for c in cells[1:]):
            break
        if len(cells) > 1 and len(header) < max_rows:
            header.append(row)
    return header


def _is_heading(line: str) -> bool:
    return (
        len(line) < 100
        and not _is_table_line(line)
        and not line.rstrip().endswith((".", ",", ";", ":"))
        and sum(c.isalpha() for c in line) > len(line) / 2
    )


def _blocks(text: str) -> Iterator[tuple[str, list[str]]]:
    """Group lines into ('table', rows) and ('para', [line]) blocks."""
    table: list[str] = []
    for line in text.split("\n"):
        if _is_table_line(line):
            table.append(line)
            continue
        if table:
            yield "table", table
            table = []
        yield "para", [line]
    if table:
        yield "table", table


def _hard_split(text: str, max_tokens: int, tok: Tokenizer) -> list[str]:
    offs = tok.offsets(text)
    return [
        text[offs[i][0] : offs[min(i + max_tokens, len(offs)) - 1][1]]
        for i in range(0, len(offs), max_tokens)
    ]


def _split_paragraph(line: str, max_tokens: int, tok: Tokenizer) -> list[str]:
    if tok.count(line) <= max_tokens:
        return [line]
    pieces, buf = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", line):
        if tok.count(sent) > max_tokens:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(_hard_split(sent, max_tokens, tok))
        elif buf and tok.count(f"{buf} {sent}") > max_tokens:
            pieces.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def _split_table(rows: list[str], max_tokens: int, tok: Tokenizer) -> list[str]:
    text = "\n".join(rows)
    if tok.count(text) <= max_tokens:
        return [text]
    header = _table_header(rows)
    budget = max_tokens - tok.count("\n".join(header))
    if budget < max_tokens // 2:  # pathological header: don't repeat it
        header, budget = [], max_tokens
    body = [r for r in rows if r not in header]

    pieces: list[str] = []
    buf: list[str] = []
    used = 0

    def emit() -> None:
        if buf:
            pieces.append("\n".join(header + buf))

    for row in body:
        n = tok.count(row)
        if n > budget:  # one giant row
            emit()
            buf, used = [], 0
            pieces.extend(_hard_split(row, max_tokens, tok))
            continue
        if used + n > budget:
            emit()
            buf, used = [], 0
        buf.append(row)
        used += n
    emit()
    return pieces


def _header(f: dict, company: str, item: str, title: str, sub: str) -> str:
    parts = [f"{company} ({f['ticker']})", f"FY{f['fiscal_year']} Form 10-K"]
    parts.append(f"Item {item}: {title}")
    if sub:
        parts.append(sub)
    return " | ".join(parts)


# ---------------------------------------------------------------------------- strategies


class _Packer:
    """Accumulates units into one chunk until the token budget is reached."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.units: list[str] = []
        self.counts: list[int] = []
        self.heading_flags: list[bool] = []
        self.used = 0
        self.subheading = ""

    def would_overflow(self, n: int) -> bool:
        return bool(self.units) and self.used + n > self.max_tokens

    def add(self, unit: str, n: int, subheading: str, is_heading: bool = False) -> None:
        if not self.units:
            self.subheading = subheading  # a chunk is labelled by where it starts
        self.units.append(unit)
        self.counts.append(n)
        self.heading_flags.append(is_heading)
        self.used += n

    def pop_trailing_headings(self) -> list[tuple[str, int]]:
        """Remove headings at the end of the chunk; they belong with what follows."""
        out = []
        while self.heading_flags and self.heading_flags[-1]:
            self.heading_flags.pop()
            n = self.counts.pop()
            out.append((self.units.pop(), n))
            self.used -= n
        return out[::-1]

    def take(self) -> tuple[str, str] | None:
        if not self.units:
            return None
        out = (self.subheading, "\n".join(self.units))
        self.units, self.counts, self.heading_flags, self.used = [], [], [], 0
        return out


def _chunk_section_strategy(
    filing: dict, cfg: ChunkingConfig, tok: Tokenizer
) -> Iterator[tuple[str, str, str, str]]:
    """Yield (item, title, subheading, text)."""
    for sec in filing["sections"]:
        if sec["item"] == PREAMBLE:
            continue
        item, title = sec["item"], sec["title"]
        packer = _Packer(cfg.max_tokens)
        sub = ""
        for kind, lines in _blocks(sec["text"]):
            heading = kind == "para" and _is_heading(lines[0])
            if heading:
                sub = lines[0]
                if packer.used > cfg.max_tokens // 2 and (done := packer.take()):
                    yield item, title, *done
            units = (
                _split_table(lines, cfg.max_tokens, tok)
                if kind == "table"
                else _split_paragraph(lines[0], cfg.max_tokens, tok)
            )
            for unit in units:
                n = tok.count(unit)
                if packer.would_overflow(n):
                    carried = packer.pop_trailing_headings()
                    if done := packer.take():
                        yield item, title, *done
                    # Re-attach the heading if it fits; if not, it still survives as the
                    # chunk's subheading in the contextual header.
                    for h, hn in carried:
                        if hn + n <= cfg.max_tokens:
                            packer.add(h, hn, sub, is_heading=True)
                packer.add(unit, n, sub, is_heading=heading)
        if done := packer.take():
            yield item, title, *done


def _chunk_fixed_strategy(
    filing: dict, cfg: ChunkingConfig, tok: Tokenizer
) -> Iterator[tuple[str, str, str, str]]:
    secs = [s for s in filing["sections"] if s["item"] != PREAMBLE]
    doc, starts = "", []
    for s in secs:
        starts.append(len(doc))
        doc += s["text"] + "\n\n"
    offs = tok.offsets(doc)
    step = cfg.max_tokens - cfg.overlap_tokens
    for i in range(0, len(offs), step):
        j = min(i + cfg.max_tokens, len(offs))
        start, end = offs[i][0], offs[j - 1][1]
        sec = secs[bisect.bisect_right(starts, start) - 1]  # section where the chunk starts
        yield sec["item"], sec["title"], "", doc[start:end].strip()
        if j == len(offs):
            break


def chunk_filing(filing: dict, company: str, cfg: ChunkingConfig, tok: Tokenizer) -> list[Chunk]:
    strategy = _chunk_section_strategy if cfg.strategy == "section" else _chunk_fixed_strategy
    chunks = []
    for item, title, sub, text in strategy(filing, cfg, tok):
        if not text.strip():
            continue
        embed_text = (
            f"{_header(filing, company, item, title, sub)}\n{text}"
            if cfg.contextual_header
            else text
        )
        chunks.append(
            Chunk(
                chunk_id=f"{filing['ticker']}_FY{filing['fiscal_year']}_{len(chunks):05d}",
                ticker=filing["ticker"],
                company=company,
                fiscal_year=filing["fiscal_year"],
                item=item,
                item_title=title,
                subheading=sub,
                text=text,
                embed_text=embed_text,
                n_tokens=tok.count(text),
                url=filing["url"],
            )
        )
    return chunks
