"""The agent's three tools, and their safety rails.

* ``search_filings``: the hybrid retriever with metadata filters. Every chunk returned
  gets a source id (S1, S2, ...) that the final answer must cite.
* ``query_financials``: read-only SQL over the XBRL table. Exact figures come from here
  rather than being read off a chunked table. Guard rails: a read-only connection,
  SELECT/WITH only, one statement, a row cap and a time limit. Model-written SQL is
  untrusted input.
* ``calculate``: exact arithmetic through a tiny AST evaluator (never ``eval``). LLMs
  are unreliable at multi-digit arithmetic, and a margin of 45.62% vs 46.52% is exactly
  the error a numeric eval catches.

Tool errors are returned to the model as ``{"error": ...}`` rather than raised, so it
can correct a bad SQL query or filter and try again.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finrag.retrieval.retriever import Retriever

MAX_SQL_ROWS = 50
SQL_TIMEOUT_S = 2.0
SEARCH_K = 5
# Each agent step resends the whole conversation, so tool output is the main token cost.
# Passages already shown are sent again as an id only, and JSON is compact.

# --------------------------------------------------------------------------- calculator

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}  # fmt: skip
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt}


def calculate(expression: str) -> float:
    """Evaluate arithmetic safely: numbers, + - * / ** % //, and a few functions."""

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*(ev(a) for a in node.args))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    expression = expression.replace(",", "") if re.search(r"\d,\d{3}", expression) else expression
    result = ev(ast.parse(expression, mode="eval"))
    return float(result)


# --------------------------------------------------------------------------------- SQL

_SQL_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def run_readonly_sql(db_path: Path, sql: str) -> dict[str, Any]:
    statement = sql.strip().rstrip(";").strip()
    if ";" in statement:
        raise ValueError("only a single SQL statement is allowed")
    if not _SQL_START.match(statement):
        raise ValueError("only SELECT (or WITH ... SELECT) queries are allowed")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        deadline = time.monotonic() + SQL_TIMEOUT_S
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        cur = conn.execute(statement)
        columns = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(MAX_SQL_ROWS + 1)
    finally:
        conn.close()
    return {
        "columns": columns,
        "rows": [list(r) for r in rows[:MAX_SQL_ROWS]],
        "truncated": len(rows) > MAX_SQL_ROWS,
    }


def financials_schema(db_path: Path) -> str:
    """Schema + metric catalogue, embedded in the tool description for the model."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        companies = ", ".join(
            f"{t} ({n}, {s})"
            for t, n, s in conn.execute("SELECT ticker, name, sector FROM companies")
        )
        metrics = "\n".join(
            f"  - {m} [{u}, {k}]: {d}"
            for m, d, u, k in conn.execute("SELECT metric, description, unit, kind FROM metrics")
        )
        years = [
            y for (y,) in conn.execute("SELECT DISTINCT fiscal_year FROM financials ORDER BY 1")
        ]
    finally:
        conn.close()
    return (
        "SQLite tables:\n"
        "  companies(ticker, cik, name, sector)\n"
        "  metrics(metric, description, unit, kind)\n"
        "  financials(ticker, fiscal_year, period_end, metric, value, original_value, unit, "
        "xbrl_concept, accession, filed)\n"
        "`value` is in raw units (USD, not millions; EPS in USD/share). `value` is the latest "
        "reported figure (restated or split-adjusted); `original_value` is as first reported "
        "in that year's own 10-K.\n"
        f"Fiscal years available: {years[0]}-{years[-1]}. A missing row means the company does "
        "not report that metric.\n"
        f"Companies: {companies}\nMetrics:\n{metrics}"
    )


# ------------------------------------------------------------------------------ toolbox


@dataclass
class Source:
    source_id: str
    kind: str  # "filing" or "financials"
    label: str  # human-readable, e.g. "AAPL FY2025 10-K, Item 7"
    text: str
    url: str = ""
    chunk_id: str = ""


@dataclass
class Toolbox:
    """Executes tool calls and keeps every source the model has seen, by id."""

    retriever: Retriever
    db_path: Path
    sources: dict[str, Source] = field(default_factory=dict)
    _n_search: int = 0
    _n_table: int = 0

    def specs(self) -> list[dict[str, Any]]:
        tickers = sorted({c.ticker for c in self.retriever.chunks})
        years = sorted({c.fiscal_year for c in self.retriever.chunks})
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_filings",
                    "description": (
                        "Search the text of 10-K filings (narrative and tables). Returns "
                        "passages with source ids to cite. Always pass tickers and "
                        "fiscal_years when the question names them; for questions about "
                        "several companies, call once per company. "
                        f"Companies: {', '.join(tickers)}. Fiscal years with text: {years}."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to look for."},
                            "tickers": {"type": "array", "items": {"type": "string"}},
                            "fiscal_years": {"type": "array", "items": {"type": "integer"}},
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "10-K Items, e.g. ['1A'] risk factors, "
                                "['7'] MD&A, ['8'] financial statements.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_financials",
                    "description": (
                        "Run one read-only SQL SELECT against standardized annual financial "
                        "data from XBRL. Use it for exact figures and multi-year trends.\n"
                        + financials_schema(self.db_path)
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": (
                        "Evaluate an arithmetic expression exactly, e.g. "
                        "'(416161 / 391035 - 1) * 100'. Use for every calculation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments or "{}")
            if name == "search_filings":
                result: Any = self._search(**args)
            elif name == "query_financials":
                result = self._sql(args["sql"])
            elif name == "calculate":
                result = {"result": calculate(args["expression"])}
            else:
                raise ValueError(f"unknown tool {name!r}")
        except Exception as exc:  # returned to the model so it can recover
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, default=str, separators=(",", ":"), ensure_ascii=False)

    def _search(
        self,
        query: str,
        tickers: list[str] | None = None,
        fiscal_years: list[int] | None = None,
        items: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        results = self.retriever.search(
            query, k=SEARCH_K, tickers=tickers, fiscal_years=fiscal_years, items=items
        )
        out = []
        for r in results:
            c = r.chunk
            existing = next((s for s in self.sources.values() if s.chunk_id == c.chunk_id), None)
            if existing is not None:
                out.append({"source_id": existing.source_id, "note": "already shown above"})
                continue
            self._n_search += 1
            sid = f"S{self._n_search}"
            label = f"{c.ticker} FY{c.fiscal_year} 10-K, Item {c.item}: {c.item_title}"
            self.sources[sid] = Source(sid, "filing", label, c.text, c.url, c.chunk_id)
            out.append(
                {"source_id": sid, "ticker": c.ticker, "fiscal_year": c.fiscal_year,
                 "item": c.item, "section": c.subheading, "text": c.text}
            )  # fmt: skip
        if not out:
            return [{"note": "no passages match these filters"}]
        return out

    def _sql(self, sql: str) -> dict[str, Any]:
        result = run_readonly_sql(self.db_path, sql)
        self._n_table += 1
        sid = f"T{self._n_table}"
        preview = json.dumps({"sql": sql, **result}, default=str)
        self.sources[sid] = Source(sid, "financials", "XBRL financial data (SEC)", preview)
        return {"source_id": sid, **result}
