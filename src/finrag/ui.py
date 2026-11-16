"""Presentation helpers for the Streamlit app. Pure functions, so they are unit-tested."""

from __future__ import annotations

import contextlib
import csv
import json
import re
from pathlib import Path
from typing import Any

from finrag.agent.agent import Step

_CITE = re.compile(r"\[([ST]\d+)\]")


def describe_step(step: Step) -> str:
    """One readable line per tool call, e.g. 'Searched AAPL FY2025, Item 7: "iPhone units"'."""
    try:
        args: dict[str, Any] = json.loads(step.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if step.tool == "search_filings":
        scope = " ".join(args.get("tickers") or ["all companies"])
        if args.get("fiscal_years"):
            scope += " " + ", ".join(f"FY{y}" for y in args["fiscal_years"])
        if args.get("items"):
            scope += ", Item " + ", ".join(args["items"])
        return f'Searched {scope}: "{args.get("query", "")}"'
    if step.tool == "query_financials":
        return f"Queried XBRL financials: `{args.get('sql', '')}`"
    if step.tool == "calculate":
        try:
            result = json.loads(step.result_preview).get("result")
        except (json.JSONDecodeError, AttributeError):
            result = None
        expr = args.get("expression", "")
        return (
            f"Calculated `{expr}` = {result:,.4g}" if result is not None else f"Calculated `{expr}`"
        )
    return f"{step.tool}({step.arguments})"


def answer_markdown(text: str) -> str:
    """Make an answer safe and readable in Streamlit markdown.

    * Escape "$": Streamlit renders $...$ as LaTeX, so "$416 billion ... $10 billion"
      would turn into a math formula.
    * Show citations as small badges.
    """
    text = text.replace("$", r"\$")
    return _CITE.sub(r" :blue-badge[\1]", text)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            with contextlib.suppress(TypeError, ValueError):  # text columns stay text
                r[k] = float(v) if v not in ("", "None") else None
    return rows


def retrieval_ladder(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ablation ladder: each step adds one change to the one before."""
    ladder = [
        ("fixed", "dense", "none", "Fixed chunks, dense"),
        ("section", "dense", "none", "+ section-aware chunks"),
        ("section", "hybrid", "none", "+ hybrid (BM25 + dense)"),
        ("section", "hybrid", "oracle", "+ company/year filters"),
    ]
    out = []
    for index, mode, filters, label in ladder:
        row = next(
            (r for r in rows if (r["index"], r["mode"], r["filters"], r["category"])
             == (index, mode, filters, "ALL")),
            None,
        )  # fmt: skip
        if row:
            out.append({"step": label, "recall@10": row["recall@10"], "MRR": row["mrr"]})
    return out
