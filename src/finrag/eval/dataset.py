"""The evaluation question set: schema, loading, and verification against the sources.

Gold evidence is stored as verbatim text from the filing, not as chunk IDs. Chunk IDs
change with every chunking config, but the text does not, so one question set can score
any index fairly: a retrieval "hit" is a retrieved chunk from the right filing that
contains the evidence text.

``finrag-eval-check`` verifies the set itself: every evidence quote must appear in its
filing, and every numeric lookup must match the XBRL table. The gold answers are checked
against the sources, not just typed in.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from finrag.config import PROJECT_ROOT, get_settings

DEFAULT_QUESTIONS = PROJECT_ROOT / "eval" / "questions.yaml"

Category = Literal[
    "fact", "numeric_lookup", "numeric_calc", "comparison", "structured", "unanswerable"
]


class Evidence(BaseModel):
    ticker: str
    fiscal_year: int
    text: str


class XbrlRef(BaseModel):
    ticker: str
    fiscal_year: int
    metric: str


class Question(BaseModel):
    id: str
    category: Category
    question: str
    answer: str
    value: float | None = None
    unit: str | None = None
    tolerance: float = 0.01  # relative tolerance for numeric grading
    evidence: list[Evidence] = Field(default_factory=list)
    xbrl: list[XbrlRef] = Field(default_factory=list)
    notes: str | None = None

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        questions = [Question.model_validate(q) for q in yaml.safe_load(f)]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return questions


def normalize(text: str) -> str:
    """Whitespace- and quote-insensitive form used for evidence matching."""
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


# ------------------------------------------------------------------------- verification


def verify(questions: list[Question], processed_dir: Path, db_path: Path) -> list[str]:
    """Return a list of problems; empty means the question set is consistent."""
    docs: dict[tuple[str, int], str] = {}
    for path in processed_dir.glob("*.json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        docs[(d["ticker"], d["fiscal_year"])] = normalize(
            "\n".join(s["text"] for s in d["sections"])
        )
    conn = sqlite3.connect(db_path)
    problems = []

    for q in questions:
        if q.answerable and not (q.evidence or q.xbrl):
            problems.append(f"{q.id}: answerable but has no evidence or xbrl reference")
        if not q.answerable and (q.evidence or q.value is not None):
            problems.append(f"{q.id}: unanswerable question should have no evidence or value")
        for ev in q.evidence:
            doc = docs.get((ev.ticker, ev.fiscal_year))
            if doc is None:
                problems.append(f"{q.id}: no filing {ev.ticker} FY{ev.fiscal_year}")
            elif normalize(ev.text) not in doc:
                problems.append(f"{q.id}: evidence not found in {ev.ticker} FY{ev.fiscal_year}: "
                                f"{ev.text[:70]!r}")  # fmt: skip
        for ref in q.xbrl:
            row = conn.execute(
                "SELECT 1 FROM financials WHERE ticker=? AND fiscal_year=? AND metric=?",
                (ref.ticker, ref.fiscal_year, ref.metric),
            ).fetchone()
            if row is None:
                problems.append(
                    f"{q.id}: no XBRL value {ref.ticker} FY{ref.fiscal_year} {ref.metric}"
                )
        # Single-reference lookups must equal the table value (or the as-reported value).
        if (
            q.category in ("numeric_lookup", "structured")
            and len(q.xbrl) == 1
            and q.value is not None
        ):
            ref = q.xbrl[0]
            v, orig = conn.execute(
                "SELECT value, original_value FROM financials "
                "WHERE ticker=? AND fiscal_year=? AND metric=?",
                (ref.ticker, ref.fiscal_year, ref.metric),
            ).fetchone()
            scale = 1 if q.unit == "USD" else 1e6
            if not any(
                abs(q.value - c / scale) <= max(abs(c / scale) * 1e-4, 0.01) for c in (v, orig)
            ):
                problems.append(
                    f"{q.id}: value {q.value} != XBRL {v / scale} (orig {orig / scale})"
                )
    return problems


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Verify the eval question set against the sources.")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = p.parse_args(argv)
    questions = load_questions(args.questions)
    problems = verify(questions, settings.data_dir / "processed", settings.financials_db_path)
    for prob in problems:
        print("FAIL", prob)
    print(f"{len(questions)} questions, {len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
