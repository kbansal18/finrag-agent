"""Grading answers: a deterministic number check plus an LLM judge.

Two independent graders, on purpose:

* ``numeric_match`` (no LLM): pull every number out of the answer, apply its scale word
  (million, billion, %...), and check whether any is within tolerance of the gold value.
  Cheap and exact, but only defined for questions with a numeric gold value.
* ``judge`` (LLM): compares the answer with the gold answer and with the source passages
  the answer cited. It decides correctness, abstention ("not available"), and whether
  the answer makes claims its sources do not support (hallucination).

The judge is a different model family from the systems under test (Qwen judging
gpt-oss), which reduces self-preference bias. Where both graders apply, their agreement
rate is reported: it is one check on whether the judge can be trusted. The other is a
human-labelled sample (see ``finrag.eval.answers``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from finrag.agent.llm import LLM
from finrag.eval.dataset import Question

UNIT_SCALE = {
    "USD": 1.0,
    "USD millions": 1e6,
    "USD billions": 1e9,
    "USD trillions": 1e12,
    "million associates": 1e6,
}
WORD_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_NUMBER = re.compile(
    r"(?<![\w.])\$?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(thousand|million|billion|trillion|[mbk]n?\b)?",
    re.IGNORECASE,
)
_ABBREV = {"k": "thousand", "m": "million", "mn": "million", "b": "billion", "bn": "billion"}


def extract_numbers(text: str) -> list[float]:
    """Every number in the text, with its scale word applied ("$416.2 billion" -> 4.162e11)."""
    out = []
    for raw, word in _NUMBER.findall(text):
        value = float(raw.replace(",", ""))
        word = _ABBREV.get(word.lower(), word.lower()) if word else ""
        out.append(value * WORD_SCALE.get(word, 1.0))
    return out


def numeric_match(question: Question, answer: str) -> bool | None:
    """True/False if the question has a numeric gold value, else None."""
    if question.value is None or not question.answerable:
        return None
    scale = UNIT_SCALE.get(question.unit or "", 1.0)
    gold = abs(question.value) * scale
    tol = question.tolerance
    for n in extract_numbers(answer):
        # A bare "416,161" in an answer about USD millions is in millions too.
        for candidate in (n, n * scale):
            if gold == 0 and candidate == 0:
                return True
            if gold and abs(abs(candidate) - gold) / gold <= tol:
                return True
    return False


# ------------------------------------------------------------------------------- judge

JUDGE_SYSTEM = """\
You grade answers from a question-answering system over SEC 10-K filings. Be strict and
consistent. Respond with a single JSON object and nothing else."""

JUDGE_TEMPLATE = """\
QUESTION:
{question}

GOLD ANSWER (written by a human from the filings):
{gold}
{gold_value}
CATEGORY: {category}
{unanswerable_note}

SYSTEM ANSWER:
{answer}

SOURCES THE SYSTEM ANSWER CITED (may be empty):
{sources}

Grade on three criteria:
1. "correct": does the system answer agree with the gold answer on the key facts and
   numbers? Minor rounding is fine (e.g. $416.2 billion for $416,161 million; 6.4% for
   6.43%). For a split-adjusted or restatement question, accept either figure if the
   answer explains it. If the gold says the question is unanswerable, "correct" means the
   system declined to give the specific fact.
2. "abstained": did the system say the information is not available / not disclosed
   instead of giving a substantive answer?
3. "unsupported_claims": list factual claims in the system answer (numbers, names,
   dates, facts) that are NOT supported by the cited sources above. Claims that merely
   restate the question or say information is unavailable are not factual claims. If
   the answer cites no sources but states facts, list those facts.

Return JSON exactly in this shape:
{{"correct": true, "abstained": false, "unsupported_claims": [], "reason": "one sentence"}}"""


@dataclass
class Judgment:
    correct: bool
    abstained: bool
    unsupported_claims: list[str] = field(default_factory=list)
    reason: str = ""
    raw: str = ""

    @property
    def hallucinated(self) -> bool:
        return bool(self.unsupported_claims)


def _parse_json_object(text: str) -> dict[str, Any]:
    """Models sometimes wrap JSON in prose or code fences; take the outermost object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def judge(
    llm: LLM,
    question: Question,
    answer: str,
    cited_sources: list[str],
    max_source_chars: int = 2000,
) -> Judgment:
    sources = "\n\n".join(s[:max_source_chars] for s in cited_sources) or "(none)"
    gold_value = (
        f"GOLD VALUE: {question.value} {question.unit or ''} (tolerance {question.tolerance:.0%})\n"
        if question.value is not None
        else ""
    )
    prompt = JUDGE_TEMPLATE.format(
        question=question.question,
        gold=question.answer,
        gold_value=gold_value,
        category=question.category,
        unanswerable_note=(
            "This question is NOT answerable from the corpus; the ideal answer declines."
            if not question.answerable
            else ""
        ),
        answer=answer or "(empty)",
        sources=sources,
    )
    resp = llm.chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}]
    )
    data = _parse_json_object(resp.content)
    return Judgment(
        correct=bool(data.get("correct")),
        abstained=bool(data.get("abstained")),
        unsupported_claims=[str(c) for c in data.get("unsupported_claims") or []],
        reason=str(data.get("reason", "")),
        raw=resp.content,
    )
