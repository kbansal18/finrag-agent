import json
import sqlite3

from finrag.eval.dataset import DEFAULT_QUESTIONS, Question, load_questions, verify


def _sources(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "AAPL_FY2025.json").write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "fiscal_year": 2025,
                "sections": [{"item": "1", "title": "Business", "text": "The Company’s\nsegments"}],
            }
        )
    )
    db = tmp_path / "f.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE financials (ticker, fiscal_year, metric, value, original_value)")
        c.execute("INSERT INTO financials VALUES ('AAPL', 2025, 'revenue', 416161e6, 416161e6)")
    return proc, db


def q(**kw):
    base = {"id": "x", "category": "numeric_lookup", "question": "?", "answer": "a"}
    return Question.model_validate(base | kw)


def test_good_question_passes(tmp_path):
    proc, db = _sources(tmp_path)
    good = q(
        value=416161,
        evidence=[{"ticker": "AAPL", "fiscal_year": 2025, "text": "the company's  segments"}],
        xbrl=[{"ticker": "AAPL", "fiscal_year": 2025, "metric": "revenue"}],
    )
    assert verify([good], proc, db) == []  # quotes and whitespace normalized


def test_bad_evidence_value_and_missing_support_caught(tmp_path):
    proc, db = _sources(tmp_path)
    bad = [
        q(id="a", evidence=[{"ticker": "AAPL", "fiscal_year": 2025, "text": "not in the filing"}]),
        q(id="b", value=999, xbrl=[{"ticker": "AAPL", "fiscal_year": 2025, "metric": "revenue"}]),
        q(id="c"),
        q(id="d", category="unanswerable", value=1),
    ]
    problems = verify(bad, proc, db)
    assert [p.split(":")[0] for p in problems] == ["a", "b", "c", "d"]


def test_repo_question_set_loads():
    qs = load_questions(DEFAULT_QUESTIONS)
    assert 50 <= len(qs) <= 100
    assert sum(not x.answerable for x in qs) >= 10  # enough to measure hallucination
