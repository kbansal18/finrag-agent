import json

from finrag.agent.agent import Step
from finrag.ui import answer_markdown, describe_step, retrieval_ladder


def step(tool, result="", **args):
    return Step(tool, json.dumps(args), result, 0.01)


def test_describe_steps():
    assert describe_step(step("search_filings", query="iPhone units", tickers=["AAPL"],
                              fiscal_years=[2025], items=["7"])) == (
        'Searched AAPL FY2025, Item 7: "iPhone units"'
    )  # fmt: skip
    assert describe_step(step("search_filings", query="CET1")) == 'Searched all companies: "CET1"'
    calc = step("calculate", result='{"result":10178000000.0}', expression="112010 - 101832")
    assert describe_step(calc) == "Calculated `112010 - 101832` = 1.018e+10"


def test_answer_markdown_escapes_dollars_and_badges_citations():
    out = answer_markdown("Apple earned $112.0B vs $101.8B [S1][T2].")
    assert r"\$112.0B" in out and r"\$101.8B" in out
    assert ":blue-badge[S1]" in out and ":blue-badge[T2]" in out


def test_retrieval_ladder_order():
    rows = [
        {"index": "section", "mode": "hybrid", "filters": "oracle", "category": "ALL",
         "recall@10": 0.84, "mrr": 0.615},
        {"index": "fixed", "mode": "dense", "filters": "none", "category": "ALL",
         "recall@10": 0.48, "mrr": 0.233},
    ]  # fmt: skip
    ladder = retrieval_ladder(rows)
    assert [r["recall@10"] for r in ladder] == [0.48, 0.84]
