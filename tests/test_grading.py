import pytest

from finrag.agent.llm import LLMResponse
from finrag.eval.answers import summarize
from finrag.eval.dataset import Question
from finrag.eval.grading import extract_numbers, judge, numeric_match


def q(**kw):
    return Question.model_validate({"id": "q", "category": "numeric_lookup", "question": "?",
                                    "answer": "a"} | kw)  # fmt: skip


def test_extract_numbers_with_scale_words():
    nums = extract_numbers("Revenue was $416.2 billion (up 6.43%), or $416,161 million; 2.1M staff")
    assert 416.2e9 in nums and 6.43 in nums and 416_161e6 in nums and 2.1e6 in nums


@pytest.mark.parametrize(
    "answer, ok",
    [
        ("Apple's total net sales were $416,161 million.", True),
        ("Apple's total net sales were $416.2 billion.", True),
        ("Total net sales | 416,161", True),  # bare number in the gold unit
        ("Apple's total net sales were $391.0 billion.", False),
    ],
)
def test_numeric_match_scales(answer, ok):
    assert numeric_match(q(value=416161, unit="USD millions"), answer) is ok


def test_numeric_match_percent_and_sign():
    calc = q(category="numeric_calc", value=-2.43, unit="%", tolerance=0.02)
    assert numeric_match(calc, "Net income decreased about 2.4%.")
    assert numeric_match(q(category="unanswerable"), "anything") is None


class FakeJudge:
    model = "judge"

    def __init__(self, content):
        self.content = content

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.prompt = messages[1]["content"]
        return LLMResponse(self.content)


def test_judge_parses_json_wrapped_in_prose():
    llm = FakeJudge('Here you go:\n```json\n{"correct": true, "abstained": false, '
                    '"unsupported_claims": ["made-up figure"], "reason": "ok"}\n```')  # fmt: skip
    j = judge(llm, q(value=1, unit="USD"), "answer", ["source text"])
    assert j.correct and j.hallucinated and "GOLD VALUE: 1.0 USD" in llm.prompt


def test_summarize_metrics():
    qs = {"a": q(id="a"), "u": q(id="u", category="unanswerable")}
    recs = [
        {"qid": "a", "cited": ["S1"], "tokens": 100, "llm_calls": 2, "seconds": 1.0,
         "hit_step_limit": False, "numeric_match": True,
         "judge": {"correct": True, "abstained": False, "hallucinated": False}},
        {"qid": "u", "cited": [], "tokens": 50, "llm_calls": 1, "seconds": 0.5,
         "hit_step_limit": False, "numeric_match": None,
         "judge": {"correct": True, "abstained": True, "hallucinated": False}},
    ]  # fmt: skip
    s = summarize(recs, qs)
    assert s["accuracy"] == 1.0 and s["correct_abstention"] == 1.0
    assert s["hallucination_rate"] == 0.0 and s["citation_rate"] == 1.0
    assert s["judge_numeric_agreement"] == 1.0 and s["avg_tokens"] == 75


def test_judge_ignores_reasoning_block_with_braces():
    llm = FakeJudge('<think>gold is {416161}; answer says 416.2B</think>'
                    '{"correct": true, "abstained": false, "unsupported_claims": []}')  # fmt: skip
    j = judge(llm, q(value=1, unit="USD"), "answer", [])
    assert j.correct and not j.hallucinated


def test_daily_limit_detection():
    from finrag.eval.answers import _is_daily_limit

    assert _is_daily_limit("Error code: 429 - ... on tokens per day (TPD): Limit 200000")
    assert not _is_daily_limit("Error code: 429 - ... on tokens per minute (TPM)")
    assert not _is_daily_limit(None)


def test_cohen_kappa():
    from finrag.eval.answers import cohen_kappa

    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohen_kappa([], []) is None
