import json
import sqlite3

import pytest

from finrag.agent.agent import MAX_STEPS, answer_agent, answer_rag, extract_citations
from finrag.agent.llm import LLMResponse, ToolCall
from finrag.agent.tools import Toolbox, calculate, run_readonly_sql
from finrag.ingest.xbrl import SCHEMA
from finrag.retrieval.chunking import Chunk
from finrag.retrieval.embeddings import HashEmbedder
from finrag.retrieval.retriever import Retriever

# ----------------------------------------------------------------------------- fixtures


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "f.db"
    with sqlite3.connect(path) as c:
        c.executescript(SCHEMA)
        c.execute("INSERT INTO companies VALUES ('AAPL', 320193, 'Apple Inc.', 'Technology')")
        c.execute("INSERT INTO metrics VALUES ('revenue', 'Revenue', 'USD', 'duration')")
        c.execute(
            "INSERT INTO financials VALUES "
            "('AAPL', 2025, '2025-09-27', 'revenue', 416161e6, 416161e6, 'USD', 'R', 'a', 'f')"
        )
    return path


@pytest.fixture
def retriever():
    chunks = [
        Chunk("c0", "AAPL", "Apple Inc.", 2025, "7", "MD&A", "", "Total net sales | 416,161",
              "Total net sales | 416,161", 5, "u"),
        Chunk("c1", "MSFT", "Microsoft", 2025, "7", "MD&A", "", "Azure grew 34%",
              "Azure grew 34%", 3, "u"),
    ]  # fmt: skip
    emb = HashEmbedder()
    return Retriever(chunks, emb.embed_passages([c.text for c in chunks]), emb)


class ScriptedLLM:
    """Plays back a fixed sequence of responses and records what it was sent."""

    model = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.calls.append({"messages": [dict(m) for m in messages], "tool_choice": tool_choice})
        return self.responses.pop(0)


def call(name, **args):
    return ToolCall(id=f"id-{name}", name=name, arguments=json.dumps(args))


# --------------------------------------------------------------------------- calculator


def test_calculate_arithmetic_and_thousands_separators():
    assert calculate("(416161 / 391035 - 1) * 100") == pytest.approx(6.4255, rel=1e-4)
    assert calculate("416,161 - 391,035") == 25126
    assert calculate("round(2/3, 2)") == 0.67


@pytest.mark.parametrize(
    "expr", ["__import__('os').system('ls')", "open('x')", "(1).__class__", "9**9**9", "x + 1"]
)
def test_calculate_rejects_anything_but_arithmetic(expr):
    with pytest.raises(ValueError):
        calculate(expr)


# --------------------------------------------------------------------------------- SQL


def test_sql_select_works(db):
    out = run_readonly_sql(db, "SELECT value FROM financials WHERE ticker = 'AAPL';")
    assert out["rows"] == [[416161e6]] and out["columns"] == ["value"]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM financials",
        "SELECT 1; DROP TABLE financials",
        "UPDATE financials SET value = 0",
        "ATTACH DATABASE 'x.db' AS x",
    ],
)
def test_sql_rejects_writes_and_multiple_statements(db, sql):
    with pytest.raises(ValueError):
        run_readonly_sql(db, sql)
    assert run_readonly_sql(db, "SELECT COUNT(*) FROM financials")["rows"] == [[1]]


def test_sql_cte_that_writes_is_blocked_by_readonly_connection(db):
    with pytest.raises(sqlite3.OperationalError):
        run_readonly_sql(db, "WITH x AS (SELECT 1) DELETE FROM financials")


# ------------------------------------------------------------------------------ toolbox


def test_toolbox_assigns_stable_source_ids_and_returns_errors(retriever, db):
    tb = Toolbox(retriever, db)
    first = json.loads(
        tb.execute("search_filings", json.dumps({"query": "net sales", "tickers": ["AAPL"]}))
    )
    assert first[0]["source_id"] == "S1" and "text" in first[0]
    assert tb.sources["S1"].chunk_id == "c0"
    err = json.loads(tb.execute("query_financials", json.dumps({"sql": "DROP TABLE x"})))
    assert "error" in err


# -------------------------------------------------------------------------------- agent


def test_citations_parsed():
    assert extract_citations("Revenue was $416B [S1, T2]. Margin 46% [S1][S3].") == [
        "S1",
        "T2",
        "S3",
    ]


def test_answer_text_normalized():
    from finrag.agent.agent import normalize_citations

    raw = "Apple had **166,000\u202ffull\u2011time** employees \u3010S1\u3011."
    assert normalize_citations(raw) == "Apple had 166,000 full-time employees [S1]."


def test_spaced_and_line_range_citations_from_gpt_oss():
    from finrag.agent.agent import normalize_citations

    text = "EPS was $7.46 [S2†L13-L16][S3†L31-L34]; segments are X [ S1 ][ T1 ]."
    assert extract_citations(text) == ["S2", "S3", "S1", "T1"]
    assert normalize_citations(text) == "EPS was $7.46 [S2][S3]; segments are X [S1][T1]."
    assert extract_citations("Banks (CCB) and [note] and [S1 and more]") == []


def test_full_width_citations_from_gpt_oss_are_recognized():
    assert extract_citations("a difference of $10.18 billion \u3010T1\u3011.") == ["T1"]


def test_agent_runs_tools_then_answers(retriever, db):
    search = call("search_filings", query="net sales", tickers=["AAPL"], fiscal_years=[2025])
    sql = call("query_financials", sql="SELECT value FROM financials")
    llm = ScriptedLLM(
        [
            LLMResponse("", [search, sql]),
            LLMResponse("", [call("calculate", expression="416161000000 / 1e9")]),
            LLMResponse("Apple's FY2025 net sales were $416.2 billion [S1][T1]. Made up [S9]."),
        ]
    )
    ans = answer_agent("Apple FY2025 net sales?", llm, Toolbox(retriever, db))
    assert [s.tool for s in ans.steps] == ["search_filings", "query_financials", "calculate"]
    assert ans.cited == ["S1", "T1"]  # S9 was never shown to the model: dropped
    assert ans.llm_calls == 3 and not ans.hit_step_limit
    tool_msgs = [m for m in llm.calls[-1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 3 and all(m["tool_call_id"] for m in tool_msgs)


def test_agent_step_limit_forces_final_answer(retriever, db):
    loop = [LLMResponse("", [call("calculate", expression="1+1")])] * MAX_STEPS
    llm = ScriptedLLM([*loop, LLMResponse("Not available in the filings.")])
    ans = answer_agent("?", llm, Toolbox(retriever, db))
    assert ans.hit_step_limit and ans.text == "Not available in the filings."
    assert llm.calls[-1]["tool_choice"] == "none"


def test_rag_puts_context_in_prompt_and_tracks_citations(retriever):
    llm = ScriptedLLM([LLMResponse("Azure grew 34% [S1].")])
    ans = answer_rag("How fast did Azure grow?", llm, retriever)
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "[S1]" in prompt and "Question: How fast did Azure grow?" in prompt
    assert ans.cited == ["S1"] and ans.mode == "rag"


def test_search_with_nothing_new_tells_model_to_conclude(retriever, db):
    tb = Toolbox(retriever, db)
    tb.execute("search_filings", json.dumps({"query": "net sales", "tickers": ["AAPL"]}))
    again = tb.execute("search_filings", json.dumps({"query": "sales", "tickers": ["AAPL"]}))
    assert "no passages you have not already seen" in json.loads(again)["note"]


def test_repeated_search_refused_even_with_reordered_words(retriever, db):
    tb = Toolbox(retriever, db)
    args = {"query": "iPhone units sold", "tickers": ["AAPL"], "fiscal_years": [2025]}
    tb.execute("search_filings", json.dumps(args))
    repeat = tb.execute("search_filings", json.dumps(args | {"query": "units sold iPhone"}))
    assert "already ran this exact search" in json.loads(repeat)["note"]


class FailingLLM:
    model = "fake"

    def chat(self, *a, **k):
        raise RuntimeError("Error code: 413 - Request too large")


def test_api_failure_is_recorded_not_raised(retriever, db):
    ans = answer_agent("?", FailingLLM(), Toolbox(retriever, db))
    assert ans.error and "413" in ans.error and ans.text == ""
    assert answer_rag("?", FailingLLM(), retriever).error


def test_replaying_recorded_steps_reproduces_source_ids(retriever, db):
    """The citation repair depends on this: same tool calls in order -> same ids."""
    s1 = call("search_filings", query="net sales", tickers=["AAPL"])
    s2 = call("search_filings", query="Azure growth", tickers=["MSFT"])
    sql = call("query_financials", sql="SELECT value FROM financials")
    llm = ScriptedLLM([LLMResponse("", [s1, s2, sql]), LLMResponse("Done [S2][T1].")])
    original = answer_agent("?", llm, Toolbox(retriever, db))

    replay = Toolbox(retriever, db)
    for st in original.steps:
        replay.execute(st.tool, st.arguments)
    assert {k: v.chunk_id for k, v in replay.sources.items()} == {
        k: v.chunk_id for k, v in original.sources.items()
    }
    assert replay.sources["S2"].text == "Azure grew 34%"
