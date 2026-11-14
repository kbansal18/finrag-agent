from finrag.eval.dataset import Question
from finrag.eval.retrieval import QuestionResult, evaluate_question, summarize
from finrag.retrieval.chunking import Chunk
from finrag.retrieval.embeddings import HashEmbedder
from finrag.retrieval.retriever import Retriever


def chunk(i, ticker, year, text):
    return Chunk(f"c{i}", ticker, ticker, year, "7", "t", "", text, text, 10, "u")


CHUNKS = [
    chunk(0, "AAPL", 2025, "Total net sales | 416,161 | 391,035"),
    chunk(1, "AAPL", 2024, "Total net sales | 391,035 | 383,285"),  # same number, wrong year
    chunk(2, "MSFT", 2025, "Total revenue | 281,724 | 245,122"),
]


def retriever():
    emb = HashEmbedder()
    return Retriever(CHUNKS, emb.embed_passages([c.text for c in CHUNKS]), emb)


def question(*evidence):
    return Question(
        id="q", category="comparison", question="Apple and Microsoft net sales revenue",
        answer="a", evidence=[{"ticker": t, "fiscal_year": y, "text": x} for t, y, x in evidence],
    )  # fmt: skip


def test_hit_requires_right_filing_and_text():
    q = question(("AAPL", 2025, "391,035"), ("MSFT", 2025, "281,724"), ("GS", 2025, "nope"))
    res = evaluate_question(retriever(), q, "bm25", oracle_filters=False)
    assert res.ranks[0] is not None
    assert res.ranks[2] is None
    assert res.recall(10) == 2 / 3


def test_right_text_in_wrong_filing_is_not_a_hit():
    # "416,161" exists only in AAPL FY2025; evidence claims FY2024, so it must miss.
    q = question(("AAPL", 2024, "416,161"))
    assert evaluate_question(retriever(), q, "bm25", oracle_filters=False).ranks == [None]


def test_oracle_filters_restrict_to_evidence_filings():
    q = question(("MSFT", 2025, "281,724"))
    res = evaluate_question(retriever(), q, "dense", oracle_filters=True)
    assert res.ranks == [1]  # only MSFT FY2025 chunks are searched


def test_summary_metrics():
    rs = [QuestionResult("a", "fact", [1]), QuestionResult("b", "fact", [4, None])]
    s = summarize(rs)
    assert s["recall@1"] == 0.5  # (1 + 0) / 2
    assert s["recall@5"] == 0.75  # (1 + 0.5) / 2
    assert s["mrr"] == (1 + (0.25 + 0) / 2) / 2
