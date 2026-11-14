import numpy as np
import pytest

from finrag.retrieval.chunking import Chunk
from finrag.retrieval.embeddings import HashEmbedder
from finrag.retrieval.retriever import Retriever, bm25_tokenize


def chunk(i, ticker, year, item, text):
    return Chunk(f"c{i}", ticker, ticker, year, item, "t", "", text, text, 10, "u")


CHUNKS = [
    chunk(0, "AAPL", 2025, "1A", "supply chain concentration in china manufacturing"),
    chunk(1, "AAPL", 2025, "7", "iphone net sales increased services revenue grew"),
    chunk(2, "JPM", 2025, "7", "net interest income CET1 capital ratio 15.7%"),
    chunk(3, "JPM", 2024, "1A", "credit risk from consumer lending and supply of liquidity"),
    chunk(4, "WMT", 2025, "7", "walmart us comparable sales grew ecommerce 391,035"),
]


@pytest.fixture
def retriever():
    emb = HashEmbedder()
    return Retriever(CHUNKS, emb.embed_passages([c.embed_text for c in CHUNKS]), emb)


def test_bm25_tokenize_numbers():
    assert bm25_tokenize("Revenue was $391,035 or 7.46 per share") == [
        "revenue", "391035", "7.46", "per", "share",
    ]  # fmt: skip


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_modes_find_obvious_match(retriever, mode):
    assert retriever.search("CET1 capital ratio", k=1, mode=mode)[0].chunk.chunk_id == "c2"


def test_bm25_matches_number_without_comma(retriever):
    assert retriever.search("391035", k=1, mode="bm25")[0].chunk.ticker == "WMT"


def test_filters_applied_before_ranking(retriever):
    res = retriever.search("supply", k=5, tickers=["jpm"], fiscal_years=[2024], items=["1a"])
    assert [r.chunk.chunk_id for r in res] == ["c3"]
    assert retriever.search("supply", tickers=["MSFT"]) == []


def test_hybrid_is_rrf_of_both_lists(retriever):
    res = retriever.search("supply chain", k=5, mode="hybrid")
    top = res[0]
    assert top.dense_rank is not None and top.bm25_rank is not None
    expected = 1 / (60 + top.dense_rank) + 1 / (60 + top.bm25_rank)
    assert top.score == pytest.approx(expected)


def test_misaligned_index_rejected():
    with pytest.raises(ValueError):
        Retriever(CHUNKS, np.zeros((2, 4)), HashEmbedder())
