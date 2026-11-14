"""Search over one chunk index: dense, BM25, or hybrid (Reciprocal Rank Fusion).

Design notes:
* Exact (brute-force) vector search. About 7,000 chunks x 384 dims is 10 MB; one matrix
  product takes about a millisecond. An approximate index (FAISS, a vector database)
  would add a dependency and recall loss for no speed benefit at this scale.
* BM25 complements embeddings on exact tokens: tickers, "Item 7A", "CET1", "$391,035".
  Small embedding models are weak on numbers and rare acronyms.
* Hybrid uses Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank). It only
  uses ranks, so it needs no tuning to reconcile BM25 scores (unbounded) with cosine
  similarities (-1..1). k = 60 is the value from the original RRF paper.
* Metadata filters (ticker, year, Item) are applied *before* ranking. "Search only
  JPM's FY2025 risk factors" then searches exactly that subset, rather than filtering a
  top-k afterwards and possibly ending up with nothing. The agent's tool exposes them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from finrag.retrieval.chunking import Chunk
from finrag.retrieval.embeddings import Embedder

SearchMode = Literal["dense", "bm25", "hybrid"]
RRF_K = 60
FUSION_DEPTH = 50  # how deep into each ranked list RRF looks

# fmt: off
_STOPWORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "our", "that", "the",
    "this", "to", "was", "we", "were", "what", "which", "with",
])
# fmt: on


def bm25_tokenize(text: str) -> list[str]:
    # Keep numbers like 391,035 and 7.46 whole; drop commas inside numbers so
    # "391,035" and "391035" match.
    tokens = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())
    return [t.replace(",", "") for t in tokens if t not in _STOPWORDS]


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int
    dense_rank: int | None = None
    bm25_rank: int | None = None


class Retriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings are misaligned")
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.embedder = embedder
        self._bm25 = BM25Okapi([bm25_tokenize(c.embed_text) for c in chunks])
        self._ticker = np.array([c.ticker for c in chunks])
        self._year = np.array([c.fiscal_year for c in chunks])
        self._item = np.array([c.item for c in chunks])

    @classmethod
    def load(cls, index_dir: Path, embedder: Embedder) -> Retriever:
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if meta["embedding_model"] != embedder.model_name:
            raise ValueError(
                f"index built with {meta['embedding_model']}, queried with {embedder.model_name}"
            )
        with (index_dir / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        return cls(chunks, np.load(index_dir / "embeddings.npy"), embedder)

    # ----------------------------------------------------------------------- search

    def _mask(
        self,
        tickers: list[str] | None,
        fiscal_years: list[int] | None,
        items: list[str] | None,
    ) -> np.ndarray:
        mask = np.ones(len(self.chunks), dtype=bool)
        if tickers:
            mask &= np.isin(self._ticker, [t.upper() for t in tickers])
        if fiscal_years:
            mask &= np.isin(self._year, fiscal_years)
        if items:
            mask &= np.isin(self._item, [i.upper() for i in items])
        return mask

    @staticmethod
    def _ranked(scores: np.ndarray, mask: np.ndarray, depth: int) -> list[int]:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        order = np.argsort(-scores[idx], kind="stable")[:depth]
        return idx[order].tolist()

    def search(
        self,
        query: str,
        k: int = 8,
        mode: SearchMode = "hybrid",
        tickers: list[str] | None = None,
        fiscal_years: list[int] | None = None,
        items: list[str] | None = None,
    ) -> list[SearchResult]:
        mask = self._mask(tickers, fiscal_years, items)
        depth = max(k, FUSION_DEPTH)

        dense: list[int] = []
        bm25: list[int] = []
        if mode in ("dense", "hybrid"):
            q = self.embedder.embed_query(query)
            dense = self._ranked(self.embeddings @ q, mask, depth)
        if mode in ("bm25", "hybrid"):
            bm25 = self._ranked(self._bm25.get_scores(bm25_tokenize(query)), mask, depth)

        dense_rank = {i: r for r, i in enumerate(dense, 1)}
        bm25_rank = {i: r for r, i in enumerate(bm25, 1)}

        if mode == "hybrid":
            fused: dict[int, float] = {}
            for ranks in (dense_rank, bm25_rank):
                for i, r in ranks.items():
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)
            order = sorted(fused, key=lambda i: -fused[i])[:k]
            scores = [fused[i] for i in order]
        else:
            order = (dense if mode == "dense" else bm25)[:k]
            scores = [1.0 / r for r in range(1, len(order) + 1)]

        return [
            SearchResult(
                chunk=self.chunks[i],
                score=s,
                rank=r,
                dense_rank=dense_rank.get(i),
                bm25_rank=bm25_rank.get(i),
            )
            for r, (i, s) in enumerate(zip(order, scores, strict=True), 1)
        ]


def main(argv: list[str] | None = None) -> None:
    """Try queries from the terminal: finrag-search "CET1 ratio" --tickers JPM."""
    import argparse

    from finrag.config import get_settings
    from finrag.retrieval.embeddings import FastEmbedder

    p = argparse.ArgumentParser(description="Search a chunk index.")
    p.add_argument("query")
    p.add_argument("--index", default="section", help="index name under data/index/")
    p.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--years", nargs="*", type=int)
    p.add_argument("--items", nargs="*")
    a = p.parse_args(argv)

    retriever = Retriever.load(get_settings().data_dir / "index" / a.index, FastEmbedder())
    for r in retriever.search(a.query, a.k, a.mode, a.tickers, a.years, a.items):
        c = r.chunk
        print(f"\n#{r.rank}  {c.ticker} FY{c.fiscal_year}  Item {c.item}  {c.subheading[:60]}")
        print(f"    dense rank={r.dense_rank}  bm25 rank={r.bm25_rank}  score={r.score:.4f}")
        print("    " + c.text[:300].replace("\n", "\n    "))
