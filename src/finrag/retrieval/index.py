"""Build one chunk index per chunking config: ``data/index/<config>/``.

Each index directory holds:
* ``chunks.jsonl``: chunk text and metadata
* ``embeddings.npy``: float32 matrix, one L2-normalised row per chunk
* ``meta.json``: embedding model and chunking config, checked when loading so an index
  is never queried with a different model than it was built with.

BM25 is rebuilt from the chunks at load time (under a second), so it is not stored.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from finrag.config import get_settings
from finrag.retrieval.chunking import CHUNKING_CONFIGS, Chunk, ChunkingConfig, chunk_filing
from finrag.retrieval.embeddings import DEFAULT_EMBEDDING_MODEL, Embedder
from finrag.retrieval.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def company_names(db_path: Path) -> dict[str, str]:
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path) as conn:
        return dict(conn.execute("SELECT ticker, name FROM companies"))


def chunk_corpus(
    processed_dir: Path, cfg: ChunkingConfig, tok: Tokenizer, names: dict[str, str]
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(processed_dir.glob("*.json")):
        filing = json.loads(path.read_text(encoding="utf-8"))
        company = names.get(filing["ticker"], filing["ticker"])
        chunks.extend(chunk_filing(filing, company, cfg, tok))
    return chunks


def build_index(
    processed_dir: Path, out_dir: Path, cfg: ChunkingConfig, tok: Tokenizer,
    embedder: Embedder, names: dict[str, str],
) -> list[Chunk]:  # fmt: skip
    t0 = time.perf_counter()
    chunks = chunk_corpus(processed_dir, cfg, tok, names)
    sizes = [c.n_tokens for c in chunks]
    logger.info(
        "[%s] %d chunks; tokens median %d, max %d",
        cfg.name, len(chunks), statistics.median(sizes), max(sizes),
    )  # fmt: skip

    t1 = time.perf_counter()
    embeddings = embedder.embed_passages([c.embed_text for c in chunks])
    logger.info("[%s] embedded in %.0fs", cfg.name, time.perf_counter() - t1)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "embedding_model": embedder.model_name,
                "chunking": asdict(cfg),
                "n_chunks": len(chunks),
                "dim": int(embeddings.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("[%s] done in %.0fs -> %s", cfg.name, time.perf_counter() - t0, out_dir)
    return chunks


def main(argv: list[str] | None = None) -> None:
    from finrag.retrieval.embeddings import FastEmbedder
    from finrag.retrieval.tokenizer import HFTokenizer

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Chunk, embed and index the parsed 10-Ks.")
    parser.add_argument("--configs", nargs="+", default=list(CHUNKING_CONFIGS))
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tok = HFTokenizer(args.model)
    embedder = FastEmbedder(args.model)
    names = company_names(settings.financials_db_path)
    for name in args.configs:
        build_index(
            settings.data_dir / "processed",
            settings.data_dir / "index" / name,
            CHUNKING_CONFIGS[name],
            tok,
            embedder,
            names,
        )


if __name__ == "__main__":
    main()
