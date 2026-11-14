"""Embedding models behind one small interface.

Default: BAAI/bge-small-en-v1.5 via ``fastembed`` (ONNX Runtime). Why:
* Runs locally and free, so the eval can be rerun as often as needed at no cost.
* 33M parameters, 384 dimensions: embeds this corpus in minutes on a laptop CPU.
* ONNX instead of PyTorch keeps the install about 10x smaller, which matters on
  Streamlit Community Cloud's limited memory and build time.

bge models expect a different input for queries than for passages (queries get an
instruction prefix). fastembed's ``query_embed``/``passage_embed`` handle that, which is
why the interface has two methods rather than one.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    model_name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class FastEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 64) -> None:
        # Single process on purpose: ONNX Runtime already uses every CPU core inside one
        # process. Adding fastembed's multi-process mode on top (one process per core,
        # each using all cores and holding its own model copy) oversubscribed the CPU
        # and exhausted memory on a laptop.
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name)
        self._batch = batch_size

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack(list(self._model.passage_embed(texts, batch_size=self._batch))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(next(iter(self._model.query_embed(text))))


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder: no downloads, for tests only."""

    model_name = "hash-test"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._one(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._one(text))
