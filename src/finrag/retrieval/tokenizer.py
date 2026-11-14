"""Token counting, so chunk sizes are measured in the embedding model's own tokens.

bge-small truncates input at 512 tokens *silently*. Sizing chunks in characters or words
risks chunks whose tail is never embedded, so we count with the model's real tokenizer.
"""

from __future__ import annotations

import re
from typing import Protocol


class Tokenizer(Protocol):
    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of every token, excluding special tokens."""
        ...

    def count(self, text: str) -> int: ...


class HFTokenizer:
    """The embedding model's WordPiece tokenizer (Rust ``tokenizers``, fast)."""

    def __init__(self, model_name: str) -> None:
        from tokenizers import Tokenizer as _Tok

        self._tok = _Tok.from_pretrained(model_name)
        self._tok.no_truncation()

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return list(self._tok.encode(text, add_special_tokens=False).offsets)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)


class WhitespaceTokenizer:
    """Dependency-free stand-in for tests: one token per whitespace-separated word."""

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"\S+", text)]

    def count(self, text: str) -> int:
        return len(text.split())
