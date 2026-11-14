from finrag.retrieval.chunking import ChunkingConfig, chunk_filing
from finrag.retrieval.tokenizer import WhitespaceTokenizer

TOK = WhitespaceTokenizer()
SECTION = ChunkingConfig("s", "section", max_tokens=50, contextual_header=True)
FIXED = ChunkingConfig("f", "fixed", max_tokens=50, overlap_tokens=10)


def filing(*sections):
    return {
        "ticker": "AAPL",
        "fiscal_year": 2025,
        "url": "u",
        "sections": [{"item": i, "title": t, "text": x} for i, t, x in sections],
    }


def words(n, w="word"):
    return " ".join(f"{w}{i}." for i in range(n))


F = filing(
    ("0", "Cover", "cover page " * 20),
    ("1A", "Risk Factors", "Supply Chain Risks\n" + words(70, "risk")),
    ("7", "MD&A", "Revenue grew strongly.\n" + words(30, "mdna")),
)


def test_section_chunks_respect_limits_and_boundaries():
    chunks = chunk_filing(F, "Apple Inc.", SECTION, TOK)
    assert all(c.n_tokens <= 50 for c in chunks)
    assert {c.item for c in chunks} == {"1A", "7"}  # preamble excluded
    for c in chunks:  # never mixes sections
        assert ("risk" in c.text) != ("mdna" in c.text)
    assert all(c.n_tokens > 5 for c in chunks)  # no orphan heading-only chunks


def test_contextual_header_with_subheading():
    c = chunk_filing(F, "Apple Inc.", SECTION, TOK)[0]
    assert c.embed_text.startswith(
        "Apple Inc. (AAPL) | FY2025 Form 10-K | Item 1A: Risk Factors | Supply Chain Risks\n"
    )
    assert not c.text.startswith("Apple Inc.")  # header is for retrieval, not the LLM text


def test_split_table_repeats_header_row():
    rows = ["Line | 2025 | 2024"] + [f"Row{i} | {i} | {i + 1}" for i in range(40)]
    f = filing(("8", "Financial Statements", "\n".join(rows)))
    chunks = chunk_filing(f, "Apple Inc.", SECTION, TOK)
    assert len(chunks) > 1
    assert all(c.text.startswith("Line | 2025 | 2024") for c in chunks)
    body = [r for c in chunks for r in c.text.split("\n")[1:]]
    assert body == rows[1:]  # every row exactly once, in order


def test_fixed_chunks_overlap_and_cover_everything():
    chunks = chunk_filing(F, "Apple Inc.", FIXED, TOK)
    assert all(c.n_tokens <= 50 for c in chunks)
    assert chunks[0].embed_text == chunks[0].text  # no header in the baseline
    seen = {w for c in chunks for w in c.text.split()}
    assert {f"risk{i}." for i in range(70)} <= seen
    a, b = chunks[0].text.split(), chunks[1].text.split()
    assert a[-10:] == b[:10]  # 10-token overlap
