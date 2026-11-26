"""Two ways to answer a question, compared head to head in the evaluation.

``answer_rag`` (baseline): retrieve 8 passages for the raw question, answer in one LLM
call. This is standard single-shot RAG.

``answer_agent``: the model plans its own retrieval with tools. It can filter by company
and year, search once per company for comparisons (the retrieval eval showed a single
search can't cover four filings), pull exact figures with SQL, and do arithmetic with a
calculator. The loop is capped at MAX_STEPS model calls so it always terminates.

Both return the same ``Answer`` (text, cited sources, full trace, token usage), so the
evaluation and the UI treat them identically.
"""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finrag.agent.llm import LLM, LLMResponse
from finrag.agent.tools import Source, Toolbox
from finrag.retrieval.retriever import Retriever

MAX_STEPS = 6
RAG_K = 8

GROUNDING_RULES = """\
Rules:
- Answer only from the sources provided by tools or context. Do not use outside knowledge
  for any fact or number.
- Cite every factual claim with its source id in plain ASCII square brackets, e.g. [S3]
  or [T1]. Plain text only: no markdown bold or tables.
- If the sources do not contain the answer, say clearly that the information is not
  available in the filings. Never guess a number.
- Fiscal years are labelled by the calendar year in which the fiscal period ends
  (Walmart's year ending January 2025 is FY2025; NVIDIA's ending January 2025 is FY2025).
- Report money with units (e.g. "$416,161 million" or "$416.2 billion").
- Be concise: lead with the answer, then one or two sentences of support."""

AGENT_SYSTEM = f"""\
You are a financial research assistant answering questions about SEC 10-K filings.

Tools:
- search_filings: passages from 10-K text. Pass tickers and fiscal_years whenever the
  question implies them. For a question about several companies, search each company
  separately.
- query_financials: exact annual figures (revenue, net income, EPS, ...) for 2021-2025.
  Prefer it for standard metrics. `value` is restated/split-adjusted; `original_value`
  is as first reported. If the question asks what a specific filing reported, use
  original_value or the filing text.
- calculate: use it for every arithmetic step (growth rates, margins, differences).

Search budget: if two searches with different wording do not find something, the filing
most likely does not disclose it. Say so instead of rephrasing the same search again.
Companies often do not disclose unit volumes, customer names, or segment-level detail.

{GROUNDING_RULES}"""

RAG_SYSTEM = f"""\
You are a financial research assistant answering questions about SEC 10-K filings using
only the context passages below.

{GROUNDING_RULES}"""

_BRACKET = re.compile(r"\[([^\[\]]{1,80})\]")
# One source id with optional "†..." suffix (gpt-oss writes [S2†L13-L16]).
_CITE_ID = re.compile(r"([ST]\d+)(?:†[^,\]]*)?")


@dataclass
class Step:
    tool: str
    arguments: str
    result_preview: str
    seconds: float


@dataclass
class Answer:
    question: str
    text: str
    mode: str
    model: str
    sources: dict[str, Source]  # every source the model saw
    cited: list[str]  # source ids actually cited in the answer
    steps: list[Step] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    hit_step_limit: bool = False
    error: str | None = None  # set when the LLM call failed (rate limit, request too large)


# Models sometimes cite with full-width or lenticular brackets (gpt-oss emits 【T1】).
_BRACKETS = str.maketrans({"\u3010": "[", "\u3011": "]", "\uff3b": "[", "\uff3d": "]"})


# gpt-oss also emits typographic Unicode (narrow no-break spaces, non-breaking hyphens)
# and markdown bold despite instructions. Normalise once, so the UI, citation parsing
# and grading all see plain text.
_TYPOGRAPHY = str.maketrans(
    {"\u202f": " ", "\u00a0": " ", "\u2009": " ", "\u2011": "-", "\u2010": "-"}
)


def _citation_ids(content: str) -> list[str] | None:
    """Source ids if a bracket holds only citations ("S1", " S1 ", "S2†L13-L16, T1")."""
    ids = _CITE_ID.findall(content)
    leftover = _CITE_ID.sub("", content).replace(",", "").strip()
    return ids if ids and not leftover else None


def normalize_citations(text: str) -> str:
    """Plain text with canonical citations: "[ S1 ]" and "[S2†L13-L16]" become "[S1]"."""
    text = text.translate(_BRACKETS).translate(_TYPOGRAPHY)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)

    def canon(m: re.Match[str]) -> str:
        ids = _citation_ids(m.group(1))
        return "".join(f"[{i}]" for i in ids) if ids else m.group(0)

    return _BRACKET.sub(canon, text)


def extract_citations(text: str) -> list[str]:
    ids: list[str] = []
    for content in _BRACKET.findall(normalize_citations(text)):
        for sid in _citation_ids(content) or []:
            if sid not in ids:
                ids.append(sid)
    return ids


def _assistant_message(resp: LLMResponse) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in resp.tool_calls
        ]
    return msg


def answer_agent(
    question: str,
    llm: LLM,
    toolbox: Toolbox,
    on_step: Callable[[Step], None] | None = None,
) -> Answer:
    """Run the tool loop. ``on_step`` is called after each tool call (the UI streams it)."""
    t0 = time.perf_counter()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question},
    ]
    ans = Answer(question, "", "agent", llm.model, toolbox.sources, [])
    try:
        _agent_loop(messages, llm, toolbox, ans, on_step)
    except Exception as exc:  # API failures become a recorded error, not a crash
        ans.error = f"{type(exc).__name__}: {exc}"
    ans.text = normalize_citations(ans.text)
    ans.cited = [s for s in extract_citations(ans.text) if s in toolbox.sources]
    ans.seconds = time.perf_counter() - t0
    return ans


def _agent_loop(
    messages: list[dict[str, Any]],
    llm: LLM,
    toolbox: Toolbox,
    ans: Answer,
    on_step: Callable[[Step], None] | None = None,
) -> None:
    tools = toolbox.specs()
    for _ in range(MAX_STEPS):
        resp = llm.chat(messages, tools=tools)
        ans.llm_calls += 1
        ans.prompt_tokens += resp.prompt_tokens
        ans.completion_tokens += resp.completion_tokens
        messages.append(_assistant_message(resp))
        if not resp.tool_calls:
            ans.text = resp.content
            break
        for call in resp.tool_calls:
            ts = time.perf_counter()
            result = toolbox.execute(call.name, call.arguments)
            step = Step(call.name, call.arguments, result[:300], time.perf_counter() - ts)
            ans.steps.append(step)
            if on_step:
                on_step(step)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    else:
        # Step budget spent: force a final answer from what has been gathered.
        ans.hit_step_limit = True
        messages.append(
            {"role": "user", "content": "Stop using tools. Answer now from the sources gathered."}
        )
        resp = llm.chat(messages, tools=tools, tool_choice="none")
        ans.llm_calls += 1
        ans.prompt_tokens += resp.prompt_tokens
        ans.completion_tokens += resp.completion_tokens
        ans.text = resp.content


def rag_sources(question: str, retriever: Retriever) -> dict[str, Source]:
    """The passages plain RAG puts in context, with ids S1..S8. Deterministic."""
    sources: dict[str, Source] = {}
    for i, r in enumerate(retriever.search(question, k=RAG_K), 1):
        c = r.chunk
        sid = f"S{i}"
        label = f"{c.ticker} FY{c.fiscal_year} 10-K, Item {c.item}: {c.item_title}"
        sources[sid] = Source(sid, "filing", label, c.text, c.url, c.chunk_id)
    return sources


def answer_rag(question: str, llm: LLM, retriever: Retriever) -> Answer:
    t0 = time.perf_counter()
    sources = rag_sources(question, retriever)
    blocks = [f"[{sid}] {s.label}\n{s.text}" for sid, s in sources.items()]
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {
            "role": "user",
            "content": "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}",
        },
    ]
    try:
        resp = llm.chat(messages)
    except Exception as exc:
        ans = Answer(question, "", "rag", llm.model, sources, [], llm_calls=1)
        ans.error = f"{type(exc).__name__}: {exc}"
        ans.seconds = time.perf_counter() - t0
        return ans
    ans = Answer(
        question,
        resp.content,
        "rag",
        llm.model,
        sources,
        [],
        llm_calls=1,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
    )
    ans.text = normalize_citations(ans.text)
    ans.cited = [s for s in extract_citations(ans.text) if s in sources]
    ans.seconds = time.perf_counter() - t0
    return ans


# ----------------------------------------------------------------------------------- CLI


def build(index: str = "section", model: str | None = None) -> tuple[LLM, Retriever, Path]:
    from finrag.agent.llm import OpenAICompatibleLLM
    from finrag.config import get_settings
    from finrag.retrieval.embeddings import FastEmbedder

    s = get_settings()
    if s.groq_api_key is None:
        raise SystemExit("GROQ_API_KEY is not set (add it to .env)")
    llm = OpenAICompatibleLLM(model or s.llm_model, s.groq_api_key.get_secret_value())
    retriever = Retriever.load(s.data_dir / "index" / index, FastEmbedder())
    return llm, retriever, s.financials_db_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ask a question about the 10-K corpus.")
    p.add_argument("question")
    p.add_argument("--mode", choices=["agent", "rag"], default="agent")
    p.add_argument("--model", help="override LLM_MODEL from .env")
    p.add_argument("--index", default="section")
    a = p.parse_args(argv)

    llm, retriever, db = build(a.index, a.model)
    if a.mode == "agent":
        ans = answer_agent(a.question, llm, Toolbox(retriever, db))
    else:
        ans = answer_rag(a.question, llm, retriever)

    for i, st in enumerate(ans.steps, 1):
        print(f"  step {i}: {st.tool}({st.arguments})  [{st.seconds * 1000:.0f} ms]")
    if ans.error:
        print(f"\nERROR: {ans.error}")
        if "413" in ans.error or "rate_limit" in ans.error or "TPM" in ans.error:
            print("Groq's free tier limits tokens per minute; see the README for the Dev tier.")
    print(f"\n{ans.text}\n")
    for sid in ans.cited:
        print(f"  [{sid}] {ans.sources[sid].label}")
    print(f"\n{ans.mode} | {ans.model} | {ans.llm_calls} LLM calls | "
          f"{ans.prompt_tokens + ans.completion_tokens:,} tokens | {ans.seconds:.1f}s")  # fmt: skip


if __name__ == "__main__":
    main()
