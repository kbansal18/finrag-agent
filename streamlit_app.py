"""finrag: ask questions about ten companies' 10-K filings.

Run locally:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="finrag: 10-K research agent", page_icon="📑", layout="wide")

# On Streamlit Community Cloud, secrets live in st.secrets rather than .env.
# Copying them into the environment lets the same typed Settings class read both.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except FileNotFoundError:  # no secrets.toml locally: .env is used instead
    pass

from finrag.agent.agent import Answer, Step, answer_agent, answer_rag  # noqa: E402
from finrag.agent.llm import OpenAICompatibleLLM  # noqa: E402
from finrag.agent.tools import Toolbox  # noqa: E402
from finrag.config import PROJECT_ROOT, get_settings  # noqa: E402
from finrag.data_bundle import ensure_data, is_ready  # noqa: E402
from finrag.retrieval.embeddings import FastEmbedder  # noqa: E402
from finrag.retrieval.retriever import Retriever  # noqa: E402
from finrag.ui import answer_markdown, describe_step, read_csv, retrieval_ladder  # noqa: E402

MODELS = {
    "gpt-oss-120b (large)": "openai/gpt-oss-120b",
    "gpt-oss-20b (small)": "openai/gpt-oss-20b",
}
MAX_QUESTIONS_PER_SESSION = 15  # the public demo shares one API key's rate limit
BLUE = "#2a78d6"  # single-series bars: one hue (reference palette, slot 1)
EXAMPLES = [
    "Which had higher net income in fiscal 2025, Apple or Microsoft, and by how much?",
    "What was NVIDIA's net profit margin in fiscal year 2025?",
    "How concentrated was NVIDIA's revenue among direct customers in fiscal 2025?",
    "Compare JPMorgan's and Goldman Sachs's CET1 ratios at the end of 2025.",
    "How did Microsoft's capital expenditures change from fiscal 2021 to 2025?",
    "How many iPhones did Apple sell in fiscal 2025?",
]

settings = get_settings()


@st.cache_resource(show_spinner="Loading the search index...")
def load_retriever() -> Retriever:
    return Retriever.load(settings.data_dir / "index" / "section", FastEmbedder())


@st.cache_resource
def load_llm(model: str) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(model, settings.groq_api_key.get_secret_value())


@st.cache_resource(show_spinner="First start: downloading the search index (~15 MB)...")
def prepare_data() -> bool:
    """On a fresh deployment, fetch the prebuilt index once; locally it already exists."""
    try:
        ensure_data(settings.data_dir, settings.data_bundle_url, settings.data_bundle_sha256)
    except Exception as exc:  # shown in the UI below
        st.session_state.data_error = str(exc)
    return is_ready(settings.data_dir)


def data_ready() -> bool:
    return prepare_data()


# ------------------------------------------------------------------------------ sidebar

with st.sidebar:
    st.header("finrag")
    st.caption(
        "A tool-calling research agent over the FY2024-2025 10-K filings of AAPL, MSFT, "
        "NVDA, GOOGL, JPM, BAC, GS, WMT, COST and TGT."
    )
    mode = st.radio(
        "Answering strategy",
        ["Agent (tools)", "Plain RAG"],
        help="Agent: plans its own searches, SQL queries and calculations. "
        "Plain RAG: one retrieval of 8 passages, one LLM call.",
    )
    model_label = st.selectbox("Model", list(MODELS))
    st.divider()
    st.caption("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            st.session_state.pending = ex

ask_tab, results_tab, about_tab = st.tabs(["Ask", "Evaluation results", "How it works"])

# ---------------------------------------------------------------------------------- ask


def render_answer(ans: Answer) -> None:
    if ans.error:
        st.error(
            "The model API call failed (often a rate limit on the shared demo key). "
            "Please try again in a minute.",
            icon="⚠️",
        )
        with st.expander("Error details"):
            st.code(ans.error, language=None)
        return
    st.markdown(answer_markdown(ans.text))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tool calls", len(ans.steps))
    c2.metric("LLM calls", ans.llm_calls)
    c3.metric("Tokens", f"{ans.prompt_tokens + ans.completion_tokens:,}")
    c4.metric("Time", f"{ans.seconds:.1f}s")
    if ans.cited:
        st.subheader("Sources", divider="gray")
        for sid in ans.cited:
            src = ans.sources[sid]
            with st.expander(f"[{sid}] {src.label}"):
                st.code(src.text, language=None, wrap_lines=True)
                if src.url:
                    st.link_button("Open the filing on SEC.gov", src.url)


with ask_tab:
    if not data_ready():
        st.error(
            "Search index not available. Locally, run `make data`. On Streamlit Cloud, set "
            "DATA_BUNDLE_URL and DATA_BUNDLE_SHA256 in the app's secrets (see README). "
            + st.session_state.get("data_error", "")
        )
        st.stop()
    if settings.groq_api_key is None:
        st.error("GROQ_API_KEY is not configured.")
        st.stop()

    st.session_state.setdefault("asked", 0)
    question = st.chat_input("Ask about revenue, risks, segments, capital ratios...")
    question = question or st.session_state.pop("pending", None)

    if question:
        if st.session_state.asked >= MAX_QUESTIONS_PER_SESSION:
            st.warning("Demo limit reached for this session. Refresh the page to continue.")
            st.stop()
        st.session_state.asked += 1
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            retriever = load_retriever()
            llm = load_llm(MODELS[model_label])
            if mode.startswith("Agent"):
                with st.status("Researching...", expanded=True) as status:

                    def show(step: Step) -> None:
                        status.write(describe_step(step))

                    ans = answer_agent(
                        question, llm, Toolbox(retriever, settings.financials_db_path), show
                    )
                    status.update(
                        label=f"Done: {len(ans.steps)} tool calls in {ans.seconds:.1f}s",
                        state="error" if ans.error else "complete",
                        expanded=False,
                    )
            else:
                with st.spinner("Retrieving and answering..."):
                    ans = answer_rag(question, llm, retriever)
            render_answer(ans)
    else:
        st.info("Ask a question below, or pick an example from the sidebar.", icon="💬")

# ------------------------------------------------------------------------------ results


def hbar(df: pd.DataFrame, value: str, label: str, fmt: str = ".2f") -> alt.LayerChart:
    """Ranked horizontal bars, one hue, value labelled at the bar end, hover tooltip."""
    base = alt.Chart(df).encode(
        y=alt.Y(f"{label}:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(
            f"{value}:Q",
            scale=alt.Scale(domain=[0, 1]),
            title=None,
            axis=alt.Axis(format="%", grid=True, tickCount=5),
        ),  # fmt: skip
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip(f"{value}:Q", format=fmt)],
    )
    bars = base.mark_bar(color=BLUE, cornerRadiusEnd=4, height=18)
    text = base.mark_text(align="left", dx=6, color="gray").encode(
        text=alt.Text(f"{value}:Q", format=fmt)
    )
    return (bars + text).properties(height=alt.Step(34))


with results_tab:
    results = PROJECT_ROOT / "results"
    st.subheader("Retrieval: what each design choice added")
    st.caption(
        "Recall@10: share of gold evidence passages found in the top 10 results, over 69 "
        "answerable questions. Each row adds one change to the row above."
    )
    ladder = retrieval_ladder(read_csv(results / "retrieval_summary.csv"))
    if ladder:
        df = pd.DataFrame(ladder)
        left, right = st.columns(2)
        left.markdown("**Recall@10**")
        left.altair_chart(hbar(df, "recall@10", "step"), width="stretch")
        right.markdown("**Mean reciprocal rank**")
        right.altair_chart(hbar(df, "MRR", "step", ".3f"), width="stretch")
    else:
        st.info("Run `make eval-retrieval` to populate this section.")

    st.subheader("Answers: agent vs plain RAG, large vs small model")
    answers = [r for r in read_csv(results / "answer_summary.csv") if r["category"] == "ALL"]
    if answers and min(r["graded"] or 0 for r in answers) < 60:
        st.warning("Answer evaluation is still in progress: some configurations have fewer "
                   "than 60 of 80 answers graded, so these numbers are provisional.")  # fmt: skip
    if answers:
        df = pd.DataFrame(answers)
        best = df.sort_values("accuracy", ascending=False).iloc[0]
        k1, k2, k3, k4 = st.columns(4)

        def pct(v: float | None) -> str:
            return "n/a" if v is None or pd.isna(v) else f"{v:.0%}"  # missing is not 0%

        k1.metric(f"Accuracy ({best['config']})", pct(best["accuracy"]),
                  help=f"Based on {int(best['graded'])} graded answers")  # fmt: skip
        k2.metric("Hallucination rate", pct(best["hallucination_rate"]))
        k3.metric("Correct abstention", pct(best["correct_abstention"]),
                  help="Unanswerable questions the system declined to answer")  # fmt: skip
        k4.metric("Avg time", f"{best['avg_seconds']:.1f}s")
        left, right = st.columns(2)
        left.markdown("**Accuracy** (higher is better)")
        left.altair_chart(hbar(df, "accuracy", "config"), width="stretch")
        right.markdown("**Hallucination rate** (lower is better)")
        right.altair_chart(hbar(df, "hallucination_rate", "config"), width="stretch")
        with st.expander("All metrics, as a table"):
            st.dataframe(df.drop(columns=["category"]), hide_index=True)
    else:
        st.info("Run `make eval-answers` to populate this section.")

# -------------------------------------------------------------------------------- about

with about_tab:
    st.markdown(
        """
**Pipeline.** SEC EDGAR 10-Ks are parsed into Item sections (tables kept as rows), split
into section-aware chunks with a contextual header, embedded with bge-small, and searched
with hybrid BM25 + dense retrieval fused by Reciprocal Rank Fusion.

**Agent.** The model plans its own retrieval with three tools: `search_filings`
(filtered by company, year and Item), `query_financials` (read-only SQL over SEC XBRL
data) and `calculate` (exact arithmetic). Every claim must cite a source, and the agent
is told to say when information is not disclosed.

**Evaluation.** 80 hand-built questions with gold answers and gold evidence, verified
against the filings by script: facts, numeric lookups, calculations, cross-company
comparisons, XBRL-only questions, and 11 unanswerable questions to measure
hallucination. Answers are graded by a different model family (Qwen) and by a
deterministic number check.

Code and full write-up: [github.com/kbansal18/finrag-agent](https://github.com/kbansal18/finrag-agent)
"""
    )
