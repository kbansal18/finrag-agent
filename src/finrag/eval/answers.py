"""End-to-end answer evaluation: generate, grade, report.

Configurations compared (all on the ``section`` index):
    rag-120b, agent-120b, rag-20b, agent-20b
i.e. plain RAG vs the tool-calling agent, on a large and a small model.

Metrics per configuration:
* accuracy: share of answerable questions judged correct
* numeric accuracy: deterministic number check (no LLM) on questions with a gold value
* correct abstention: share of unanswerable questions the system declined
* over-abstention: share of answerable questions it wrongly declined
* hallucination rate: share of answers stating a fact not supported by their citations
* citation rate: share of substantive answers that cite at least one source
* cost and speed: tokens, LLM calls, seconds, step-limit hits, API errors

Everything is cached per question in ``results/answers/<config>.jsonl``. Rerunning skips
finished questions, so a run interrupted by a rate limit simply resumes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from finrag.config import PROJECT_ROOT, get_settings
from finrag.eval.dataset import DEFAULT_QUESTIONS, Question, load_questions
from finrag.eval.grading import judge, numeric_match

RESULTS = PROJECT_ROOT / "results"
ANSWERS_DIR = RESULTS / "answers"
# Agent configs first: under per-day token quotas, the order decides which configs get
# each day's budget, and the agent results are the headline comparison.
CONFIGS: dict[str, tuple[str, str]] = {
    "agent-120b": ("agent", "openai/gpt-oss-120b"),
    "agent-20b": ("agent", "openai/gpt-oss-20b"),
    "rag-120b": ("rag", "openai/gpt-oss-120b"),
    "rag-20b": ("rag", "openai/gpt-oss-20b"),
}
# A different model family from the systems under test (gpt-oss), to limit
# self-preference bias. Check availability with --list-models; it varies by account tier.
DEFAULT_JUDGE = "qwen/qwen3.8-27b"
JUDGE_MAX_TOKENS = 400


class DailyLimitReached(RuntimeError):
    """The provider's tokens-per-day quota is spent: retrying today is pointless."""


def _is_daily_limit(error: str | None) -> bool:
    return bool(error) and ("tokens per day" in error or "(TPD)" in error)


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["qid"]: r for r in map(json.loads, f)}


def _save_cache(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------- generation


def generate(config: str, questions: list[Question], sleep_s: float) -> None:
    from finrag.agent.agent import answer_agent, answer_rag
    from finrag.agent.llm import OpenAICompatibleLLM
    from finrag.agent.tools import Toolbox
    from finrag.retrieval.embeddings import FastEmbedder
    from finrag.retrieval.retriever import Retriever

    s = get_settings()
    mode, model = CONFIGS[config]
    llm = OpenAICompatibleLLM(model, s.groq_api_key.get_secret_value())
    retriever = Retriever.load(s.data_dir / "index" / "section", FastEmbedder())
    path = ANSWERS_DIR / f"{config}.jsonl"
    cache = _load_cache(path)

    todo = [q for q in questions if q.id not in cache or cache[q.id].get("error")]
    print(f"[{config}] {len(todo)} to generate ({len(questions) - len(todo)} cached)")
    for i, q in enumerate(todo, 1):
        if mode == "agent":
            ans = answer_agent(q.question, llm, Toolbox(retriever, s.financials_db_path))
        else:
            ans = answer_rag(q.question, llm, retriever)
        cache[q.id] = {
            "qid": q.id,
            "config": config,
            "category": q.category,
            "answer": ans.text,
            "cited": ans.cited,
            "cited_texts": [ans.sources[c].text for c in ans.cited],
            "steps": [asdict(st) | {"result_preview": ""} for st in ans.steps],
            "llm_calls": ans.llm_calls,
            "tokens": ans.prompt_tokens + ans.completion_tokens,
            "seconds": round(ans.seconds, 2),
            "hit_step_limit": ans.hit_step_limit,
            "error": ans.error,
        }
        if _is_daily_limit(ans.error):
            del cache[q.id]  # not a real failure of the system under test
            _save_cache(path, cache)
            raise DailyLimitReached(f"{model}: {ans.error}")
        _save_cache(path, cache)  # after every question: interruptions lose nothing
        flag = "ERROR" if ans.error else ""
        print(f"  {i}/{len(todo)} {q.id} {ans.seconds:.1f}s {flag}")
        if sleep_s:
            time.sleep(sleep_s)


# ------------------------------------------------------------------------------- repair


def repair(config: str, questions: list[Question]) -> int:
    """Re-parse citations in cached answers, with no LLM calls.

    An early parser missed gpt-oss citation formats ("[ S1 ]", "[S2†L13-L16]"), so some
    correctly cited answers were stored as citing nothing, and the judge then flagged
    their facts as unsupported. Sources are rebuilt deterministically: plain RAG by
    re-running retrieval, the agent by replaying its recorded tool calls in order, which
    reproduces the same source ids. Records whose citations change are re-graded.
    """
    from finrag.agent.agent import extract_citations, normalize_citations, rag_sources
    from finrag.agent.tools import Toolbox
    from finrag.retrieval.embeddings import FastEmbedder
    from finrag.retrieval.retriever import Retriever

    s = get_settings()
    retriever = Retriever.load(s.data_dir / "index" / "section", FastEmbedder())
    by_id = {q.id: q for q in questions}
    path = ANSWERS_DIR / f"{config}.jsonl"
    cache = _load_cache(path)
    mode = CONFIGS[config][0]
    changed = 0
    for rec in cache.values():
        if rec.get("error") or not rec.get("answer"):
            continue
        text = normalize_citations(rec["answer"])
        ids = extract_citations(text)
        if text == rec["answer"] and ids == rec["cited"]:
            continue
        if mode == "rag":
            sources = rag_sources(by_id[rec["qid"]].question, retriever)
        else:
            toolbox = Toolbox(retriever, s.financials_db_path)
            for st in rec["steps"]:
                toolbox.execute(st["tool"], st["arguments"])
            sources = toolbox.sources
        cited = [i for i in ids if i in sources]
        if cited != rec["cited"]:
            rec.pop("judge", None)  # verdict was based on the wrong sources
            rec.pop("numeric_match", None)
            changed += 1
        rec |= {"answer": text, "cited": cited, "cited_texts": [sources[i].text for i in cited]}
    _save_cache(path, cache)
    print(f"[{config}] repaired citations in {changed} answers (queued for re-grading)")
    return changed


# ------------------------------------------------------------------------------ grading


def check_models(models: list[str]) -> None:
    """Fail fast, before any tokens are spent, if a model is not available to this key."""
    from finrag.agent.llm import OpenAICompatibleLLM

    s = get_settings()
    available = OpenAICompatibleLLM("", s.groq_api_key.get_secret_value()).available_models()
    missing = [m for m in models if m not in available]
    if missing:
        raise SystemExit(
            f"Not available to your API key: {missing}\nAvailable models:\n  "
            + "\n  ".join(available)
        )


def grade(config: str, questions: list[Question], judge_model: str, sleep_s: float) -> None:
    from finrag.agent.llm import OpenAICompatibleLLM

    s = get_settings()
    # The verdict is ~60 tokens of JSON. Without these settings a reasoning model asks for
    # ~2,000 output tokens per call, which Groq's free tier (1,000 output tokens/minute)
    # rejects outright.
    judge_llm = OpenAICompatibleLLM(
        judge_model,
        s.groq_api_key.get_secret_value(),
        max_tokens=JUDGE_MAX_TOKENS,
        extra_body={"reasoning_effort": "none"},
    )
    path = ANSWERS_DIR / f"{config}.jsonl"
    cache = _load_cache(path)
    by_id = {q.id: q for q in questions}
    todo = [r for r in cache.values() if "judge" not in r and not r.get("error")]
    print(f"[{config}] {len(todo)} to grade with {judge_model}")
    for rec in todo:
        q = by_id[rec["qid"]]
        rec["numeric_match"] = numeric_match(q, rec["answer"])
        try:
            j = judge(judge_llm, q, rec["answer"], rec["cited_texts"])
            rec["judge"] = asdict(j) | {"hallucinated": j.hallucinated, "model": judge_model}
        except Exception as exc:
            if _is_daily_limit(str(exc)):
                raise DailyLimitReached(f"{judge_model}: {exc}") from exc
            print(f"  judge failed on {q.id}: {exc}")
            continue
        _save_cache(path, cache)
        if sleep_s:
            time.sleep(sleep_s)


# ------------------------------------------------------------------------------ report


def _mean(xs: list[float]) -> float | None:
    return round(statistics.mean(xs), 3) if xs else None


def summarize(records: list[dict[str, Any]], questions: dict[str, Question]) -> dict[str, Any]:
    graded = [r for r in records if "judge" in r]
    ans = [r for r in graded if questions[r["qid"]].answerable]
    unans = [r for r in graded if not questions[r["qid"]].answerable]
    substantive = [r for r in graded if not r["judge"]["abstained"]]
    numeric = [r for r in ans if r.get("numeric_match") is not None]
    agree = [r for r in numeric if r["numeric_match"] == r["judge"]["correct"]]
    return {
        "n": len(records),
        "graded": len(graded),
        "accuracy": _mean([r["judge"]["correct"] for r in ans]),
        "numeric_accuracy": _mean([r["numeric_match"] for r in numeric]),
        "correct_abstention": _mean([r["judge"]["abstained"] for r in unans]),
        "over_abstention": _mean([r["judge"]["abstained"] for r in ans]),
        "hallucination_rate": _mean([r["judge"]["hallucinated"] for r in graded]),
        "citation_rate": _mean([bool(r["cited"]) for r in substantive]),
        "judge_numeric_agreement": round(len(agree) / len(numeric), 3) if numeric else None,
        "avg_tokens": round(statistics.mean(r["tokens"] for r in records)) if records else None,
        "avg_llm_calls": _mean([r["llm_calls"] for r in records]),
        "avg_seconds": _mean([r["seconds"] for r in records]),
        "step_limit_rate": _mean([r["hit_step_limit"] for r in records]),
        "errors": sum(bool(r.get("error")) for r in records),
    }


def report(questions: list[Question]) -> None:
    """Summarize every config with saved answers, not just the ones run this time.

    A run limited with --configs must not drop the other configs from the summary file
    the app and README read.
    """
    qmap = {q.id: q for q in questions}
    rows = []
    for config in CONFIGS:
        records = list(_load_cache(ANSWERS_DIR / f"{config}.jsonl").values())
        if not records:
            continue
        rows.append({"config": config, "category": "ALL"} | summarize(records, qmap))
        for cat in sorted({q.category for q in questions}):
            sub = [r for r in records if r["category"] == cat]
            if sub:
                rows.append({"config": config, "category": cat} | summarize(sub, qmap))
    if not rows:
        print("no answers yet")
        return
    with (RESULTS / "answer_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    cols = [
        "graded", "accuracy", "numeric_accuracy", "correct_abstention",
        "hallucination_rate", "citation_rate", "avg_tokens", "avg_seconds", "errors",
    ]  # fmt: skip
    print("| config | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for r in rows:
        if r["category"] == "ALL":
            print(f"| {r['config']} | " + " | ".join(str(r[c]) for c in cols) + " |")
    print(f"\nWrote {RESULTS / 'answer_summary.csv'}")


def export_audit(configs: list[str], n: int, seed: int = 0) -> Path:
    """Random sample of judged answers for a human to label, to validate the judge."""
    rows = []
    for config in configs:
        rows += [r for r in _load_cache(ANSWERS_DIR / f"{config}.jsonl").values() if "judge" in r]
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    out = RESULTS / "judge_audit.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["config", "qid", "answer", "judge_correct", "judge_hallucinated",
                    "human_correct", "human_hallucinated"])  # fmt: skip
        for r in sample:
            w.writerow([r["config"], r["qid"], r["answer"], r["judge"]["correct"],
                        r["judge"]["hallucinated"], "", ""])  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate, grade and report answers.")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE)
    p.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between calls (free tier)")
    p.add_argument("--stage", choices=["all", "repair", "generate", "grade", "report", "audit"],
                   default="all")  # fmt: skip
    p.add_argument("--list-models", action="store_true", help="show models your key can use")
    a = p.parse_args(argv)

    if a.list_models:
        from finrag.agent.llm import OpenAICompatibleLLM

        key = get_settings().groq_api_key.get_secret_value()
        print("\n".join(OpenAICompatibleLLM("", key).available_models()))
        return
    needed = []
    if a.stage in ("all", "generate"):
        needed += [CONFIGS[c][1] for c in a.configs]
    if a.stage in ("all", "grade"):
        needed.append(a.judge_model)
    if needed:
        check_models(sorted(set(needed)))

    questions = load_questions(a.questions)
    if a.limit:
        questions = questions[: a.limit]
    # Each model has its own daily quota (free tier: 200k tokens/model/day). When one is
    # spent, skip only the work that needs that model, so every quota is used each day.
    if a.stage == "repair":
        for config in a.configs:
            repair(config, load_questions(a.questions))
        return
    exhausted: set[str] = set()
    for config in a.configs:
        model = CONFIGS[config][1]
        if a.stage in ("all", "generate") and model not in exhausted:
            try:
                generate(config, questions, a.sleep)
            except DailyLimitReached:
                print(f"  daily quota spent for {model}; skipping its remaining configs")
                exhausted.add(model)
        if a.stage in ("all", "grade") and a.judge_model not in exhausted:
            try:
                grade(config, questions, a.judge_model, a.sleep)
            except DailyLimitReached:
                print(f"  daily quota spent for judge {a.judge_model}; grading resumes next run")
                exhausted.add(a.judge_model)
    if exhausted:
        print(f"\nQuota reached for: {', '.join(sorted(exhausted))}. Progress is saved; "
              "rerun the same command tomorrow.")  # fmt: skip
    if a.stage in ("all", "report"):
        report(load_questions(a.questions))
    if a.stage == "audit":
        print(f"Wrote {export_audit(a.configs, 25)}: fill in the human_* columns")


if __name__ == "__main__":
    main()
