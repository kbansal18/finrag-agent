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
CONFIGS: dict[str, tuple[str, str]] = {
    "rag-120b": ("rag", "openai/gpt-oss-120b"),
    "agent-120b": ("agent", "openai/gpt-oss-120b"),
    "rag-20b": ("rag", "openai/gpt-oss-20b"),
    "agent-20b": ("agent", "openai/gpt-oss-20b"),
}
# A different model family from the systems under test (gpt-oss), to limit
# self-preference bias. Check availability with --list-models; it varies by account tier.
DEFAULT_JUDGE = "qwen/qwen3.8-27b"


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
        _save_cache(path, cache)  # after every question: interruptions lose nothing
        flag = "ERROR" if ans.error else ""
        print(f"  {i}/{len(todo)} {q.id} {ans.seconds:.1f}s {flag}")
        if sleep_s:
            time.sleep(sleep_s)


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
    judge_llm = OpenAICompatibleLLM(judge_model, s.groq_api_key.get_secret_value())
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


def report(configs: list[str], questions: list[Question]) -> None:
    qmap = {q.id: q for q in questions}
    rows = []
    for config in configs:
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

    cols = ["accuracy", "numeric_accuracy", "correct_abstention", "hallucination_rate",
            "citation_rate", "avg_tokens", "avg_seconds", "errors"]  # fmt: skip
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
    p.add_argument("--stage", choices=["all", "generate", "grade", "report", "audit"],
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
    for config in a.configs:
        if a.stage in ("all", "generate"):
            generate(config, questions, a.sleep)
        if a.stage in ("all", "grade"):
            grade(config, questions, a.judge_model, a.sleep)
    if a.stage in ("all", "report"):
        report(a.configs, load_questions(a.questions))
    if a.stage == "audit":
        print(f"Wrote {export_audit(a.configs, 25)}: fill in the human_* columns")


if __name__ == "__main__":
    main()
