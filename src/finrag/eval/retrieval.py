"""Retrieval evaluation: does the right passage come back, and how high?

For each answerable question and each gold evidence quote, find the rank of the first
retrieved chunk that (a) comes from the right filing and (b) contains the quote. Then:

* recall@k: fraction of evidence quotes found in the top k (averaged over questions).
  A comparison question with four quotes and two found scores 0.5, not 0 or 1.
* MRR: mean of 1/rank of the first hit per quote (0 if not found within the depth).
  It rewards ranking the evidence first, not just somewhere in the top 10.

Two settings:
* ``none``: the raw question, no filters. Tests the retriever alone.
* ``oracle``: filter to the tickers and years the evidence comes from. This is an upper
  bound on what an agent gains by extracting "JPM" and "2025" from the question and
  passing them as search filters, which is exactly what the agent's tool will allow.

No LLM is involved, so a full run is free and takes about a minute.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from finrag.config import PROJECT_ROOT, get_settings
from finrag.eval.dataset import DEFAULT_QUESTIONS, Question, load_questions, normalize
from finrag.retrieval.retriever import Retriever, SearchMode

RESULTS_DIR = PROJECT_ROOT / "results"
KS = (1, 5, 10)
DEPTH = 20


@dataclass
class QuestionResult:
    qid: str
    category: str
    ranks: list[int | None]  # per evidence quote: 1-based rank of first hit, or None
    hit_ids: list[str | None] = field(default_factory=list)  # chunk that matched, for audits

    def recall(self, k: int) -> float:
        return sum(r is not None and r <= k for r in self.ranks) / len(self.ranks)

    def mrr(self) -> float:
        return statistics.mean(1 / r if r else 0.0 for r in self.ranks)


def evaluate_question(
    retriever: Retriever, q: Question, mode: SearchMode, oracle_filters: bool
) -> QuestionResult:
    tickers = sorted({e.ticker for e in q.evidence}) if oracle_filters else None
    years = sorted({e.fiscal_year for e in q.evidence}) if oracle_filters else None
    results = retriever.search(q.question, k=DEPTH, mode=mode, tickers=tickers, fiscal_years=years)
    texts = [(r.chunk.ticker, r.chunk.fiscal_year, normalize(r.chunk.text)) for r in results]

    ranks: list[int | None] = []
    hit_ids: list[str | None] = []
    for ev in q.evidence:
        needle = normalize(ev.text)
        rank = next(
            (
                i
                for i, (t, y, txt) in enumerate(texts, 1)
                if t == ev.ticker and y == ev.fiscal_year and needle in txt
            ),
            None,
        )
        ranks.append(rank)
        hit_ids.append(results[rank - 1].chunk.chunk_id if rank else None)
    return QuestionResult(q.id, q.category, ranks, hit_ids)


def summarize(results: list[QuestionResult]) -> dict[str, float]:
    out = {f"recall@{k}": statistics.mean(r.recall(k) for r in results) for k in KS}
    out["mrr"] = statistics.mean(r.mrr() for r in results)
    out["n"] = len(results)
    return out


def run(
    indexes: list[str], modes: list[SearchMode], questions: list[Question], data_dir: Path
) -> tuple[list[dict], list[dict]]:
    from finrag.retrieval.embeddings import FastEmbedder

    embedder = FastEmbedder()
    # Questions answered only from the XBRL table have no text evidence to retrieve.
    qs = [q for q in questions if q.evidence]
    summary_rows, detail_rows = [], []
    for index in indexes:
        retriever = Retriever.load(data_dir / "index" / index, embedder)
        for mode in modes:
            for oracle in (False, True):
                t0 = time.perf_counter()
                res = [evaluate_question(retriever, q, mode, oracle) for q in qs]
                setting = {"index": index, "mode": mode, "filters": "oracle" if oracle else "none"}
                summary_rows.append(
                    setting | {"category": "ALL"} | summarize(res)
                    | {"ms_per_query": round(1000 * (time.perf_counter() - t0) / len(qs), 1)}
                )  # fmt: skip
                by_cat: dict[str, list[QuestionResult]] = defaultdict(list)
                for r in res:
                    by_cat[r.category].append(r)
                for cat, rs in sorted(by_cat.items()):
                    summary_rows.append(setting | {"category": cat} | summarize(rs))
                detail_rows += [
                    setting | {"qid": r.qid, "ranks": r.ranks, "hit_ids": r.hit_ids} for r in res
                ]
    return summary_rows, detail_rows


def _fmt(rows: list[dict]) -> str:
    head = "| index | mode | filters | recall@1 | recall@5 | recall@10 | MRR | ms/query |"
    lines = [head, "|" + "---|" * 8]
    for r in rows:
        if r["category"] != "ALL":
            continue
        lines.append(
            f"| {r['index']} | {r['mode']} | {r['filters']} | {r['recall@1']:.2f} | "
            f"{r['recall@5']:.2f} | {r['recall@10']:.2f} | {r['mrr']:.3f} | {r['ms_per_query']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate retrieval across indexes and modes.")
    p.add_argument("--indexes", nargs="+", default=["fixed", "section"])
    p.add_argument("--modes", nargs="+", default=["dense", "bm25", "hybrid"])
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = p.parse_args(argv)

    summary, detail = run(
        args.indexes, args.modes, load_questions(args.questions), get_settings().data_dir
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    fields = ["index", "mode", "filters", "category", "n", "recall@1", "recall@5",
              "recall@10", "mrr", "ms_per_query"]  # fmt: skip
    with (RESULTS_DIR / "retrieval_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in summary:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with (RESULTS_DIR / "retrieval_details.jsonl").open("w") as f:
        for r in detail:
            f.write(json.dumps(r) + "\n")
    print(_fmt(summary))
    print(f"\nWrote {RESULTS_DIR / 'retrieval_summary.csv'} and retrieval_details.jsonl")


if __name__ == "__main__":
    main()
