"""Retrieval eval harness — the differentiator.

Loads a labelled test set (question -> the source(s) that should be retrieved),
runs retrieval, and computes Precision@k, Recall@k, and MRR. Persists the last
run so the dashboard can read it.

Why this matters: anyone can wire up RAG. Proving retrieval quality with numbers
is what separates a production engineer from a tutorial follower.
"""
from __future__ import annotations

import json
from pathlib import Path

import rag
from settings import settings

TESTSET = Path(__file__).parent.parent / "eval_data" / "testset.json"
RESULTS = Path(__file__).parent.parent / "eval_data" / "last_run.json"
K = settings.eval_k


def _relevant_sources(hits: list[dict], expected: list[str]) -> list[bool]:
    """Per-hit relevance: does the hit's source match any expected source?"""
    exp = {e.lower() for e in expected}
    return [h["source"].lower() in exp for h in hits]


def run() -> dict:
    if not TESTSET.exists():
        return {"error": f"no test set at {TESTSET}"}
    cases = json.loads(TESTSET.read_text())
    details, precisions, recalls, rr = [], [], [], []

    for case in cases:
        q, expected = case["question"], case["expected_sources"]
        hits = rag.retrieve(q, K)
        flags = _relevant_sources(hits, expected)
        n_rel = sum(flags)

        precision = n_rel / K
        recall = (len(set(h["source"].lower() for h, f in zip(hits, flags) if f))
                  / max(len(set(e.lower() for e in expected)), 1))
        rank = next((i + 1 for i, f in enumerate(flags) if f), None)
        precisions.append(precision)
        recalls.append(min(recall, 1.0))
        rr.append(1.0 / rank if rank else 0.0)
        details.append({
            "question": q, "hit": rank is not None, "rank": rank,
            "top_score": round(hits[0]["score"], 3) if hits else None,
        })

    n = len(cases) or 1
    metrics = {
        "k": K, "n": len(cases),
        "precision_at_k": round(sum(precisions) / n, 3),
        "recall_at_k": round(sum(recalls) / n, 3),
        "mrr": round(sum(rr) / n, 3),
    }
    out = {"metrics": metrics, "details": details}
    RESULTS.write_text(json.dumps(out, indent=2))
    return out


def latest() -> dict | None:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return None


if __name__ == "__main__":
    print(json.dumps(run()["metrics"], indent=2))
