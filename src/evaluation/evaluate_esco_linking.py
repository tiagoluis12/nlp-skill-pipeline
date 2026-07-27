from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from src.esco.ranker import load_span_ranker, rank_candidate_texts
from src.extraction.lexicon_baseline import build_lexicon_index, load_esco_skills, match_span_detail


def _load_review(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _acc_at_k(predictions: Sequence[str], gold: str, k: int) -> bool:
    return gold in predictions[:k]


def _mrr(predictions: Sequence[str], gold: str) -> float:
    for rank, uri in enumerate(predictions, start=1):
        if uri == gold:
            return 1.0 / rank
    return 0.0


def _evaluate_rows(rows: list[dict[str, str]], system: str, ranker, lexicon_index: dict[str, str]) -> dict[str, float]:
    gold_rows = [row for row in rows if row.get("gold_concept_uri")]
    if not gold_rows:
        raise ValueError("No gold_concept_uri values found in the review file.")

    acc1 = 0
    acc3 = 0
    mrr_sum = 0.0

    for row in gold_rows:
        gold = row["gold_concept_uri"].strip()
        span_text = row["span_text"]

        if system == "baseline1":
            result = match_span_detail(span_text, lexicon_index)
            predictions = [result.concept_uri] if result.concept_uri else []
        elif system == "baseline2":
            scores, indices = rank_candidate_texts([span_text], ranker, top_k=5)
            predictions = [ranker.artifacts.concept_uris[idx] for idx in indices[0] if idx >= 0]
        else:
            raise ValueError(f"Unsupported system: {system}")

        acc1 += int(_acc_at_k(predictions, gold, 1))
        acc3 += int(_acc_at_k(predictions, gold, 3))
        mrr_sum += _mrr(predictions, gold)

    total = len(gold_rows)
    return {
        "documents": total,
        "acc_at_1": acc1 / total,
        "acc_at_3": acc3 / total,
        "mrr": mrr_sum / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ESCO concept linking from a reviewed gold file.")
    parser.add_argument("--review-csv", type=Path, default=Path("results/gold_esco_review.csv"))
    parser.add_argument("--esco-csv", type=Path, default=Path("data/benchmarks/esco/skills_en.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("models/esco_embeddings.npy"))
    parser.add_argument("--index", type=Path, default=Path("models/esco_faiss.index"))
    parser.add_argument("--labels", type=Path, default=Path("models/esco_embedding_labels.json"))
    parser.add_argument("--system", choices=["baseline1", "baseline2"], default="baseline2")
    parser.add_argument("--output", type=Path, default=Path("results/esco_linking_metrics.json"))
    args = parser.parse_args()

    rows = _load_review(args.review_csv)
    skills = load_esco_skills(args.esco_csv)
    lexicon_index = build_lexicon_index(skills)
    ranker = load_span_ranker(args.embeddings, args.index, args.labels)
    report = _evaluate_rows(rows, args.system, ranker, lexicon_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
