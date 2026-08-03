"""Run ablation A2: Baseline 2 spans with the full ESCO ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.esco.ranker import load_span_ranker, rank_document_candidates, rank_span_candidates, select_predictions
from src.evaluation.evaluate_phase5 import (
    _link_predictions,
    _read_json,
    _read_jsonl,
    _review_index,
)
from src.evaluation.run_ablation_a1 import (
    DEFAULT_ESCO_CSV,
    DEFAULT_FULL_TABLE,
    _deltas,
    _extraction_metrics,
    _load_full_pipeline_row,
    _numeric,
)
from src.extraction.lexicon_baseline import build_lexicon_index, load_esco_skills


DEFAULT_GOLD_JSONL = Path("data/annotated/gold_annotator1.normalized.jsonl")
DEFAULT_EMBEDDINGS = Path("models/esco_embeddings.npy")
DEFAULT_INDEX = Path("models/esco_faiss.index")
DEFAULT_LABELS = Path("models/esco_embedding_labels.json")
DEFAULT_OUTPUT = Path("results/ablation_a2.json")
DEFAULT_THRESHOLD_REPORT = Path("results/b2_threshold_search.json")
DEFAULT_EXTRACTION_REPORT = Path("results/baseline2_gold.json")
DEFAULT_REVIEW_CSV = Path("results/gold_esco_review_priority.csv")


def _a2_end_to_end(
    gold_rows: list[dict[str, Any]],
    ranker: Any,
    lexicon_index: dict[str, str],
    review_by_span: dict[tuple[str, int, int], str],
    threshold: float,
) -> dict[str, int | float]:
    correct_postings: set[str] = set()
    postings: set[str] = set()
    predicted_spans = 0
    exact_gold_spans = 0
    correct_links = 0

    for row in gold_rows:
        posting_id = str(row["id"])
        postings.add(posting_id)
        candidates = rank_document_candidates(row["tokens"], ranker)
        selected = select_predictions(candidates, threshold)
        for candidate in selected:
            predicted_spans += 1
            ranked = rank_span_candidates(candidate.text, ranker, lexicon_index=lexicon_index, top_k=5)
            predicted_uri = str(ranked[0]["concept_uri"]) if ranked else None
            gold_uri = review_by_span.get((posting_id, candidate.start, candidate.end))
            if not gold_uri:
                continue
            exact_gold_spans += 1
            if predicted_uri == gold_uri:
                correct_links += 1
                correct_postings.add(posting_id)

    return {
        "correct_postings": len(correct_postings),
        "postings": len(postings),
        "coverage": len(correct_postings) / len(postings) if postings else 0.0,
        "predicted_spans": predicted_spans,
        "exact_gold_spans": exact_gold_spans,
        "correct_links_on_exact_spans": correct_links,
        "exact_span_link_accuracy": correct_links / exact_gold_spans if exact_gold_spans else 0.0,
    }


def run_ablation_a2(
    gold_jsonl: Path = DEFAULT_GOLD_JSONL,
    extraction_report_path: Path = DEFAULT_EXTRACTION_REPORT,
    threshold_report_path: Path = DEFAULT_THRESHOLD_REPORT,
    full_table_path: Path = DEFAULT_FULL_TABLE,
    review_csv: Path = DEFAULT_REVIEW_CSV,
    esco_csv: Path = DEFAULT_ESCO_CSV,
    embeddings: Path = DEFAULT_EMBEDDINGS,
    index: Path = DEFAULT_INDEX,
    labels: Path = DEFAULT_LABELS,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    gold_rows = _read_jsonl(gold_jsonl)
    extraction_report = _read_json(extraction_report_path)
    threshold = float(_read_json(threshold_report_path)["best_threshold"])
    full_row = _load_full_pipeline_row(full_table_path)
    review_rows, review_by_span = _review_index(review_csv)
    lexicon_index = build_lexicon_index(load_esco_skills(esco_csv))
    ranker = load_span_ranker(embeddings, index, labels)

    # This is intentionally the same ranker evaluated by the full pipeline.
    linking, _ = _link_predictions(
        review_rows,
        "Proposed_NER_Ranking",
        ranker=ranker,
        lexicon_index=lexicon_index,
    )
    end_to_end = _a2_end_to_end(gold_rows, ranker, lexicon_index, review_by_span, threshold)
    report: dict[str, Any] = {
        "ablation": "A2",
        "description": "Baseline 2 noun-phrase spans + full exact/fuzzy and embedding ESCO ranker; NER removed",
        "dataset": "gold",
        "threshold": threshold,
        "threshold_source": str(threshold_report_path),
        "review_csv": str(review_csv),
        "extraction": _extraction_metrics(extraction_report),
        "esco_linking_on_reviewed_gold_spans": linking,
        "end_to_end": end_to_end,
        "full_pipeline_reference": {
            "system": "Proposed_NER_Ranking",
            "dataset": "gold",
            "span_exact_f1": _numeric(full_row["span_exact_f1"]),
            "span_relaxed_f1": _numeric(full_row["span_relaxed_f1"]),
            "esco_acc_at_1": _numeric(full_row["esco_acc_at_1"]),
            "esco_acc_at_3": _numeric(full_row["esco_acc_at_3"]),
            "mrr": _numeric(full_row["esco_mrr"]),
            "end_to_end_coverage": _numeric(full_row["end_to_end_coverage"]),
        },
        "interpretation": (
            "The reviewed-gold linking scores are expected to equal the full pipeline because both use "
            "the same ranker on the same reviewed spans. The end-to-end coverage isolates the effect of "
            "replacing NER with Baseline 2 noun-phrase spans."
        ),
    }
    report["delta_vs_full_pipeline"] = _deltas(
        {
            "extraction": report["extraction"],
            "esco_linking": report["esco_linking_on_reviewed_gold_spans"],
            "end_to_end": report["end_to_end"],
        },
        full_row,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5.3 ablation A2.")
    parser.add_argument("--gold-jsonl", type=Path, default=DEFAULT_GOLD_JSONL)
    parser.add_argument("--extraction-report", type=Path, default=DEFAULT_EXTRACTION_REPORT)
    parser.add_argument("--threshold-report", type=Path, default=DEFAULT_THRESHOLD_REPORT)
    parser.add_argument("--full-table", type=Path, default=DEFAULT_FULL_TABLE)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--esco-csv", type=Path, default=DEFAULT_ESCO_CSV)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_ablation_a2(
        gold_jsonl=args.gold_jsonl,
        extraction_report_path=args.extraction_report,
        threshold_report_path=args.threshold_report,
        full_table_path=args.full_table,
        review_csv=args.review_csv,
        esco_csv=args.esco_csv,
        embeddings=args.embeddings,
        index=args.index,
        labels=args.labels,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
