"""Run ablation A1: NER extraction with exact/fuzzy ESCO matching only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_phase5 import _link_predictions, _read_json, _read_jsonl, _review_index
from src.extraction.lexicon_baseline import build_lexicon_index, load_esco_skills, match_span_detail


DEFAULT_NER_REPORT = Path("results/ner_gold.json")
DEFAULT_FULL_TABLE = Path("results/final_results_table.csv")
DEFAULT_REVIEW_CSV = Path("results/gold_esco_review_priority.csv")
DEFAULT_ESCO_CSV = Path("data/benchmarks/esco/skills_en.csv")
DEFAULT_AUDIT_DIR = Path("results/audit_traces")
DEFAULT_OUTPUT = Path("results/ablation_a1.json")


def _load_full_pipeline_row(path: Path) -> dict[str, str]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("system") == "Proposed_NER_Ranking" and row.get("dataset") == "gold":
                return row
    raise ValueError(f"Full proposed gold row not found in {path}")


def _numeric(value: Any) -> float:
    if value in (None, ""):
        raise ValueError(f"Expected numeric value, got {value!r}")
    return float(value)


def _a1_coverage(audit_dir: Path, lexicon_index: dict[str, str]) -> dict[str, int | float]:
    correct_postings: set[str] = set()
    postings: set[str] = set()
    predicted_spans = 0
    matched_spans = 0

    for path in sorted(audit_dir.glob("*.jsonl")):
        if path.stem.isdigit():
            postings.add(path.stem)
        for record in _read_jsonl(path):
            posting_id = str(record.get("posting_id", ""))
            if posting_id:
                postings.add(posting_id)
            if record.get("record_type") != "predicted_span":
                continue
            predicted_spans += 1
            result = match_span_detail(str(record.get("span", "")), lexicon_index)
            if result.concept_uri is None:
                continue
            matched_spans += 1
            gold_uri = record.get("gold_uri")
            exact_boundary = (
                str(record.get("span_start")) == str(record.get("gold_span_start"))
                and str(record.get("span_end")) == str(record.get("gold_span_end"))
            )
            if gold_uri and result.concept_uri == gold_uri and exact_boundary:
                correct_postings.add(posting_id)

    return {
        "correct_postings": len(correct_postings),
        "postings": len(postings),
        "coverage": len(correct_postings) / len(postings) if postings else 0.0,
        "predicted_spans": predicted_spans,
        "lexicon_matched_spans": matched_spans,
    }


def _extraction_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "documents": report.get("documents"),
        "token": report.get("token_metrics", {}),
        "span_exact": report.get("span_exact", {}),
        "span_relaxed": report.get("span_relaxed", {}),
        "per_class": (report.get("classification_report") or {}).get("per_label", {}),
    }


def _deltas(a1: dict[str, Any], full: dict[str, str]) -> dict[str, float]:
    return {
        "span_exact_f1": float(a1["extraction"]["span_exact"]["f1"]) - _numeric(full["span_exact_f1"]),
        "span_relaxed_f1": float(a1["extraction"]["span_relaxed"]["f1"]) - _numeric(full["span_relaxed_f1"]),
        "esco_acc_at_1": float(a1["esco_linking"]["acc_at_1"]) - _numeric(full["esco_acc_at_1"]),
        "esco_acc_at_3": float(a1["esco_linking"]["acc_at_3"]) - _numeric(full["esco_acc_at_3"]),
        "mrr": float(a1["esco_linking"]["mrr"]) - _numeric(full["esco_mrr"]),
        "end_to_end_coverage": float(a1["end_to_end"]["coverage"]) - _numeric(full["end_to_end_coverage"]),
    }


def run_ablation_a1(
    ner_report_path: Path = DEFAULT_NER_REPORT,
    full_table_path: Path = DEFAULT_FULL_TABLE,
    review_csv: Path = DEFAULT_REVIEW_CSV,
    esco_csv: Path = DEFAULT_ESCO_CSV,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    ner_report = _read_json(ner_report_path)
    full_row = _load_full_pipeline_row(full_table_path)
    review_rows, _ = _review_index(review_csv)
    lexicon_index = build_lexicon_index(load_esco_skills(esco_csv))

    linking, _ = _link_predictions(
        review_rows,
        "Baseline1_Lexicon",
        ranker=None,
        lexicon_index=lexicon_index,
    )
    coverage = _a1_coverage(audit_dir, lexicon_index)
    report: dict[str, Any] = {
        "ablation": "A1",
        "description": "NER extractor + exact/fuzzy ESCO lexicon matching; embedding ranking removed",
        "dataset": "gold",
        "review_csv": str(review_csv),
        "extraction": _extraction_metrics(ner_report),
        "esco_linking": linking,
        "end_to_end": coverage,
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
    }
    report["delta_vs_full_pipeline"] = _deltas(report, full_row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5.2 ablation A1.")
    parser.add_argument("--ner-report", type=Path, default=DEFAULT_NER_REPORT)
    parser.add_argument("--full-table", type=Path, default=DEFAULT_FULL_TABLE)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--esco-csv", type=Path, default=DEFAULT_ESCO_CSV)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_ablation_a1(
        ner_report_path=args.ner_report,
        full_table_path=args.full_table,
        review_csv=args.review_csv,
        esco_csv=args.esco_csv,
        audit_dir=args.audit_dir,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
