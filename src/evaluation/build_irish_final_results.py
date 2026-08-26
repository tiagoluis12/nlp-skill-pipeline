"""Build the frozen Phase 5.1 results table for the Irish evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from sklearn.metrics import f1_score

from src.extraction.lexicon_baseline import build_lexicon_index, load_esco_skills, match_span_detail


SYSTEMS = ("Baseline1_Lexicon", "Baseline2_Embedding", "Proposed_NER_Ranking")
OUTPUT_FIELDS = [
    "system", "dataset", "documents", "token_precision", "token_recall", "token_f1",
    "span_exact_precision", "span_exact_recall", "span_exact_f1",
    "span_relaxed_precision", "span_relaxed_recall", "span_relaxed_f1",
    "skill_tech_precision", "skill_tech_recall", "skill_tech_f1",
    "skill_trans_precision", "skill_trans_recall", "skill_trans_f1",
    "esco_acc_at_1", "esco_acc_at_3", "esco_mrr", "esco_macro_f1", "esco_evaluation_units",
    "end_to_end_coverage", "end_to_end_correct_postings", "end_to_end_postings", "notes",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _value(value: Any) -> float | int | str:
    return "" if value is None else value


def _metric(report: dict[str, Any], section: str, key: str) -> float | str:
    return _value((report.get(section) or {}).get(key))


def _class_metric(report: dict[str, Any], label: str, key: str) -> float | str:
    values = ((report.get("classification_report") or {}).get("per_label") or {}).get(label) or {}
    return _value(values.get(key))


def _normalise_baseline2_test(report: dict[str, Any]) -> dict[str, Any]:
    if "thresholds" not in report:
        return report
    result = dict(report["thresholds"][0])
    result["documents"] = report["documents"]
    result["source"] = "skillspan"
    return result


def _linking_macro(rows: Sequence[dict[str, str]], system: str) -> float:
    reviewed = [row for row in rows if (row.get("gold_concept_uri") or "").strip()]
    gold: list[str] = []
    predicted: list[str] = []
    for row in reviewed:
        gold_uri = row["gold_concept_uri"].strip()
        if system == "Baseline1_Lexicon":
            uri = (row.get("lexicon_uri") or "").strip()
        elif system == "Baseline2_Embedding":
            uri = (row.get("candidate_1_uri") or "").strip()
        else:
            uri = (row.get("lexicon_uri") or row.get("candidate_1_uri") or "").strip()
        gold.append(gold_uri)
        predicted.append(uri or "__NO_PREDICTION__")
    return float(f1_score(gold, predicted, labels=sorted(set(gold)), average="macro", zero_division=0))


def _review_index(rows: Sequence[dict[str, str]]) -> dict[tuple[str, int, int], str]:
    return {
        (str(row["posting_id"]), int(row["span_start"]), int(row["span_end"])): row["gold_concept_uri"].strip()
        for row in rows
        if (row.get("gold_concept_uri") or "").strip()
    }


def _coverage_from_predictions(path: Path, review: dict[tuple[str, int, int], str]) -> tuple[int, int]:
    correct: set[str] = set()
    postings: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("role_type"):
            continue
        posting_id = str(record.get("id"))
        postings.add(posting_id)
        for span in record.get("predicted_spans") or []:
            key = (posting_id, int(span["start"]), int(span["end"]))
            if review.get(key) and review[key] == span.get("concept_uri"):
                correct.add(posting_id)
    return len(correct), len(postings)


def _coverage_from_audit(path: Path) -> tuple[int, int]:
    correct: set[str] = set()
    postings: set[str] = set()
    for trace_file in path.glob("*.jsonl"):
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            posting_id = str(row.get("posting_id") or "")
            if not posting_id:
                continue
            postings.add(posting_id)
            if (
                row.get("record_type") == "predicted_span"
                and row.get("gold_uri")
                and row.get("predicted_uri") == row.get("gold_uri")
                and str(row.get("span_start")) == str(row.get("gold_span_start"))
                and str(row.get("span_end")) == str(row.get("gold_span_end"))
            ):
                correct.add(posting_id)
    return len(correct), len(postings)


def _row(
    system: str,
    dataset: str,
    report: dict[str, Any],
    linking: dict[str, Any] | None = None,
    coverage: tuple[int, int] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "system": system,
        "dataset": dataset,
        "documents": report.get("documents", ""),
        "token_precision": _metric(report, "token_metrics", "precision"),
        "token_recall": _metric(report, "token_metrics", "recall"),
        "token_f1": _metric(report, "token_metrics", "f1"),
        "span_exact_precision": _metric(report, "span_exact", "precision"),
        "span_exact_recall": _metric(report, "span_exact", "recall"),
        "span_exact_f1": _metric(report, "span_exact", "f1"),
        "span_relaxed_precision": _metric(report, "span_relaxed", "precision"),
        "span_relaxed_recall": _metric(report, "span_relaxed", "recall"),
        "span_relaxed_f1": _metric(report, "span_relaxed", "f1"),
        "skill_tech_precision": _class_metric(report, "SKILL_TECH", "precision"),
        "skill_tech_recall": _class_metric(report, "SKILL_TECH", "recall"),
        "skill_tech_f1": _class_metric(report, "SKILL_TECH", "f1"),
        "skill_trans_precision": _class_metric(report, "SKILL_TRANS", "precision"),
        "skill_trans_recall": _class_metric(report, "SKILL_TRANS", "recall"),
        "skill_trans_f1": _class_metric(report, "SKILL_TRANS", "f1"),
        "esco_acc_at_1": "", "esco_acc_at_3": "", "esco_mrr": "", "esco_macro_f1": "",
        "esco_evaluation_units": "", "end_to_end_coverage": "",
        "end_to_end_correct_postings": "", "end_to_end_postings": "",
        "notes": "",
    }
    if linking is not None:
        row.update({
            "esco_acc_at_1": linking.get("acc_at_1", ""),
            "esco_acc_at_3": linking.get("acc_at_3", ""),
            "esco_mrr": linking.get("mrr", ""),
            "esco_macro_f1": linking.get("macro_f1", ""),
            "esco_evaluation_units": linking.get("documents", ""),
        })
    if coverage is not None:
        correct, total = coverage
        row.update({
            "end_to_end_coverage": correct / total if total else 0.0,
            "end_to_end_correct_postings": correct,
            "end_to_end_postings": total,
        })
    return row


def build_table(results_dir: Path, review_csv: Path, output: Path, audit_dir: Path) -> list[dict[str, Any]]:
    reports = {
        ("Baseline1_Lexicon", "skillspan_test"): _read_json(results_dir / "baseline1_skillspan.json"),
        ("Baseline2_Embedding", "skillspan_test"): _normalise_baseline2_test(_read_json(results_dir / "baseline2_skillspan.json")),
        ("Proposed_NER_Ranking", "skillspan_test"): _read_json(results_dir / "ner_skillspan.json"),
        ("Baseline1_Lexicon", "gold"): _read_json(results_dir / "baseline1_irish_gold.json"),
        ("Baseline2_Embedding", "gold"): _read_json(results_dir / "baseline2_irish_gold.json"),
        ("Proposed_NER_Ranking", "gold"): _read_json(results_dir / "ner_irish_gold.json"),
    }
    linking_paths = {
        "Baseline1_Lexicon": "esco_linking_metrics_baseline1_irish.json",
        "Baseline2_Embedding": "esco_linking_metrics_baseline2_irish.json",
        "Proposed_NER_Ranking": "esco_linking_metrics_proposed_irish.json",
    }
    with review_csv.open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    review_index = _review_index(review_rows)
    links: dict[str, dict[str, Any]] = {}
    for system, filename in linking_paths.items():
        links[system] = _read_json(results_dir / filename)
        links[system]["macro_f1"] = _linking_macro(review_rows, system)

    baseline1_coverage = _coverage_from_predictions(results_dir / "baseline1_irish_predictions.jsonl", review_index)
    baseline2_predictions = results_dir / "baseline2_irish_predictions.jsonl"
    baseline2_coverage = (
        _coverage_from_predictions(baseline2_predictions, review_index)
        if baseline2_predictions.exists()
        else None
    )
    proposed_coverage = _coverage_from_audit(audit_dir)
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        rows.append(_row(system, "skillspan_test", reports[(system, "skillspan_test")]))
        coverage = (
            proposed_coverage
            if system == "Proposed_NER_Ranking"
            else baseline1_coverage
            if system == "Baseline1_Lexicon"
            else baseline2_coverage
        )
        row = _row(system, "gold", reports[(system, "gold")], links[system], coverage)
        row["notes"] = "Irish gold extraction and reviewed ESCO linking; 1,404 mapped spans, 88 documented coverage gaps excluded."
        if system == "Baseline2_Embedding" and baseline2_coverage is None:
            row["notes"] += " Baseline 2 posting coverage is unavailable because the frozen aggregate report has no per-posting predictions."
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Irish Phase 5.1 final results table.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--review-csv", type=Path, default=Path("results/irish_gold_esco_review_priority.csv"))
    parser.add_argument("--audit-dir", type=Path, default=Path("results/audit_traces_irish"))
    parser.add_argument("--output", type=Path, default=Path("results/final_results_table_irish.csv"))
    args = parser.parse_args()
    rows = build_table(args.results_dir, args.review_csv, args.output, args.audit_dir)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
