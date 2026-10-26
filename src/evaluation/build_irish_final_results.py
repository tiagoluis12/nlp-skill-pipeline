"""Rebuild the Irish results table from public aggregate reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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
        "end_to_end_correct_postings": "", "end_to_end_postings": "", "notes": "",
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


def build_table(results_dir: Path, output: Path) -> list[dict[str, Any]]:
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
    frozen = _read_json(results_dir / "irish_aggregate_linking.json")
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        rows.append(_row(system, "skillspan_test", reports[(system, "skillspan_test")]))
        linking = _read_json(results_dir / linking_paths[system])
        linking["macro_f1"] = frozen["macro_f1"][system]
        coverage_data = frozen["end_to_end_coverage"][system]
        coverage = (int(coverage_data["correct_postings"]), int(coverage_data["postings"]))
        row = _row(system, "gold", reports[(system, "gold")], linking, coverage)
        row["notes"] = "Irish gold extraction and reviewed ESCO linking; 1,404 mapped spans, 88 documented coverage gaps excluded."
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the Irish results table from aggregate reports.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/final_results_table_irish.csv"))
    args = parser.parse_args()
    rows = build_table(args.results_dir, args.output)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
