from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _apply_extraction_report(row: dict[str, str], report: dict[str, object]) -> None:
    row["dataset"] = _format_value(report.get("source"))
    row["documents"] = _format_value(report.get("documents"))

    token_metrics = report.get("token_metrics") or {}
    span_exact = report.get("span_exact") or {}
    span_relaxed = report.get("span_relaxed") or {}

    row["token_precision"] = _format_value(token_metrics.get("precision"))
    row["token_recall"] = _format_value(token_metrics.get("recall"))
    row["token_f1"] = _format_value(token_metrics.get("f1"))
    row["span_exact_precision"] = _format_value(span_exact.get("precision"))
    row["span_exact_recall"] = _format_value(span_exact.get("recall"))
    row["span_exact_f1"] = _format_value(span_exact.get("f1"))
    row["span_relaxed_precision"] = _format_value(span_relaxed.get("precision"))
    row["span_relaxed_recall"] = _format_value(span_relaxed.get("recall"))
    row["span_relaxed_f1"] = _format_value(span_relaxed.get("f1"))


def _apply_linking_report(row: dict[str, str], report: dict[str, object]) -> None:
    row["esco_acc_at_1"] = _format_value(report.get("acc_at_1"))
    row["esco_acc_at_3"] = _format_value(report.get("acc_at_3"))
    row["esco_mrr"] = _format_value(report.get("mrr"))


def refresh_baseline_comparison(
    comparison_csv: Path,
    output_csv: Path,
    baseline1_extraction: Path,
    baseline2_extraction: Path,
    proposed_extraction: Path,
    baseline1_linking: Path,
    baseline2_linking: Path,
    proposed_linking: Path,
) -> list[dict[str, str]]:
    rows = list(csv.DictReader(comparison_csv.read_text(encoding="utf-8").splitlines()))
    reports = {
        "Baseline1_Lexicon": (_load_json(baseline1_extraction), _load_json(baseline1_linking)),
        "Baseline2_Embedding": (_load_json(baseline2_extraction), _load_json(baseline2_linking)),
        "Proposed_Pipeline": (_load_json(proposed_extraction), _load_json(proposed_linking)),
    }

    for row in rows:
        system = row.get("system", "")
        if system not in reports:
            continue
        extraction_report, linking_report = reports[system]
        _apply_extraction_report(row, extraction_report)
        _apply_linking_report(row, linking_report)
        if system == "Proposed_Pipeline":
            row["notes"] = "NER evaluated on gold corpus; ESCO linking on reviewed gold spans"
        else:
            row["notes"] = "Gold corpus extraction and reviewed ESCO linking"

    fieldnames = list(rows[0].keys()) if rows else []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh results/baseline_comparison.csv from Phase 4 outputs.")
    parser.add_argument("--comparison-csv", type=Path, default=Path("results/baseline_comparison.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline_comparison.csv"))
    parser.add_argument("--baseline1-extraction", type=Path, default=Path("results/baseline1_gold.json"))
    parser.add_argument("--baseline2-extraction", type=Path, default=Path("results/baseline2_gold.json"))
    parser.add_argument("--proposed-extraction", type=Path, default=Path("results/ner_gold.json"))
    parser.add_argument("--baseline1-linking", type=Path, default=Path("results/esco_linking_metrics_baseline1.json"))
    parser.add_argument("--baseline2-linking", type=Path, default=Path("results/esco_linking_metrics_baseline2.json"))
    parser.add_argument("--proposed-linking", type=Path, default=Path("results/esco_linking_metrics.json"))
    args = parser.parse_args()

    rows = refresh_baseline_comparison(
        comparison_csv=args.comparison_csv,
        output_csv=args.output,
        baseline1_extraction=args.baseline1_extraction,
        baseline2_extraction=args.baseline2_extraction,
        proposed_extraction=args.proposed_extraction,
        baseline1_linking=args.baseline1_linking,
        baseline2_linking=args.baseline2_linking,
        proposed_linking=args.proposed_linking,
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
