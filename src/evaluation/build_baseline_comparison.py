from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "system",
    "dataset",
    "documents",
    "token_precision",
    "token_recall",
    "token_f1",
    "span_exact_precision",
    "span_exact_recall",
    "span_exact_f1",
    "span_relaxed_precision",
    "span_relaxed_recall",
    "span_relaxed_f1",
    "esco_acc_at_1",
    "esco_acc_at_3",
    "esco_mrr",
    "notes",
]


def _load_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(system: str, dataset: str, report: dict[str, object], notes: str) -> dict[str, object]:
    token = report.get("token_metrics") or {}
    exact = report.get("span_exact") or {}
    relaxed = report.get("span_relaxed") or {}
    return {
        "system": system,
        "dataset": dataset,
        "documents": report.get("documents"),
        "token_precision": token.get("precision"),
        "token_recall": token.get("recall"),
        "token_f1": token.get("f1"),
        "span_exact_precision": exact.get("precision"),
        "span_exact_recall": exact.get("recall"),
        "span_exact_f1": exact.get("f1"),
        "span_relaxed_precision": relaxed.get("precision"),
        "span_relaxed_recall": relaxed.get("recall"),
        "span_relaxed_f1": relaxed.get("f1"),
        "esco_acc_at_1": "",
        "esco_acc_at_3": "",
        "esco_mrr": "",
        "notes": notes,
    }


def build_baseline_comparison(
    baseline1_report: Path,
    baseline2_report: Path,
    output: Path,
) -> list[dict[str, object]]:
    baseline1 = _load_report(baseline1_report)
    baseline2 = _load_report(baseline2_report)

    rows = [
        _row("Baseline1_Lexicon", str(baseline1.get("source", "gold")), baseline1, "ESCO gold labels not available in current gold standard"),
        _row("Baseline2_Embedding", str(baseline2.get("source", "gold")), baseline2, "ESCO gold labels not available in current gold standard"),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 3 baseline comparison table.")
    parser.add_argument("--baseline1", type=Path, default=Path("results/baseline1_gold.json"))
    parser.add_argument("--baseline2", type=Path, default=Path("results/baseline2_gold.json"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline_comparison.csv"))
    args = parser.parse_args()

    rows = build_baseline_comparison(args.baseline1, args.baseline2, args.output)
    print(json.dumps({"output": str(args.output), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
