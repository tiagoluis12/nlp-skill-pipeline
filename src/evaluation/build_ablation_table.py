"""Build the Phase 5.6 ablation summary table from JSON reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "ablation_table.csv"

FIELDNAMES = [
    "ablation",
    "config_description",
    "delta_reference",
    "span_f1_delta",
    "esco_acc1_delta",
    "esco_acc3_delta",
    "mrr_delta",
    "interpretation",
]


def _read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: float) -> str:
    return f"{value:.6f}"


def _full_pipeline_row(report: dict[str, Any]) -> dict[str, str]:
    delta = report["delta_vs_full_pipeline"]
    return {
        "ablation": str(report["ablation"]),
        "config_description": str(report["description"]),
        "delta_reference": "full proposed pipeline",
        "span_f1_delta": _number(float(delta["span_exact_f1"])),
        "esco_acc1_delta": _number(float(delta["esco_acc_at_1"])),
        "esco_acc3_delta": _number(float(delta["esco_acc_at_3"])),
        "mrr_delta": _number(float(delta["mrr"])),
        "interpretation": _interpretation(report),
    }


def _interpretation(report: dict[str, Any]) -> str:
    value = report.get("interpretation")
    if value:
        return str(value)
    ablation = str(report["ablation"])
    if ablation == "A1":
        return "Removing embedding ranking reduces ESCO linking performance while extraction remains unchanged."
    return "Reported delta against the corrected full proposed pipeline."


def _a3_row(report: dict[str, Any]) -> dict[str, str]:
    gap = report["domain_transfer_gap"]
    return {
        "ablation": "A3",
        "config_description": str(report["description"]),
        "delta_reference": "SKILLSPAN test minus gold standard",
        "span_f1_delta": _number(float(gap["span_exact"]["f1"])),
        "esco_acc1_delta": "N/A",
        "esco_acc3_delta": "N/A",
        "mrr_delta": "N/A",
        "interpretation": _interpretation(report),
    }


def _a4_row(report: dict[str, Any]) -> dict[str, str]:
    delta = report["delta_minilm_minus_mpnet"]
    return {
        "ablation": "A4",
        "config_description": str(report["description"]),
        "delta_reference": "MiniLM minus MPNet",
        "span_f1_delta": _number(0.0),
        "esco_acc1_delta": _number(float(delta["acc_at_1"])),
        "esco_acc3_delta": _number(float(delta["acc_at_3"])),
        "mrr_delta": _number(float(delta["mrr"])),
        "interpretation": _interpretation(report),
    }


def build_ablation_rows(results_dir: Path = DEFAULT_RESULTS_DIR) -> list[dict[str, str]]:
    reports = {name: _read_report(results_dir / f"ablation_{name.lower()}.json") for name in ("A1", "A2", "A3", "A4")}
    return [
        _full_pipeline_row(reports["A1"]),
        _full_pipeline_row(reports["A2"]),
        _a3_row(reports["A3"]),
        _a4_row(reports["A4"]),
    ]


def write_ablation_table(rows: list[dict[str, str]], output_path: Path = DEFAULT_OUTPUT) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 5.6 ablation summary table.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = build_ablation_rows(args.results_dir)
    write_ablation_table(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
