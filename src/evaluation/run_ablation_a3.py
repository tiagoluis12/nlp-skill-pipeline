"""Run ablation A3: NER domain-transfer analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_phase5 import _read_json


DEFAULT_IN_DISTRIBUTION = Path("results/ner_skillspan.json")
DEFAULT_OUT_OF_DISTRIBUTION = Path("results/ner_gold.json")
DEFAULT_OUTPUT = Path("results/ablation_a3.json")


METRIC_SECTIONS = ("token_metrics", "span_exact", "span_relaxed")
METRIC_NAMES = ("precision", "recall", "f1")
LABELS = ("SKILL_TECH", "SKILL_TRANS")


def _metric_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "source": report.get("source"),
        "split": report.get("split"),
        "documents": report.get("documents"),
    }
    for section in METRIC_SECTIONS:
        values = report.get(section) or {}
        snapshot[section] = {
            name: values.get(name) for name in METRIC_NAMES
        }
    per_label = ((report.get("classification_report") or {}).get("per_label") or {})
    snapshot["per_class"] = {
        label: {
            name: (per_label.get(label) or {}).get(name)
            for name in METRIC_NAMES
        }
        for label in LABELS
    }
    return snapshot


def _gap_value(in_value: Any, out_value: Any) -> float | None:
    if in_value is None or out_value is None:
        return None
    return float(in_value) - float(out_value)


def _transfer_gaps(in_report: dict[str, Any], out_report: dict[str, Any]) -> dict[str, Any]:
    gaps: dict[str, Any] = {}
    for section in METRIC_SECTIONS:
        gaps[section] = {
            name: _gap_value(
                (in_report.get(section) or {}).get(name),
                (out_report.get(section) or {}).get(name),
            )
            for name in METRIC_NAMES
        }

    in_labels = ((in_report.get("classification_report") or {}).get("per_label") or {})
    out_labels = ((out_report.get("classification_report") or {}).get("per_label") or {})
    gaps["per_class"] = {
        label: {
            name: _gap_value(
                (in_labels.get(label) or {}).get(name),
                (out_labels.get(label) or {}).get(name),
            )
            for name in METRIC_NAMES
        }
        for label in LABELS
    }
    return gaps


def run_ablation_a3(
    in_distribution_path: Path = DEFAULT_IN_DISTRIBUTION,
    out_of_distribution_path: Path = DEFAULT_OUT_OF_DISTRIBUTION,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    in_report = _read_json(in_distribution_path)
    out_report = _read_json(out_of_distribution_path)
    report: dict[str, Any] = {
        "ablation": "A3",
        "description": "Domain transfer of the SKILLSPAN-trained NER model",
        "model": out_report.get("model_name") or in_report.get("model_name"),
        "in_distribution": _metric_snapshot(in_report),
        "out_of_distribution": _metric_snapshot(out_report),
        "domain_transfer_gap": _transfer_gaps(in_report, out_report),
        "gap_definition": "in_distribution minus out_of_distribution; positive values indicate performance loss after domain transfer",
        "interpretation": (
            "The gold-standard corpus is out-of-distribution relative to SKILLSPAN. "
            "A positive gap indicates degradation on the collected job-posting corpus; "
            "a zero gap indicates no change."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5.4 ablation A3.")
    parser.add_argument("--in-distribution", type=Path, default=DEFAULT_IN_DISTRIBUTION)
    parser.add_argument("--out-of-distribution", type=Path, default=DEFAULT_OUT_OF_DISTRIBUTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_ablation_a3(
        in_distribution_path=args.in_distribution,
        out_of_distribution_path=args.out_of_distribution,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
