"""Evaluate an independent Irish gold spot-check against the original annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.build_iaa_disagreement_report import build_report, write_markdown
from src.evaluation.iaa import load_aligned_jsonl_labels
from src.preprocess.labelstudio_to_conll import export_conll, load_tasks


def source_id(task: dict) -> str:
    return str((task.get("data") or {}).get("source_id") or task.get("id"))


def evaluate_spotcheck(
    original_export: Path,
    spotcheck_export: Path,
    original_jsonl: Path,
    spotcheck_jsonl: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict:
    original_tasks = {source_id(task): task for task in load_tasks(original_export)}
    spotcheck_tasks = load_tasks(spotcheck_export)
    spotcheck_ids = [source_id(task) for task in spotcheck_tasks]
    if len(spotcheck_ids) != len(set(spotcheck_ids)):
        raise ValueError("Spot-check export contains duplicate task IDs")
    missing = sorted(set(spotcheck_ids) - set(original_tasks))
    if missing:
        raise ValueError(f"Spot-check tasks are missing from original export: {missing}")

    original_subset = [original_tasks[task_id] for task_id in spotcheck_ids]
    export_conll(original_subset, original_jsonl.with_suffix(".conll"), original_jsonl)
    export_conll(spotcheck_tasks, spotcheck_jsonl.with_suffix(".conll"), spotcheck_jsonl)

    labels_a, labels_b, task_count = load_aligned_jsonl_labels(original_jsonl, spotcheck_jsonl)
    report = build_report(original_jsonl, spotcheck_jsonl)
    report["tasks"] = task_count
    report["original_export"] = str(original_export)
    report["spotcheck_export"] = str(spotcheck_export)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, output_markdown)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Irish gold spot-check against the original annotations.")
    parser.add_argument("--original-export", type=Path, required=True)
    parser.add_argument("--spotcheck-export", type=Path, required=True)
    parser.add_argument("--original-jsonl", type=Path, required=True)
    parser.add_argument("--spotcheck-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_spotcheck(
        args.original_export,
        args.spotcheck_export,
        args.original_jsonl,
        args.spotcheck_jsonl,
        args.output_json,
        args.output_markdown,
    )
    print(json.dumps({key: report[key] for key in ("tasks", "tokens", "agreement", "kappa", "per_role")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
