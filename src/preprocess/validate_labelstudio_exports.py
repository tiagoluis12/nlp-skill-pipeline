from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def load_export(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Unsupported Label Studio export format in {path}")


def task_key(row: dict) -> str:
    data = row.get("data") or {}
    return str(data.get("source_id") or row.get("id"))


def count_span_results(annotations: Iterable[dict]) -> int:
    total = 0
    for annotation in annotations:
        for result in annotation.get("result", []):
            if result.get("type") == "labels":
                value = result.get("value") or {}
                if value.get("start") is not None and value.get("end") is not None:
                    total += 1
    return total


def summarize_export(path: Path) -> dict:
    rows = load_export(path)
    keys = [task_key(row) for row in rows]
    key_counts = Counter(keys)
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    annotations = [annotation for row in rows for annotation in row.get("annotations", []) if not annotation.get("was_cancelled")]
    span_results = count_span_results(annotations)
    tasks_with_spans = sum(
        1
        for row in rows
        if count_span_results(annotation for annotation in row.get("annotations", []) if not annotation.get("was_cancelled")) > 0
    )
    empty_tasks = [key for key, row in zip(keys, rows, strict=True) if count_span_results(row.get("annotations", [])) == 0]

    return {
        "path": str(path),
        "rows": len(rows),
        "unique_tasks": len(key_counts),
        "duplicate_task_ids": duplicate_keys,
        "duplicate_task_count": len(duplicate_keys),
        "annotations": len(annotations),
        "tasks_with_spans": tasks_with_spans,
        "span_results": span_results,
        "empty_task_ids": empty_tasks,
    }


def validate_exports(paths: list[Path], expected_unique_tasks: int | None = None) -> dict:
    reports = [summarize_export(path) for path in paths]
    errors: list[str] = []

    if expected_unique_tasks is not None:
        for report in reports:
            if report["unique_tasks"] != expected_unique_tasks:
                errors.append(
                    f"{report['path']}: expected {expected_unique_tasks} unique tasks, found {report['unique_tasks']}"
                )

    for report in reports:
        if report["duplicate_task_count"] > 0:
            errors.append(f"{report['path']}: duplicate task ids detected: {', '.join(report['duplicate_task_ids'])}")
        if report["span_results"] == 0:
            errors.append(f"{report['path']}: no span annotations were exported")

    if len(reports) > 1:
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        for index, current_hash in enumerate(hashes):
            for other_index in range(index):
                if current_hash == hashes[other_index]:
                    errors.append(
                        f"{paths[index]} and {paths[other_index]} are byte-for-byte identical; "
                        "independent annotator exports are required"
                    )

        base_keys = set(task_key(row) for row in load_export(paths[0]))
        for other in paths[1:]:
            other_keys = set(task_key(row) for row in load_export(other))
            if base_keys != other_keys:
                missing = sorted(base_keys - other_keys)
                extra = sorted(other_keys - base_keys)
                errors.append(
                    f"{paths[0]} vs {other}: task sets differ; missing={missing[:5]} extra={extra[:5]}"
                )

    return {"reports": reports, "errors": errors, "valid": not errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one or more Label Studio export JSON files.")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Label Studio export JSON files")
    parser.add_argument("--expected-unique-tasks", type=int, default=30)
    parser.add_argument("--output", type=Path, help="Optional JSON report output path")
    args = parser.parse_args()

    report = validate_exports(args.input, expected_unique_tasks=args.expected_unique_tasks)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
