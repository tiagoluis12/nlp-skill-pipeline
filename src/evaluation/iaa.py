from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from sklearn.metrics import cohen_kappa_score


def load_conll_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        labels.append(parts[-1])
    return labels


def load_jsonl_tasks(path: Path) -> dict[str, dict[str, list[str]]]:
    tasks: dict[str, dict[str, list[str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("source_id", row.get("id"))
        if task_id is None:
            raise ValueError(f"Missing source_id/id in {path}")
        task_key = str(task_id)
        tasks[task_key] = {
            "tokens": row.get("tokens") or [],
            "labels": row.get("labels") or [],
        }
    return tasks


def sort_task_ids(task_ids: Iterable[str]) -> list[str]:
    def task_key(value: str) -> tuple[int, int | str]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted(task_ids, key=task_key)


def load_aligned_jsonl_labels(path_a: Path, path_b: Path) -> tuple[list[str], list[str], int]:
    tasks_a = load_jsonl_tasks(path_a)
    tasks_b = load_jsonl_tasks(path_b)

    ids_a = set(tasks_a)
    ids_b = set(tasks_b)
    if ids_a != ids_b:
        missing_from_a = sort_task_ids(ids_b - ids_a)
        missing_from_b = sort_task_ids(ids_a - ids_b)
        raise ValueError(
            "Task id mismatch between annotation files: "
            f"missing_from_a={missing_from_a}, missing_from_b={missing_from_b}"
        )

    labels_a: list[str] = []
    labels_b: list[str] = []
    ordered_task_ids = sort_task_ids(ids_a)
    for task_id in ordered_task_ids:
        task_a = tasks_a[task_id]
        task_b = tasks_b[task_id]
        tokens_a = task_a["tokens"]
        tokens_b = task_b["tokens"]
        if tokens_a != tokens_b:
            raise ValueError(f"Token mismatch for task {task_id}")

        task_labels_a = task_a["labels"]
        task_labels_b = task_b["labels"]
        if len(task_labels_a) != len(task_labels_b):
            raise ValueError(
                f"Label length mismatch for task {task_id}: "
                f"{len(task_labels_a)} vs {len(task_labels_b)}"
            )

        labels_a.extend(task_labels_a)
        labels_b.extend(task_labels_b)

    return labels_a, labels_b, len(ordered_task_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute token-level Cohen's kappa for two CoNLL files.")
    parser.add_argument("--labels-a", type=Path, required=True)
    parser.add_argument("--labels-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_count: int | None = None
    if args.labels_a.suffix == ".jsonl" and args.labels_b.suffix == ".jsonl":
        labels_a, labels_b, task_count = load_aligned_jsonl_labels(args.labels_a, args.labels_b)
    else:
        labels_a = load_conll_labels(args.labels_a)
        labels_b = load_conll_labels(args.labels_b)

    if len(labels_a) != len(labels_b):
        raise ValueError(f"Label length mismatch: {len(labels_a)} vs {len(labels_b)}")

    kappa = cohen_kappa_score(labels_a, labels_b)
    agreement = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a) if labels_a else 0.0
    result = {
        "labels_a": str(args.labels_a),
        "labels_b": str(args.labels_b),
        "tokens": len(labels_a),
        "accuracy": agreement,
        "kappa": kappa,
    }
    if task_count is not None:
        result["tasks"] = task_count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
