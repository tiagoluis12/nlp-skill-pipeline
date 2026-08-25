"""Build a human-review queue from two aligned BIO annotation exports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.evaluation.build_iaa_disagreement_report import disagreement_groups, load_tasks


ROLE_PRIORITY = {
    "software_dev": 1,
    "qa_engineer": 2,
    "cloud_support": 3,
    "cybersecurity": 3,
    "data_analyst": 3,
}


FIELDNAMES = [
    "priority",
    "role_type",
    "source_id",
    "token_start",
    "token_end",
    "text",
    "context",
    "annotator1_labels",
    "annotator2_labels",
    "adjudicated_labels",
    "decision",
    "decision_reason",
]


def build_queue(labels_a: Path, labels_b: Path) -> list[dict[str, str | int]]:
    tasks_a = load_tasks(labels_a)
    tasks_b = load_tasks(labels_b)
    if set(tasks_a) != set(tasks_b):
        raise ValueError("The two normalized exports do not contain the same task IDs")

    rows: list[dict[str, str | int]] = []
    for source_id in sorted(tasks_a):
        task_a = tasks_a[source_id]
        task_b = tasks_b[source_id]
        if task_a["tokens"] != task_b["tokens"]:
            raise ValueError(f"Token mismatch for task {source_id}")
        if len(task_a["labels"]) != len(task_b["labels"]):
            raise ValueError(f"Label length mismatch for task {source_id}")

        role = task_a.get("role_type") or "unknown"
        for group in disagreement_groups(task_a["tokens"], task_a["labels"], task_b["labels"]):
            rows.append(
                {
                    "priority": ROLE_PRIORITY.get(role, 4),
                    "role_type": role,
                    "source_id": source_id,
                    "token_start": group["token_start"],
                    "token_end": group["token_end"],
                    "text": group["text"],
                    "context": group["context"],
                    "annotator1_labels": " ".join(group["labels_a"]),
                    "annotator2_labels": " ".join(group["labels_b"]),
                    "adjudicated_labels": "",
                    "decision": "",
                    "decision_reason": "",
                }
            )

    rows.sort(key=lambda row: (row["priority"], str(row["role_type"]), str(row["source_id"]), row["token_start"]))
    return rows


def write_queue(rows: list[dict[str, str | int]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CSV adjudication queue from two normalized BIO exports.")
    parser.add_argument("--labels-a", type=Path, required=True)
    parser.add_argument("--labels-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = build_queue(args.labels_a, args.labels_b)
    write_queue(rows, args.output)
    print(f"Wrote {len(rows)} disagreement groups to {args.output}")


if __name__ == "__main__":
    main()
