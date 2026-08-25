"""Summarize aligned pilot BIO disagreements for annotation calibration."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            tasks[str(row["source_id"])] = row
    return tasks


def disagreement_groups(tokens: list[str], labels_a: list[str], labels_b: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    index = 0
    while index < len(tokens):
        if labels_a[index] == labels_b[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(tokens) and labels_a[index + 1] != labels_b[index + 1]:
            index += 1
        end = index + 1
        context_start = max(0, start - 5)
        context_end = min(len(tokens), end + 5)
        groups.append(
            {
                "token_start": start,
                "token_end": end,
                "text": " ".join(tokens[start:end]),
                "context": " ".join(tokens[context_start:context_end]),
                "labels_a": labels_a[start:end],
                "labels_b": labels_b[start:end],
            }
        )
        index += 1
    return groups


def build_report(path_a: Path, path_b: Path, max_examples: int = 100) -> dict[str, Any]:
    tasks_a = load_tasks(path_a)
    tasks_b = load_tasks(path_b)
    if set(tasks_a) != set(tasks_b):
        raise ValueError("The two normalized exports do not contain the same task IDs")

    all_a: list[str] = []
    all_b: list[str] = []
    role_labels: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    transitions: Counter[tuple[str, str]] = Counter()
    examples: list[dict[str, Any]] = []
    task_disagreements: Counter[str] = Counter()

    for source_id in sorted(tasks_a):
        task_a = tasks_a[source_id]
        task_b = tasks_b[source_id]
        if task_a["tokens"] != task_b["tokens"]:
            raise ValueError(f"Token mismatch for task {source_id}")
        labels_a = task_a["labels"]
        labels_b = task_b["labels"]
        if len(labels_a) != len(labels_b):
            raise ValueError(f"Label length mismatch for task {source_id}")
        role = task_a.get("role_type") or "unknown"
        role_a, role_b = role_labels[role]
        role_a.extend(labels_a)
        role_b.extend(labels_b)
        all_a.extend(labels_a)
        all_b.extend(labels_b)
        for label_a, label_b in zip(labels_a, labels_b, strict=True):
            if label_a != label_b:
                transitions[(label_a, label_b)] += 1
        groups = disagreement_groups(task_a["tokens"], labels_a, labels_b)
        if groups:
            task_disagreements[source_id] = sum(group["token_end"] - group["token_start"] for group in groups)
            for group in groups:
                examples.append(
                    {
                        "source_id": source_id,
                        "role_type": role,
                        "title": (task_a.get("text") or "")[:120],
                        **group,
                    }
                )

    per_role = {}
    for role, (labels_a, labels_b) in sorted(role_labels.items()):
        per_role[role] = {
            "tokens": len(labels_a),
            "agreement": sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / len(labels_a),
            "kappa": cohen_kappa_score(labels_a, labels_b),
            "disagreements": sum(a != b for a, b in zip(labels_a, labels_b, strict=True)),
        }
    examples.sort(key=lambda row: (-len(row["text"].split()), row["source_id"], row["token_start"]))
    return {
        "tasks": len(tasks_a),
        "tokens": len(all_a),
        "agreement": sum(a == b for a, b in zip(all_a, all_b, strict=True)) / len(all_a),
        "kappa": cohen_kappa_score(all_a, all_b),
        "label_counts_a": dict(Counter(all_a)),
        "label_counts_b": dict(Counter(all_b)),
        "top_transitions": [
            {"annotator_a": a, "annotator_b": b, "count": count}
            for (a, b), count in transitions.most_common()
            if a != b
        ],
        "per_role": per_role,
        "tasks_by_disagreement": [
            {"source_id": source_id, "disagreement_tokens": count}
            for source_id, count in task_disagreements.most_common()
        ],
        "examples": examples[:max_examples],
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Irish Pilot Disagreement Report",
        "",
        f"- Tasks: {report['tasks']}",
        f"- Tokens: {report['tokens']}",
        f"- Token agreement: {report['agreement']:.4f}",
        f"- Cohen's kappa: {report['kappa']:.4f}",
        "",
        "## Per Role",
        "",
        "| Role | Tokens | Agreement | Kappa | Disagreements |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for role, values in report["per_role"].items():
        lines.append(f"| {role} | {values['tokens']} | {values['agreement']:.4f} | {values['kappa']:.4f} | {values['disagreements']} |")
    lines.extend(["", "## Main Label Transitions", "", "| Annotator 1 | Annotator 2 | Count |", "| --- | --- | ---: |"])
    for transition in report["top_transitions"][:15]:
        lines.append(f"| `{transition['annotator_a']}` | `{transition['annotator_b']}` | {transition['count']} |")
    lines.extend(["", "## Calibration Examples", ""])
    for example in report["examples"][:30]:
        lines.extend(
            [
                f"- `{example['source_id']}` ({example['role_type']})",
                f"  - Context: `{example['context']}`",
                f"  - Span: `{example['text']}`",
                f"  - Annotator 1: `{', '.join(example['labels_a'])}`; Annotator 2: `{', '.join(example['labels_b'])}`",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a disagreement report from two normalized BIO exports.")
    parser.add_argument("--labels-a", type=Path, required=True)
    parser.add_argument("--labels-b", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.labels_a, args.labels_b)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.markdown_output)
    print(json.dumps({key: report[key] for key in ("tasks", "tokens", "agreement", "kappa", "per_role")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
