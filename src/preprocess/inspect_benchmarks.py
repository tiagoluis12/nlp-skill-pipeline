from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_from_disk


def inspect_skillspan(path: Path) -> dict[str, object]:
    dataset = load_from_disk(str(path))
    summary = {}
    for split_name, split in dataset.items():
        label_fields = tuple(field for field in ("tags_skill", "tags_knowledge", "ner_tags", "tags", "labels", "label_ids", "label") if field in split.column_names)
        labels = {field: Counter() for field in label_fields}
        token_counts = Counter()
        for record in split:
            for field in label_fields:
                value = record.get(field)
                if isinstance(value, list):
                    labels[field].update(value)
                    token_counts[field] += len(value)
                elif value is not None:
                    labels[field].update([value])
                    token_counts[field] += 1
        summary[split_name] = {
            "rows": len(split),
            "columns": list(split.column_names),
            "token_counts": dict(token_counts),
            "label_counts": {field: dict(counts) for field, counts in labels.items()},
        }
    return summary


def inspect_esco(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    return {
        "rows": len(frame),
        "columns": frame.columns.tolist(),
    }


def main() -> None:
    summary = {}
    skillspan_path = Path("data/benchmarks/skillspan")
    esco_path = Path("data/benchmarks/esco/skills_en.csv")

    if skillspan_path.exists():
        summary["skillspan"] = inspect_skillspan(skillspan_path)
    if esco_path.exists():
        summary["esco"] = inspect_esco(esco_path)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
