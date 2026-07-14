from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import requests
from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_url

DEFAULT_REPO_ID = "jjzha/skillspan"
DEFAULT_FILES = ("train.json", "dev.json", "test.json")


def download_repo_file(repo_id: str, filename: str, output_path: Path) -> None:
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
    response = requests.get(url, headers={"Accept-Encoding": "identity"}, timeout=120)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def load_local_dataset(files: dict[str, Path]) -> DatasetDict:
    data_files = {split: str(path) for split, path in files.items()}
    return load_dataset("json", data_files=data_files)


def summarize_split(split) -> dict[str, object]:
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

    return {
        "rows": len(split),
        "columns": list(split.column_names),
        "token_counts": dict(token_counts),
        "label_counts": {field: dict(counts) for field, counts in labels.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and inspect the SKILLSPAN benchmark.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/skillspan"))
    parser.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    local_files: dict[str, Path] = {}
    for filename in args.files:
        split = Path(filename).stem
        target = args.output / filename
        download_repo_file(args.repo_id, filename, target)
        local_files[split] = target

    dataset = load_local_dataset(local_files)
    dataset.save_to_disk(str(args.output))

    summary = {name: summarize_split(split) for name, split in dataset.items()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
