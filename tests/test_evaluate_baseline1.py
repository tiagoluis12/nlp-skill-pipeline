from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

from src.evaluation.evaluate_baseline1 import run_gold_evaluation, run_skillspan_evaluation


def _write_esco_csv(path: Path) -> None:
    rows = [
        {
            "conceptUri": "http://data.europa.eu/esco/skill/1",
            "preferredLabel": "Python",
            "altLabels": "Python programming",
        },
        {
            "conceptUri": "http://data.europa.eu/esco/skill/2",
            "preferredLabel": "SQL",
            "altLabels": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["conceptUri", "preferredLabel", "altLabels"])
        writer.writeheader()
        writer.writerows(rows)


def test_run_skillspan_evaluation_reports_metrics(tmp_path: Path) -> None:
    esco_csv = tmp_path / "skills_en.csv"
    _write_esco_csv(esco_csv)

    from datasets import Dataset, DatasetDict, load_from_disk

    dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "idx": [1],
                    "tokens": [["Python", "and", "SQL"]],
                    "tags_skill": [["B", "O", "B"]],
                    "tags_knowledge": [["O", "O", "O"]],
                    "source": ["tech"],
                }
            )
        }
    )
    disk = tmp_path / "skillspan"
    dataset.save_to_disk(str(disk))

    report = run_skillspan_evaluation(disk, "test", esco_csv)

    assert report["documents"] == 1
    assert report["token_metrics"]["f1"] > 0.0
    assert report["span_exact"]["tp"] == 2


def test_run_gold_evaluation_reads_normalized_jsonl(tmp_path: Path) -> None:
    esco_csv = tmp_path / "skills_en.csv"
    _write_esco_csv(esco_csv)
    gold_jsonl = tmp_path / "gold.jsonl"
    gold_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": 1,
                        "tokens": ["Python", "and", "SQL"],
                        "labels": ["B-SKILL_TECH", "O", "B-SKILL_TECH"],
                        "role_type": "software_dev",
                        "source_id": "a",
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_gold_evaluation(gold_jsonl, esco_csv)

    assert report["documents"] == 1
    assert report["span_exact"]["tp"] == 2
