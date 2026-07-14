from __future__ import annotations

import json

from src.preprocess.labelstudio_to_conll import export_conll, load_tasks


def test_labelstudio_to_conll_exports_bio(tmp_path) -> None:
    input_path = tmp_path / "labelstudio.json"
    conll_path = tmp_path / "pilot.conll"
    jsonl_path = tmp_path / "pilot.jsonl"
    input_path.write_text(
        json.dumps(
            [
                {
                    "id": "task-1",
                    "data": {"text": "Python and SQL are required.", "role_type": "software_dev", "source_id": "a1"},
                    "annotations": [
                        {
                            "was_cancelled": False,
                            "result": [
                                {"type": "labels", "value": {"start": 0, "end": 6, "labels": ["SKILL_TECH"]}},
                                {"type": "labels", "value": {"start": 11, "end": 14, "labels": ["SKILL_TECH"]}},
                            ],
                        }
                    ],
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    tasks = load_tasks(input_path)
    summary = export_conll(tasks, conll_path, jsonl_path)

    assert summary["tasks"] == 1
    assert summary["tokens"] > 0
    assert conll_path.read_text(encoding="utf-8").splitlines()[:4] == [
        "Python B-SKILL_TECH",
        "and O",
        "SQL B-SKILL_TECH",
        "are O",
    ]
    jsonl = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert jsonl["labels"][0] == "B-SKILL_TECH"
    assert jsonl["labels"][2] == "B-SKILL_TECH"
