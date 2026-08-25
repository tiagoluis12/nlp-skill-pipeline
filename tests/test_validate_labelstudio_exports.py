from __future__ import annotations

import json

from src.preprocess.validate_labelstudio_exports import summarize_export, validate_exports


def _write_export(path, rows) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def test_validate_labelstudio_exports_accepts_labeled_unique_tasks(tmp_path) -> None:
    export_a = tmp_path / "annotator1.json"
    export_b = tmp_path / "annotator2.json"
    rows = [
        {
            "id": 1,
            "data": {"source_id": "a"},
            "annotations": [
                {
                    "was_cancelled": False,
                    "result": [
                        {"type": "labels", "value": {"start": 0, "end": 6, "labels": ["SKILL_TECH"]}}
                    ],
                }
            ],
        },
        {
            "id": 2,
            "data": {"source_id": "b"},
            "annotations": [
                {
                    "was_cancelled": False,
                    "result": [
                        {"type": "labels", "value": {"start": 0, "end": 4, "labels": ["SKILL_TRANS"]}}
                    ],
                }
            ],
        },
    ]
    _write_export(export_a, rows)
    rows[1]["annotations"][0]["result"][0]["value"]["end"] = 5
    _write_export(export_b, rows)

    report = validate_exports([export_a, export_b], expected_unique_tasks=2)

    assert report["valid"] is True
    assert report["errors"] == []
    assert summarize_export(export_a)["span_results"] == 2


def test_validate_labelstudio_exports_rejects_identical_annotator_exports(tmp_path) -> None:
    export_a = tmp_path / "annotator1.json"
    export_b = tmp_path / "annotator2.json"
    rows = [
        {
            "id": 1,
            "data": {"source_id": "a"},
            "annotations": [
                {
                    "was_cancelled": False,
                    "result": [
                        {"type": "labels", "value": {"start": 0, "end": 6, "labels": ["SKILL_TECH"]}}
                    ],
                }
            ],
        }
    ]
    _write_export(export_a, rows)
    _write_export(export_b, rows)

    report = validate_exports([export_a, export_b], expected_unique_tasks=1)

    assert report["valid"] is False
    assert any("byte-for-byte identical" in error for error in report["errors"])


def test_validate_labelstudio_exports_rejects_empty_and_duplicate_tasks(tmp_path) -> None:
    export_path = tmp_path / "annotator.json"
    rows = [
        {
            "id": 1,
            "data": {"source_id": "a"},
            "annotations": [{"was_cancelled": False, "result": []}],
        },
        {
            "id": 2,
            "data": {"source_id": "a"},
            "annotations": [{"was_cancelled": False, "result": []}],
        },
    ]
    _write_export(export_path, rows)

    report = validate_exports([export_path], expected_unique_tasks=2)

    assert report["valid"] is False
    assert any("duplicate task ids detected" in error for error in report["errors"])
    assert any("no span annotations were exported" in error for error in report["errors"])
