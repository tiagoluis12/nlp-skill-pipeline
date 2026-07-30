from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import src.evaluation.run_integration_gate as gate


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_select_unseen_postings_excludes_gold_ids(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    fixtures = {
        "cloud_support": [{"id": "c1", "role_type": "cloud_support", "sentences": ["Cloud."], "raw_text": "Cloud."}],
        "cybersecurity": [{"id": "c2", "role_type": "cybersecurity", "sentences": ["Security."], "raw_text": "Security."}],
        "data_analyst": [{"id": "c3", "role_type": "data_analyst", "sentences": ["Data."], "raw_text": "Data."}],
        "qa_engineer": [
            {"id": "b1", "role_type": "qa_engineer", "sentences": ["Testing."], "raw_text": "Testing."},
            {"id": "b2", "role_type": "qa_engineer", "sentences": ["Automation."], "raw_text": "Automation."},
        ],
        "software_dev": [
            {"id": "a1", "role_type": "software_dev", "sentences": ["Python."], "raw_text": "Python."},
            {"id": "a2", "role_type": "software_dev", "sentences": ["SQL."], "raw_text": "SQL."},
        ],
    }
    for role, rows in fixtures.items():
        _write_jsonl(clean_dir / f"{role}.jsonl", rows)

    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(gold_path, [{"id": "a1"}])

    selected = gate.select_unseen_postings(clean_dir=clean_dir, gold_jsonl=gold_path, per_role=1)

    assert [row["id"] for row in selected] == ["c1", "c2", "c3", "b1", "a2"]


def test_run_integration_record_populates_uris(monkeypatch) -> None:
    record = {
        "id": "x1",
        "role_type": "software_dev",
        "sentences": ["Python testing is important."],
        "raw_text": "Python testing is important.",
    }

    monkeypatch.setattr(
        gate,
        "predict_text_spans",
        lambda bundle, text: [
            SimpleNamespace(text="Python testing", label="SKILL_TECH", confidence=0.9),
        ],
    )
    monkeypatch.setattr(
        gate,
        "rank_span_candidates",
        lambda span_text, ranker, lexicon_index=None, top_k=5: [
            {"label": "Python testing", "concept_uri": "uri-1", "score": 1.0, "method": "exact"}
        ],
    )

    report = gate.run_integration_record(record, bundle=object(), ranker=object(), lexicon_index={})

    assert report["span_count"] == 1
    assert report["uri_populated"] == 1
    assert report["failure_reasons"] == []
    assert report["traces"][0]["predicted_uri"] == "uri-1"


def test_write_integration_outputs_writes_manifest_and_traces(tmp_path: Path, monkeypatch) -> None:
    records = [
        {"id": "x1", "role_type": "software_dev", "sentences": ["Python testing."], "raw_text": "Python testing."}
    ]

    monkeypatch.setattr(
        gate,
        "run_integration_record",
        lambda record, bundle, ranker, lexicon_index, top_k=5: {
            "posting_id": record["id"],
            "role_type": record["role_type"],
            "span_count": 1,
            "uri_populated": 1,
            "failure_reasons": [],
            "traces": [
                {
                    "posting_id": "x1",
                    "role_type": "software_dev",
                    "raw_text": "Python testing.",
                    "sentence": "Python testing.",
                    "span": "Python testing",
                    "span_confidence": 0.9,
                    "ner_label": "SKILL_TECH",
                    "top5_candidates": [{"concept_uri": "uri-1"}],
                    "predicted_uri": "uri-1",
                    "error_category": "integration_ok",
                }
            ],
        },
    )

    manifest = gate.write_integration_outputs(records, bundle=object(), ranker=object(), lexicon_index={}, output_dir=tmp_path)

    assert manifest["postings"] == 1
    assert manifest["predicted_uris"] == 1
    assert (tmp_path / "x1.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()
