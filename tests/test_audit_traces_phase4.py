from __future__ import annotations

import json
from pathlib import Path

import src.evaluation.build_audit_traces as audit_traces


class DummyRanker:
    pass


def test_build_audit_trace_records_emits_predicted_and_missed_records(monkeypatch) -> None:
    documents = [
        {
            "id": 1,
            "source_id": "source-1",
            "source": "software_dev",
            "source_text": "Python testing and teamwork",
            "tokens": ["Python", "testing", "and", "teamwork"],
            "predicted_spans": [
                {
                    "start": 0,
                    "end": 2,
                    "label": "SKILL_TECH",
                    "text": "Python testing",
                    "confidence": 0.91,
                    "source_sentence": "Python testing and teamwork",
                }
            ],
        }
    ]
    gold_rows = [
        {
            "posting_id": "1",
            "span_start": "0",
            "span_end": "2",
            "span_label": "SKILL_TECH",
            "span_text": "Python testing",
            "gold_concept_uri": "uri-1",
        },
        {
            "posting_id": "1",
            "span_start": "2",
            "span_end": "4",
            "span_label": "SKILL_TRANS",
            "span_text": "teamwork",
            "gold_concept_uri": "uri-2",
        },
    ]

    def fake_rank(span_text, ranker, lexicon_index=None, top_k=5):
        if span_text == "Python testing":
            return [{"concept_uri": "uri-1", "label": "Python testing", "score": 1.0, "method": "exact"}]
        if span_text == "teamwork":
            return [{"concept_uri": "uri-2", "label": "teamwork", "score": 1.0, "method": "exact"}]
        return []

    monkeypatch.setattr(audit_traces, "rank_span_candidates", fake_rank)

    traces = audit_traces.build_audit_trace_records(documents, gold_rows, DummyRanker(), lexicon_index={})

    assert "1" in traces
    assert len(traces["1"]) == 2
    assert traces["1"][0]["record_type"] == "predicted_span"
    assert traces["1"][0]["error_category"] == "correct"
    assert traces["1"][1]["record_type"] == "missed_gold_span"
    assert traces["1"][1]["error_category"] == "missed_extraction"


def test_write_audit_traces_writes_manifest_and_jsonl(tmp_path: Path) -> None:
    traces = {
        "1": [
            {
                "record_type": "predicted_span",
                "posting_id": "1",
                "source_id": "source-1",
                "role_type": "software_dev",
                "sentence": "Python testing and teamwork",
                "span_start": 0,
                "span_end": 2,
                "span": "Python testing",
                "span_confidence": 0.91,
                "ner_label": "SKILL_TECH",
                "top5_candidates": [{"concept_uri": "uri-1"}],
                "predicted_uri": "uri-1",
                "gold_uri": "uri-1",
                "gold_label": "SKILL_TECH",
                "gold_span_start": 0,
                "gold_span_end": 2,
                "gold_text": "Python testing",
                "error_category": "correct",
            }
        ]
    }

    manifest = audit_traces.write_audit_traces(traces, tmp_path)

    assert manifest["postings"] == 1
    assert manifest["trace_records"] == 1
    assert (tmp_path / "1.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["trace_records"] == 1
