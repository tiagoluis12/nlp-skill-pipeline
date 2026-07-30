from __future__ import annotations

import numpy as np
from datasets import Dataset, DatasetDict

import src.extraction.ner_model as ner_model
from src.extraction.ner_model import bio_to_spans, combine_skillspan_tags, resolve_model_bundle


def test_combine_skillspan_tags_prioritizes_skill_layer() -> None:
    combined = combine_skillspan_tags(
        ["B", "I", "O", "O"],
        ["O", "O", "B", "I"],
    )

    assert combined == ["B-SKILL_TECH", "I-SKILL_TECH", "B-SKILL_TRANS", "I-SKILL_TRANS"]


def test_bio_to_spans_recovers_contiguous_entities() -> None:
    spans = bio_to_spans(["O", "B-SKILL_TECH", "I-SKILL_TECH", "O", "B-SKILL_TRANS"])

    assert spans == [(1, 3, "SKILL_TECH"), (4, 5, "SKILL_TRANS")]


def test_resolve_model_bundle_uses_offline_fallback(monkeypatch) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "tokens": [["Python", "testing"]],
                    "tags_skill": [["B", "I"]],
                    "tags_knowledge": [["O", "O"]],
                    "source": ["tech"],
                }
            ),
            "dev": Dataset.from_dict(
                {
                    "tokens": [["SQL"]],
                    "tags_skill": [["O"]],
                    "tags_knowledge": [["B"]],
                    "source": ["tech"],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "tokens": [["Cloud"]],
                    "tags_skill": [["O"]],
                    "tags_knowledge": [["B"]],
                    "source": ["tech"],
                }
            ),
        }
    )

    monkeypatch.setattr(ner_model, "_is_pretrained_model_available", lambda model_name: False)
    bundle = resolve_model_bundle(dataset, allow_offline_fallback=True)

    assert bundle.offline_fallback is True
    assert bundle.model_name.endswith("offline-wordlevel-fallback")
    assert bundle.tokenizer.pad_token == "[PAD]"
    assert bundle.model.config.num_labels == len(ner_model.LABEL_NAMES)


def test_evaluate_records_reports_gold_source() -> None:
    records = [
        {
            "id": 1,
            "role_type": "software_dev",
            "text": "Python testing",
            "tokens": ["Python", "testing"],
            "labels": ["B-SKILL_TECH", "I-SKILL_TECH"],
        }
    ]
    predictions = [
        {
            "tokens": ["Python", "testing"],
            "labels": ["B-SKILL_TECH", "I-SKILL_TECH"],
            "confidences": [0.9, 0.8],
        }
    ]

    report = ner_model._evaluate_records(records, predictions, source="gold", checkpoint_name="checkpoint")

    assert report["source"] == "gold"
    assert report["documents"] == 1
    assert report["model_name"] == "checkpoint"
    assert report["document_records"][0]["source"] == "software_dev"
    assert report["document_records"][0]["source_text"] == "Python testing"
    assert report["document_records"][0]["predicted_spans"][0]["text"] == "Python testing"
