from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.evaluation import evaluate_esco_linking as linking
import src.evaluation.export_gold_esco_review as review_export
from src.evaluation.export_gold_esco_review import build_gold_esco_review
from src.esco.ranker import RankerArtifacts, SpanRanker


class DummyIndex:
    def search(self, vectors, top_k):
        scores = np.array([[1.0, 0.8][:top_k]], dtype=np.float32)
        indices = np.array([[0, 1][:top_k]], dtype=np.int64)
        return scores, indices


class DummyModel:
    def encode(self, items, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        return np.array([[1.0, 0.0] for _ in items], dtype=np.float32)


def _dummy_ranker() -> SpanRanker:
    artifacts = RankerArtifacts(
        model_name="dummy",
        labels=["Python", "SQL"],
        concept_uris=["uri-1", "uri-2"],
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        index=DummyIndex(),
    )
    return SpanRanker(artifacts=artifacts, model=DummyModel(), model_name="dummy")


def test_build_gold_esco_review_exports_candidate_columns(tmp_path, monkeypatch) -> None:
    gold_jsonl = tmp_path / "gold.jsonl"
    gold_jsonl.write_text(
        """
{"id": 1, "tokens": ["Python", "developer"], "labels": ["B-SKILL_TECH", "O"], "role_type": "software_dev", "source_id": "s1"}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    esco_csv = tmp_path / "skills_en.csv"
    esco_csv.write_text(
        "conceptUri,preferredLabel,altLabels\nuri-1,Python,\nuri-2,SQL,\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(review_export, "load_span_ranker", lambda *args, **kwargs: _dummy_ranker())
    monkeypatch.setattr(
        review_export,
        "rank_candidate_texts",
        lambda texts, ranker, top_k=5, batch_size=64: (
            np.array([[1.0, 0.8][:top_k]], dtype=np.float32),
            np.array([[0, 1][:top_k]], dtype=np.int64),
        ),
    )

    rows = build_gold_esco_review(gold_jsonl, esco_csv, tmp_path / "embeddings.npy", tmp_path / "index.faiss", tmp_path / "labels.json", top_k=2)
    assert len(rows) == 1
    assert rows[0]["lexicon_uri"] == "uri-1"
    assert rows[0]["candidate_1_uri"] == "uri-1"
    assert rows[0]["candidate_2_uri"] == "uri-2"


def test_evaluate_esco_linking_reports_expected_metrics(tmp_path, monkeypatch) -> None:
    review_csv = tmp_path / "review.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "span_text",
                "gold_concept_uri",
                "reviewer_notes",
            ],
        )
        writer.writeheader()
        writer.writerow({"span_text": "Python", "gold_concept_uri": "uri-1", "reviewer_notes": ""})

    esco_csv = tmp_path / "skills_en.csv"
    esco_csv.write_text(
        "conceptUri,preferredLabel,altLabels\nuri-1,Python,\nuri-2,SQL,\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(linking, "load_span_ranker", lambda *args, **kwargs: _dummy_ranker())
    monkeypatch.setattr(
        linking,
        "rank_candidate_texts",
        lambda texts, ranker, top_k=5: (
            np.array([[1.0, 0.8][:top_k]], dtype=np.float32),
            np.array([[0, 1][:top_k]], dtype=np.int64),
        ),
    )

    skills = linking.load_esco_skills(esco_csv)
    lexicon_index = linking.build_lexicon_index(skills)
    ranker = _dummy_ranker()
    report = linking._evaluate_rows(linking._load_review(review_csv), "baseline2", ranker, lexicon_index)
    assert report["acc_at_1"] == 1.0
    assert report["acc_at_3"] == 1.0
    assert report["mrr"] == 1.0
