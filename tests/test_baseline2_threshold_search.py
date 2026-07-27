from __future__ import annotations

import numpy as np
import pandas as pd

import src.esco.ranker as ranker_mod
from src.esco.ranker import CandidateSpan, RankerArtifacts, SpanRanker, generate_candidate_spans, select_predictions
from src.evaluation.evaluate_baseline2 import run_threshold_search


class DummyModel:
    def encode(self, items, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        vectors = []
        for item in items:
            text = str(item).lower()
            if "python" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.array(vectors, dtype=np.float32)


class DummyFaissIndex:
    def search(self, vectors, top_k):
        scores = []
        indices = []
        for vector in vectors:
            if vector[0] >= vector[1]:
                scores.append([1.0, 0.2][:top_k])
                indices.append([0, 1][:top_k])
            else:
                scores.append([1.0, 0.2][:top_k])
                indices.append([1, 0][:top_k])
        return np.array(scores, dtype=np.float32), np.array(indices, dtype=np.int64)


def test_generate_candidate_spans_prefers_content_windows() -> None:
    spans = generate_candidate_spans(["the", "Python", "developer", "role"])
    texts = [span[2] for span in spans]
    assert "Python" in texts
    assert "the" not in texts


def test_select_predictions_filters_overlap_by_score() -> None:
    candidates = [
        CandidateSpan(0, 2, "python developer", 0.9, "uri-1", []),
        CandidateSpan(0, 1, "python", 0.8, "uri-2", []),
        CandidateSpan(3, 4, "sql", 0.7, "uri-3", []),
    ]
    selected = select_predictions(candidates, threshold=0.75)
    assert [span.text for span in selected] == ["python developer"]


def test_run_threshold_search_with_dummy_ranker(tmp_path) -> None:
    artifacts = RankerArtifacts(
        model_name="dummy",
        labels=["Python", "SQL"],
        concept_uris=["uri-1", "uri-2"],
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        index=DummyFaissIndex(),
    )
    ranker = SpanRanker(artifacts=artifacts, model=DummyModel(), model_name="dummy")

    from datasets import Dataset, DatasetDict

    dataset = DatasetDict(
        {
            "dev": Dataset.from_dict(
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

    payload = run_threshold_search(disk, "dev", ranker, [0.5, 0.8])
    assert payload["best_threshold"] == 0.5
    assert payload["thresholds"][0]["documents"] == 1
