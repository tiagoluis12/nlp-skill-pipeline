from __future__ import annotations

import numpy as np

import src.esco.ranker as ranker_module
from src.esco.embedding_index import build_faiss_index
from src.esco.ranker import RankerArtifacts, SpanRanker, rank_span_candidates


class DummyModel:
    def encode(self, items, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        return np.array([[1.0, 0.0] for _ in items], dtype=np.float32)


def test_rank_span_candidates_prefers_lexicon_then_embedding(monkeypatch) -> None:
    artifacts = RankerArtifacts(
        model_name="dummy",
        labels=["Root cause analysis", "Python"],
        concept_uris=["uri-exact", "uri-embed"],
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        index=build_faiss_index(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)),
    )
    ranker = SpanRanker(artifacts=artifacts, model=DummyModel())

    def fake_rank_candidate_texts(texts, ranker, top_k=5, batch_size=64):
        scores = np.array([[0.92, 0.81]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int64)
        return scores, indices

    monkeypatch.setattr(ranker_module, "rank_candidate_texts", fake_rank_candidate_texts)

    candidates = rank_span_candidates(
        "Root cause analysis",
        ranker,
        lexicon_index={"root cause analysis": "uri-exact"},
        top_k=3,
    )

    assert candidates[0]["concept_uri"] == "uri-exact"
    assert candidates[0]["method"] == "exact"
    assert candidates[1]["concept_uri"] == "uri-embed"
    assert candidates[1]["method"] == "embedding"
