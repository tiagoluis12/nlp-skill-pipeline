from __future__ import annotations

import json

import numpy as np
import pandas as pd

import src.esco.embedding_index as embedding_index
from src.esco.embedding_index import (
    EmbeddingIndexArtifacts,
    build_embedding_artifacts,
    build_faiss_index,
    save_embedding_artifacts,
)


class DummyModel:
    def encode(self, items, batch_size=64, show_progress_bar=True, convert_to_numpy=True):
        vectors = []
        for item in items:
            text = str(item).lower()
            if "python" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.array(vectors, dtype=np.float32)


def test_build_faiss_index_uses_inner_product() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = build_faiss_index(embeddings)

    assert index.ntotal == 2


def test_save_embedding_artifacts_writes_expected_files(tmp_path, monkeypatch) -> None:
    artifacts = EmbeddingIndexArtifacts(
        model_name="dummy",
        labels=["Python", "SQL"],
        concept_uris=["uri-1", "uri-2"],
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        faiss_index=build_faiss_index(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)),
    )

    metadata = save_embedding_artifacts(
        artifacts,
        tmp_path / "embeddings.npy",
        tmp_path / "index.faiss",
        tmp_path / "labels.json",
        tmp_path / "meta.json",
    )

    assert metadata["labels"] == 2
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "labels.json").exists()
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["dimension"] == 2


def test_build_embedding_artifacts_with_monkeypatched_model(tmp_path, monkeypatch) -> None:
    skills = pd.DataFrame(
        [
            {"conceptUri": "uri-1", "preferredLabel": "Python"},
            {"conceptUri": "uri-2", "preferredLabel": "SQL"},
        ]
    )
    csv_path = tmp_path / "skills_en.csv"
    skills.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        embedding_index,
        "build_embedding_matrix",
        lambda labels, model_name="dummy", batch_size=1, show_progress_bar=False: np.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        ),
    )
    artifacts = build_embedding_artifacts(csv_path, model_name="dummy", batch_size=1, show_progress_bar=False)

    assert artifacts.embeddings.shape == (2, 2)
    assert artifacts.faiss_index.ntotal == 2
