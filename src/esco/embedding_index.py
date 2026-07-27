from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from src.extraction.lexicon_baseline import load_esco_skills


DEFAULT_MODEL_NAME = "all-mpnet-base-v2"


@dataclass(frozen=True)
class EmbeddingIndexArtifacts:
    model_name: str
    labels: list[str]
    concept_uris: list[str]
    embeddings: np.ndarray
    faiss_index: faiss.Index


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def build_embedding_matrix(
    labels: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 16,
    show_progress_bar: bool = False,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(
        list(labels),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32, copy=False)
    return _normalize_embeddings(embeddings)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, found shape {embeddings.shape}")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def build_embedding_artifacts(
    skills_csv: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 16,
    show_progress_bar: bool = False,
) -> EmbeddingIndexArtifacts:
    skills = load_esco_skills(skills_csv)
    labels = skills["preferredLabel"].astype(str).tolist()
    concept_uris = skills["conceptUri"].astype(str).tolist()
    embeddings = build_embedding_matrix(
        labels,
        model_name=model_name,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    index = build_faiss_index(embeddings)
    return EmbeddingIndexArtifacts(
        model_name=model_name,
        labels=labels,
        concept_uris=concept_uris,
        embeddings=embeddings,
        faiss_index=index,
    )


def save_embedding_artifacts(
    artifacts: EmbeddingIndexArtifacts,
    embeddings_path: Path,
    index_path: Path,
    labels_path: Path,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, artifacts.embeddings)
    faiss.write_index(artifacts.faiss_index, str(index_path))
    labels_path.write_text(
        json.dumps(
            {"labels": artifacts.labels, "concept_uris": artifacts.concept_uris},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "model_name": artifacts.model_name,
        "labels": len(artifacts.labels),
        "dimension": int(artifacts.embeddings.shape[1]),
    }
    if metadata_path is not None:
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_embedding_artifacts(
    embeddings_path: Path,
    index_path: Path,
    labels_path: Path,
) -> EmbeddingIndexArtifacts:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = payload["labels"]
    concept_uris = payload["concept_uris"]
    embeddings = np.load(embeddings_path)
    index = faiss.read_index(str(index_path))
    return EmbeddingIndexArtifacts(
        model_name=DEFAULT_MODEL_NAME,
        labels=labels,
        concept_uris=concept_uris,
        embeddings=embeddings,
        faiss_index=index,
    )


def query_embedding_index(
    query: str,
    artifacts: EmbeddingIndexArtifacts,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = 5,
) -> list[dict[str, object]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32, copy=False)
    query_embedding = _normalize_embeddings(query_embedding)
    scores, indices = artifacts.faiss_index.search(query_embedding, top_k)

    results: list[dict[str, object]] = []
    for score, idx in zip(scores[0], indices[0], strict=True):
        if idx < 0:
            continue
        results.append(
            {
                "rank": len(results) + 1,
                "label": artifacts.labels[idx],
                "concept_uri": artifacts.concept_uris[idx],
                "score": float(score),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ESCO embedding index for Baseline 2.")
    parser.add_argument("--skills-csv", type=Path, required=True, help="Path to ESCO skills_en.csv")
    parser.add_argument("--embeddings", type=Path, default=Path("models/esco_embeddings.npy"))
    parser.add_argument("--index", type=Path, default=Path("models/esco_faiss.index"))
    parser.add_argument("--labels", type=Path, default=Path("models/esco_embedding_labels.json"))
    parser.add_argument("--metadata", type=Path, default=Path("models/esco_embedding_index.json"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()

    artifacts = build_embedding_artifacts(
        args.skills_csv,
        model_name=args.model_name,
        batch_size=args.batch_size,
        show_progress_bar=False,
    )
    metadata = save_embedding_artifacts(artifacts, args.embeddings, args.index, args.labels, args.metadata)

    payload: dict[str, object] = {"metadata": metadata}
    if args.query:
      payload["matches"] = {query: query_embedding_index(query, artifacts, model_name=args.model_name) for query in args.query}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
