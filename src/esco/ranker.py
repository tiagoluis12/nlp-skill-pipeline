from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

from src.extraction.lexicon_baseline import normalize_text


DEFAULT_MODEL_NAME = "all-mpnet-base-v2"


@dataclass(frozen=True)
class RankerArtifacts:
    model_name: str
    labels: list[str]
    concept_uris: list[str]
    embeddings: np.ndarray
    index: object


@dataclass(frozen=True)
class CandidateSpan:
    start: int
    end: int
    text: str
    score: float
    predicted_uri: str | None
    top5: list[dict[str, object]]


@dataclass(frozen=True)
class SpanRanker:
    artifacts: RankerArtifacts
    model: object
    model_name: str = DEFAULT_MODEL_NAME


class HashingTextEncoder:
    def __init__(self, n_features: int = 2048) -> None:
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            alternate_sign=False,
            norm=None,
            n_features=n_features,
        )

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        matrix = self.vectorizer.transform(list(texts))
        dense = matrix.astype(np.float32).toarray()
        return _normalize_vectors(dense)


class NumpyIndex:
    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(queries, dtype=np.float32) @ self.embeddings.T
        if scores.ndim == 1:
            scores = scores[None, :]
        top_k = min(top_k, scores.shape[1])
        indices = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(scores, indices, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        return top_scores.astype(np.float32, copy=False), indices.astype(np.int64, copy=False)


def _load_label_metadata(labels_path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    return list(payload["labels"]), list(payload["concept_uris"])


def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu", local_files_only=True)


def _sentence_transformer_is_cached(model_name: str) -> bool:
    from huggingface_hub import snapshot_download

    repo_id = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    try:
        snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception:
        return False
    return True


def load_span_ranker(
    embeddings_path: Path = Path("models/esco_embeddings.npy"),
    index_path: Path = Path("models/esco_faiss.index"),
    labels_path: Path = Path("models/esco_embedding_labels.json"),
    model_name: str = DEFAULT_MODEL_NAME,
) -> SpanRanker:
    if _sentence_transformer_is_cached(model_name):
        model = _load_sentence_transformer(model_name)
        from src.esco.embedding_index import load_embedding_artifacts

        artifacts = load_embedding_artifacts(embeddings_path, index_path, labels_path)
        artifacts = RankerArtifacts(
            model_name=artifacts.model_name,
            labels=list(artifacts.labels),
            concept_uris=list(artifacts.concept_uris),
            embeddings=np.asarray(artifacts.embeddings, dtype=np.float32),
            index=artifacts.faiss_index,
        )
    else:
        labels, concept_uris = _load_label_metadata(labels_path)
        model = HashingTextEncoder()
        embeddings = model.encode(labels)
        artifacts = RankerArtifacts(
            model_name=f"{model_name}-fallback-hashing",
            labels=labels,
            concept_uris=concept_uris,
            embeddings=embeddings,
            index=NumpyIndex(embeddings),
        )
    return SpanRanker(artifacts=artifacts, model=model, model_name=model_name)


def _token_is_content(token: str) -> bool:
    normalized = normalize_text(token)
    return bool(normalized) and normalized not in STOP_WORDS and any(ch.isalnum() for ch in normalized)


def generate_candidate_spans(tokens: Sequence[str], max_ngram: int = 5) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []
    token_count = len(tokens)
    for start in range(token_count):
        for end in range(start + 1, min(token_count, start + max_ngram) + 1):
            window = tokens[start:end]
            normalized = [normalize_text(token) for token in window]
            if not any(_token_is_content(token) for token in window):
                continue
            if end - start > 1 and normalized[0] in STOP_WORDS:
                continue
            if end - start > 1 and normalized[-1] in STOP_WORDS:
                continue
            if sum(1 for token in normalized if token in STOP_WORDS) > (end - start) // 2:
                continue
            candidates.append((start, end, " ".join(window)))
    return candidates


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def rank_candidate_texts(
    texts: Sequence[str],
    ranker: SpanRanker,
    top_k: int = 5,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = ranker.model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    embeddings = _normalize_vectors(embeddings)
    scores, indices = ranker.artifacts.index.search(embeddings, top_k)
    return scores, indices


def _top_k_candidates(
    scores: np.ndarray,
    indices: np.ndarray,
    ranker: SpanRanker,
) -> list[list[dict[str, object]]]:
    output: list[list[dict[str, object]]] = []
    for row_scores, row_indices in zip(scores, indices, strict=True):
        candidates: list[dict[str, object]] = []
        for rank, (score, idx) in enumerate(zip(row_scores, row_indices, strict=True), start=1):
            if idx < 0:
                continue
            candidates.append(
                {
                    "rank": rank,
                    "label": ranker.artifacts.labels[idx],
                    "concept_uri": ranker.artifacts.concept_uris[idx],
                    "score": float(score),
                }
            )
        output.append(candidates)
    return output


def rank_document_candidates(
    tokens: Sequence[str],
    ranker: SpanRanker,
    top_k: int = 5,
    batch_size: int = 64,
    max_ngram: int = 5,
) -> list[CandidateSpan]:
    spans = generate_candidate_spans(tokens, max_ngram=max_ngram)
    if not spans:
        return []

    texts = [span_text for _, _, span_text in spans]
    scores, indices = rank_candidate_texts(texts, ranker, top_k=top_k, batch_size=batch_size)
    top_candidates = _top_k_candidates(scores, indices, ranker)

    ranked: list[CandidateSpan] = []
    for (start, end, text), candidates in zip(spans, top_candidates, strict=True):
        best = candidates[0] if candidates else None
        ranked.append(
            CandidateSpan(
                start=start,
                end=end,
                text=text,
                score=float(best["score"]) if best else 0.0,
                predicted_uri=str(best["concept_uri"]) if best else None,
                top5=candidates,
            )
        )
    return ranked


def _overlaps(a: CandidateSpan, b: CandidateSpan) -> bool:
    return min(a.end, b.end) > max(a.start, b.start)


def select_predictions(
    candidates: Sequence[CandidateSpan],
    threshold: float,
) -> list[CandidateSpan]:
    ranked = sorted(
        (candidate for candidate in candidates if candidate.score >= threshold),
        key=lambda candidate: (-candidate.score, -(candidate.end - candidate.start), candidate.start, candidate.end),
    )
    selected: list[CandidateSpan] = []
    for candidate in ranked:
        if any(_overlaps(candidate, accepted) for accepted in selected):
            continue
        selected.append(candidate)
    selected.sort(key=lambda candidate: (candidate.start, candidate.end))
    return selected


def document_trace(
    tokens: Sequence[str],
    ranker: SpanRanker,
    threshold: float,
    top_k: int = 5,
    batch_size: int = 64,
    max_ngram: int = 5,
) -> dict[str, object]:
    candidates = rank_document_candidates(tokens, ranker, top_k=top_k, batch_size=batch_size, max_ngram=max_ngram)
    selected = select_predictions(candidates, threshold)
    return {
        "threshold": threshold,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "candidates": [
            {
                "start": candidate.start,
                "end": candidate.end,
                "text": candidate.text,
                "score": candidate.score,
                "predicted_uri": candidate.predicted_uri,
                "top5": candidate.top5,
            }
            for candidate in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidate spans against the ESCO embedding index.")
    parser.add_argument("--embeddings", type=Path, default=Path("models/esco_embeddings.npy"))
    parser.add_argument("--index", type=Path, default=Path("models/esco_faiss.index"))
    parser.add_argument("--labels", type=Path, default=Path("models/esco_embedding_labels.json"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()

    ranker = load_span_ranker(args.embeddings, args.index, args.labels, model_name=args.model_name)
    payload = {"matches": {}}
    for query in args.query:
        scores, indices = rank_candidate_texts([query], ranker)
        payload["matches"][query] = _top_k_candidates(scores, indices, ranker)[0]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
