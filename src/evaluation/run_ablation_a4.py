"""Run ablation A4: compare MPNet and MiniLM ESCO embedding models."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.esco.ranker import NumpyIndex, RankerArtifacts, SpanRanker
from src.evaluation.evaluate_phase5 import _link_predictions, _read_json, _review_index
from src.extraction.lexicon_baseline import build_lexicon_index, load_esco_skills


DEFAULT_MODEL_CACHE = Path(".cache/huggingface/hub")
DEFAULT_ESCO_CSV = Path("data/benchmarks/esco/skills_en.csv")
DEFAULT_REVIEW_CSV = Path("results/gold_esco_review_priority.csv")
DEFAULT_MPNet_EMBEDDINGS = Path("models/esco_embeddings.npy")
DEFAULT_MPNet_INDEX = Path("models/esco_faiss.index")
DEFAULT_LABELS = Path("models/esco_embedding_labels.json")
DEFAULT_MINILM_DIR = Path("models/a4_minilm")
DEFAULT_OUTPUT = Path("results/ablation_a4.json")
MPNET_MODEL_ID = "all-mpnet-base-v2"
MINILM_MODEL_ID = "all-MiniLM-L6-v2"


def _resolve_snapshot(model_id: str, project_cache: Path = DEFAULT_MODEL_CACHE) -> Path:
    from huggingface_hub import snapshot_download

    repo_id = f"sentence-transformers/{model_id}"
    attempts = ({"local_files_only": True}, {"local_files_only": True, "cache_dir": str(project_cache)})
    for kwargs in attempts:
        try:
            return Path(snapshot_download(repo_id, **kwargs))
        except Exception:
            continue
    raise RuntimeError(
        f"{model_id} is not available locally. Cache it before running A4: {repo_id}"
    )


def _load_ranker(
    model_path: Path,
    model_id: str,
    embeddings_path: Path,
    index_path: Path,
    labels_path: Path,
) -> SpanRanker:
    from sentence_transformers import SentenceTransformer

    _ = index_path  # A4 uses a NumPy exact-IP index because FAISS cannot initialise on this host.
    label_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
    model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    ranker_artifacts = RankerArtifacts(
        model_name=model_id,
        labels=list(label_payload["labels"]),
        concept_uris=list(label_payload["concept_uris"]),
        embeddings=embeddings,
        index=NumpyIndex(embeddings),
    )
    return SpanRanker(artifacts=ranker_artifacts, model=model, model_name=model_id)


def _ensure_minilm_index(
    model_path: Path,
    esco_csv: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Path | int | str]:
    embeddings_path = output_dir / "esco_embeddings.npy"
    index_path = output_dir / "esco_numpy_ip.index.json"
    labels_path = output_dir / "esco_embedding_labels.json"
    metadata_path = output_dir / "esco_embedding_index.json"
    if not all(path.exists() for path in (embeddings_path, index_path, labels_path, metadata_path)):
        from sentence_transformers import SentenceTransformer

        skills = load_esco_skills(esco_csv)
        labels = skills["preferredLabel"].astype(str).tolist()
        concept_uris = skills["conceptUri"].astype(str).tolist()
        model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
        embeddings = model.encode(
            labels,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings = embeddings / norms
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, embeddings)
        labels_path.write_text(
            json.dumps({"labels": labels, "concept_uris": concept_uris}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index_path.write_text(
            json.dumps({"type": "numpy_exact_inner_product", "embeddings": str(embeddings_path)}, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "model_id": MINILM_MODEL_ID,
                    "model_path": str(model_path),
                    "dimension": int(embeddings.shape[1]),
                    "labels": len(labels),
                    "index_type": "numpy_exact_inner_product",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    metadata = _read_metadata(metadata_path)
    return {
        "embeddings": embeddings_path,
        "index": index_path,
        "labels": labels_path,
        "metadata": metadata_path,
        "dimension": int(metadata["dimension"]),
    }


def _read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "KMP_USE_SHM": "0",
            "OMP_WAIT_POLICY": "PASSIVE",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def _run_child(arguments: list[str]) -> None:
    subprocess.run(
        [sys.executable, "-m", "src.evaluation.run_ablation_a4", *arguments],
        check=True,
        env=_child_environment(),
    )


def _run_single_model(
    model_id: str,
    model_path: Path,
    esco_csv: Path,
    review_csv: Path,
    embeddings: Path,
    index: Path,
    labels: Path,
    output_path: Path,
) -> dict[str, Any]:
    ranker = _load_ranker(model_path, model_id, embeddings, index, labels)
    review_rows, _ = _review_index(review_csv)
    lexicon_index = build_lexicon_index(load_esco_skills(esco_csv))
    metrics, _ = _link_predictions(review_rows, "Proposed_NER_Ranking", ranker, lexicon_index)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _deltas(minilm: dict[str, Any], mpnet: dict[str, Any]) -> dict[str, float]:
    return {
        "acc_at_1": float(minilm["acc_at_1"]) - float(mpnet["acc_at_1"]),
        "acc_at_3": float(minilm["acc_at_3"]) - float(mpnet["acc_at_3"]),
        "mrr": float(minilm["mrr"]) - float(mpnet["mrr"]),
        "macro_f1": float(minilm["macro_f1"]) - float(mpnet["macro_f1"]),
    }


def run_ablation_a4(
    esco_csv: Path = DEFAULT_ESCO_CSV,
    review_csv: Path = DEFAULT_REVIEW_CSV,
    mpnet_embeddings: Path = DEFAULT_MPNet_EMBEDDINGS,
    mpnet_index: Path = DEFAULT_MPNet_INDEX,
    labels: Path = DEFAULT_LABELS,
    minilm_dir: Path = DEFAULT_MINILM_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    batch_size: int = 64,
    project_cache: Path = DEFAULT_MODEL_CACHE,
) -> dict[str, Any]:
    mpnet_path = _resolve_snapshot(MPNET_MODEL_ID, project_cache)
    minilm_path = _resolve_snapshot(MINILM_MODEL_ID, project_cache)
    minilm_files = (
        minilm_dir / "esco_embeddings.npy",
        minilm_dir / "esco_numpy_ip.index.json",
        minilm_dir / "esco_embedding_labels.json",
        minilm_dir / "esco_embedding_index.json",
    )
    if not all(path.exists() for path in minilm_files):
        _run_child(
            [
                "--build-index",
                "--model-path",
                str(minilm_path),
                "--esco-csv",
                str(esco_csv),
                "--minilm-dir",
                str(minilm_dir),
                "--batch-size",
                str(batch_size),
            ]
        )
    minilm_index = _ensure_minilm_index(minilm_path, esco_csv, minilm_dir, batch_size)

    with tempfile.TemporaryDirectory(prefix="phase5_a4_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        mpnet_metrics_path = temporary_path / "mpnet_metrics.json"
        minilm_metrics_path = temporary_path / "minilm_metrics.json"
        common_arguments = [
            "--esco-csv",
            str(esco_csv),
            "--review-csv",
            str(review_csv),
            "--labels",
            str(labels),
        ]
        _run_child(
            [
                "--single-model",
                MPNET_MODEL_ID,
                "--model-path",
                str(mpnet_path),
                "--embeddings",
                str(mpnet_embeddings),
                "--index",
                str(mpnet_index),
                "--metrics-output",
                str(mpnet_metrics_path),
                *common_arguments,
            ]
        )
        _run_child(
            [
                "--single-model",
                MINILM_MODEL_ID,
                "--model-path",
                str(minilm_path),
                "--embeddings",
                str(minilm_index["embeddings"]),
                "--index",
                str(minilm_index["index"]),
                "--metrics-output",
                str(minilm_metrics_path),
                *common_arguments,
            ]
        )
        mpnet_metrics = _read_json(mpnet_metrics_path)
        minilm_metrics = _read_json(minilm_metrics_path)

    review_rows, _ = _review_index(review_csv)

    report: dict[str, Any] = {
        "ablation": "A4",
        "description": "Replace all-mpnet-base-v2 with all-MiniLM-L6-v2 in the ESCO ranker",
        "dataset": "gold",
        "review_csv": str(review_csv),
        "reviewed_rows": len(review_rows),
        "reference_mpnet": {
            "model_id": MPNET_MODEL_ID,
            "model_path": str(mpnet_path),
            "embeddings": str(mpnet_embeddings),
            "source_index_artifact": str(mpnet_index),
            "runtime_index": "numpy_exact_inner_product",
            "metrics": mpnet_metrics,
        },
        "minilm": {
            "model_id": MINILM_MODEL_ID,
            "model_path": str(minilm_path),
            "index_dir": str(minilm_dir),
            "dimension": minilm_index["dimension"],
            "runtime_index": "numpy_exact_inner_product",
            "metrics": minilm_metrics,
        },
        "delta_minilm_minus_mpnet": _deltas(minilm_metrics, mpnet_metrics),
        "comparison_note": (
            "This is a controlled MPNet versus MiniLM comparison: both models use local "
            "SentenceTransformer snapshots, the same ESCO labels, the same reviewed spans, and the "
            "same lexicon-first ranking procedure."
        ),
        "interpretation": (
            "Positive deltas indicate that MiniLM outperformed MPNet; negative deltas indicate a decrease. "
            "Both models were loaded from local SentenceTransformer snapshots and evaluated on the same "
            f"{int(mpnet_metrics['documents'])} URI-mapped spans with the same lexicon-first ranking procedure. "
            "FAISS was unavailable on "
            "this macOS host, so both models use the mathematically equivalent NumPy exact inner-product index."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5.5 ablation A4.")
    parser.add_argument("--single-model", choices=[MPNET_MODEL_ID, MINILM_MODEL_ID])
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--esco-csv", type=Path, default=DEFAULT_ESCO_CSV)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--mpnet-embeddings", type=Path, default=DEFAULT_MPNet_EMBEDDINGS)
    parser.add_argument("--mpnet-index", type=Path, default=DEFAULT_MPNet_INDEX)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--minilm-dir", type=Path, default=DEFAULT_MINILM_DIR)
    parser.add_argument("--project-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.build_index:
        if args.model_path is None:
            parser.error("--build-index requires --model-path")
        _ensure_minilm_index(args.model_path, args.esco_csv, args.minilm_dir, args.batch_size)
        return
    if args.single_model:
        if args.model_path is None or args.embeddings is None or args.index is None or args.metrics_output is None:
            parser.error("--single-model requires --model-path, --embeddings, --index, and --metrics-output")
        _run_single_model(
            args.single_model,
            args.model_path,
            args.esco_csv,
            args.review_csv,
            args.embeddings,
            args.index,
            args.labels,
            args.metrics_output,
        )
        return
    report = run_ablation_a4(
        esco_csv=args.esco_csv,
        review_csv=args.review_csv,
        mpnet_embeddings=args.mpnet_embeddings,
        mpnet_index=args.mpnet_index,
        labels=args.labels,
        minilm_dir=args.minilm_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        project_cache=args.project_cache,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
