from pathlib import Path

from src.esco.ranker import _model_cache_directory, resolve_local_sentence_transformer


def test_model_cache_directory_uses_huggingface_layout() -> None:
    assert _model_cache_directory("all-mpnet-base-v2") == "models--sentence-transformers--all-mpnet-base-v2"


def test_resolve_local_sentence_transformer_finds_project_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--sentence-transformers--all-mpnet-base-v2" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)

    resolved = resolve_local_sentence_transformer("all-mpnet-base-v2", project_cache=tmp_path)

    assert resolved == snapshot


def test_resolve_local_sentence_transformer_accepts_direct_model_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    assert resolve_local_sentence_transformer(str(model_path)) == model_path
