from pathlib import Path

from src.evaluation.build_adjudication_queue import build_queue, write_queue


def _write_jsonl(path: Path, labels: list[str]) -> None:
    path.write_text(
        '{"source_id":"task-1","role_type":"software_dev","tokens":["Python","and","SQL"],'
        f'"labels":{labels!r}'.replace("'", '"') + "}\n",
        encoding="utf-8",
    )


def test_build_queue_preserves_both_labels_and_blank_decision_fields(tmp_path: Path) -> None:
    labels_a = tmp_path / "a.jsonl"
    labels_b = tmp_path / "b.jsonl"
    _write_jsonl(labels_a, ["B-SKILL_TECH", "O", "O"])
    _write_jsonl(labels_b, ["O", "O", "B-SKILL_TECH"])

    rows = build_queue(labels_a, labels_b)
    output = tmp_path / "queue.csv"
    write_queue(rows, output)

    assert len(rows) == 2
    assert rows[0]["role_type"] == "software_dev"
    assert rows[0]["annotator1_labels"] == "B-SKILL_TECH"
    assert rows[0]["annotator2_labels"] == "O"
    assert rows[0]["adjudicated_labels"] == ""
    assert output.read_text(encoding="utf-8").startswith("priority,role_type,source_id")
