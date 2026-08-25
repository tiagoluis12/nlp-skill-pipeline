import json
from pathlib import Path

from src.evaluation.apply_adjudication import apply_queue, write_conll


def test_apply_queue_replaces_disagreement_range(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    base.write_text(
        json.dumps(
            {
                "source_id": "task-1",
                "role_type": "software_dev",
                "tokens": ["Python", "and", "SQL"],
                "labels": ["B-SKILL_TECH", "O", "O"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    queue = tmp_path / "queue.csv"
    queue.write_text(
        "source_id,token_start,token_end,annotator1_labels,adjudicated_labels,decision,decision_reason\n"
        'task-1,2,3,O,B-SKILL_TECH,annotator2,named skill\n',
        encoding="utf-8",
    )
    output = tmp_path / "gold.jsonl"

    summary = apply_queue(base, queue, output)

    assert summary == {"tasks": 1, "adjudication_groups": 1}
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["labels"] == ["B-SKILL_TECH", "O", "B-SKILL_TECH"]

    conll = tmp_path / "gold.conll"
    write_conll(output, conll)
    assert "Python B-SKILL_TECH" in conll.read_text(encoding="utf-8")
