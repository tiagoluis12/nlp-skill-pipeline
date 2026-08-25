from __future__ import annotations

from src.evaluation.build_iaa_disagreement_report import build_report, disagreement_groups


def test_disagreement_groups_are_contiguous() -> None:
    groups = disagreement_groups(["a", "b", "c", "d"], ["O", "B-X", "O", "B-X"], ["O", "O", "O", "O"])

    assert len(groups) == 2
    assert groups[0]["text"] == "b"
    assert groups[1]["text"] == "d"


def test_build_report_returns_per_role_metrics(tmp_path) -> None:
    rows_a = [
        {"source_id": "s1", "role_type": "software_dev", "tokens": ["Python", "skills"], "labels": ["B-SKILL_TECH", "O"]}
    ]
    rows_b = [
        {"source_id": "s1", "role_type": "software_dev", "tokens": ["Python", "skills"], "labels": ["B-SKILL_TECH", "O"]}
    ]
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("\n".join(__import__("json").dumps(row) for row in rows_a) + "\n", encoding="utf-8")
    b.write_text("\n".join(__import__("json").dumps(row) for row in rows_b) + "\n", encoding="utf-8")

    report = build_report(a, b)

    assert report["tasks"] == 1
    assert report["kappa"] == 1.0
    assert report["per_role"]["software_dev"]["disagreements"] == 0
