from __future__ import annotations

from src.preprocess.build_irish_review_queue import build_queues, rank_record


def _record(title: str, role: str, **overrides):
    record = {
        "id": title,
        "title": title,
        "candidate_role": role,
        "screening_decision": "manual_review",
        "role_confidence": "high",
        "entry_signal": False,
        "company": "Example",
        "raw_text": "Role description.",
    }
    record.update(overrides)
    return record


def test_entry_level_technical_title_ranks_above_senior_experience_signal() -> None:
    junior_score, _ = rank_record(_record("Junior Software Developer", "software_dev", entry_signal=True), "software_dev")
    experienced_score, _ = rank_record(
        _record("Software Engineer", "software_dev", raw_text="Requires 5 years experience."), "software_dev"
    )

    assert junior_score > experienced_score


def test_queue_marks_only_requested_number_for_review() -> None:
    queues, summary = build_queues(
        [
            _record("Junior Software Developer", "software_dev", entry_signal=True),
            _record("Software Engineer", "software_dev"),
            _record("Senior Cloud Engineer", "cloud_support", screening_decision="excluded"),
        ],
        per_role=1,
    )

    assert summary["software_dev"]["recommended_for_review"] == 1
    assert queues["software_dev"][0]["recommended_for_review"] is True
    assert queues["software_dev"][1]["recommended_for_review"] is False
    assert queues["cloud_support"] == []
