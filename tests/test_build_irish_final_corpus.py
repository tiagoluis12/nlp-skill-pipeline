from __future__ import annotations

from src.preprocess.build_irish_final_corpus import build_pilot_tasks, materialize_records


class DummyNLP:
    def __call__(self, text):
        class Sentence:
            def __init__(self, value):
                self.text = value

        class Document:
            sents = [Sentence(part.strip()) for part in text.split(".") if part.strip()]

        return Document()


def _candidate(identifier: str, role: str) -> dict:
    return {
        "id": identifier,
        "country": "IE",
        "screening_decision": "manual_review",
        "title": f"{role} posting",
        "company": "Example Ltd",
        "location": "Dublin",
        "published_at": "2026-07-01",
        "source": "test",
        "source_file": "test.json",
        "url": "https://example.test/job",
        "raw_text": "Python and communication skills are required. Apply online.",
    }


def test_materialize_records_assigns_queue_role_and_sentences() -> None:
    accepted = {"software_dev": ["s1"], "data_analyst": [], "cloud_support": [], "cybersecurity": [], "qa_engineer": []}
    candidates = {"s1": _candidate("s1", "software_dev")}
    records = materialize_records(accepted, candidates, DummyNLP(), target_counts={role: 1 if role == "software_dev" else 0 for role in accepted})

    assert len(records) == 1
    assert records[0]["role_type"] == "software_dev"
    assert records[0]["sentences"] == ["Python and communication skills are required", "Apply online"]


def test_build_pilot_tasks_is_stratified() -> None:
    roles = ("cloud_support", "cybersecurity", "data_analyst", "qa_engineer", "software_dev")
    records = [{**_candidate(f"{role}-{index}", role), "role_type": role, "sentences": ["Text."]} for role in roles for index in range(2)]
    tasks = build_pilot_tasks(records, per_role=2)

    assert len(tasks) == 10
    assert [task["data"]["role_type"] for task in tasks[:2]] == ["cloud_support", "cloud_support"]
    assert len({task["data"]["source_id"] for task in tasks}) == 10
