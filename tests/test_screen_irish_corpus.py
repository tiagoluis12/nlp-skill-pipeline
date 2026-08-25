from __future__ import annotations

from datetime import date
from pathlib import Path

from src.preprocess.screen_irish_corpus import (
    IRISH_TARGET_RECORDS,
    IRISH_TARGET_ROLE_COUNTS,
    canonicalize_record,
    deduplicate,
    screen_record,
    write_reports,
)


def test_irish_sampling_target_is_documented() -> None:
    assert IRISH_TARGET_RECORDS == 115
    assert sum(IRISH_TARGET_ROLE_COUNTS.values()) == IRISH_TARGET_RECORDS


def test_canonicalize_indeed_removes_pii_and_reads_nested_country() -> None:
    record = {
        "jobKey": "abc123",
        "title": "Junior Software Developer",
        "companyName": "Example Ltd",
        "location": {"countryCode": "IE", "formattedAddressLong": "Galway, Ireland"},
        "descriptionText": "Contact hiring@example.ie. Build Python applications and APIs for customers.",
        "datePublished": "2026-07-30",
    }
    row = canonicalize_record(record, Path("dataset_indeed_software_dev.json"))

    assert row["id"] == "indeed:abc123"
    assert row["country"] == "IE"
    assert "hiring@example.ie" not in row["raw_text"]
    assert "[EMAIL]" in row["raw_text"]


def test_canonicalize_preserves_url_and_publication_timestamp() -> None:
    row = canonicalize_record(
        {
            "jobId": "107724287",
            "title": "Junior Software Developer",
            "country": "IE",
            "url": "https://example.test/job/107724287",
            "publishDate": "2026-07-20T08:23:01.35Z",
            "description": "Build Python applications and APIs for customers. " * 10,
        },
        Path("dataset_irishjobs.json"),
    )

    assert row["url"] == "https://example.test/job/107724287"
    assert row["published_at"] == "2026-07-20T08:23:01.35Z"


def test_screen_excludes_senior_and_keeps_uncertain_for_manual_review() -> None:
    cutoff = date(2025, 1, 1)
    senior = screen_record(
        {
            "title": "Senior Cloud Engineer",
            "industry": None,
            "role_hint": "cloud_support",
            "raw_text": "Cloud platform engineering " * 20,
            "country": "IE",
            "published_at": "2026-01-01",
            "expired": False,
        },
        cutoff,
    )
    uncertain = screen_record(
        {
            "title": "Software Engineer",
            "industry": None,
            "role_hint": "software_dev",
            "raw_text": "Develop software services using Python and SQL. " * 20,
            "country": "IE",
            "published_at": "2026-01-01",
            "expired": False,
        },
        cutoff,
    )

    assert senior["screening_decision"] == "excluded"
    assert "senior_title" in senior["screening_reasons"]
    assert uncertain["screening_decision"] == "manual_review"
    assert uncertain["candidate_role"] == "software_dev"


def test_deduplicate_uses_normalized_description_digest() -> None:
    rows, removed = deduplicate(
        [
            {"text_digest": "same", "source": "a", "source_job_id": "1", "url": None},
            {"text_digest": "same", "source": "b", "source_job_id": "2", "url": None},
            {"text_digest": "other", "source": "b", "source_job_id": "3", "url": None},
        ]
    )

    assert len(rows) == 2
    assert removed == 1


def test_write_reports_does_not_include_raw_text_in_csv(tmp_path) -> None:
    record = {
        "id": "indeed:1",
        "source": "indeed",
        "source_file": "source.json",
        "source_job_id": "1",
        "url": "https://example.test/job/1",
        "title": "Junior Software Developer",
        "company": "Example",
        "country": "IE",
        "location": "Galway",
        "published_at": "2026-07-30",
        "role_hint": "software_dev",
        "industry": None,
        "raw_text": "Private source text",
        "text_digest": "digest",
        "expired": False,
        "candidate_role": "software_dev",
        "role_confidence": "high",
        "matching_roles": ["software_dev"],
        "entry_signal": True,
        "senior_signal": False,
        "screening_reasons": [],
        "screening_decision": "eligible",
    }
    csv_path = tmp_path / "report.csv"
    write_reports([record], input_count=1, deduplicated=1, cutoff=date(2025, 1, 1), csv_path=csv_path, json_path=tmp_path / "report.json")

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "Private source text" not in csv_text
    assert "Junior Software Developer" in csv_text
