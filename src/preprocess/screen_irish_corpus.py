"""Normalize and screen Irish job-posting sources for manual corpus selection.

The source exports have incompatible schemas and may contain contact information.
This module creates an auditable, PII-redacted canonical dataset without changing
any source file. It deliberately sends uncertain roles and seniority cases to
manual review rather than silently excluding them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from src.preprocess.clean import normalize_text, redact_pii, strip_html

ROLE_TYPES = ("cloud_support", "cybersecurity", "data_analyst", "qa_engineer", "software_dev")
IRISH_TARGET_RECORDS = 115
IRISH_TARGET_ROLE_COUNTS = {
    "cloud_support": 24,
    "cybersecurity": 23,
    "data_analyst": 24,
    "qa_engineer": 21,
    "software_dev": 23,
}
DEFAULT_INPUTS = (
    "data/raw/dataset_irishjobs.json",
    "data/raw/dataset_irishjobs_ie.json",
    "data/raw/dataset_linkedin.json",
    "data/raw/dataset_jobs-ie.json",
    "data/raw/dataset_indeed_ cybersecurity.json",
    "data/raw/dataset_indeed_cybersecurity_2.json",
    "data/raw/dataset_indeed_ data_analyst.json",
    "data/raw/dataset_indeed_data analyst_2.json",
    "data/raw/dataset_indeed_ qa_engineer.json",
    "data/raw/dataset_indeed_cloud_support.json",
    "data/raw/dataset_indeed_software_dev.json",
)

SENIOR_TITLE_RE = re.compile(
    r"\b(?:senior|lead|principal|staff|manager|director|architect|head|vice president|vp|expert)\b", re.I
)
ENTRY_TITLE_RE = re.compile(
    r"\b(?:junior|graduate|entry[ -]level|trainee|intern(?:ship)?|apprentice|early careers?|new grad|associate)\b", re.I
)
ROLE_PATTERNS = {
    "cloud_support": re.compile(r"\b(?:cloud|aws|azure|gcp|devops|site reliability|sre|infrastructure)\b", re.I),
    "cybersecurity": re.compile(r"\b(?:cyber(?:security)?|information security|infosec|security operations|soc|penetration test)\b", re.I),
    "data_analyst": re.compile(r"\b(?:data analyst|analytics|business intelligence|\bbi\b|reporting analyst|data science)\b", re.I),
    "qa_engineer": re.compile(r"\b(?:quality assurance|\bqa\b|test(?:ing|er| engineer| analyst| automation)|validation engineer)\b", re.I),
    "software_dev": re.compile(r"\b(?:software (?:engineer|developer|development)|application(?:s)? engineer|full[ -]?stack|backend|front[ -]?end|web developer|programmer)\b", re.I),
}


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return normalize_text(strip_html(value))


def _first_text(record: dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return _text(value)
    return ""


def _description(record: dict[str, Any]) -> str:
    return redact_pii(_first_text(record, "descriptionText", "description", "descriptionMarkdown", "descriptionHtml", "summary"))


def _location(record: dict[str, Any]) -> str:
    location = record.get("location")
    if isinstance(location, dict):
        return _text(location.get("formattedAddressLong") or location.get("fullAddress") or location.get("city"))
    return _text(location)


def _country(record: dict[str, Any], source: str, location: str) -> str:
    country = str(record.get("country") or "").strip().upper()
    if country in {"IE", "IRELAND", "REPUBLIC OF IRELAND"}:
        return "IE"
    nested = record.get("location")
    if isinstance(nested, dict) and str(nested.get("countryCode") or "").upper() == "IE":
        return "IE"
    if source == "linkedin" and re.search(r"\b(?:ireland|éire)\b", location, re.I):
        return "IE"
    return country or "UNKNOWN"


def source_name(path: Path) -> str:
    name = path.name.lower()
    if "indeed" in name:
        return "indeed"
    if "linkedin" in name:
        return "linkedin"
    if "jobs-ie" in name:
        return "jobs_ie"
    if "irishjobs" in name:
        return "irishjobs"
    return path.stem


def source_role_hint(path: Path) -> str | None:
    normalized = re.sub(r"[^a-z]+", "_", path.stem.lower()).strip("_")
    for role_type in ROLE_TYPES:
        if role_type in normalized:
            return role_type
    return None


def canonicalize_record(record: dict[str, Any], path: Path) -> dict[str, Any]:
    source = source_name(path)
    location = _location(record)
    source_id = str(record.get("jobId") or record.get("jobKey") or record.get("id") or "")
    raw_text = _description(record)
    title = _text(record.get("title"))
    company = _first_text(record, "companyName", "company")
    url = _first_text(record, "jobUrl", "url", "link", "originalUrl")
    published_at = _first_text(record, "datePublished", "publishDateISO", "publishDate", "postedAt", "originalPublishDate")
    digest = hashlib.sha256(raw_text.casefold().encode("utf-8")).hexdigest() if raw_text else ""
    return {
        "id": f"{source}:{source_id}" if source_id else f"{source}:{digest[:16]}",
        "source": source,
        "source_file": path.name,
        "source_job_id": source_id or None,
        "url": url or None,
        "title": title,
        "company": company or None,
        "country": _country(record, source, location),
        "location": location or None,
        "published_at": published_at or None,
        "role_hint": source_role_hint(path),
        "industry": _text(record.get("industry") or record.get("companyIndustry")) or None,
        "raw_text": raw_text,
        "text_digest": digest,
        "expired": bool(record.get("expired", False)),
    }


def _published_before(value: str | None, cutoff: date) -> bool:
    if not value:
        return False
    try:
        return date.fromisoformat(value[:10]) < cutoff
    except ValueError:
        return False


def classify_role(record: dict[str, Any]) -> tuple[str | None, str, list[str]]:
    title = record["title"]
    context = " ".join(filter(None, (title, record.get("industry"), record.get("role_hint"))))
    full_context = f"{context} {record['raw_text'][:2000]}"
    scores = {role: int(bool(pattern.search(context))) * 3 + int(bool(pattern.search(full_context))) for role, pattern in ROLE_PATTERNS.items()}
    best_score = max(scores.values())
    if best_score == 0:
        return None, "none", []
    best = sorted(role for role, score in scores.items() if score == best_score)
    confidence = "high" if len(best) == 1 and best_score >= 4 else "medium" if len(best) == 1 else "low"
    return best[0], confidence, best


def screen_record(record: dict[str, Any], cutoff: date) -> dict[str, Any]:
    row = dict(record)
    candidate_role, role_confidence, matching_roles = classify_role(row)
    title = row["title"]
    entry_signal = bool(ENTRY_TITLE_RE.search(title))
    senior_signal = bool(SENIOR_TITLE_RE.search(title))
    reasons: list[str] = []
    if row["country"] != "IE":
        reasons.append("non_ireland")
    if len(row["raw_text"]) < 200:
        reasons.append("missing_or_short_description")
    if row["expired"]:
        reasons.append("expired")
    if _published_before(row["published_at"], cutoff):
        reasons.append("older_than_cutoff")
    if senior_signal:
        reasons.append("senior_title")
    if candidate_role is None:
        reasons.append("not_target_tech_role")

    hard_exclusions = {"non_ireland", "missing_or_short_description", "expired", "older_than_cutoff", "senior_title", "not_target_tech_role"}
    decision = "excluded" if any(reason in hard_exclusions for reason in reasons) else "eligible"
    if decision == "eligible" and (not entry_signal or role_confidence != "high"):
        decision = "manual_review"
    row.update(
        {
            "candidate_role": candidate_role,
            "role_confidence": role_confidence,
            "matching_roles": matching_roles,
            "entry_signal": entry_signal,
            "senior_signal": senior_signal,
            "screening_reasons": reasons,
            "screening_decision": decision,
        }
    )
    return row


def deduplicate(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed = 0
    for record in records:
        key = record["text_digest"] or (record["source"], record["source_job_id"]) or record["url"]
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(record)
    return kept, removed


def load_sources(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"Expected a JSON array in {path}")
        records.extend(canonicalize_record(row, path) for row in payload if isinstance(row, dict))
    return records


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            output = {key: value for key, value in record.items() if key not in {"text_digest", "expired"}}
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def write_reports(records: list[dict[str, Any]], *, input_count: int, deduplicated: int, cutoff: date, csv_path: Path, json_path: Path) -> dict[str, Any]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "source", "source_file", "title", "company", "country", "location", "published_at", "role_hint", "candidate_role", "role_confidence", "entry_signal", "senior_signal", "screening_decision", "screening_reasons", "url"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            report_row = {column: record.get(column) for column in columns}
            report_row["screening_reasons"] = ";".join(record["screening_reasons"])
            writer.writerow(report_row)

    by_decision = Counter(record["screening_decision"] for record in records)
    by_role = {decision: dict(Counter(record["candidate_role"] or "unclassified" for record in records if record["screening_decision"] == decision)) for decision in sorted(by_decision)}
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        by_source[record["source"]][record["screening_decision"]] += 1
    summary = {
        "screening_date_cutoff": cutoff.isoformat(),
        "input_records": input_count,
        "deduplicated_records": deduplicated,
        "duplicates_removed": input_count - deduplicated,
        "decision_counts": dict(by_decision),
        "role_counts_by_decision": by_role,
        "source_counts_by_decision": {source: dict(counts) for source, counts in sorted(by_source.items())},
        "target_records": IRISH_TARGET_RECORDS,
        "target_role_counts": IRISH_TARGET_ROLE_COUNTS,
        "manual_review_instruction": "The approved Irish target is 115 postings with the documented role allocation; retain only validated entry-level records and report the availability limitation.",
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize, deduplicate, and screen Irish job-posting exports.")
    parser.add_argument("--inputs", nargs="+", type=Path, default=[Path(path) for path in DEFAULT_INPUTS])
    parser.add_argument("--output", type=Path, default=Path("data/clean/irish_screened_candidates.jsonl"))
    parser.add_argument("--report-csv", type=Path, default=Path("results/irish_corpus_screening.csv"))
    parser.add_argument("--report-json", type=Path, default=Path("results/irish_corpus_screening.json"))
    parser.add_argument("--published-on-or-after", default="2025-01-01", help="ISO cutoff; older dated postings are excluded.")
    args = parser.parse_args()
    cutoff = date.fromisoformat(args.published_on_or_after)
    missing = [str(path) for path in args.inputs if not path.exists()]
    if missing:
        parser.error(f"Input files not found: {', '.join(missing)}")

    source_records = load_sources(args.inputs)
    unique_records, duplicates_removed = deduplicate(source_records)
    screened = [screen_record(record, cutoff) for record in unique_records]
    candidates = [record for record in screened if record["screening_decision"] != "excluded"]
    write_jsonl(candidates, args.output)
    summary = write_reports(
        screened,
        input_count=len(source_records),
        deduplicated=len(unique_records),
        cutoff=cutoff,
        csv_path=args.report_csv,
        json_path=args.report_json,
    )
    summary.update({"candidate_output": str(args.output), "candidate_records": len(candidates)})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
