from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable

import requests

ROLE_QUERIES = {
    "software_dev": [
        "junior software developer",
        "graduate software engineer",
        "entry level software engineer",
        "junior python developer",
        "junior backend developer",
        "junior frontend developer",
        "junior full stack developer",
        "trainee developer",
    ],
    "data_analyst": [
        "junior data analyst",
        "graduate data analyst",
        "entry level data analyst",
        "junior business analyst",
        "junior reporting analyst",
        "junior BI analyst",
        "graduate analyst",
        "trainee analyst",
    ],
    "qa_engineer": [
        "junior QA engineer",
        "junior test engineer",
        "entry level software tester",
        "quality assurance analyst",
        "quality assurance engineer",
        "test analyst",
        "automation tester",
        "SDET",
        "software development engineer in test",
        "test automation engineer",
        "manual tester",
        "QA tester",
        "software tester",
        "quality engineer",
        "QA technician",
        "trainee QA",
    ],
    "cloud_support": [
        "cloud support engineer",
        "cloud engineer",
        "technical support engineer cloud",
        "IT support engineer cloud",
        "cloud operations engineer",
        "cloud technician",
        "junior systems engineer cloud",
        "service desk engineer cloud",
    ],
    "cybersecurity": [
        "junior cybersecurity analyst",
        "information security analyst",
        "SOC analyst",
        "security operations analyst",
        "junior security engineer",
        "cyber security analyst",
        "vulnerability analyst",
        "threat analyst",
    ],
}
ROLE_TYPES = tuple(ROLE_QUERIES.keys())


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def md5_digest(job_id: str | None, country: str, title: str, company: str, description: str) -> str:
    if job_id:
        key = f"{country}|{job_id}"
    else:
        key = f"{country}|{title}|{company}|{description[:200]}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def fetch_adzuna(
    query: str,
    page: int,
    app_id: str,
    app_key: str,
    country: str = "gb",
    where: str | None = None,
    results_per_page: int = 50,
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> dict:
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": query,
        "results_per_page": results_per_page,
        "content-type": "application/json",
    }
    if where:
        params["where"] = where
    for attempt in range(1, retries + 1):
        response = requests.get(url, params=params, timeout=60)
        if response.ok:
            return response.json()
        if attempt == retries:
            response.raise_for_status()
        time.sleep(backoff_seconds * attempt)
    raise RuntimeError("Unreachable retry loop")


def iter_job_ads(
    role_type: str,
    app_id: str,
    app_key: str,
    country: str,
    where: str | None,
    pages: int,
    results_per_page: int,
    query_limit: int | None,
    target_records: int | None,
    sleep_seconds: float,
) -> Iterable[dict]:
    seen = set()
    emitted = 0
    queries = ROLE_QUERIES[role_type]
    if query_limit is not None:
        queries = queries[:query_limit]
    for page in range(1, pages + 1):
        for query in queries:
            payload = fetch_adzuna(
                query=query,
                page=page,
                app_id=app_id,
                app_key=app_key,
                country=country,
                where=where,
                results_per_page=results_per_page,
            )
            for item in payload.get("results", []):
                title = item.get("title", "")
                company = (item.get("company") or {}).get("display_name", "")
                description = item.get("description", "")
                job_id = item.get("id") or item.get("adref")
                record_id = md5_digest(job_id, country, title, company, description)
                if record_id in seen:
                    continue
                seen.add(record_id)
                yield {
                    "id": record_id,
                    "job_id": job_id,
                    "country": country,
                    "role_type": role_type,
                    "query": query,
                    "page": page,
                    "title": title,
                    "company": company,
                    "description": description,
                    "created": item.get("created"),
                    "location": (item.get("location") or {}).get("display_name"),
                    "source": "adzuna",
                }
                emitted += 1
                if target_records is not None and emitted >= target_records:
                    return
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)


def collect_role_type(
    role_type: str,
    output_path: Path,
    app_id: str,
    app_key: str,
    countries: list[str],
    where: str | None,
    pages: int,
    results_per_page: int,
    query_limit: int | None,
    target_records: int | None,
    sleep_seconds: float,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    unique_titles = set()
    with output_path.open("w", encoding="utf-8") as fh:
        for country in countries:
            for record in iter_job_ads(
                role_type=role_type,
                app_id=app_id,
                app_key=app_key,
                country=country,
                where=where,
                pages=pages,
                results_per_page=results_per_page,
                query_limit=query_limit,
                target_records=None if target_records is None else max(0, target_records - count),
                sleep_seconds=sleep_seconds,
            ):
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                unique_titles.add(record["title"])
                if target_records is not None and count >= target_records:
                    break
            if target_records is not None and count >= target_records:
                break
    summary = {
        "role_type": role_type,
        "countries": countries,
        "where": where,
        "output": str(output_path),
        "records": count,
        "unique_titles": len(unique_titles),
        "target_records": target_records,
        "pages": pages,
        "results_per_page": results_per_page,
        "query_limit": query_limit,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Collect entry-level job postings from Adzuna.")
    parser.add_argument("--role-type", choices=(*ROLE_TYPES, "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--country", default="gb", help="Single country code to collect from if --countries is not supplied.")
    parser.add_argument("--countries", nargs="+", default=None, help="Optional list of country codes to collect from.")
    parser.add_argument("--where", default=None, help="Optional Adzuna location filter, e.g. 'Republic of Ireland'.")
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--results-per-page", type=int, default=50)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--target-records", type=int, default=600)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise SystemExit("Set ADZUNA_APP_ID and ADZUNA_APP_KEY before running collect.py.")

    countries = args.countries or [args.country]
    role_types = ROLE_TYPES if args.role_type == "all" else (args.role_type,)
    for role_type in role_types:
        output_path = args.output_dir / f"{role_type}.jsonl"
        collect_role_type(
            role_type=role_type,
            output_path=output_path,
            app_id=app_id,
            app_key=app_key,
            countries=countries,
            where=args.where,
            pages=args.pages,
            results_per_page=args.results_per_page,
            query_limit=args.query_limit,
            target_records=args.target_records,
            sleep_seconds=args.sleep_seconds,
        )


if __name__ == "__main__":
    main()
