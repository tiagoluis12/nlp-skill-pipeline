from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

ESCO_NS = "http://data.europa.eu/esco/model#"
SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _lang_of(element: ET.Element) -> str:
    return element.attrib.get(f"{{{XML_NS}}}lang", "").lower()


def _dedupe(values: Iterable[str]) -> list[str]:
    seen = OrderedDict()
    for value in values:
        cleaned = " ".join(str(value).split()).strip()
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen.keys())


def parse_esco_rdf_stream(rdf_stream) -> list[dict[str, str]]:
    skills: list[dict[str, str]] = []

    context = ET.iterparse(rdf_stream, events=("end",))
    for _, elem in context:
        if _local_name(elem.tag) != "Concept":
            continue

        concept_uri = elem.attrib.get(f"{{{RDF_NS}}}about", "").strip()
        if "/skill/" not in concept_uri:
            elem.clear()
            continue

        preferred_labels: list[str] = []
        alt_labels: list[str] = []
        for child in list(elem):
            name = _local_name(child.tag)
            if name == "prefLabel" and _lang_of(child) == "en" and child.text:
                preferred_labels.append(child.text)
            elif name == "altLabel" and _lang_of(child) == "en" and child.text:
                alt_labels.append(child.text)

        preferred_label = _dedupe(preferred_labels)[0] if preferred_labels else ""
        skills.append(
            {
                "conceptUri": concept_uri,
                "preferredLabel": preferred_label,
                "altLabels": "\n".join(_dedupe(alt_labels)),
            }
        )

        elem.clear()

    return skills


def extract_esco_skills_from_zip(zip_path: Path, output_csv: Path, output_jsonl: Path | None = None) -> dict[str, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        member = next((name for name in archive.namelist() if name.endswith(".rdf")), None)
        if member is None:
            raise ValueError(f"No .rdf file found inside {zip_path}")

        with archive.open(member) as rdf_stream:
            skills = parse_esco_rdf_stream(rdf_stream)

    fieldnames = ["conceptUri", "preferredLabel", "altLabels"]
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skills)

    if output_jsonl is not None:
        with output_jsonl.open("w", encoding="utf-8") as fh:
            for row in skills:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {"skills": len(skills)}
    (output_csv.parent / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the ESCO skills CSV from the official RDF package.")
    parser.add_argument("--rdf-zip", type=Path, required=True, help="Official ESCO RDF zip archive")
    parser.add_argument("--output-csv", type=Path, default=Path("data/benchmarks/esco/skills_en.csv"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/benchmarks/esco/skills_en.jsonl"))
    args = parser.parse_args()

    summary = extract_esco_skills_from_zip(args.rdf_zip, args.output_csv, args.output_jsonl)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
