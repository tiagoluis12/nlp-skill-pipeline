from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

from src.preprocess.extract_esco_skills import extract_esco_skills_from_zip, parse_esco_rdf_stream


def test_parse_esco_rdf_stream_extracts_skill_labels() -> None:
    rdf = """<?xml version='1.0' encoding='UTF-8'?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:skos="http://www.w3.org/2004/02/skos/core#">
  <skos:Concept rdf:about="http://data.europa.eu/esco/skill/123">
    <skos:prefLabel xml:lang="en">Cloud support</skos:prefLabel>
    <skos:altLabel xml:lang="en">Cloud infrastructure support</skos:altLabel>
    <skos:altLabel xml:lang="en">Cloud operations support</skos:altLabel>
    <skos:prefLabel xml:lang="fr">Support cloud</skos:prefLabel>
  </skos:Concept>
</rdf:RDF>
"""
    rows = parse_esco_rdf_stream(io.BytesIO(rdf.encode("utf-8")))
    assert rows == [
        {
            "conceptUri": "http://data.europa.eu/esco/skill/123",
            "preferredLabel": "Cloud support",
            "altLabels": "Cloud infrastructure support\nCloud operations support",
        }
    ]


def test_extract_esco_skills_from_zip_writes_csv(tmp_path: Path) -> None:
    rdf = """<?xml version='1.0' encoding='UTF-8'?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:skos="http://www.w3.org/2004/02/skos/core#">
  <skos:Concept rdf:about="http://data.europa.eu/esco/skill/123">
    <skos:prefLabel xml:lang="en">Cloud support</skos:prefLabel>
    <skos:altLabel xml:lang="en">Cloud infrastructure support</skos:altLabel>
  </skos:Concept>
</rdf:RDF>
"""
    zip_path = tmp_path / "esco.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("esco-v1.2.0.rdf", rdf)

    csv_path = tmp_path / "skills_en.csv"
    jsonl_path = tmp_path / "skills_en.jsonl"
    summary = extract_esco_skills_from_zip(zip_path, csv_path, jsonl_path)

    assert summary["skills"] == 1
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows == [
        {
            "conceptUri": "http://data.europa.eu/esco/skill/123",
            "preferredLabel": "Cloud support",
            "altLabels": "Cloud infrastructure support",
        }
    ]
    assert jsonl_path.read_text(encoding="utf-8").strip().startswith("{")
