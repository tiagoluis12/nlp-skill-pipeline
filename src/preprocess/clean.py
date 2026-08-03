from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}")
WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text


def load_sentence_nlp() -> "spacy.language.Language":
    import spacy

    for model_name in ("en_core_web_trf", "en_core_web_sm"):
        try:
            return spacy.load(model_name, exclude=["ner", "tagger", "lemmatizer", "attribute_ruler"])
        except Exception:
            continue
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def clean_text(text: str, nlp) -> list[str]:
    text = strip_html(text)
    text = normalize_text(text)
    text = redact_pii(text)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw job postings and segment sentences.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nlp = load_sentence_nlp()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()

    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            source_text = record.get("raw_text", "")
            sentences = clean_text(source_text, nlp)
            cleaned_text = " ".join(sentences)
            if cleaned_text in seen:
                continue
            seen.add(cleaned_text)
            out = {
                "id": record.get("id"),
                "role_type": record.get("role_type"),
                "sentences": sentences,
                "raw_text": cleaned_text,
            }
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
