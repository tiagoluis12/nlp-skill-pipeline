from __future__ import annotations

import json

from src.preprocess.clean import clean_text, redact_pii, strip_html
from src.preprocess.clean_corpus import clean_file


def test_strip_html_removes_markup() -> None:
    assert strip_html("<p>Hello <strong>world</strong></p>") == "Hello world"


def test_redact_pii_removes_email_and_phone() -> None:
    text = "Email me at test@example.com or call +353 87 123 4567."
    redacted = redact_pii(text)
    assert "[EMAIL]" in redacted
    assert "[PHONE]" in redacted


def test_clean_text_returns_sentences() -> None:
    class DummyNLP:
        def __call__(self, text: str):
            class DummyDoc:
                def __init__(self, text: str):
                    self._text = text

                @property
                def sents(self):
                    class Sent:
                        def __init__(self, text: str):
                            self.text = text

                    return [Sent(part.strip()) for part in self._text.split(".") if part.strip()]

            return DummyDoc(text)

    sentences = clean_text("<p>One sentence. Two sentence.</p>", DummyNLP())
    assert sentences == ["One sentence", "Two sentence"]


def test_clean_file_preserves_expected_keys(tmp_path) -> None:
    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "clean.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"id":"1","role_type":"software_dev","country":"gb","description":"<p>Email test@example.com. Python required.</p>"}',
                '{"id":"2","role_type":"software_dev","country":"gb","description":"<p>Email test@example.com. Python required.</p>"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = clean_file(raw, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()

    assert summary["input_rows"] == 2
    assert summary["output_rows"] == 1
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["raw_text"].startswith("<p>Email test@example.com")
    assert "[EMAIL]" in " ".join(row["sentences"])
    assert "Python required" in " ".join(row["sentences"])
