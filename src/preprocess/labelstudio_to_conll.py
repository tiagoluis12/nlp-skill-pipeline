from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def load_tasks(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    return payload


def tokenize(text: str) -> list[Token]:
    try:
        import spacy

        doc = spacy.blank("en")(text)
        return [Token(token.text, token.idx, token.idx + len(token.text)) for token in doc if not token.is_space]
    except Exception:
        tokens: list[Token] = []
        cursor = 0
        for part in text.split():
            start = text.find(part, cursor)
            end = start + len(part)
            tokens.append(Token(part, start, end))
            cursor = end
        return tokens


def extract_spans(task: dict) -> list[Span]:
    annotations = task.get("annotations") or []
    if not annotations:
        return []

    annotation = next((item for item in reversed(annotations) if not item.get("was_cancelled")), annotations[-1])
    spans: list[Span] = []
    for result in annotation.get("result", []):
        if result.get("type") != "labels":
            continue
        value = result.get("value") or {}
        labels = value.get("labels") or []
        if not labels:
            continue
        start = value.get("start")
        end = value.get("end")
        if start is None or end is None:
            continue
        spans.append(Span(start=int(start), end=int(end), label=str(labels[0])))
    spans.sort(key=lambda span: (span.start, span.end))
    return spans


def label_tokens(tokens: Iterable[Token], spans: list[Span]) -> list[str]:
    token_list = list(tokens)
    labels = ["O"] * len(token_list)
    for span in spans:
        matched = [index for index, token in enumerate(token_list) if token.end > span.start and token.start < span.end]
        if not matched:
            continue
        for offset, token_index in enumerate(matched):
            prefix = "B" if offset == 0 else "I"
            if labels[token_index] == "O":
                labels[token_index] = f"{prefix}-{span.label}"
    return labels


def export_conll(tasks: list[dict], output_path: Path, jsonl_path: Path | None = None) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    total_tokens = 0

    with output_path.open("w", encoding="utf-8") as conll_fh:
        jsonl_fh = jsonl_path.open("w", encoding="utf-8") if jsonl_path is not None else None
        try:
            for task in tasks:
                text = (task.get("data") or {}).get("text") or ""
                tokens = tokenize(text)
                spans = extract_spans(task)
                labels = label_tokens(tokens, spans)

                for token, label in zip(tokens, labels, strict=True):
                    conll_fh.write(f"{token.text} {label}\n")
                conll_fh.write("\n")

                if jsonl_fh is not None:
                    jsonl_fh.write(
                        json.dumps(
                            {
                                "id": task.get("id"),
                                "text": text,
                                "tokens": [token.text for token in tokens],
                                "labels": labels,
                                "role_type": (task.get("data") or {}).get("role_type"),
                                "source_id": (task.get("data") or {}).get("source_id"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                total_tasks += 1
                total_tokens += len(tokens)
        finally:
            if jsonl_fh is not None:
                jsonl_fh.close()

    return {"tasks": total_tasks, "tokens": total_tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Label Studio annotations to CoNLL BIO format.")
    parser.add_argument("--input", type=Path, required=True, help="Label Studio JSON export file")
    parser.add_argument("--output", type=Path, required=True, help="CoNLL output path")
    parser.add_argument("--jsonl-output", type=Path, help="Optional JSONL output with tokens and labels")
    args = parser.parse_args()

    tasks = load_tasks(args.input)
    summary = export_conll(tasks, args.output, args.jsonl_output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
