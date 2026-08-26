from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_DATASET_DIR = Path("data/benchmarks/skillspan")
DEFAULT_OUTPUT_DIR = Path("models/ner/best_checkpoint")
DEFAULT_PRIMARY_MODEL = "jjzha/jobbert-base-cased"
DEFAULT_FALLBACK_MODEL = "bert-base-uncased"
DEFAULT_CACHE_DIR = Path(".cache/huggingface")
DEFAULT_MAX_LENGTH = 128
DEFAULT_MAX_PER_DEVICE_BATCH = 4
DEFAULT_GOLD_JSONL = Path("data/annotated/gold_annotator1.normalized.jsonl")

os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_CACHE_DIR / "datasets"))

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL_NAMES = ("O", "B-SKILL_TECH", "I-SKILL_TECH", "B-SKILL_TRANS", "I-SKILL_TRANS")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class NerBundle:
    model_name: str
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    offline_fallback: bool


@dataclass(frozen=True)
class SpanPrediction:
    start_token: int
    end_token: int
    label: str
    text: str
    confidence: float


def _configure_cache() -> None:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_CACHE_DIR / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_CACHE_DIR / "datasets"))


def _is_pretrained_model_available(model_name: str) -> bool:
    from huggingface_hub import snapshot_download

    repo_id = model_name if "/" in model_name else model_name
    try:
        snapshot_download(repo_id=repo_id, local_files_only=True, cache_dir=str(DEFAULT_CACHE_DIR))
    except Exception:
        return False
    return True


def load_skillspan_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR) -> DatasetDict:
    dataset = load_from_disk(str(dataset_dir))
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {dataset_dir}, found {type(dataset).__name__}")
    return dataset


def combine_skillspan_tags(
    skill_tags: Sequence[str],
    knowledge_tags: Sequence[str],
) -> list[str]:
    if len(skill_tags) != len(knowledge_tags):
        raise ValueError("SKILLSPAN token/label length mismatch")

    combined: list[str] = []
    for skill_tag, knowledge_tag in zip(skill_tags, knowledge_tags, strict=True):
        skill_tag = str(skill_tag)
        knowledge_tag = str(knowledge_tag)
        if skill_tag != "O":
            combined.append(f"{skill_tag}-SKILL_TECH")
        elif knowledge_tag != "O":
            combined.append(f"{knowledge_tag}-SKILL_TRANS")
        else:
            combined.append("O")
    return combined


def add_combined_tags(dataset: DatasetDict) -> DatasetDict:
    def _map_batch(batch: dict[str, list[object]]) -> dict[str, list[object]]:
        batch["ner_tags"] = [
            combine_skillspan_tags(skill_tags, knowledge_tags)
            for skill_tags, knowledge_tags in zip(batch["tags_skill"], batch["tags_knowledge"], strict=True)
        ]
        return batch

    return dataset.map(_map_batch, batched=True, load_from_cache_file=False, keep_in_memory=True)


def summarize_skillspan_label_overlap(dataset: DatasetDict) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split_name, split in dataset.items():
        overlap_sentences = 0
        overlap_tokens = 0
        total_tokens = 0
        for row in split:
            skill_tags = row["tags_skill"]
            knowledge_tags = row["tags_knowledge"]
            total_tokens += len(skill_tags)
            sentence_overlap = sum(
                skill_tag != "O" and knowledge_tag != "O"
                for skill_tag, knowledge_tag in zip(skill_tags, knowledge_tags, strict=True)
            )
            if sentence_overlap:
                overlap_sentences += 1
                overlap_tokens += sentence_overlap
        summary[split_name] = {
            "sentences_with_overlap": overlap_sentences,
            "overlap_tokens": overlap_tokens,
            "total_tokens": total_tokens,
        }
    return summary


def build_token_vocab(dataset: DatasetDict) -> dict[str, int]:
    unique_tokens: set[str] = set()
    for split in dataset.values():
        for tokens in split["tokens"]:
            unique_tokens.update(str(token) for token in tokens)

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
    }
    for token in sorted(unique_tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def build_offline_tokenizer(dataset: DatasetDict) -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    vocab = build_token_vocab(dataset)
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=256,
    )


def resolve_model_bundle(
    dataset: DatasetDict,
    model_name: str = DEFAULT_PRIMARY_MODEL,
    fallback_model_name: str = DEFAULT_FALLBACK_MODEL,
    allow_offline_fallback: bool = True,
) -> NerBundle:
    _configure_cache()

    candidate_models = [model_name]
    if fallback_model_name not in candidate_models:
        candidate_models.append(fallback_model_name)

    for candidate in candidate_models:
        if not _is_pretrained_model_available(candidate):
            continue
        tokenizer = AutoTokenizer.from_pretrained(
            candidate,
            local_files_only=True,
            use_fast=True,
            cache_dir=str(DEFAULT_CACHE_DIR),
        )
        if getattr(tokenizer, "model_max_length", None) is None or int(tokenizer.model_max_length) > 4096:
            tokenizer.model_max_length = 512
        model = AutoModelForTokenClassification.from_pretrained(
            candidate,
            local_files_only=True,
            num_labels=len(LABEL_NAMES),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            cache_dir=str(DEFAULT_CACHE_DIR),
        )
        return NerBundle(model_name=candidate, tokenizer=tokenizer, model=model, offline_fallback=False)

    if not allow_offline_fallback:
        raise RuntimeError(
            "No pretrained transformer checkpoint is cached locally. "
            f"Tried: {', '.join(candidate_models)}"
        )

    tokenizer = build_offline_tokenizer(dataset)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        num_labels=len(LABEL_NAMES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=256,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id or 0,
    )
    model = BertForTokenClassification(config)
    return NerBundle(
        model_name=f"{model_name}::offline-wordlevel-fallback",
        tokenizer=tokenizer,
        model=model,
        offline_fallback=True,
    )


def load_checkpoint_bundle(checkpoint_dir: Path = DEFAULT_OUTPUT_DIR) -> NerBundle:
    """Load the trained NER checkpoint for inference-only consumers.

    This is intentionally separate from ``resolve_model_bundle`` so training
    always starts from the configured base model rather than reusing a prior
    output checkpoint or silently falling back to an untrained model.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Trained NER checkpoint not found: {checkpoint_dir}. "
            "Train the model or provide the released checkpoint before running inference."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(checkpoint_dir), local_files_only=True)
    return NerBundle(
        model_name=str(checkpoint_dir),
        tokenizer=tokenizer,
        model=model,
        offline_fallback=False,
    )


def tokenize_and_align_labels(
    examples: dict[str, list[Sequence[str]]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, list[list[int]]]:
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    aligned_labels: list[list[int]] = []
    for batch_index, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids: list[int] = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
                continue
            if word_id != previous_word_id:
                label_ids.append(LABEL_TO_ID[str(word_labels[word_id])])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        aligned_labels.append(label_ids)
    tokenized["labels"] = aligned_labels
    return tokenized


def _ensure_length_match(expected: Sequence[str], actual: Sequence[str]) -> None:
    if len(expected) != len(actual):
        raise ValueError(f"Length mismatch: expected {len(expected)} tokens, got {len(actual)}")


def bio_to_spans(labels: Sequence[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    current_label: str | None = None

    for index, label in enumerate(list(labels) + ["O"]):
        if label.startswith("B-"):
            if start is not None and current_label is not None:
                spans.append((start, index, current_label))
            start = index
            current_label = label[2:]
            continue

        if label.startswith("I-"):
            entity = label[2:]
            if start is None or current_label != entity:
                if start is not None and current_label is not None:
                    spans.append((start, index, current_label))
                start = index
                current_label = entity
            continue

        if start is not None and current_label is not None:
            spans.append((start, index, current_label))
        start = None
        current_label = None

    return spans


def spans_from_prediction(
    tokens: Sequence[str],
    labels: Sequence[str],
    confidences: Sequence[float] | None = None,
) -> list[SpanPrediction]:
    _ensure_length_match(tokens, labels)
    if confidences is not None:
        _ensure_length_match(tokens, confidences)

    spans: list[SpanPrediction] = []
    start: int | None = None
    current_label: str | None = None
    current_scores: list[float] = []

    def _flush(end_index: int) -> None:
        nonlocal start, current_label, current_scores
        if start is None or current_label is None:
            return
        span_tokens = tokens[start:end_index]
        confidence = float(np.mean(current_scores)) if current_scores else 0.0
        spans.append(
            SpanPrediction(
                start_token=start,
                end_token=end_index,
                label=current_label,
                text=" ".join(str(token) for token in span_tokens),
                confidence=confidence,
            )
        )
        start = None
        current_label = None
        current_scores = []

    for index, label in enumerate(list(labels) + ["O"]):
        score = confidences[index] if confidences is not None and index < len(confidences) else 0.0
        if label.startswith("B-"):
            _flush(index)
            start = index
            current_label = label[2:]
            current_scores = [float(score)]
            continue

        if label.startswith("I-"):
            entity = label[2:]
            if start is None or current_label != entity:
                _flush(index)
                start = index
                current_label = entity
                current_scores = [float(score)]
            else:
                current_scores.append(float(score))
            continue

        _flush(index)
        if index < len(labels):
            start = None
            current_label = None
            current_scores = []

    return spans


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def predict_token_sequences(
    bundle: NerBundle,
    token_sequences: Sequence[Sequence[str]],
    batch_size: int = 16,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[dict[str, object]]:
    """Predict complete sequences with overlapping windows.

    JobBERT is limited to a short maximum sequence length. Overflow windows
    prevent long job postings from being silently padded with ``O`` labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle.model.to(device)
    bundle.model.eval()

    outputs: list[dict[str, object]] = []
    stride = min(32, max(1, max_length // 4))
    for tokens in token_sequences:
        tokens = list(tokens)
        if not tokens:
            outputs.append({"tokens": [], "labels": [], "confidences": []})
            continue

        encoding = bundle.tokenizer(
            [tokens],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        # Overflow metadata identifies the source sequence but is not a model input.
        inputs = {
            key: value.to(device)
            for key, value in encoding.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.no_grad():
            logits = bundle.model(**inputs).logits
        probs = _softmax(logits)
        predictions = torch.argmax(logits, dim=-1)

        best_predictions: dict[int, tuple[float, str]] = {}
        for window_index in range(logits.shape[0]):
            previous_word_id = None
            word_ids = encoding.word_ids(batch_index=window_index)
            for token_index, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id == previous_word_id:
                    continue
                predicted_id = int(predictions[window_index, token_index].item())
                label = ID_TO_LABEL[predicted_id]
                score = float(probs[window_index, token_index, predicted_id].item())
                current = best_predictions.get(word_id)
                if current is None or score > current[0]:
                    best_predictions[word_id] = (score, label)
                previous_word_id = word_id

        word_labels = [best_predictions.get(index, (0.0, "O"))[1] for index in range(len(tokens))]
        word_scores = [best_predictions.get(index, (0.0, "O"))[0] for index in range(len(tokens))]
        outputs.append({"tokens": tokens, "labels": word_labels, "confidences": word_scores})

    return outputs


def predict_text_spans(
    bundle: NerBundle,
    text: str,
    nlp: object | None = None,
    batch_size: int = 1,
) -> list[SpanPrediction]:
    try:
        import spacy

        if nlp is None:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_space]
    except Exception:
        tokens = text.split()

    if not tokens:
        return []

    predictions = predict_token_sequences(bundle, [tokens], batch_size=batch_size)[0]
    return spans_from_prediction(predictions["tokens"], predictions["labels"], predictions["confidences"])


def _sequence_metrics(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]]) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def _classification_report_dict(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]]) -> dict[str, object]:
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    per_label: dict[str, dict[str, float]] = {}
    for key, value in report.items():
        if key.endswith("avg"):
            continue
        if not isinstance(value, dict):
            continue
        per_label[key] = {
            "precision": float(value.get("precision", 0.0)),
            "recall": float(value.get("recall", 0.0)),
            "f1": float(value.get("f1-score", 0.0)),
            "support": float(value.get("support", 0.0)),
        }
    return {
        "per_label": per_label,
        "macro": {
            "precision": float(report.get("macro avg", {}).get("precision", 0.0)),
            "recall": float(report.get("macro avg", {}).get("recall", 0.0)),
            "f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
            "support": float(report.get("macro avg", {}).get("support", 0.0)),
        },
        "weighted": {
            "precision": float(report.get("weighted avg", {}).get("precision", 0.0)),
            "recall": float(report.get("weighted avg", {}).get("recall", 0.0)),
            "f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
            "support": float(report.get("weighted avg", {}).get("support", 0.0)),
        },
        "micro": {
            "precision": float(report.get("micro avg", {}).get("precision", 0.0)),
            "recall": float(report.get("micro avg", {}).get("recall", 0.0)),
            "f1": float(report.get("micro avg", {}).get("f1-score", 0.0)),
            "support": float(report.get("micro avg", {}).get("support", 0.0)),
        },
    }


def _score_spans_exact(gold: Sequence[tuple[int, int, str]], pred: Sequence[tuple[int, int, str]]) -> dict[str, float]:
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _spans_overlap(a: tuple[int, int, str], b: tuple[int, int, str]) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0])


def _source_sentence(tokens: Sequence[str], start: int, end: int) -> str:
    sentence_end_tokens = {".", "!", "?"}
    left = start
    while left > 0 and tokens[left - 1] not in sentence_end_tokens:
        left -= 1

    right = end
    while right < len(tokens) and tokens[right] not in sentence_end_tokens:
        right += 1
    if right < len(tokens):
        right += 1

    return " ".join(str(token) for token in tokens[left:right]).strip()


def _score_spans_relaxed(gold: Sequence[tuple[int, int, str]], pred: Sequence[tuple[int, int, str]]) -> dict[str, float]:
    gold_by_label: dict[str, list[int]] = {}
    for index, span in enumerate(gold):
        gold_by_label.setdefault(span[2], []).append(index)

    matched_gold: set[int] = set()
    tp = 0
    fp = 0
    for pred_span in pred:
        candidates = gold_by_label.get(pred_span[2], [])
        matched_index = None
        for gold_index in candidates:
            if gold_index in matched_gold:
                continue
            if _spans_overlap(pred_span, gold[gold_index]):
                matched_index = gold_index
                break
        if matched_index is None:
            fp += 1
        else:
            matched_gold.add(matched_index)
            tp += 1

    fn = len(gold) - len(matched_gold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _prediction_records(
    examples: Sequence[dict[str, object]],
    predictions: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for example, prediction in zip(examples, predictions, strict=True):
        gold_labels = [str(label) for label in example["ner_tags"]]
        pred_labels = [str(label) for label in prediction["labels"]]
        gold_spans = bio_to_spans(gold_labels)
        pred_spans = bio_to_spans(pred_labels)
        records.append(
            {
                "id": example.get("idx"),
                "source": example.get("source"),
                "tokens": list(example["tokens"]),
                "gold_labels": gold_labels,
                "predicted_labels": pred_labels,
                "gold_spans": [
                    {"start": start, "end": end, "label": label}
                    for start, end, label in gold_spans
                ],
                "predicted_spans": [
                    {
                        "start": span.start_token,
                        "end": span.end_token,
                        "label": span.label,
                        "text": span.text,
                        "confidence": span.confidence,
                    }
                    for span in spans_from_prediction(prediction["tokens"], pred_labels, prediction["confidences"])
                ],
            }
        )
    return records


def _evaluate_records(
    records: Sequence[dict[str, object]],
    predictions: Sequence[dict[str, object]],
    source: str,
    checkpoint_name: str,
) -> dict[str, object]:
    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    gold_spans: list[tuple[int, int, str]] = []
    pred_spans: list[tuple[int, int, str]] = []
    document_records: list[dict[str, object]] = []

    for example, prediction in zip(records, predictions, strict=True):
        gold_labels = [str(label) for label in example["labels"]]
        pred_labels = [str(label) for label in prediction["labels"]]
        y_true.append(gold_labels)
        y_pred.append(pred_labels)
        gold_span_list = bio_to_spans(gold_labels)
        pred_span_list = bio_to_spans(pred_labels)
        gold_spans.extend(gold_span_list)
        pred_spans.extend(pred_span_list)
        source_text = str(example.get("text", ""))
        document_records.append(
            {
                "id": example.get("id"),
                "source": example.get("role_type"),
                "source_id": example.get("source_id"),
                "source_text": source_text,
                "tokens": list(example["tokens"]),
                "gold_labels": gold_labels,
                "predicted_labels": pred_labels,
                "gold_spans": [
                    {
                        "start": start,
                        "end": end,
                        "label": label,
                        "text": " ".join(str(token) for token in example["tokens"][start:end]),
                        "source_sentence": _source_sentence(example["tokens"], start, end),
                    }
                    for start, end, label in gold_span_list
                ],
                "predicted_spans": [
                    {
                        "start": span.start_token,
                        "end": span.end_token,
                        "label": span.label,
                        "text": span.text,
                        "confidence": span.confidence,
                        "source_sentence": _source_sentence(example["tokens"], span.start_token, span.end_token),
                    }
                    for span in spans_from_prediction(prediction["tokens"], pred_labels, prediction["confidences"])
                ],
            }
        )

    token_metrics = _sequence_metrics(y_true, y_pred)
    exact = _score_spans_exact(gold_spans, pred_spans)
    relaxed = _score_spans_relaxed(gold_spans, pred_spans)
    report = _classification_report_dict(y_true, y_pred)

    return {
        "source": source,
        "documents": len(document_records),
        "model_name": checkpoint_name,
        "token_metrics": token_metrics,
        "span_exact": exact,
        "span_relaxed": relaxed,
        "classification_report": report,
        "document_records": document_records,
    }


def load_gold_jsonl(gold_jsonl: Path = DEFAULT_GOLD_JSONL) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with gold_jsonl.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "labels" not in record:
                raise KeyError(f"Missing 'labels' in {gold_jsonl} line {line_number}")
            if "tokens" not in record:
                raise KeyError(f"Missing 'tokens' in {gold_jsonl} line {line_number}")
            records.append(record)
    return records


def evaluate_gold_checkpoint(
    checkpoint_dir: Path,
    gold_jsonl: Path = DEFAULT_GOLD_JSONL,
    batch_size: int = 16,
) -> dict[str, object]:
    records = load_gold_jsonl(gold_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(checkpoint_dir), local_files_only=True)
    bundle = NerBundle(model_name=str(checkpoint_dir), tokenizer=tokenizer, model=model, offline_fallback=False)

    token_predictions = predict_token_sequences(
        bundle,
        [row["tokens"] for row in records],
        batch_size=min(batch_size, DEFAULT_MAX_PER_DEVICE_BATCH),
        max_length=DEFAULT_MAX_LENGTH,
    )

    report = _evaluate_records(records, token_predictions, source="gold", checkpoint_name=str(checkpoint_dir))
    return report


def train_ner_model(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_model_name: str = DEFAULT_PRIMARY_MODEL,
    fallback_model_name: str = DEFAULT_FALLBACK_MODEL,
    learning_rate: float = 3e-5,
    batch_size: int = 16,
    epochs: int = 3,
    seed: int = 42,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    allow_offline_fallback: bool = True,
) -> dict[str, object]:
    set_seed(seed)
    dataset = add_combined_tags(load_skillspan_dataset(dataset_dir))
    bundle = resolve_model_bundle(
        dataset,
        model_name=base_model_name,
        fallback_model_name=fallback_model_name,
        allow_offline_fallback=allow_offline_fallback,
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    tokenized_train = train_dataset.map(
        lambda batch: tokenize_and_align_labels(batch, bundle.tokenizer, max_length=max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    tokenized_eval = eval_dataset.map(
        lambda batch: tokenize_and_align_labels(batch, bundle.tokenizer, max_length=max_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=bundle.tokenizer)

    actual_batch_size = min(batch_size, DEFAULT_MAX_PER_DEVICE_BATCH)
    gradient_accumulation_steps = max(1, batch_size // actual_batch_size)

    def compute_metrics(eval_prediction) -> dict[str, float]:
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids
        predictions = np.argmax(logits, axis=-1)
        y_true: list[list[str]] = []
        y_pred: list[list[str]] = []
        for example_index in range(len(labels)):
            true_labels: list[str] = []
            pred_labels: list[str] = []
            for token_index, label_id in enumerate(labels[example_index]):
                if label_id == -100:
                    continue
                true_labels.append(ID_TO_LABEL[int(label_id)])
                pred_labels.append(ID_TO_LABEL[int(predictions[example_index, token_index])])
            y_true.append(true_labels)
            y_pred.append(pred_labels)
        return _sequence_metrics(y_true, y_pred)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=actual_batch_size,
        per_device_eval_batch_size=actual_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
        seed=seed,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=bundle.model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=bundle.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    bundle.tokenizer.save_pretrained(str(output_dir))

    best_checkpoint = trainer.state.best_model_checkpoint or str(output_dir)
    summary = {
        "model_name": bundle.model_name,
        "offline_fallback": bundle.offline_fallback,
        "best_checkpoint": best_checkpoint,
        "output_dir": str(output_dir),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "per_device_batch_size": actual_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "epochs": epochs,
        "train_metrics": {key: float(value) if isinstance(value, (int, float, np.floating)) else value for key, value in train_result.metrics.items()},
        "eval_metrics": {key: float(value) if isinstance(value, (int, float, np.floating)) else value for key, value in eval_metrics.items()},
    }
    return summary


def evaluate_ner_checkpoint(
    checkpoint_dir: Path,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    split: str = "test",
    batch_size: int = 16,
) -> dict[str, object]:
    dataset = add_combined_tags(load_skillspan_dataset(dataset_dir))
    if split not in dataset:
        raise KeyError(f"Split {split!r} not found in {dataset_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(checkpoint_dir), local_files_only=True)
    bundle = NerBundle(model_name=str(checkpoint_dir), tokenizer=tokenizer, model=model, offline_fallback=False)

    split_dataset = dataset[split]
    token_predictions = predict_token_sequences(
        bundle,
        [row["tokens"] for row in split_dataset],
        batch_size=min(batch_size, DEFAULT_MAX_PER_DEVICE_BATCH),
        max_length=DEFAULT_MAX_LENGTH,
    )

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    gold_spans: list[tuple[int, int, str]] = []
    pred_spans: list[tuple[int, int, str]] = []
    records: list[dict[str, object]] = []

    for example, prediction in zip(split_dataset, token_predictions, strict=True):
        gold_labels = [str(label) for label in example["ner_tags"]]
        pred_labels = [str(label) for label in prediction["labels"]]
        y_true.append(gold_labels)
        y_pred.append(pred_labels)
        gold_span_list = bio_to_spans(gold_labels)
        pred_span_list = bio_to_spans(pred_labels)
        gold_spans.extend(gold_span_list)
        pred_spans.extend(pred_span_list)
        records.append(
            {
                "id": example.get("idx"),
                "source": example.get("source"),
                "tokens": list(example["tokens"]),
                "gold_labels": gold_labels,
                "predicted_labels": pred_labels,
                "predicted_spans": [
                    {
                        "start": span.start_token,
                        "end": span.end_token,
                        "label": span.label,
                        "text": span.text,
                        "confidence": span.confidence,
                        "source_sentence": _source_sentence(example["tokens"], span.start_token, span.end_token),
                    }
                    for span in spans_from_prediction(prediction["tokens"], pred_labels, prediction["confidences"])
                ],
            }
        )

    token_metrics = _sequence_metrics(y_true, y_pred)
    exact = _score_spans_exact(gold_spans, pred_spans)
    relaxed = _score_spans_relaxed(gold_spans, pred_spans)
    report = _classification_report_dict(y_true, y_pred)

    return {
        "source": "skillspan",
        "split": split,
        "documents": len(split_dataset),
        "model_name": str(checkpoint_dir),
        "token_metrics": token_metrics,
        "span_exact": exact,
        "span_relaxed": relaxed,
        "classification_report": report,
        "document_records": records,
    }


def write_hp_search_csv(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No hyperparameter rows were provided")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_hyperparameter_search(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    search_output: Path = Path("results/ner_hp_search.csv"),
    base_model_name: str = DEFAULT_PRIMARY_MODEL,
    fallback_model_name: str = DEFAULT_FALLBACK_MODEL,
    learning_rates: Sequence[float] = (2e-5, 3e-5),
    batch_sizes: Sequence[int] = (16, 32),
    epochs_list: Sequence[int] = (3, 5),
    seed: int = 42,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    allow_offline_fallback: bool = True,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                candidate_output = output_dir / f"lr-{learning_rate:g}_bs-{batch_size}_ep-{epochs}"
                summary = train_ner_model(
                    dataset_dir=dataset_dir,
                    output_dir=candidate_output,
                    base_model_name=base_model_name,
                    fallback_model_name=fallback_model_name,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs,
                    seed=seed,
                    max_train_samples=max_train_samples,
                    max_eval_samples=max_eval_samples,
                    allow_offline_fallback=allow_offline_fallback,
                )
                row = {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "per_device_batch_size": summary["per_device_batch_size"],
                    "gradient_accumulation_steps": summary["gradient_accumulation_steps"],
                    "epochs": epochs,
                    "dev_precision": summary["eval_metrics"].get("eval_precision"),
                    "dev_recall": summary["eval_metrics"].get("eval_recall"),
                    "dev_f1": summary["eval_metrics"].get("eval_f1"),
                    "best_checkpoint": summary["best_checkpoint"],
                    "model_name": summary["model_name"],
                    "offline_fallback": summary["offline_fallback"],
                }
                rows.append(row)
                if best_row is None or float(row["dev_f1"] or 0.0) > float(best_row["dev_f1"] or 0.0):
                    best_row = row

    best_checkpoint_dir = output_dir / "best_checkpoint"
    if best_row is not None and best_row.get("best_checkpoint"):
        source = Path(str(best_row["best_checkpoint"]))
        if source.exists():
            best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, best_checkpoint_dir, dirs_exist_ok=True)
            best_row["best_checkpoint"] = str(best_checkpoint_dir)

    write_hp_search_csv(rows, search_output)
    return {
        "search_output": str(search_output),
        "best": best_row,
        "runs": rows,
    }


def _save_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Phase 4 NER model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run the SKILLSPAN hyperparameter search.")
    train_parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent)
    train_parser.add_argument("--search-output", type=Path, default=Path("results/ner_hp_search.csv"))
    train_parser.add_argument("--base-model", default=DEFAULT_PRIMARY_MODEL)
    train_parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL)
    train_parser.add_argument("--max-train-samples", type=int)
    train_parser.add_argument("--max-eval-samples", type=int)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--no-offline-fallback", action="store_false", dest="allow_offline_fallback")
    train_parser.set_defaults(allow_offline_fallback=True)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on a dataset split.")
    eval_parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    eval_parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    eval_parser.add_argument("--split", default="test")
    eval_parser.add_argument("--output", type=Path, default=Path("results/ner_skillspan.json"))

    gold_parser = subparsers.add_parser("evaluate-gold", help="Evaluate a checkpoint on the annotated gold corpus.")
    gold_parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    gold_parser.add_argument("--gold-jsonl", type=Path, default=DEFAULT_GOLD_JSONL)
    gold_parser.add_argument("--output", type=Path, default=Path("results/ner_gold.json"))

    text_parser = subparsers.add_parser("predict-text", help="Run NER on raw text.")
    text_parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    text_parser.add_argument("--text", required=True)

    args = parser.parse_args()

    if args.command == "train":
        report = run_hyperparameter_search(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            search_output=args.search_output,
            base_model_name=args.base_model,
            fallback_model_name=args.fallback_model,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            seed=args.seed,
            allow_offline_fallback=args.allow_offline_fallback,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "evaluate":
        report = evaluate_ner_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            dataset_dir=args.dataset_dir,
            split=args.split,
        )
        _save_json(report, args.output)
        print(json.dumps({key: value for key, value in report.items() if key != "document_records"}, indent=2, sort_keys=True))
        return

    if args.command == "evaluate-gold":
        report = evaluate_gold_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            gold_jsonl=args.gold_jsonl,
        )
        _save_json(report, args.output)
        print(json.dumps({key: value for key, value in report.items() if key != "document_records"}, indent=2, sort_keys=True))
        return

    if args.command == "predict-text":
        dataset = add_combined_tags(load_skillspan_dataset())
        bundle = resolve_model_bundle(dataset, allow_offline_fallback=True)
        spans = predict_text_spans(bundle, args.text)
        print(
            json.dumps(
                [
                    {
                        "start_token": span.start_token,
                        "end_token": span.end_token,
                        "label": span.label,
                        "text": span.text,
                        "confidence": span.confidence,
                    }
                    for span in spans
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
