# nlp-skill-pipeline

Development and evaluation of an NLP pipeline for extracting and classifying skill requirements from entry-level technology job postings.

## Phase 1 Status

The repository now contains the Phase 1 scaffold:

- project directory structure
- pinned Python requirements
- benchmark download helpers
- preprocessing utilities
- draft annotation guidelines

## Layout

```text
data/
  raw/
  clean/
  annotated/
  benchmarks/
models/
notebooks/
results/
docs/
src/
tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
set -a && . ./.env && set +a
```

## Phase 1 Scripts

- `python -m src.preprocess.download_skillspan --output data/benchmarks/skillspan`
- `python -m src.preprocess.download_green_et_al --zip-url <url> --output data/benchmarks/green_et_al`
- `python -m src.preprocess.extract_esco_skills --rdf-zip data/benchmarks/esco/esco_v1.2.0_classification_rdf.zip --output-csv data/benchmarks/esco/skills_en.csv`
- `python -m src.preprocess.inspect_benchmarks`

## Phase 2 Entry Point

- `set -a && . ./.env && set +a && .venv/bin/python -m src.preprocess.collect --role-type all --output-dir data/raw --country gb --where "Republic of Ireland"`

## Phase 2 Logging

- `.venv/bin/python -m src.preprocess.corpus_stats --raw-dir data/raw --clean-dir data/clean --output results/corpus_stats.json`

## Phase 3 Entry Point

- `.venv/bin/python -m src.extraction.lexicon_baseline --skills-csv data/benchmarks/esco/skills_en.csv --output-index models/esco_lexicon_index.json --query "root cause analysis" --query "cloud infrastructure support"`

## Phase 3 Status

Phase 3 is implemented through the baseline comparison layer:

- `3.1` ESCO lexicon index built and tested
- `3.2` Baseline 1 evaluated on `SKILLSPAN` test
- `3.3` ESCO embedding index built and tested
- `3.4` Threshold search implemented on `SKILLSPAN` dev
- `3.5` Baseline comparison table generated
- `3.6` Baseline 2 evaluation on `SKILLSPAN` test and gold
- `3.7` Baseline comparison table committed to `results/baseline_comparison.csv`
- `3.8` Unit tests for the baseline modules passing

Current outputs:

- `results/baseline1_skillspan.json`
- `results/baseline1_gold.json`
- `results/b2_threshold_search.json`
- `results/baseline2_skillspan.json`
- `results/baseline2_gold.json`
- `results/baseline_comparison.csv`
- `results/gold_esco_review.csv`
- `results/gold_esco_review_priority.csv`

Note:

- Best Baseline 2 threshold on `SKILLSPAN` dev is `0.60`; on `SKILLSPAN` test it is `0.65` under the current fallback encoder.
- The current gold annotations are span-only, so ESCO `Acc@1`, `Acc@3`, and `MRR` are not computable yet without concept-linked gold labels.
- `results/gold_esco_review_priority.csv` is the review file for filling those concept URIs.
- If the MPNet model is not cached locally, the ranker falls back to a deterministic offline encoder on this machine. Re-run the embedding steps with the real cached model before final dissertation freeze.

## Phase 4 Status

See [docs/phase4_status.md](</Users/tiagoluis/Developer/nlp-skill-pipeline/docs/phase4_status.md>) for the full task list and execution order.

## Label Studio

- `BASE_DATA_DIR=.cache/label-studio LATEST_VERSION_CHECK=false .venv/bin/label-studio --help`
- `BASE_DATA_DIR=.cache/label-studio LATEST_VERSION_CHECK=false .venv/bin/label-studio start`
