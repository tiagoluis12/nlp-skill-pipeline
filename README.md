# nlp-skill-pipeline

Development and evaluation of an NLP pipeline for extracting and classifying skill requirements from entry-level technology job postings.

## Current Status

Phases 1-5 are implemented, with the final audit gates documented in
`docs/phase5_status.md`. Phase 6 is not yet release-ready: the corpus geography,
ESCO gold-label adjudication, privacy scrub, and model-artifact release still
require completion.

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

## Phase 2 Collection Entry Point

- Intended Ireland collection command: `set -a && . ./.env && set +a && .venv/bin/python -m src.preprocess.collect --role-type all --output-dir data/raw --country gb --where "Republic of Ireland"`
- Verify the resulting records contain the intended country/location before annotation. The committed corpus currently contains US records and must not be described as Irish-market data.

## Phase 2 Logging

- `.venv/bin/python -m src.preprocess.corpus_stats --raw-dir data/raw --clean-dir data/clean --output results/corpus_stats.json`
- Before committing collected records, scrub source JSONL with `.venv/bin/python -m src.preprocess.scrub_pii --input-dir data/raw --output-dir /tmp/nlp-skill-pipeline-raw-scrubbed` and replace only after verifying the output.

## Phase 3 Entry Point

- `.venv/bin/python -m src.extraction.lexicon_baseline --skills-csv data/benchmarks/esco/skills_en.csv --output-index models/esco_lexicon_index.json --query "root cause analysis" --query "cloud infrastructure support"`

## Phase 3-5 Outputs

The baseline, proposed pipeline, ablation, taxonomy, and comparison artefacts are committed:

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

The final reported Phase 5 metrics use the project-local MPNet snapshot. Do not
use the deterministic hashing fallback for final evaluation.

## Phase 4 Status

See [docs/phase4_status.md](</Users/tiagoluis/Developer/nlp-skill-pipeline/docs/phase4_status.md>) for the full task list and execution order.

## Label Studio

Label Studio should run in a separate environment because its current dependency
requirements conflict with the pinned ML environment:

```bash
python -m venv .labelstudio-venv
source .labelstudio-venv/bin/activate
pip install -r requirements-label-studio.txt
label-studio start
```

The main `.venv` is for model training, evaluation, and tests.
