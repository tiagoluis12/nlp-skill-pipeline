# NLP Skill Pipeline

A research prototype for extracting skill mentions from job advertisements and linking them to concepts in the European Skills, Competences, Qualifications and Occupations (ESCO) classification. The project compares lexicon matching, embedding-based retrieval, and named-entity recognition (NER) followed by ESCO ranking. The conference paper and its LaTeX source are in [`paper/conference/`](paper/conference/).

## Features

- Clean and sentence-segment JSONL supplied by the user.
- Extract skill spans with a lexicon baseline or a trained NER model.
- Rank skill phrases against ESCO concepts.
- Evaluate span extraction, ESCO linking, ablations, and aggregate end-to-end results.
- Inspect local predictions in a Streamlit interface.
- Rebuild the checked-in aggregate Irish results table from frozen metric reports.

## Data and licensing

The Irish job-advertisement corpus, annotations, per-posting predictions, review queues, and audit traces are not distributed in this repository. Written permission to redistribute the source postings has not been established. The paper reports aggregate results for the study sample; this repository does not contain the underlying Irish records needed to rerun those experiments. Use only data that you are authorized to collect and process.

The separate SkillSpan benchmark is included under CC BY 4.0. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and the license link. ESCO files are included for taxonomy lookup; see the same notice for their source and attribution. Third-party dataset terms apply independently of this repository's software license.

## Requirements and installation

- Python 3.11
- Git and pip
- Internet access when downloading model dependencies
- CPU is sufficient to rebuild the aggregate results table. Model training and transformer inference may require substantial memory and time.

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Rebuild the main results table

Run:

```bash
python -m src.evaluation.build_irish_final_results
```

This rebuilds `results/final_results_table_irish.csv` from the checked-in aggregate extraction reports, ESCO linking metrics, and frozen posting-level coverage counts. It does not train models or regenerate predictions, and the source Irish postings and record-level outputs are not included.

The conference paper's aggregate tables and figures are in `results/` and `paper/conference/`. To rebuild the paper PDF, install a LaTeX distribution with `latexmk`, then run:

```bash
cd paper/conference
latexmk -pdf nlp_skill_pipeline_conference_paper_.tex
```

## Run the project

Run commands from the repository root. Command-line scripts show their options with `--help`.

### Preprocess an authorized JSONL input

```bash
python -m src.preprocess.clean \
  --input /path/to/authorized_posts.jsonl \
  --output /tmp/clean_posts.jsonl
```

The Adzuna collector is optional and requires API credentials. Use it only in accordance with the API terms and applicable permissions:

```bash
python -m src.preprocess.collect --help
```

### Query the ESCO lexicon

```bash
python -m src.extraction.lexicon_baseline \
  --skills-csv data/benchmarks/esco/skills_en.csv \
  --output-index /tmp/esco_lexicon_index.json \
  --query "cloud infrastructure support"
```

### Rank a phrase against ESCO

```bash
python -m src.esco.ranker --query "cloud infrastructure support"
```

### Train the NER model on SkillSpan

```bash
python -m src.extraction.ner_model train \
  --dataset-dir data/benchmarks/skillspan \
  --output-dir models/ner \
  --search-output /tmp/ner_hp_search.csv \
  --seed 42
```

Model checkpoints are not included. Training downloads the configured pretrained transformer if it is not cached. See [`models/ner/README.md`](models/ner/README.md) for checkpoint details.

### Launch the inspection interface

```bash
streamlit run src/app.py
```

The interface requires a trained NER checkpoint and the ESCO embedding files included under `models/`.

### Inspect evaluation commands

```bash
python -m src.evaluation.build_irish_final_results --help
python -m src.evaluation.run_integration_gate --help
python -m src.evaluation.build_error_taxonomy --help
```

## Citation and license

Use [`CITATION.cff`](CITATION.cff) to cite this project. The project software is licensed under the MIT License; see [`LICENSE`](LICENSE). Dataset and taxonomy licenses are described in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
