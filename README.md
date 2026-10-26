# NLP Skill Pipeline

A research prototype for extracting skill mentions from job advertisements and linking them to concepts in the European Skills, Competences, Qualifications and Occupations (ESCO) classification. The project compares transparent lexicon matching, embedding-based retrieval, and supervised named-entity recognition (NER) followed by ESCO ranking.

The evaluation uses the SkillSpan benchmark and an availability-constrained Irish technology-posting sample. It reports extraction, taxonomy-linking, and end-to-end results separately so that a strong ranker is not mistaken for a strong complete pipeline.

## Features

- Clean and sentence-segment job-posting JSONL, with basic email and phone redaction.
- Extract skill spans using lexicon matching or a trained NER model.
- Rank candidate ESCO concepts and return their labels and URIs.
- Evaluate exact and relaxed span metrics, ESCO ranking metrics, ablations, and audit traces.
- Inspect NER and ESCO predictions in a Streamlit interface.
- Rebuild the final Irish results table from the checked-in evaluation artifacts.

## System requirements

- Python 3.11
- Git and pip
- Internet access for model downloads, benchmark downloads, or Adzuna collection
- CPU is sufficient for the table-reproduction command. NER training and transformer inference can require substantial memory and time.

## Install

Run these commands from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The main evaluation-table command uses checked-in reports and traces. It does not train or run the models. Collection from Adzuna is optional and requires `ADZUNA_APP_ID` and `ADZUNA_APP_KEY`; copy `.env.example` to `.env` and add credentials before using the collector.

## Reproduce the main results table

From the repository root, run:

```bash
python -m src.evaluation.build_irish_final_results
```

This writes `results/final_results_table_irish.csv` using the frozen metric reports, reviewed ESCO mappings, and audit traces in `results/`. It rebuilds the reported table; it does not retrain models or regenerate predictions.

## Run the project

Run the examples from the repository root. Command-line scripts that accept options show them with `--help`.

### Preprocess job postings

Clean and sentence-segment a JSONL input, writing the output outside the repository:

```bash
python -m src.preprocess.clean \
  --input data/raw/cloud_support.jsonl \
  --output /tmp/cloud_support_clean.jsonl
```

The Adzuna collector requires credentials. See its available filters before collecting:

```bash
python -m src.preprocess.collect --help
```

Check the returned locations and country fields before using collected postings as a geography-specific evaluation set.

### Run the ESCO lexicon baseline

```bash
python -m src.extraction.lexicon_baseline \
  --skills-csv data/benchmarks/esco/skills_en.csv \
  --output-index /tmp/esco_lexicon_index.json \
  --query "cloud infrastructure support"
```

### Rank a phrase against ESCO

The project includes the ESCO embedding artifacts. The configured sentence-transformer may be downloaded the first time it is used.

```bash
python -m src.esco.ranker --query "cloud infrastructure support"
```

### Train the NER model

Training uses the local SkillSpan data and downloads the configured pretrained transformer if it is not cached. Training may take substantial time and compute:

```bash
python -m src.extraction.ner_model train \
  --dataset-dir data/benchmarks/skillspan \
  --output-dir models/ner \
  --search-output results/ner_hp_search.csv
```

The trained NER checkpoint is excluded from Git because it is large. A fresh clone needs that checkpoint to run NER inference or the demo; see [models/ner/README.md](models/ner/README.md) for its expected location and regeneration notes.

### Launch the inspection interface

```bash
streamlit run src/app.py
```

The app opens locally in a browser. It requires the NER checkpoint at `models/ner/best_checkpoint/`, the ESCO embedding files under `models/`, and the ESCO CSV under `data/benchmarks/esco/`.

### See evaluation commands

```bash
python -m src.evaluation.build_irish_final_results --help
python -m src.evaluation.run_integration_gate --help
python -m src.evaluation.build_error_taxonomy --help
```

## Data and results

The Irish evaluation comprises 85 main-gold postings. A separate 30-posting pilot was used for calibration; together these make the 115-posting screened sample. The sample is availability-constrained and should not be treated as representative of all Irish vacancies. See the [sampling decision](docs/irish_corpus_sampling_decision.md) and the [final results table](results/final_results_table_irish.csv).

The research paper, its LaTeX source, and referenced figures are in [`paper/conference/`](paper/conference/). The compiled paper is [available as a PDF](paper/conference/NLP_Techniques_for_Information_Extraction_and_Skill_Taxonomy_Construction_from_Unstructured_Job_Advertisement_Data.pdf); the LaTeX source is [`nlp_skill_pipeline_conference_paper_.tex`](paper/conference/nlp_skill_pipeline_conference_paper_.tex).

To rebuild the paper PDF, install a LaTeX distribution with `latexmk` and run:

```bash
cd paper/conference
latexmk -pdf nlp_skill_pipeline_conference_paper_.tex
```

The included `IEEEtran.cls` supplies the IEEE conference document class. The Streamlit interface is intended for local project demonstration; no separate cloud deployment configuration is included.

## Data privacy

Review job-posting data, annotation exports, and repository history before public distribution. The JSONL scrubber targets email addresses and phone numbers; it does not guarantee removal of every personal identifier and does not rewrite Git history.

## License

See [LICENSE](LICENSE).
