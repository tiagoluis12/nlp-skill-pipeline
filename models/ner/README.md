# NER Checkpoint

The Phase 4 best checkpoint is intentionally kept out of Git because its model
and optimizer files exceed GitHub's 100 MB per-file limit. The local checkpoint
is stored at `models/ner/best_checkpoint/`.

The selected reduced-search configuration and evaluation metrics are recorded
in `results/ner_hp_search_reduced.csv`, `results/ner_skillspan.json`, and
`results/ner_gold.json`. A fresh clone can regenerate a checkpoint by running
the repository's eight-configuration search (2 learning rates × 2 batch sizes
× 2 epoch counts):

```bash
python -m src.extraction.ner_model train \
  --dataset-dir data/benchmarks/skillspan \
  --output-dir models/ner \
  --search-output results/ner_hp_search.csv \
  --base-model jjzha/jobbert-base-cased \
  --fallback-model bert-base-uncased \
  --seed 42 \
  --no-offline-fallback
```

The search downloads JobBERT if it is not cached and selects the best run on
the SkillSpan development split. It stores the selected checkpoint at
`models/ner/best_checkpoint/`. Training is compute-intensive, and the result
may vary with software and hardware versions. The paper's frozen metrics are
available in `results/`; reproducing those tables does not require retraining.

To run the local inspection app after training:

```bash
streamlit run src/app.py
```

The checkpoint is intentionally excluded from Git because the local model and
optimizer files are large. A separate release asset can be added if direct
download is preferred over retraining.
