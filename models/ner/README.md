# NER Checkpoint

The Phase 4 best checkpoint is intentionally kept out of Git because its model
and optimizer files exceed GitHub's 100 MB per-file limit. The local checkpoint
is stored at `models/ner/best_checkpoint/`.

The selected reduced-search configuration and evaluation metrics are recorded
in `results/ner_hp_search_reduced.csv`, `results/ner_skillspan.json`, and
`results/ner_gold.json`. Regenerate a checkpoint with:

```bash
.venv/bin/python src/extraction/ner_model.py train --help
```
