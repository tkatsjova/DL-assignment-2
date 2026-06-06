# MEG Brain State Classification (INFOMDLR)

Classify MEG brain recordings into 4 states: `rest`, `math`, `memory`, `motor`.  
Dataset: 248 sensors × ~17.5 s per file, stored as `.h5`. Place it under `Final Project data/` (not tracked by git).

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the project

**EDA** (exploratory plots saved to `outputs/eda/`):
```bash
python main.py
```

**Train all baseline models** (intra + cross-subject):
```bash
python run_experiments.py
python run_experiments.py --cross
```

**Run improvement experiments** (dropout / weight_decay / overlap / window sweep on CNN-GRU-Attn):
```bash
python run_experiments.py --improve
python run_experiments.py --improve --cross
```

**Evaluate saved checkpoints** and generate result plots:
```bash
python evaluate.py          # intra-subject
python evaluate.py --cross  # cross-subject
python evaluate.py --both   # both
```

**Print summary of all existing results:**
```bash
python run_experiments.py --summary
```

Experiments that already have a saved JSON are skipped automatically — safe to re-run after interruption.

---

## Project structure

```
src/
  data/       — loading, preprocessing, windowing, EDA
  models/     — model architectures, training loop
  evaluate/   — evaluation, plots
outputs/      — checkpoints (.pt), result JSONs, plots
run_experiments.py  — experiment runner (baseline + improvement grid)
main.py             — EDA entry point
evaluate.py         — evaluation entry point
```

---

**Deadline:** June 17, 2026 — IEEE paper (4–6 pages) submitted to Brightspace.
