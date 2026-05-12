# DL Assignment 2 — MEG Brain State Classification (INFOMDLR)

**Course:** Deep Learning (INFOMDLR) — Dr. Siamak Mehrkanoon  
**Deadline:** June 17, 2026  
**Task:** Classify MEG brain recordings into 4 states: `rest`, `math`, `memory`, `motor`

---

## Dataset

MEG (Magnetoencephalography) data — brain magnetic field recordings via 248 scalp sensors.  
Each file: `248 sensors × 35624 time steps` (~17.5 s at 2034 Hz), stored as `.h5`.

**Structure after download:**
```
data/
├── Intra/
│   ├── train/   # 1 subject
│   └── test/    # same subject
└── Cross/
    ├── train/   # 2 subjects
    ├── test1/   # unseen subject
    ├── test2/   # unseen subject
    └── test3/   # unseen subject
```

File naming: `taskType_subjectID_chunk.h5`  
Task types: `rest`, `task_motor`, `task_story_math`, `task_working_memory`

---

## Work Distribution

| Person | Responsibility |
|--------|---------------|
| **Person 1** | **Data pipeline** — preprocessing (normalization, downsampling), windowing into segments, `DataLoader`, exploratory analysis |
| **Person 2** | **Model** — architecture design (1D CNN / EEGNet), literature review, hyperparameter search |
| **Person 3** | **Intra-subject experiments** — train/test on same subject, accuracy analysis, results writeup |
| **Person 4** | **Cross-subject experiments** — train on subjects A+B, test on unseen subjects, tackle overfitting/domain shift, model improvement |

---

## Pipeline Overview

1. **Preprocess** — Z-score or min-max normalization (time-wise), downsample from 2034 Hz
2. **Window** — slice recordings into fixed-length segments for training
3. **Model** — 1D CNN or EEGNet over 248-channel MEG input
4. **Intra-subject** — train and test on `Intra/train` → `Intra/test` (same subject)
5. **Cross-subject** — train on `Cross/train`, evaluate on `Cross/test1/2/3` (unseen subjects)
6. **Improve** — address overfitting / domain shift, justify approach
7. **Report** — IEEE format, 4–6 pages

---

## Setup

```bash
pip install h5py torch numpy scikit-learn matplotlib
```

Place the downloaded dataset under `data/` (excluded from git via `.gitignore`).

Run the full pipeline:
```bash
python main.py
```

---

## Report

IEEE format, 4–6 pages. Due **June 17, 2026**.  
See [DL_project.pdf](DL_project.pdf) for full assignment specification.
