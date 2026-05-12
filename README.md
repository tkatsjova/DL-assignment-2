# DL Assignment 2 — MEG Brain State Classification (INFOMDLR)

**Course:** Deep Learning (INFOMDLR) — Dr. Siamak Mehrkanoon  
**Deadline:** June 17, 2026  
**Task:** Classify MEG brain recordings into 4 states: `rest`, `math`, `memory`, `motor`

---

## Project Instructions

The project is done in teams of 4 students. The goal is to apply deep learning methods on a research-oriented problem. For technical/programming questions, you can go to the Computer Lab and talk with your TA (optional).

### Final Deliverable

A **4–6 page double-column IEEE paper** submitted as PDF to Brightspace by **June 17, 2026, 23:59**. A GitHub link to the code should be sent as supplementary material (not graded). Maximum 3 submission attempts — only the last one is graded.

**IEEE LaTeX / Word template:** https://www.ieee.org/conferences/publishing/templates.html

**Report must include:**
1. Title, Authors, Abstract
2. Introduction — problem statement and motivation
3. Related Work — relevant literature
4. Approach — methodology, technical details, and choices made
5. Results — qualitative and quantitative analysis
6. Conclusion — summary of findings
7. References

**Grading criteria:** clarity of the document, technical content, critical analysis, and performance.

### Implementation

Use any suitable library — PyTorch or TensorFlow/Keras recommended.

Useful resources:
- https://github.com/ChristosChristofidis/awesome-deep-learning
- https://paperswithcode.com/

---

## Dataset

MEG (Magnetoencephalography) data — brain magnetic field recordings via 248 scalp sensors.  
Each file: `248 sensors × 35624 time steps` (~17.5 s at 2034 Hz), stored as `.h5`.  
Download the dataset and place it under `data/` (excluded from git).

---

## Work Distribution

| Person | Responsibility |
|--------|---------------|
| **Person 1** | **Data pipeline** — preprocessing (normalization, downsampling), windowing into segments, `DataLoader`, exploratory analysis |
| **Person 2** | **Model** — architecture design (1D CNN / EEGNet), literature review, hyperparameter search |
| **Person 3** | **Intra-subject experiments** — train/test on same subject, accuracy analysis, results writeup |
| **Person 4** | **Cross-subject experiments** — train on subjects A+B, test on unseen subjects, tackle overfitting/domain shift, model improvement |

---

## Workflow

Below is the recommended order of work across the team. Steps 1–2 must be done before any experiments; steps 3–5 can run in parallel once the pipeline is ready.

### Step 1 — Data exploration (Person 1)
- Download and unzip the dataset into `data/`
- Open a few `.h5` files, check shapes (should be `248 × 35624`) and value ranges (~10e-15 T)
- Plot raw signals from several sensors to get intuition about the data
- Check class balance across train files

### Step 2 — Preprocessing & DataLoader (Person 1)
- Apply **Z-score normalisation** per sensor per file (time-wise, not global)
- **Downsample** from 2034 Hz to ~200–500 Hz (e.g. keep every 4th sample) to reduce compute
- **Window** each recording into fixed-length segments (e.g. 256 samples, 50% overlap) to get more training examples
- Build a `torch.utils.data.Dataset` that returns `(segment, label)` pairs
- For Cross data (64 train files) load in chunks of ~8 files per iteration to avoid OOM

### Step 3 — Model architecture (Person 2)
- Start with a **1D CNN** baseline: temporal convolutions over the 248-channel input
  - E.g. 3 conv blocks (Conv1d → BatchNorm → ReLU → MaxPool) + GlobalAvgPool + Linear head
- Alternatively try **EEGNet** (lightweight depthwise separable CNN from the BCI literature)
- Document the choice with references (why this architecture fits MEG/EEG data)
- Set up the training loop: Adam, CrossEntropyLoss, LR scheduler, early stopping

### Step 4 — Intra-subject experiments (Person 3)
- Train on `Intra/train`, evaluate on `Intra/test` (same subject)
- Run for multiple epochs, track train/val loss and accuracy curves
- Report per-class accuracy and confusion matrix
- Tune hyperparameters (kernel size, learning rate, window size, downsample factor)
- Expected result: high accuracy since train and test come from the same person

### Step 5 — Cross-subject experiments (Person 4)
- Train on `Cross/train` (2 subjects), evaluate on `Cross/test1`, `test2`, `test3` (unseen subjects)
- Compare accuracy to Intra-subject — expect a drop due to **domain shift**
- Analyse which classes suffer most (inter-subject variability differs per task)
- Try at least one improvement strategy:
  - Data augmentation (Gaussian noise, channel dropout)
  - Subject-level batch normalisation
  - Domain-adversarial training or feature alignment
  - Fine-tuning on a small amount of target-subject data
- Document the improvement and justify the choice

### Step 6 — Analysis & report (all)
- Summarise results in tables: Intra vs Cross accuracy per class and overall
- Plot learning curves, confusion matrices, and any ablation results
- Each person writes their section (Introduction/Related Work → Person 2, Approach → Person 1+2, Results → Person 3+4, Conclusion → all)
- Proofread for the IEEE format and page limit (4–6 pages)
- Submit PDF to Brightspace + GitHub link before **June 17, 2026, 23:59**

---

## Setup

```bash
pip install h5py torch numpy scikit-learn matplotlib
```

Run the full pipeline:
```bash
python main.py
```

---

## Report

IEEE format, 4–6 pages. Due **June 17, 2026**.  
See [DL_project.pdf](DL_project.pdf) for the full assignment specification.
