# Business Entity Resolution — Team Cheethas

This repository contains the complete, reproducible end-to-end entity resolution pipeline for ML Challenge 2026.

---

## 1. System Requirements & Environment Setup

* **Python Version**: Python 3.8+ (tested on Python 3.10 and 3.11)
* **Operating System**: Linux / macOS / Windows

### Installation
From this directory (`code/business_entity_resolution/`), install the pinned dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Directory Structure

```
business_entity_resolution/
├── README.md               # Reproduction guide
├── requirements.txt        # Pinned dependencies
├── run_pipeline.py         # Master entrypoint script
└── src/
    ├── config.py           # Hyperparameters and path configurations
    ├── preprocessing/      # Legal suffix stripper (US, India, France) & address parser
    ├── blocking/           # High-speed inverted index candidate generator
    ├── features/           # RapidFuzz similarity and address feature extractor
    ├── models/             # LightGBM classifier training & batch inference
    ├── postprocessing/     # Precision-thresholding & TSV output formatters
    └── utils/              # Memory-safe TSV streaming & Macro F_0.5 evaluator
```

---

## 3. End-to-End Execution

To train the model and generate both final output files (`matching_results.tsv` and `candidate_pairs.tsv`):

```bash
python run_pipeline.py
```

### Options & Flags

* **Run Training Only**:
  ```bash
  python run_pipeline.py --train --train-sample-size 200000
  ```
* **Final full-data training with held-out validation and threshold tuning**:
  ```bash
  python run_pipeline.py --train --whole-data --val-fraction 0.20
  ```
  This uses all Source 1 entities, trains on 80%, and reserves 20% for
  validation and threshold calibration. It saves the trained model and
  validation metadata; it does not retrain on the held-out entities afterward.
  The full-data run is substantially more memory- and time-intensive than a
  sampled run. Do not add `--tune` unless there is time for extra cross-validation.
* **Run Test Prediction Only** (using pre-trained model):
  ```bash
  python run_pipeline.py --predict
  ```
* **Run Entity-Level Validation and Blocking Diagnostics**:
  ```bash
  python -m src.models.validation_runner --sample-size 100000
  ```
  This randomly samples Source 1 entities, uses the complete Source 2 and Source 3
  target files, reports candidate recall and full match-set coverage, and tunes
  the threshold with the familiar coarse-to-fine sweep and per-threshold scores. It also reports
  per-country metrics, cross-country/missing-target ground-truth counts, and
  name/address signatures for true pairs missed by blocking. It reports a
  candidate-oracle Macro F_0.5 to estimate the ceiling if candidate pairs were
  classified perfectly.
  Training uses the same hard-negative selection and class-weight calculation
  as the main pipeline, then compares a pooled threshold with thresholds
  calibrated independently by country. The main training pipeline saves
  country thresholds in model metadata and uses them only when they improve
  the pooled validation score; otherwise it retains the pooled threshold.
  `--threshold` explicitly overrides calibration during prediction.
  Use
  `--sample-size 0` to use all Source 1 entities. Validation writes a temporary
  model and does not replace the saved prediction model. An optional custom
  threshold sweep can be supplied with `--threshold-range START STOP STEP`.
  Candidate generation currently allows up to 300 candidates per Source 1 entity
  and reserves part of that budget for exact normalized-name, postal-code, and
  compound-address candidates. A bounded address-number channel shares part of
  that existing budget to retrieve records whose street names or postal codes
  differ. Prefix-based retrieval is disabled by default; in the latest
  10,000-entity sample, enabling it reduced F_0.5. Use `--ablations` to compare
  it with the default blocker.
  Add `--ablations` to compare the full blocker against removing each retrieval
  channel, re-enabling prefix retrieval, and 150/300/500-candidate budgets. Each ablation reports candidate
  recall, full-match coverage, candidate volume, and its validation Macro F_0.5
  with a retuned threshold. The 500-candidate setting raised candidate recall
  by about 0.5 percentage points but increased candidate volume by about 65%
  for only +0.00035 Macro F_0.5 on the 10,000-entity sample, so the production
  cap remains 300. For a quicker diagnostic run:
  ```bash
  python -m src.models.validation_runner --sample-size 10000 --ablations
  ```
* **Specify Custom Data / Output Directories**:
  ```bash
  python run_pipeline.py \
      --train-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/train \
      --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test \
      --output-dir ../../output
  ```

---

## 4. Submission Validation

Once inference finishes, validate both output files using the official validator:

```bash
python ../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
