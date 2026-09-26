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
* **Run Test Prediction Only** (using pre-trained model):
  ```bash
  python run_pipeline.py --predict
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
