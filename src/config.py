"""
Configuration settings for Business Entity Resolution pipeline.
"""

from pathlib import Path
import os

# Project Roots
BASE_DIR = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = BASE_DIR.parent.parent

# Dataset paths: check default locations
DEFAULT_DATA_DIR = Path("C:/Projects/business-pair/6ab10eb3b23ba_student_resource/student_resource/dataset")
if not DEFAULT_DATA_DIR.exists():
    # Fallback to relative path from workspace root
    DEFAULT_DATA_DIR = SUBMISSION_DIR.parent / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"

TRAIN_DIR = DEFAULT_DATA_DIR / "train"
TEST_DIR = DEFAULT_DATA_DIR / "test"

# Output Paths
OUTPUT_DIR = SUBMISSION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Model Artifacts Directory
ARTIFACTS_DIR = BASE_DIR / "src" / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = ARTIFACTS_DIR / "lightgbm_er_model.txt"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

# Candidate Generation (Blocking) Hyperparameters
MAX_CANDIDATES_PER_S1 = 75  # Increased to 75 for >90% recall coverage
MIN_TOKEN_LEN = 2

# Classifier & Metric Hyperparameters
RANDOM_STATE = 42
N_JOBS = os.cpu_count() or 4
MATCH_THRESHOLD = 0.50  # Default threshold (will be auto-tuned during training)
