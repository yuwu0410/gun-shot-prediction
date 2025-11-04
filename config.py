"""
Project configuration - paths and constants
"""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_CSV_PATH = RAW_DATA_DIR / "Guns_incident_Data.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# Outputs
PREPROCESSOR_PATH = PROCESSED_DATA_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = PROCESSED_DATA_DIR / "feature_names.csv"
DATA_DICT_PATH = PROCESSED_DATA_DIR / "data_dictionary.json"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.parquet"
X_VALID_PATH = PROCESSED_DATA_DIR / "X_valid.parquet"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.parquet"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
Y_VALID_PATH = PROCESSED_DATA_DIR / "y_valid.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

MODEL_PATH = MODELS_DIR / "decision_tree_model.joblib"

# Constants
RANDOM_SEED = 42

