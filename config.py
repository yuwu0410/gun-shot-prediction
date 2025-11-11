"""
Project configuration - paths and constants
"""
from pathlib import Path

# Define project root
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Processed data files
X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.parquet"
X_VALID_PATH = PROCESSED_DATA_DIR / "X_valid.parquet"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.parquet"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
Y_VALID_PATH = PROCESSED_DATA_DIR / "y_valid.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"
PREPROCESSOR_PATH = PROCESSED_DATA_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = PROCESSED_DATA_DIR / "feature_names.csv"

# Models directory and paths
MODELS_DIR = PROJECT_ROOT / "models"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
DECISION_TREE_MODEL_PATH = MODELS_DIR / "decision_tree_model.joblib"
DECISION_TREE_WEIGHTED_MODEL_PATH = MODELS_DIR / "decision_tree_weighted_model.joblib"
SMOTE_DECISION_TREE_MODEL_PATH = MODELS_DIR / "smote_decision_tree_model.joblib"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"

# Performance/Output directory
PERFORMANCE_DIR = PROJECT_ROOT / "performance"

# Constants
RANDOM_SEED = 42
