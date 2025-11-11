"""
XGBoost Model Training Script with SMOTE and Grid Search (Corrected)

This script loads preprocessed data, performs label encoding on the target variable,
uses SMOTE for oversampling, and trains an XGBoost classifier to predict gun
incident reasons. It employs GridSearchCV for hyperparameter optimization.

Key Corrections:
- Added LabelEncoder to convert string targets ('accidental', 'homicide', etc.)
  into integer labels (0, 1, 2, ...) as required by XGBoost.
- The fitted LabelEncoder is saved for later use in evaluation/prediction.

The script performs:
- SMOTE applied correctly within each cross-validation fold using a Pipeline.
- 5-fold cross-validation over a grid of key XGBoost hyperparameters.
- Optimizes for F1-weighted score.
- Tunes n_estimators, max_depth, learning_rate, subsample, and colsample_bytree.

Usage:
    python models/02_train_xgboost.py
    
Expected runtime: 20-60+ minutes, depending on hardware.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# --- Key Additions for Label Encoding, SMOTE, and XGBoost ---
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
# -----------------------------------------------------------

from config import (
    X_TRAIN_PATH, X_VALID_PATH, 
    Y_TRAIN_PATH, Y_VALID_PATH,
    MODELS_DIR,
    RANDOM_SEED
)

# Define specific paths for the XGBoost model and the label encoder
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"


def load_data():
    """
    Load preprocessed training and validation data, and perform label encoding
    on the target variable.
    """
    print("Loading preprocessed data...")
    
    # Load features
    X_train = pd.read_parquet(X_TRAIN_PATH)
    X_valid = pd.read_parquet(X_VALID_PATH)
    
    # Load targets (as strings)
    y_train_str = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_valid_str = pd.read_csv(Y_VALID_PATH).squeeze()
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_valid shape: {X_valid.shape}")
    print(f"  y_train shape: {y_train_str.shape}")
    print(f"  y_valid shape: {y_valid_str.shape}")
    
    # --- Label Encoding for XGBoost ---
    print("\nPerforming Label Encoding for target variable 'y'...")
    encoder = LabelEncoder()
    
    # Fit on training data and transform both training and validation data
    y_train = encoder.fit_transform(y_train_str)
    y_valid = encoder.transform(y_valid_str)
    
    # Print the mapping for clarity
    print("  Label mapping created:")
    for i, class_name in enumerate(encoder.classes_):
        print(f"    '{class_name}' -> {i}")
        
    # Save the fitted encoder for later use (e.g., in the evaluation script)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, LABEL_ENCODER_PATH)
    print(f"  LabelEncoder saved to: {LABEL_ENCODER_PATH}")
    # ------------------------------------
    
    return X_train, X_valid, y_train, y_valid


def train_xgboost(X_train, y_train, X_valid, y_valid):
    """
    Train an XGBoost classifier with SMOTE using GridSearchCV for hyperparameter optimization.
    
    Assumes y_train and y_valid are already label-encoded integers.
    """
    print("\n" + "="*60)
    print("Grid Search for Optimal Hyperparameters (SMOTE + XGBoost)")
    print("="*60)
    
    # --- Define the Pipeline ---
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=RANDOM_SEED)),
        ('classifier', XGBClassifier(
            random_state=RANDOM_SEED,
            use_label_encoder=False,  # Correctly set to False as we pre-encoded labels
            eval_metric='mlogloss'    # Specify metric for multi-class classification
        ))
    ])
    
    # --- Define a focused parameter grid for XGBoost ---
    param_grid = {
        'classifier__n_estimators': [100, 150],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }
    
    num_combinations = 2 * 3 * 2 * 2 * 2
    print(f"\nSearching over {num_combinations} combinations...")
    print(f"Parameter grid (note the 'classifier__' prefix for pipeline):")
    for param, values in param_grid.items():
        print(f"  - {param}: {values}")
    
    # Grid search with cross-validation on the entire pipeline
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    # Execute grid search
    print("\nStarting grid search (this may take 20-60+ minutes)...")
    grid_search.fit(X_train, y_train) # Now y_train is correctly encoded
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Display results
    print("\n" + "="*60)
    print("Grid Search Results")
    print("="*60)
    print(f"\nBest Parameters:")
    best_params_display = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
    for param, value in best_params_display.items():
        print(f"  - {param}: {value}")
    
    print(f"\nBest Cross-Validation F1 Score (weighted): {grid_search.best_score_:.4f}")
    
    # Show top 5 configurations
    print("\nTop 5 Configurations:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score']
    ]
    for idx, row in top_5.iterrows():
        params_display = {k.replace('classifier__', ''): v for k, v in row['params'].items()}
        print(f"  {idx+1}. F1={row['mean_test_score']:.4f} (±{row['std_test_score']:.4f}) | {params_display}")
    
    # Evaluate on training set
    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Evaluate on validation set
    y_valid_pred = best_model.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    
    print("\n" + "="*60)
    print("Best Model Performance")
    print("="*60)
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Valid Accuracy: {valid_accuracy:.4f}")
    print("="*60 + "\n")
    
    # Show classification report for validation set
    # We need the original string labels for the report
    encoder = joblib.load(LABEL_ENCODER_PATH)
    y_valid_pred_str = encoder.inverse_transform(y_valid_pred)
    y_valid_str = encoder.inverse_transform(y_valid) # Re-create for consistency
    
    print("Validation Set Classification Report:")
    print(classification_report(y_valid_str, y_valid_pred_str, zero_division=0))
    
    return best_model


def save_model(model):
    """Save trained pipeline to disk."""
    joblib.dump(model, XGBOOST_MODEL_PATH)
    print(f"\nModel saved to: {XGBOOST_MODEL_PATH}")


def main():
    """Main training pipeline with Label Encoding, SMOTE, XGBoost, and grid search."""
    print("="*60)
    print("SMOTE + XGBoost Training Pipeline with Grid Search (Corrected)")
    print("="*60)
    
    # Load data (this now includes label encoding)
    X_train, X_valid, y_train, y_valid = load_data()
    
    # Train model with grid search
    model = train_xgboost(X_train, y_train, X_valid, y_valid)
    
    # Save model
    save_model(model)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review the XGBoost performance and compare it with previous models.")
    print("  2. The evaluation script (03_evaluate.py) will need to be updated to load the LabelEncoder.")
    print("  3. Teammate: Complete detailed evaluation and analysis.")


if __name__ == "__main__":
    main()