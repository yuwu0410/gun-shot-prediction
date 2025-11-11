"""
Random Forest Model Training Script with Grid Search

This script loads preprocessed data and trains a Random Forest classifier
to predict gun incident reasons using GridSearchCV for hyperparameter optimization.

The script performs:
- 5-fold cross-validation over a grid of hyperparameter combinations
- Optimizes for F1-weighted score (better for imbalanced data)
- Tests n_estimators, max_depth, min_samples_split, min_samples_leaf, and criterion
- Uses class_weight='balanced' to handle 42:1 class imbalance

Usage:
    python models/02_train_random_forest.py
    
Expected runtime: Varies based on grid size, can be longer than Decision Tree
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from config import (
    X_TRAIN_PATH, X_VALID_PATH, 
    Y_TRAIN_PATH, Y_VALID_PATH,
    MODELS_DIR,
    RANDOM_SEED
)

# Define a specific path for the Random Forest model to avoid overwriting other models
RF_MODEL_PATH = MODELS_DIR / "model_a.joblib"


def load_data():
    """Load preprocessed training and validation data."""
    print("Loading preprocessed data...")
    
    # Load features
    X_train = pd.read_parquet(X_TRAIN_PATH)
    X_valid = pd.read_parquet(X_VALID_PATH)
    
    # Load targets
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()  # Convert to Series
    y_valid = pd.read_csv(Y_VALID_PATH).squeeze()
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_valid shape: {X_valid.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_valid shape: {y_valid.shape}")
    print(f"  Target classes: {sorted(y_train.unique())}")
    
    return X_train, X_valid, y_train, y_valid


def train_random_forest(X_train, y_train, X_valid, y_valid):
    """
    Train a Random Forest classifier using GridSearchCV for hyperparameter optimization.
    
    Uses 5-fold cross-validation to find the best hyperparameters for:
    - n_estimators: The number of trees in the forest
    - max_depth: Tree depth to control complexity
    - min_samples_split: Minimum samples required to split a node
    - min_samples_leaf: Minimum samples required at a leaf node
    - criterion: Split quality measure (gini vs entropy)
    
    Fixed parameters:
    - class_weight='balanced': Handle class imbalance (42:1 ratio)
    - random_state: For reproducibility
    """
    print("\n" + "="*60)
    print("Grid Search for Optimal Hyperparameters (Random Forest)")
    print("="*60)
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [15, 25, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 5],
        'criterion': ['gini', 'entropy']
    }
    
    num_combinations = (
        len(param_grid['n_estimators']) * 
        len(param_grid['max_depth']) * 
        len(param_grid['min_samples_split']) * 
        len(param_grid['min_samples_leaf']) * 
        len(param_grid['criterion'])
    )
    
    print(f"\nSearching over {num_combinations} combinations...")
    print(f"Parameter grid:")
    for param, values in param_grid.items():
        print(f"  - {param}: {values}")
    
    # Base estimator with fixed parameters
    base_model = RandomForestClassifier(
        random_state=RANDOM_SEED,
        class_weight='balanced'  # Essential for handling 42:1 class imbalance
    )
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,                      # 5-fold cross-validation
        scoring='f1_weighted',     # Better metric for imbalanced data
        n_jobs=-1,                 # Use all CPU cores
        verbose=1,                 # Show progress
        return_train_score=True
    )
    
    # Execute grid search
    print("\nStarting grid search (this may take a while)...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Display results
    print("\n" + "="*60)
    print("Grid Search Results")
    print("="*60)
    print(f"\nBest Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")
    
    print(f"\nBest Cross-Validation F1 Score (weighted): {grid_search.best_score_:.4f}")
    
    # Show top 5 configurations
    print("\nTop 5 Configurations:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score']
    ]
    for idx, row in top_5.iterrows():
        print(f"  {idx+1}. F1={row['mean_test_score']:.4f} (±{row['std_test_score']:.4f}) | {row['params']}")
    
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
    print("Validation Set Classification Report:")
    print(classification_report(y_valid, y_valid_pred, zero_division=0))
    
    return best_model


def save_model(model):
    """Save trained model to disk."""
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model to its specific path
    joblib.dump(model, RF_MODEL_PATH)
    print(f"\nModel saved to: {RF_MODEL_PATH}")


def main():
    """Main training pipeline with grid search optimization for Random Forest."""
    print("="*60)
    print("Random Forest Training Pipeline with Grid Search")
    print("="*60)
    
    # Load data
    X_train, X_valid, y_train, y_valid = load_data()
    
    # Train model with grid search
    model = train_random_forest(X_train, y_train, X_valid, y_valid)
    
    # Save model
    save_model(model)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review the best parameters and cross-validation scores above")
    print("  2. Compare Random Forest performance with the Decision Tree model")
    print("  3. Run evaluation script: python models/03_evaluate.py")
    print("  4. Teammate: Complete detailed evaluation and analysis")


if __name__ == "__main__":
    main()