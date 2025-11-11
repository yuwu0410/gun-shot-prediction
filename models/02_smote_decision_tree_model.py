"""
Decision Tree Model Training Script with SMOTE and Grid Search

This script loads preprocessed data, uses SMOTE for oversampling the minority class,
and trains a decision tree classifier to predict gun incident reasons.
It uses GridSearchCV for hyperparameter optimization.

The script performs:
- SMOTE applied correctly within each cross-validation fold using a Pipeline.
- 5-fold cross-validation over 72 hyperparameter combinations.
- Optimizes for F1-weighted score (better for imbalanced data).
- Tests max_depth, min_samples_split, min_samples_leaf, and criterion.

Usage:
    python models/02_train_smote_decision_tree.py
    
Expected runtime: 5-15 minutes (slightly longer than without SMOTE)
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# --- Key Additions for SMOTE ---
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
# -------------------------------

from config import (
    X_TRAIN_PATH, X_VALID_PATH, 
    Y_TRAIN_PATH, Y_VALID_PATH,
    MODELS_DIR,
    RANDOM_SEED
)

# Define a specific path for the SMOTE + Decision Tree model
SMOTE_DT_MODEL_PATH = MODELS_DIR / "smote_decision_tree_model.joblib"


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


def train_smote_decision_tree(X_train, y_train, X_valid, y_valid):
    """
    Train a decision tree classifier with SMOTE using GridSearchCV for hyperparameter optimization.
    
    Uses a scikit-learn Pipeline to ensure SMOTE is only applied to the training
    data within each fold of the cross-validation process, preventing data leakage.
    
    Hyperparameters tuned for the Decision Tree:
    - max_depth: Tree depth to control complexity
    - min_samples_split: Minimum samples required to split a node
    - min_samples_leaf: Minimum samples required at a leaf node
    - criterion: Split quality measure (gini vs entropy)
    """
    print("\n" + "="*60)
    print("Grid Search for Optimal Hyperparameters (SMOTE + Decision Tree)")
    print("="*60)
    
    # --- Define the Pipeline ---
    # This pipeline chains SMOTE and the Decision Tree classifier.
    # This is the correct way to use oversampling with cross-validation.
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=RANDOM_SEED)),
        ('classifier', DecisionTreeClassifier(random_state=RANDOM_SEED)) # Note: class_weight is removed
    ])
    
    # --- Define parameter grid for the classifier within the pipeline ---
    # Note the 'classifier__' prefix to target the Decision Tree step.
    param_grid = {
        'classifier__max_depth': [15, 20, 25, 30],
        'classifier__min_samples_split': [5, 10, 20],
        'classifier__min_samples_leaf': [2, 5, 10],
        'classifier__criterion': ['gini', 'entropy']
    }
    
    print(f"\nSearching over {4*3*3*2} combinations...")
    print(f"Parameter grid (note the 'classifier__' prefix for pipeline):")
    for param, values in param_grid.items():
        print(f"  - {param}: {values}")
    
    # Grid search with cross-validation on the entire pipeline
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,                      # 5-fold cross-validation
        scoring='f1_weighted',     # Better metric for imbalanced data
        n_jobs=-1,                 # Use all CPU cores
        verbose=1,                 # Show progress
        return_train_score=True
    )
    
    # Execute grid search
    print("\nStarting grid search (this may take 5-15 minutes)...")
    grid_search.fit(X_train, y_train)
    
    # Get best model (which is the entire best pipeline)
    best_model = grid_search.best_estimator_
    
    # Display results
    print("\n" + "="*60)
    print("Grid Search Results")
    print("="*60)
    print(f"\nBest Parameters:")
    # Clean up parameter names for display
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
        # Clean up param names for display
        params_display = {k.replace('classifier__', ''): v for k, v in row['params'].items()}
        print(f"  {idx+1}. F1={row['mean_test_score']:.4f} (±{row['std_test_score']:.4f}) | {params_display}")
    
    # Evaluate on training set
    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Evaluate on validation set
    y_valid_pred = best_model.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    
    # Access the trained classifier from the pipeline to get its properties
    best_tree = best_model.named_steps['classifier']
    
    print("\n" + "="*60)
    print("Best Model Performance")
    print("="*60)
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Valid Accuracy: {valid_accuracy:.4f}")
    print(f"  Tree Depth: {best_tree.get_depth()}")
    print(f"  Number of Leaves: {best_tree.get_n_leaves()}")
    print("="*60 + "\n")
    
    # Show classification report for validation set
    print("Validation Set Classification Report:")
    print(classification_report(y_valid, y_valid_pred, zero_division=0))
    
    return best_model


def save_model(model):
    """Save trained pipeline to disk."""
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the entire pipeline object
    joblib.dump(model, SMOTE_DT_MODEL_PATH)
    print(f"\nModel saved to: {SMOTE_DT_MODEL_PATH}")


def main():
    """Main training pipeline with SMOTE and grid search optimization."""
    print("="*60)
    print("SMOTE + Decision Tree Training Pipeline with Grid Search")
    print("="*60)
    
    # Load data
    X_train, X_valid, y_train, y_valid = load_data()
    
    # Train model with grid search
    model = train_smote_decision_tree(X_train, y_train, X_valid, y_valid)
    
    # Save model
    save_model(model)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review the best parameters and cross-validation scores above")
    print("  2. Compare these results with the original Decision Tree model")
    print("  3. Run evaluation script: python models/03_evaluate.py")
    print("  4. Teammate: Complete detailed evaluation and analysis")


if __name__ == "__main__":
    main()