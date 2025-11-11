"""
Decision Tree Model Training with Class Weighting

This script trains a Decision Tree classifier for predicting gun incident reasons.
Instead of using SMOTE for oversampling, this approach utilizes the built-in
'class_weight' parameter of the classifier to handle class imbalance at the
algorithmic level.

The script performs:
- Grid search over a set of Decision Tree hyperparameters.
- The 'class_weight' parameter is set to 'balanced' to penalize misclassifications
  of minority classes more heavily.
- 5-fold cross-validation is used to find the optimal model configuration.
- Optimizes for F1-weighted score, which is suitable for imbalanced datasets.

Usage:
    python models/03_train_decision_tree_weighted.py
    
Expected runtime: 1-5 minutes.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# --- Import the Decision Tree Classifier ---
from sklearn.tree import DecisionTreeClassifier
# -----------------------------------------

from config import (
    X_TRAIN_PATH, X_VALID_PATH, 
    Y_TRAIN_PATH, Y_VALID_PATH,
    MODELS_DIR,
    RANDOM_SEED
)

# Define a specific path for the weighted Decision Tree model
DECISION_TREE_WEIGHTED_MODEL_PATH = MODELS_DIR / "decision_tree_weighted_model.joblib"


def load_data():
    """
    Load preprocessed training and validation data.
    DecisionTreeClassifier handles string labels, so no encoding is needed.
    """
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


def train_decision_tree_weighted(X_train, y_train, X_valid, y_valid):
    """
    Train a Decision Tree classifier using 'class_weight' and GridSearchCV for
    hyperparameter optimization.
    
    Hyperparameters tuned:
    - criterion: The function to measure the quality of a split.
    - max_depth: The maximum depth of the tree.
    - min_samples_split: The minimum number of samples required to split an internal node.
    - min_samples_leaf: The minimum number of samples required to be at a leaf node.
    """
    print("\n" + "="*60)
    print("Grid Search for Optimal Hyperparameters (Decision Tree with Class Weight)")
    print("="*60)
    
    # --- Define the classifier with class_weight='balanced' ---
    # This is the core of this approach.
    dt_classifier = DecisionTreeClassifier(
        class_weight='balanced',
        random_state=RANDOM_SEED
    )
    
    # --- Define the parameter grid for the Decision Tree ---
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    num_combinations = 2 * 5 * 3 * 3
    print(f"\nSearching over {num_combinations} combinations...")
    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  - {param}: {values}")
    
    # --- Set up and run GridSearchCV ---
    # No pipeline is needed since we are not using SMOTE.
    grid_search = GridSearchCV(
        estimator=dt_classifier,
        param_grid=param_grid,
        cv=5,                      # 5-fold cross-validation
        scoring='f1_weighted',     # Good metric for imbalanced data
        n_jobs=-1,                 # Use all CPU cores
        verbose=2,                 # Show detailed progress
        return_train_score=True
    )
    
    # Execute grid search
    print("\nStarting grid search (this should be relatively fast)...")
    grid_search.fit(X_train, y_train)
    
    # Get the best model found by grid search
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
    """Save the trained model to disk."""
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the best estimator found by GridSearchCV
    joblib.dump(model, DECISION_TREE_WEIGHTED_MODEL_PATH)
    print(f"\nModel saved to: {DECISION_TREE_WEIGHTED_MODEL_PATH}")


def main():
    """Main training pipeline for Decision Tree with class weighting."""
    print("="*60)
    print("Decision Tree Training Pipeline with Class Weighting")
    print("="*60)
    
    # Load data
    X_train, X_valid, y_train, y_valid = load_data()
    
    # Train model with grid search
    model = train_decision_tree_weighted(X_train, y_train, X_valid, y_valid)
    
    # Save model
    save_model(model)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Compare the performance of this weighted model with the SMOTE-based models.")
    print("     - Pay attention to the F1-scores for minority classes in the classification report.")
    print("  2. Run evaluation script: python models/03_evaluate.py (you might need to adapt it for this new model).")


if __name__ == "__main__":
    main()