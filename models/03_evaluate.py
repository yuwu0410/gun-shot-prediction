"""
Model Evaluation Script

This script loads all trained models and evaluates their performance on the
unseen test set.

It performs the following steps for each model:
1.  Loads the model from a .joblib file.
2.  Makes predictions on the test data.
3.  Calculates and prints a detailed classification report (accuracy, precision,
    recall, F1-score).
4.  Generates and saves a confusion matrix visualization.
5.  Generates and saves a feature importance plot (for tree-based models).

Finally, it compiles a summary table comparing the key performance metrics
across all models to identify the best-performing one.

Usage:
    python models/03_evaluate.py
"""
import sys
from pathlib import Path
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Import configuration ---
from config import (
    X_TEST_PATH, Y_TEST_PATH,
    FEATURE_NAMES_PATH, LABEL_ENCODER_PATH,
    PERFORMANCE_DIR,
    DECISION_TREE_MODEL_PATH,
    DECISION_TREE_WEIGHTED_MODEL_PATH,
    SMOTE_DECISION_TREE_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    XGBOOST_MODEL_PATH
)

# --- Define models to evaluate ---
# A dictionary mapping a descriptive name to the model's path
MODELS_TO_EVALUATE = {
    "Decision Tree (Baseline)": DECISION_TREE_MODEL_PATH,
    "Decision Tree (SMOTE)": SMOTE_DECISION_TREE_MODEL_PATH,
    "Decision Tree (Weighted)": DECISION_TREE_WEIGHTED_MODEL_PATH,
    "Random Forest": RANDOM_FOREST_MODEL_PATH,
    "XGBoost": XGBOOST_MODEL_PATH,
}


def load_test_data_and_helpers():
    """
    Loads the test dataset, feature names, and label encoder.
    """
    print("Loading test data and helper files...")
    
    X_test = pd.read_parquet(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    
    # Feature names are needed for importance plots
    feature_names = pd.read_csv(FEATURE_NAMES_PATH).squeeze().tolist()
    
    # Label encoder is crucial for interpreting XGBoost predictions and plots
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Target classes: {label_encoder.classes_}")
    
    return X_test, y_test, feature_names, label_encoder


def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    """
    Generates, displays, and saves a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {model_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the figure
    save_path = PERFORMANCE_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path)
    print(f"  - Confusion matrix saved to: {save_path}")
    plt.close() # Close the plot to free memory


def plot_feature_importance(model, feature_names, model_name):
    """
    Generates, displays, and saves a feature importance plot for tree-based models.
    """
    # Check for feature_importances_ attribute. Handle pipelines correctly.
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps and hasattr(model.named_steps['classifier'], 'feature_importances_'):
        # This handles the SMOTE pipeline
        importances = model.named_steps['classifier'].feature_importances_
    else:
        print(f"  - Feature importance not available for {model_name}.")
        return

    # Create a DataFrame for easier plotting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    # Plot top 20 features
    top_n = 20
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(top_n), palette='viridis')
    plt.title(f'Top {top_n} Feature Importances: {model_name}', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    # Save the figure
    save_path = PERFORMANCE_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path)
    print(f"  - Feature importance plot saved to: {save_path}")
    plt.close()


def evaluate_model(model, model_name, X_test, y_test, feature_names, label_encoder):
    """
    Performs a full evaluation of a single model.
    """
    print("\n" + "="*80)
    print(f"EVALUATING MODEL: {model_name}")
    print("="*80)

    # Make predictions
    y_pred_raw = model.predict(X_test)

    # Decode predictions if necessary (e.g., for XGBoost)
    # Check if predictions are integers (encoded)
    if y_pred_raw.dtype == 'int':
        print("  - Model output is integer-encoded. Decoding labels...")
        y_pred = label_encoder.inverse_transform(y_pred_raw)
    else:
        y_pred = y_pred_raw

    # 1. Classification Report
    print("\nClassification Report (on Test Set):")
    report = classification_report(y_test, y_pred, labels=label_encoder.classes_, zero_division=0)
    print(report)

    # 2. Confusion Matrix
    print("Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_, model_name)

    # 3. Feature Importance
    plot_feature_importance(model, feature_names, model_name)
    
    # 4. Calculate metrics for summary table
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "F1 (Weighted)": f1_weighted,
        "F1 (Macro)": f1_macro
    }


def main():
    """
    Main function to run the evaluation pipeline.
    """
    print("Starting model evaluation pipeline...")
    
    # Ensure performance directory exists
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_test, y_test, feature_names, label_encoder = load_test_data_and_helpers()

    # Store results for final summary
    all_results = []

    # Evaluate each model
    for model_name, model_path in MODELS_TO_EVALUATE.items():
        if not model_path.exists():
            print(f"\n[WARNING] Model file not found, skipping: {model_path}")
            continue
        
        model = joblib.load(model_path)
        results = evaluate_model(model, model_name, X_test, y_test, feature_names, label_encoder)
        all_results.append(results)

    # --- Final Performance Summary ---
    print("\n" + "="*80)
    print("OVERALL MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    if not all_results:
        print("No models were evaluated. Exiting.")
        return
        
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values(by="F1 (Macro)", ascending=False).reset_index(drop=True)
    
    print(summary_df.to_string())
    
    # Save summary to a CSV file
    summary_path = PERFORMANCE_DIR / "model_performance_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary table saved to: {summary_path}")
    
    print("\nEvaluation complete.")
    print(f"All plots and summary have been saved to the '{PERFORMANCE_DIR.name}' directory.")


if __name__ == "__main__":
    main()