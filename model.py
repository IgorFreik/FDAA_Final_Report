"""
Caffeine Prediction Model

This module analyzes sleep and heart rate data to predict caffeine consumption.
It uses machine learning to identify features most predictive of caffeine intake
and evaluates model performance using cross-validation.
"""

import os
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Configuration constants
TARGET_VAR: str = 'is_coffee'
HR_PATH: str = 'raw_data/bpm_data.csv'
SLEEP_PATH: str = 'raw_data/sleep_data.csv'
RANDOM_SEED: int = 42
PLOTS_DIR: str = 'plots'


def load_data(hr_path: str, sleep_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load heart rate and sleep data from CSV files.

    Args:
        hr_path: Path to the heart rate data CSV file
        sleep_path: Path to the sleep data CSV file

    Returns:
        Tuple containing heart rate DataFrame and sleep DataFrame
    """
    df_bpm = pd.read_csv(hr_path)
    df_sleep = pd.read_csv(sleep_path)
    return df_bpm, df_sleep


def preprocess_data(df_bpm: pd.DataFrame, df_sleep: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess and merge heart rate and sleep data.

    Args:
        df_bpm: DataFrame containing heart rate measurements
        df_sleep: DataFrame containing sleep metrics

    Returns:
        Preprocessed and merged DataFrame with feature engineering
    """
    # Rename columns to avoid confusion
    df_sleep = df_sleep.rename(columns={'avg_hr': 'sleep_avg_hr'})

    # Aggregate BPM data
    df_bpm_grouped = df_bpm.groupby(
        ['Uid', 'Date', 'is_coffee']
    ).agg(
        {'bpm': 'mean', 'bpm_delta': 'mean'}
    ).reset_index()

    # Merge datasets
    df = df_sleep.merge(df_bpm_grouped, on=['Uid', 'Date', 'is_coffee'], how='inner')

    # Feature engineering
    df['deep_ratio'] = df['sleep_deep_duration'] / df['sleep_duration']
    df['rem_ratio'] = df['sleep_rem_duration'] / df['sleep_duration']

    # Convert bedtime to seconds since midnight
    df['bedtime'] = pd.to_datetime(df['bedtime'])
    df['bedtime'] = (df['bedtime'].dt.hour * 3600 +
                     df['bedtime'].dt.minute * 60 +
                     df['bedtime'].dt.second)

    return df


def prepare_features_and_target(
        df: pd.DataFrame,
        target_var: str,
        feature_list: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target variable from the DataFrame.

    Args:
        df: Preprocessed DataFrame
        target_var: Name of the target variable column
        feature_list: List of feature columns to use (optional)

    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the target Series
    """
    # Default features if none provided
    if feature_list is None:
        feature_list = [
            'deep_ratio',
            'rem_ratio',
            'bedtime',
            'sleep_duration',
            'sleep_deep_duration',
            'sleep_rem_duration',
            'awake_count',
            'bpm',
            'bpm_delta',
            'sleep_avg_hr',
        ]

    # Extract target and features
    y = df[target_var]
    X = df[feature_list]

    return X, y


def train_and_evaluate_model(
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'decision_tree',
        cv_splits: int = 3,
        random_state: int = 42
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Train a model and evaluate its performance using cross-validation.

    Args:
        X: Feature DataFrame
        y: Target Series
        model_type: Type of model to use ('decision_tree' or 'random_forest')
        cv_splits: Number of cross-validation splits
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing (trained model, cross-validation scores, predictions)
    """
    # Select model type
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=random_state)
    else:  # Default to decision tree
        model = DecisionTreeClassifier(max_depth=3, random_state=random_state)

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # Cross-validated accuracy scores
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # Cross-validated predictions (for confusion matrix & classification report)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    # Fit final model for feature importance
    model.fit(X, y)

    return model, cv_scores, y_pred


def print_model_evaluation(
        y_true: pd.Series,
        y_pred: np.ndarray,
        cv_scores: np.ndarray,
        target_names: List[str]
) -> None:
    """
    Print evaluation metrics for the model.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        cv_scores: Cross-validation accuracy scores
        target_names: Names of the target classes
    """
    print(f"Cross-Validated Accuracy Scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.3f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))


def visualize_feature_importance(
        model: Any,
        feature_names: List[str],
        save_path: Optional[str] = None
) -> None:
    """
    Visualize and save feature importance plot.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of the features
        save_path: Path to save the plot (optional)
    """
    # Extract feature importances
    feature_importances = model.feature_importances_

    # Sort features by importance (optional)
    sorted_idx = feature_importances.argsort()
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = feature_importances[sorted_idx]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Caffeine Prediction')
    plt.tight_layout()

    # Save the plot if path provided
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path)

    # Show plot
    plt.show()

    # Print feature importances in the console
    print("\n📈 Feature Importances:")
    for name, importance in zip(feature_names, feature_importances):
        print(f"- {name}: {importance:.3f}")


def visualize_confusion_matrix(
        y_true: pd.Series,
        y_pred: np.ndarray,
        target_names: List[str],
        save_path: Optional[str] = None
) -> None:
    """
    Visualize and save confusion matrix plot.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        target_names: Names of the target classes
        save_path: Path to save the plot (optional)
    """
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names
    )

    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Caffeine Prediction')
    plt.tight_layout()

    # Save the plot if path provided
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path)

    # Show plot
    plt.show()


def main() -> None:
    """
    Main function to run the entire data processing and modeling pipeline.
    """
    # Load data
    df_bpm, df_sleep = load_data(HR_PATH, SLEEP_PATH)

    # Preprocess data
    df = preprocess_data(df_bpm, df_sleep)

    # Prepare features and target
    X, y = prepare_features_and_target(df, TARGET_VAR)

    # Train and evaluate model
    model, cv_scores, y_pred = train_and_evaluate_model(
        X, y, model_type='decision_tree', cv_splits=3, random_state=RANDOM_SEED
    )

    # Print evaluation metrics
    print_model_evaluation(y, y_pred, cv_scores, target_names=["No Caffeine", "Caffeine"])

    # Ensure plots directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Visualize feature importance
    feature_imp_path = os.path.join(PLOTS_DIR, 'feature_importance_plot.png')
    visualize_feature_importance(model, list(X.columns), feature_imp_path)

    # Visualize confusion matrix
    conf_matrix_path = os.path.join(PLOTS_DIR, 'confusion_matrix_plot.png')
    visualize_confusion_matrix(y, y_pred, target_names=["No Caffeine", "Caffeine"], save_path=conf_matrix_path)


if __name__ == '__main__':
    main()
