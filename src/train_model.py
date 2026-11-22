"""
RetainIQ - Telecom Customer Churn Prediction System
Machine Learning Model Training, Cross-Validation, Evaluation, and Serialization Module

This module implements:
1. Training three classification algorithms inside unified scikit-learn Pipelines:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
2. 5-Fold Stratified Cross-Validation on training data evaluating Churn F1-score.
3. Model selection based on cross-validation performance.
4. Final evaluation on held-out test set (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix).
5. Generation of publication-quality evaluation figures:
   - Confusion matrices for each algorithm
   - Model metrics comparison bar charts
   - ROC curves comparison
   - Top 10 feature importance bar chart
6. Serialization of the best fitted pipeline to models/best_model.pkl.
7. Exporting structured metrics to outputs/model_metrics.json, outputs/model_comparison.csv,
   and outputs/feature_importance.csv.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, List
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from sklearn.inspection import permutation_importance

from src.data_loader import load_raw_data
from src.preprocessing import (
    clean_telco_data,
    separate_features_and_target,
    build_preprocessor,
    split_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
MODEL_METRICS_PATH = os.path.join(OUTPUTS_DIR, "model_metrics.json")
MODEL_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUTS_DIR, "feature_importance.csv")

RANDOM_STATE = 42


def get_candidate_models() -> Dict[str, Any]:
    """
    Initializes candidate classifiers with controlled hyperparameters and
    balanced class weights to appropriately address churn class imbalance.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


def perform_cross_validation(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Dict[str, Dict[str, float]]:
    """
    Executes 5-Fold Stratified Cross-Validation on the training dataset strictly.
    Evaluates F1-score for the positive churn class (1).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    print("\n" + "=" * 70)
    print("5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING SET (CHURN F1-SCORE)")
    print("=" * 70)

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", clf)
        ])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        cv_results[name] = {
            "cv_f1_mean": round(mean_score, 4),
            "cv_f1_std": round(std_score, 4),
            "fold_scores": [round(float(s), 4) for s in scores]
        }
        print(f"[{name:20s}] CV F1: {mean_score:.4f} (+/- {std_score:.4f}) | Folds: {cv_results[name]['fold_scores']}")

    return cv_results


def evaluate_models_on_test(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Fits each model pipeline on training data and evaluates on held-out test data.
    Computes Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
    """
    fitted_pipelines = {}
    test_metrics = {}
    comparison_rows = []

    print("\n" + "=" * 70)
    print("HELD-OUT TEST SET EVALUATION (CHURN POSITIVE CLASS = 1)")
    print("=" * 70)

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", clf)
        ])
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else 0.0
        cm = confusion_matrix(y_test, y_pred).tolist()

        test_metrics[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_prob": y_prob
        }

        comparison_rows.append({
            "Algorithm": name,
            "Accuracy": round(acc, 4),
            "Precision (Churn)": round(prec, 4),
            "Recall (Churn)": round(rec, 4),
            "F1-Score (Churn)": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "True Negatives": cm[0][0],
            "False Positives": cm[0][1],
            "False Negatives": cm[1][0],
            "True Positives": cm[1][1]
        })

        print(f"[{name:20s}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    comp_df = pd.DataFrame(comparison_rows)
    return fitted_pipelines, test_metrics, comp_df


def plot_confusion_matrices(test_metrics: Dict[str, Dict[str, Any]]) -> None:
    """Generates and saves confusion matrix heatmaps for each model."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for name, metrics in test_metrics.items():
        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"],
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"}
        )
        safe_name = name.lower().replace(" ", "_")
        ax.set_title(f"Confusion Matrix: {name}\n(Test Set: n={cm.sum()})", fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Customer Status", labelpad=8)
        ax.set_ylabel("Actual Customer Status", labelpad=8)
        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{safe_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {save_path}")


def plot_model_comparison_charts(comp_df: pd.DataFrame) -> None:
    """Generates comprehensive comparison bar charts across evaluated models."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Multi-metric comparison chart
    metrics_to_plot = ["Accuracy", "Precision (Churn)", "Recall (Churn)", "F1-Score (Churn)", "ROC-AUC"]
    melted = comp_df.melt(id_vars=["Algorithm"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    palette = ["#2b5c8f", "#e67e22", "#27ae60", "#8e44ad", "#34495e"]
    sns.barplot(data=melted, x="Algorithm", y="Score", hue="Metric", palette=palette, ax=ax)
    ax.set_title("Comprehensive Model Performance Comparison (Held-out Test Set)", fontweight="bold", pad=12)
    ax.set_ylabel("Metric Score (0.0 to 1.0)")
    ax.set_ylim(0, 1.05)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2, h + 0.015), ha="center", va="bottom", fontsize=8)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine(ax=ax)
    plt.tight_layout()
    p1 = os.path.join(FIGURES_DIR, "model_comparison_metrics.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    logger.info(f"Saved: {p1}")

    # 2. Accuracy comparison chart
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sns.barplot(
        data=comp_df,
        x="Algorithm",
        y="Accuracy",
        hue="Algorithm",
        palette=["#2b5c8f", "#27ae60", "#e67e22"],
        ax=ax,
        width=0.5,
        legend=False
    )
    ax.set_title("Model Accuracy Comparison (Held-out Test Set)", fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h*100:.2f}%", (p.get_x() + p.get_width() / 2, h + 0.02), ha="center", va="bottom", fontsize=10, fontweight="bold")
    sns.despine(ax=ax)
    plt.tight_layout()
    p2 = os.path.join(FIGURES_DIR, "model_accuracy_comparison.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    logger.info(f"Saved: {p2}")

    # 3. Precision, Recall, and F1 comparison
    prf_df = comp_df.melt(
        id_vars=["Algorithm"],
        value_vars=["Precision (Churn)", "Recall (Churn)", "F1-Score (Churn)"],
        var_name="Metric",
        value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.barplot(data=prf_df, x="Algorithm", y="Score", hue="Metric", palette=["#3498db", "#e74c3c", "#2ecc71"], ax=ax)
    ax.set_title("Precision, Recall, and F1-Score Comparison for Churn Class", fontweight="bold", pad=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2, h + 0.015), ha="center", va="bottom", fontsize=8)
    ax.legend(title="Metric", frameon=True)
    sns.despine(ax=ax)
    plt.tight_layout()
    p3 = os.path.join(FIGURES_DIR, "model_precision_recall_f1.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    logger.info(f"Saved: {p3}")


def plot_roc_curves(test_metrics: Dict[str, Dict[str, Any]], y_test: pd.Series) -> None:
    """Plots and saves ROC curves for all candidate models."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = {"Logistic Regression": "#2b5c8f", "Decision Tree": "#e67e22", "Random Forest": "#27ae60"}

    for name, metrics in test_metrics.items():
        if metrics["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
            auc = metrics["roc_auc"]
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=colors.get(name, "#333333"), linewidth=2.0)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier (AUC = 0.5000)", linewidth=1.2)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.legend(loc="lower right", frameon=True)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "model_roc_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {save_path}")


def analyze_feature_importance(
    best_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Computes permutation importance on the held-out test set through the complete fitted pipeline.
    Uses F1-score for the churn class as the scoring function.
    Saves feature importances to outputs/feature_importance.csv and creates a top-10 horizontal bar chart.
    """
    logger.info("Computing permutation feature importance on held-out test set...")
    result = permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    fi_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std
    }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    logger.info(f"Saved feature importance table to '{FEATURE_IMPORTANCE_PATH}'.")

    # Top 10 Features Horizontal Bar Chart
    top10 = fi_df.head(10).iloc[::-1]  # Reverse for ascending horizontal display
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    bars = ax.barh(top10["Feature"], top10["Importance_Mean"], xerr=top10["Importance_Std"], color="#2b5c8f", alpha=0.85, capsize=3)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.4f}", (w + 0.002, bar.get_y() + bar.get_height() / 2), va="center", fontsize=9, fontweight="bold")

    ax.set_title("Top 10 Most Influential Features (Permutation Importance - F1 Metric)", fontweight="bold", pad=12)
    ax.set_xlabel("Mean Decrease in Test F1-Score when Feature is Shuffled")
    ax.set_ylabel("Input Feature")
    sns.despine(ax=ax)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "feature_importance_top10.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved top 10 feature importance chart to '{save_path}'.")
    logger.info("Note: Permutation importance indicates predictive attribution within this fitted pipeline, not causal intervention.")
    return fi_df


def train_and_select_best_model():
    """
    Main orchestration routine for model training, CV selection, testing, and artifact saving.
    """
    raw_df = load_raw_data()
    cleaned_df = clean_telco_data(raw_df, save_processed=True)
    X, y = separate_features_and_target(cleaned_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=RANDOM_STATE)

    candidate_models = get_candidate_models()

    # Step 1: Stratified Cross-Validation strictly on Training Data
    cv_results = perform_cross_validation(candidate_models, X_train, y_train)

    # Step 2: Select Best Model based on highest Training CV F1-score
    best_model_name = max(cv_results.items(), key=lambda x: x[1]["cv_f1_mean"])[0]
    best_cv_score = cv_results[best_model_name]["cv_f1_mean"]
    logger.info(f"Selected best model: '{best_model_name}' (Training 5-Fold CV F1: {best_cv_score:.4f})")

    # Step 3: Fit candidate models on training set and evaluate on held-out test set
    fitted_pipelines, test_metrics, comp_df = evaluate_models_on_test(
        candidate_models, X_train, y_train, X_test, y_test
    )

    # Add CV scores to comparison dataframe
    comp_df["CV F1 (Train Mean)"] = comp_df["Algorithm"].map(lambda a: cv_results[a]["cv_f1_mean"])
    comp_df["CV F1 (Train Std)"] = comp_df["Algorithm"].map(lambda a: cv_results[a]["cv_f1_std"])
    comp_df = comp_df[[
        "Algorithm", "CV F1 (Train Mean)", "CV F1 (Train Std)", "Accuracy",
        "Precision (Churn)", "Recall (Churn)", "F1-Score (Churn)", "ROC-AUC",
        "True Negatives", "False Positives", "False Negatives", "True Positives"
    ]]

    # Step 4: Save model comparison table
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    comp_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info(f"Saved model comparison table to '{MODEL_COMPARISON_PATH}'.")

    # Step 5: Save best fitted pipeline
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_pipeline = fitted_pipelines[best_model_name]
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    logger.info(f"Saved best model pipeline to '{BEST_MODEL_PATH}'.")

    # Step 6: Save best model metrics JSON
    best_test = test_metrics[best_model_name]
    metrics_summary = {
        "selected_model": best_model_name,
        "selection_criterion": "Highest 5-Fold Stratified Cross-Validation F1-score on training set",
        "random_seed": RANDOM_STATE,
        "positive_class": "Churn = 1 (Customer left)",
        "train_samples_count": int(len(X_train)),
        "test_samples_count": int(len(X_test)),
        "training_cv_results": cv_results,
        "held_out_test_metrics": {
            "accuracy": best_test["accuracy"],
            "precision": best_test["precision"],
            "recall": best_test["recall"],
            "f1_score": best_test["f1_score"],
            "roc_auc": best_test["roc_auc"],
            "confusion_matrix": best_test["confusion_matrix"]
        },
        "all_models_test_comparison": comp_df.to_dict(orient="records")
    }

    with open(MODEL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    logger.info(f"Saved detailed metrics to '{MODEL_METRICS_PATH}'.")

    # Step 7: Generate figures
    plot_confusion_matrices(test_metrics)
    plot_model_comparison_charts(comp_df)
    plot_roc_curves(test_metrics, y_test)

    # Step 8: Feature importance on held-out test set
    analyze_feature_importance(best_pipeline, X_test, y_test)

    print("\n" + "=" * 70)
    print("FINAL MODEL TRAINING & EVALUATION SUMMARY")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    print("=" * 70)
    print(f"Selected Model: {best_model_name}")
    print(f"Held-out Test Accuracy:  {best_test['accuracy']:.4f}")
    print(f"Held-out Test Precision: {best_test['precision']:.4f}")
    print(f"Held-out Test Recall:    {best_test['recall']:.4f}")
    print(f"Held-out Test F1-Score:  {best_test['f1_score']:.4f}")
    print(f"Held-out Test ROC-AUC:   {best_test['roc_auc']:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    train_and_select_best_model()
