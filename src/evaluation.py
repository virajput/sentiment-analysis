"""
==============================================================================
EVALUATION MODULE - Task 3: Sentiment Prediction & Visualization
==============================================================================

This module handles evaluation of sentiment predictions and generates
comprehensive visualizations to understand model performance.

Key Responsibilities:
  1. Compute evaluation metrics (accuracy, precision, recall, F1-score)
  2. Generate confusion matrices for error analysis
  3. Create visualizations showing:
     - Sentiment distribution across languages
     - Probability score distributions
     - Per-language model performance
     - Model comparison charts
  4. Create detailed evaluation reports

All visualizations are saved as high-resolution PNG files (150 DPI) for
inclusion in reports and presentations.

Author: NLP Assignment Team
Date: 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────

# Sentiment class ordering (used consistently across all metrics and plots)
LABEL_ORDER = ["positive", "neutral", "negative"]

# Color scheme for visualizations (accessible and consistent)
COLOR_MAP = {
    "positive": "#2ecc71",   # Green
    "neutral": "#3498db",    # Blue
    "negative": "#e74c3c",   # Red
}

OUTPUT_DIR = "outputs"





# ─────────────────────────────────────────────────────────────────────────
# FUNCTION: ensure_output_dir()
# ─────────────────────────────────────────────────────────────────────────
def ensure_output_dir():
    """
    Create the output directory if it doesn't exist.
    
    All generated visualizations and plots are saved here.
    This is idempotent - safe to call multiple times.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# FUNCTION: compute_metrics(y_true, y_pred, labels)
# ─────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: list, y_pred: list, labels=None) -> dict:
    """
    Compute comprehensive classification metrics.
    
    This function calculates:
    - Accuracy: Overall correctness (TP + TN) / Total
    - Macro Precision: Average precision across all classes (unweighted)
    - Macro Recall: Average recall across all classes (unweighted)
    - Macro F1: Harmonic mean of macro precision and recall
    - Detailed Classification Report: Per-class metrics
    
    Macro averaging treats all classes equally, regardless of support.
    This is useful for imbalanced datasets.
    
    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        labels (list): Class labels to consider (default: LABEL_ORDER)
    
    Returns:
        dict with keys:
            'accuracy': Overall accuracy (0.0 to 1.0)
            'macro_precision': Macro-averaged precision (0.0 to 1.0)
            'macro_recall': Macro-averaged recall (0.0 to 1.0)
            'macro_f1': Macro-averaged F1-score (0.0 to 1.0)
            'report': String with detailed per-class metrics
    
    Example:
        >>> y_true = ['positive', 'positive', 'negative', 'neutral']
        >>> y_pred = ['positive', 'neutral', 'negative', 'neutral']
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']}")
        Accuracy: 0.75
    """
    if labels is None:
        labels = LABEL_ORDER
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(precision_score(y_true, y_pred, average="macro",
                                                  labels=labels, zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, average="macro",
                                           labels=labels, zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0), 4),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }
    return metrics




def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix",
                          filename="confusion_matrix.png") -> str:
    """
    Generate and save a confusion matrix heatmap.
    
    The confusion matrix shows how predictions compare to ground truth:
    - Diagonal elements: Correct predictions (true positives)
    - Off-diagonal: Misclassifications (false positives/negatives)
    
    A well-performing model should have a strong diagonal.
    Patterns in off-diagonal elements reveal which classes are confused.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        title (str): Plot title
        filename (str): Output filename in OUTPUT_DIR (default: "confusion_matrix.png")
    
    Returns:
        str: Full path to saved PNG file
    
    Visualization Notes:
        - Uses Blues color map (darker = more samples)
        - Annotations show actual count of samples
        - Size: 7x6 inches at 150 DPI
        - Grid lines separate cells for clarity
    """
    ensure_output_dir()
    labels = [l for l in LABEL_ORDER if l in set(y_true) | set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor="gray")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sentiment_distribution(df: pd.DataFrame,
                                filename="sentiment_distribution.png") -> str:
    """Bar chart of sentiment distribution per language."""
    ensure_output_dir()
    counts = df.groupby(["language", "sentiment"]).size().unstack(fill_value=0)
    # Reorder columns
    present = [l for l in LABEL_ORDER if l in counts.columns]
    counts = counts[present]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLOR_MAP.get(l, "gray") for l in present]
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white",
                width=0.7, rot=0)
    ax.set_title("Sentiment Distribution by Language", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Language", fontsize=11)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_probability_scores(predictions: list, sample_size: int = 30,
                            filename="probability_scores.png") -> str:
    """
    Stacked bar chart showing probability scores for a sample of predictions.
    """
    ensure_output_dir()
    sample = predictions[:sample_size]
    indices = list(range(len(sample)))

    pos_scores = [p["scores"].get("positive", 0) for p in sample]
    neu_scores = [p["scores"].get("neutral", 0) for p in sample]
    neg_scores = [p["scores"].get("negative", 0) for p in sample]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(indices, pos_scores, label="Positive", color=COLOR_MAP["positive"], alpha=0.9)
    ax.bar(indices, neu_scores, bottom=pos_scores, label="Neutral",
           color=COLOR_MAP["neutral"], alpha=0.9)
    ax.bar(indices, neg_scores,
           bottom=[p + n for p, n in zip(pos_scores, neu_scores)],
           label="Negative", color=COLOR_MAP["negative"], alpha=0.9)

    ax.set_title(f"Probability Scores (first {len(sample)} samples)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample Index", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_language_metrics(results_by_lang: dict,
                              filename="per_language_metrics.png") -> str:
    """Grouped bar chart of F1, precision, recall per language."""
    ensure_output_dir()
    langs = list(results_by_lang.keys())
    metrics = ["macro_precision", "macro_recall", "macro_f1"]
    metric_labels = ["Precision", "Recall", "F1-Score"]
    x = np.arange(len(langs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, label) in enumerate(zip(metrics, metric_labels)):
        values = [results_by_lang[l].get(m, 0) for l in langs]
        ax.bar(x + i * width, values, width, label=label, alpha=0.85)

    ax.set_title("Model Performance by Language", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([l.capitalize() for l in langs])
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_evaluation(df: pd.DataFrame, predictions: list) -> dict:
    """Full evaluation pipeline; returns metrics dict."""
    df = df.copy()
    df["predicted_sentiment"] = [p["sentiment"] for p in predictions]
    df["confidence"] = [p["confidence"] for p in predictions]

    y_true = df["sentiment"].tolist()
    y_pred = df["predicted_sentiment"].tolist()

    overall = compute_metrics(y_true, y_pred)
    print("\n=== Overall Metrics (LLM) ===")
    print(f"Accuracy : {overall['accuracy']:.4f}")
    print(f"Precision: {overall['macro_precision']:.4f}")
    print(f"Recall   : {overall['macro_recall']:.4f}")
    print(f"F1-Score : {overall['macro_f1']:.4f}")
    print("\nClassification Report:\n", overall["report"])

    per_lang = {}
    for lang in df["language"].unique():
        mask = df["language"] == lang
        per_lang[lang] = compute_metrics(
            df.loc[mask, "sentiment"].tolist(),
            df.loc[mask, "predicted_sentiment"].tolist(),
        )
        print(f"\n[{lang.upper()}] Accuracy={per_lang[lang]['accuracy']:.4f} "
              f"F1={per_lang[lang]['macro_f1']:.4f}")

    # Plots
    cm_path = plot_confusion_matrix(y_true, y_pred, "LLM Confusion Matrix")
    dist_path = plot_sentiment_distribution(df)
    prob_path = plot_probability_scores(predictions)
    lang_path = plot_per_language_metrics(per_lang)

    print(f"\nPlots saved:")
    for p in [cm_path, dist_path, prob_path, lang_path]:
        print(f"  {p}")

    return {
        "overall": overall,
        "per_language": per_lang,
        "dataframe": df,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.create_dataset import generate_dataset
    from src.preprocessing import preprocess_dataframe
    from src.sentiment_model import SentimentAnalyzer

    df = generate_dataset(output_path="data/multilingual_sentiment_dataset.csv")
    df = preprocess_dataframe(df)

    # Use a small subset for quick testing
    df_sample = df.groupby(["language", "sentiment"]).head(10).reset_index(drop=True)
    print(f"Evaluating on {len(df_sample)} samples")

    analyzer = SentimentAnalyzer()
    preds = analyzer.predict_batch(df_sample["text"].tolist(),
                                   df_sample["language"].tolist())

    results = run_evaluation(df_sample, preds)
