"""
Task 5: Analysis & Comparison
Train Naive Bayes and Logistic Regression on the same dataset.
Compare metrics against LLM results.
List 2 advantages, 2 disadvantages of LLMs, and one limitation.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

OUTPUT_DIR = "outputs"
LABEL_ORDER = ["positive", "neutral", "negative"]


def train_traditional_models(df: pd.DataFrame, text_col: str = "cleaned_text",
                              label_col: str = "sentiment") -> dict:
    """
    Train Naive Bayes and Logistic Regression; return train/test split and metrics.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X = df[text_col].fillna("")
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                     sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                     sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                                       solver="lbfgs")),
        ]),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "macro_precision": round(precision_score(y_test, y_pred, average="macro",
                                                     zero_division=0), 4),
            "macro_recall": round(recall_score(y_test, y_pred, average="macro",
                                               zero_division=0), 4),
            "macro_f1": round(f1_score(y_test, y_pred, average="macro",
                                       zero_division=0), 4),
            "model": model,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }
        print(f"\n[{name}]")
        print(f"  Accuracy : {results[name]['accuracy']:.4f}")
        print(f"  Precision: {results[name]['macro_precision']:.4f}")
        print(f"  Recall   : {results[name]['macro_recall']:.4f}")
        print(f"  F1-Score : {results[name]['macro_f1']:.4f}")

    return results


def plot_model_comparison(traditional_results: dict, llm_metrics: dict,
                          filename="model_comparison.png") -> str:
    """
    Side-by-side bar chart comparing LLM vs traditional models on all 4 metrics.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metric_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    all_models = {
        "LLM\n(XLM-RoBERTa)": llm_metrics,
        **{k: v for k, v in traditional_results.items()},
    }

    x = np.arange(len(metric_labels))
    width = 0.25
    colors = ["#9b59b6", "#e67e22", "#27ae60"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (model_name, metrics) in enumerate(all_models.items()):
        values = [metrics.get(k, 0) for k in metric_keys]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=model_name,
                      color=colors[i % len(colors)], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("LLM vs Traditional Models: Performance Comparison",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def print_llm_analysis():
    """Print advantages, disadvantages, and limitation of LLMs."""
    print("""
=== LLM vs Traditional Methods: Analysis ===

ADVANTAGES of LLMs:
  1. Multilingual understanding without language-specific feature engineering:
     XLM-RoBERTa processes English, Hindi, Spanish, and French in a single
     unified model, whereas traditional methods require separate stopword lists,
     tokenizers, and TF-IDF vocabularies per language.
  2. Context-aware semantics and idiom handling:
     LLMs capture sentence-level context and handle sarcasm, negation, and
     idiomatic expressions (e.g., "not bad" → positive) far better than
     bag-of-words TF-IDF representations used in Naive Bayes / Logistic Regression.

DISADVANTAGES of LLMs:
  1. Computational cost and latency:
     LLMs (hundreds of millions of parameters) require significantly more memory
     and inference time compared to lightweight TF-IDF + Naive Bayes pipelines,
     making real-time, high-throughput deployments expensive.
  2. Black-box interpretability:
     Understanding why an LLM assigns a specific sentiment is non-trivial.
     Traditional models expose feature weights directly (top TF-IDF tokens per class),
     which is crucial in regulated domains (finance, healthcare).

ONE KEY LIMITATION:
  Domain shift and label imbalance sensitivity:
  Pre-trained LLMs are fine-tuned on social-media corpora (Twitter). When applied
  to formal text (legal documents, academic reviews) or low-resource language variants
  (Hindi dialects, colloquial Spanish), performance degrades noticeably. Fine-tuning
  on domain-specific data mitigates this but requires labeled data and GPU resources.
""")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.create_dataset import generate_dataset
    from src.preprocessing import preprocess_dataframe

    df = generate_dataset(output_path="data/multilingual_sentiment_dataset.csv")
    df = preprocess_dataframe(df)

    trad_results = train_traditional_models(df)

    # Dummy LLM metrics for standalone comparison demo
    dummy_llm = {"accuracy": 0.78, "macro_precision": 0.76,
                 "macro_recall": 0.75, "macro_f1": 0.75}
    path = plot_model_comparison(trad_results, dummy_llm)
    print(f"\nComparison chart saved: {path}")
    print_llm_analysis()
