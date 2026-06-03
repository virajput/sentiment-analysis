"""
Main pipeline: runs all 5 tasks end-to-end and saves results + plots.
Run: python run_pipeline.py
"""
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data.create_dataset import generate_dataset
from src.preprocessing import preprocess_dataframe
from src.sentiment_model import SentimentAnalyzer
from src.evaluation import run_evaluation
from src.traditional_models import (
    train_traditional_models, plot_model_comparison, print_llm_analysis
)

DATA_PATH = "data/multilingual_sentiment_dataset.csv"
PREPROCESSED_PATH = "data/preprocessed_dataset.csv"
RESULTS_PATH = "outputs/results_summary.json"
EVAL_SAMPLE_SIZE = 200   # samples per language for evaluation (reduce if slow)


def main():
    os.makedirs("outputs", exist_ok=True)

    # ── Task 1: Data Loading & Preprocessing ──────────────────────────────────
    print("=" * 60)
    print("TASK 1: Data Loading & Preprocessing")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        df_raw = generate_dataset(output_path=DATA_PATH)
    else:
        df_raw = pd.read_csv(DATA_PATH)
        print(f"Loaded existing dataset: {len(df_raw)} samples")

    df = preprocess_dataframe(df_raw)
    df.to_csv(PREPROCESSED_PATH, index=False)
    print(f"Preprocessed dataset saved: {PREPROCESSED_PATH}")
    print(f"Columns: {list(df.columns)}")
    print(df.groupby(["language", "sentiment"]).size().to_string())

    # ── Task 2: LLM Model Integration ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 2: LLM Model Integration")
    print("=" * 60)
    analyzer = SentimentAnalyzer()

    # Subsample for evaluation (full 2000 samples can be slow on CPU)
    df_eval = (
        df.groupby(["language", "sentiment"])
        .head(EVAL_SAMPLE_SIZE // 3)
        .reset_index(drop=True)
    )
    print(f"Evaluation subset: {len(df_eval)} samples")

    print("\nRunning LLM predictions...")
    predictions = analyzer.predict_batch(
        df_eval["text"].tolist(),
        df_eval["language"].tolist(),
    )

    # ── Task 3: Sentiment Prediction & Visualization ──────────────────────────
    print("\n" + "=" * 60)
    print("TASK 3: Sentiment Prediction & Visualization")
    print("=" * 60)
    eval_results = run_evaluation(df_eval, predictions)
    llm_overall = eval_results["overall"]

    # ── Task 5: Analysis & Comparison (traditional models) ────────────────────
    print("\n" + "=" * 60)
    print("TASK 5: Analysis & Comparison")
    print("=" * 60)
    trad_results = train_traditional_models(df)
    trad_for_plot = {k: {
        "accuracy": v["accuracy"],
        "macro_precision": v["macro_precision"],
        "macro_recall": v["macro_recall"],
        "macro_f1": v["macro_f1"],
    } for k, v in trad_results.items()}

    comparison_path = plot_model_comparison(trad_for_plot, llm_overall)
    print(f"Comparison plot: {comparison_path}")
    print_llm_analysis()

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "llm_overall": {k: v for k, v in llm_overall.items() if k != "report"},
        "traditional_models": trad_for_plot,
        "per_language_llm": {
            lang: {k: v for k, v in metrics.items() if k != "report"}
            for lang, metrics in eval_results["per_language"].items()
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults summary saved: {RESULTS_PATH}")

    print("\n" + "=" * 60)
    print("Pipeline complete. Outputs in ./outputs/")
    print("=" * 60)
    print("To run the Streamlit app:  streamlit run app.py")


if __name__ == "__main__":
    main()
