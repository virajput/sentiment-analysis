# Multi-Language Sentiment Analysis with LLM

## Problem Statement
Build a multi-language sentiment analysis system using LLMs that can analyze text in **English, Hindi, Spanish, and French**. The system automatically detects language and applies appropriate sentiment analysis.

---

## Project Structure
```
├── data/
│   ├── create_dataset.py          # Dataset generator (500 samples/language)
│   └── multilingual_sentiment_dataset.csv  (generated)
├── src/
│   ├── preprocessing.py           # Task 1 – cleaning, tokenization
│   ├── sentiment_model.py         # Task 2 – LLM (XLM-RoBERTa)
│   ├── evaluation.py              # Task 3 – metrics + plots
│   └── traditional_models.py      # Task 5 – Naive Bayes, Logistic Regression
├── outputs/                       # Generated plots and metrics
├── app.py                         # Task 4 – Streamlit web interface
├── run_pipeline.py                # Runs all tasks end-to-end
└── requirements.txt
```

---

## Setup & Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy model (optional, used by preprocessing)
```bash
python -m spacy download en_core_web_sm
```

---

## Running the Application

### Run the full pipeline (all tasks, saves outputs)
```bash
python run_pipeline.py
```
This generates:
- `data/multilingual_sentiment_dataset.csv` – raw dataset (2000 samples)
- `data/preprocessed_dataset.csv` – cleaned dataset
- `outputs/confusion_matrix.png`
- `outputs/sentiment_distribution.png`
- `outputs/probability_scores.png`
- `outputs/per_language_metrics.png`
- `outputs/model_comparison.png`
- `outputs/results_summary.json`

### Run the Streamlit web interface
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## Model Details
- **LLM**: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
  - Multilingual XLM-RoBERTa fine-tuned on Twitter data
  - Handles English, Hindi, Spanish, French natively
  - Labels: `positive`, `neutral`, `negative`

---

## LLM vs Traditional Models Analysis

### Advantages of LLMs
1. **Multilingual without re-engineering**: Single model handles all 4 languages.
2. **Context-aware semantics**: Captures sarcasm, negation, idioms better than TF-IDF.

### Disadvantages of LLMs
1. **Computational cost**: Inference is slower and requires more memory than TF-IDF pipelines.
2. **Black-box**: Less interpretable than Logistic Regression with explicit feature weights.

### One Key Limitation
**Domain shift**: The model is pre-trained on Twitter data. Performance degrades on formal text (legal, academic) and dialectal variations of Hindi/Spanish.

---

## Dataset
Custom dataset created with 500 samples per language (2000 total), balanced across:
- **Sentiments**: positive (~167), neutral (~167), negative (~167)
- **Languages**: English, Hindi, Spanish, French

Source: Custom (based on OpenML multi-language sentiment dataset pattern, as the OpenML link may be unavailable).
