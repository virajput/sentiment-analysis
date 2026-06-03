"""
Task 4: Interactive Web Interface (Streamlit)
Features:
  - Text input and file upload (.txt / .csv)
  - Automatic language detection
  - Color-coded sentiment display
  - Confidence score and probability bar chart
  - Batch analysis with downloadable results
"""
import os
import sys
import io
import json
import tempfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Language Sentiment Analyzer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.positive-box  { background:#d4efdf; border-left:6px solid #27ae60;
                 padding:16px 20px; border-radius:8px; margin:8px 0; }
.negative-box  { background:#fadbd8; border-left:6px solid #e74c3c;
                 padding:16px 20px; border-radius:8px; margin:8px 0; }
.neutral-box   { background:#d6eaf8; border-left:6px solid #2980b9;
                 padding:16px 20px; border-radius:8px; margin:8px 0; }
.sentiment-label { font-size:24px; font-weight:700; }
.confidence-text { font-size:15px; color:#555; margin-top:4px; }
.section-header { font-size:18px; font-weight:600; color:#2c3e50;
                  margin:12px 0 4px 0; }
</style>
""", unsafe_allow_html=True)


# ─── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading sentiment model…")
def load_model():
    from src.sentiment_model import SentimentAnalyzer
    return SentimentAnalyzer()


# ─── Helper: render sentiment card ────────────────────────────────────────────
EMOJI = {"positive": "😊", "negative": "😞", "neutral": "😐"}
SENTIMENT_CSS = {"positive": "positive-box", "negative": "negative-box",
                 "neutral": "neutral-box"}


def render_sentiment_card(result: dict, text_preview: str = ""):
    sentiment = result["sentiment"]
    confidence = result["confidence"]
    language = result.get("language", "—")
    css_class = SENTIMENT_CSS.get(sentiment, "neutral-box")
    emoji = EMOJI.get(sentiment, "")

    st.markdown(f"""
<div class="{css_class}">
  <div class="sentiment-label">{emoji} {sentiment.upper()}</div>
  <div class="confidence-text">Confidence: <b>{confidence:.1%}</b> &nbsp;|&nbsp;
       Detected language: <b>{language.capitalize()}</b></div>
  {f'<div style="margin-top:8px;font-size:13px;color:#666;">"{text_preview[:120]}{"…" if len(text_preview)>120 else ""}"</div>' if text_preview else ""}
</div>
""", unsafe_allow_html=True)


def render_prob_chart(scores: dict):
    labels = list(scores.keys())
    values = list(scores.values())
    colors = {"positive": "#27ae60", "neutral": "#2980b9", "negative": "#e74c3c"}
    bar_colors = [colors.get(l, "#95a5a6") for l in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=bar_colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Sentiment Probability Scores",
        yaxis=dict(range=[0, 1.15], tickformat=".0%"),
        plot_bgcolor="white",
        height=300,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    mode = st.radio("Mode", ["Single Text", "File Upload", "Batch Demo"])
    st.markdown("---")
    st.markdown("**Supported Languages**")
    st.markdown("🇬🇧 English  🇮🇳 Hindi  \n " \
                "🇪🇸 Spanish  🇫🇷 French")
    st.markdown("---")
    st.code("Assignment: \n"
            "NLP Applications PS-45", language=None)
    st.markdown("---")

# ─── Title ────────────────────────────────────────────────────────────────────
st.title("🌐 Multi-Language Sentiment Analysis")
st.markdown("Analyze sentiment in **English, Hindi, Spanish, and French** "
            "using a pre-trained multilingual LLM (XLM-RoBERTa).")
st.markdown("---")


# ─── Single Text Mode ─────────────────────────────────────────────────────────
if mode == "Single Text":
    st.subheader("Analyze a Single Text")

    sample_texts = {
        "": "",
        "English – Positive": "This product is absolutely amazing and exceeded all my expectations!",
        "English – Negative": "Terrible product, broke after just two days of normal use.",
        "Hindi – Positive": "यह उत्पाद बिल्कुल अद्भुत है और मेरी सभी उम्मीदों से बेहतर निकला।",
        "Hindi – Negative": "बहुत निराश हूं, उत्पाद क्षतिग्रस्त और अनुपयोगी आया।",
        "Spanish – Positive": "Este producto es absolutamente increíble y superó todas mis expectativas.",
        "Spanish – Neutral": "El producto llegó a tiempo y coincide con la descripción exactamente.",
        "French – Negative": "Produit terrible, cassé après seulement deux jours d'utilisation.",
        "French – Neutral": "Le produit est correct, rien de spécial mais rien de mauvais non plus.",
    }

    chosen = st.selectbox("Load a sample text (optional)", list(sample_texts.keys()))
    default_text = sample_texts[chosen]

    user_text = st.text_area(
        "Enter text to analyze",
        value=default_text,
        height=120,
        placeholder="Type or paste text here…",
    )

    col_lang, col_btn = st.columns([2, 1])
    with col_lang:
        manual_lang = st.selectbox(
            "Language (leave Auto for detection)",
            ["Auto-detect", "english", "hindi", "spanish", "french"],
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze_btn and user_text.strip():
        with st.spinner("Analyzing…"):
            analyzer = load_model()
            lang = None if manual_lang == "Auto-detect" else manual_lang
            result = analyzer.predict(user_text, language=lang)

        st.markdown("#### Result")
        render_sentiment_card(result, user_text)
        render_prob_chart(result["scores"])

    elif analyze_btn:
        st.warning("Please enter some text before clicking Analyze.")


# ─── File Upload Mode ─────────────────────────────────────────────────────────
elif mode == "File Upload":
    st.subheader("Analyze a File")
    st.markdown("Upload a `.txt` (one sentence per line) or `.csv` "
                "(must have a `text` column; optional `language` column).")

    uploaded = st.file_uploader("Choose a file", type=["txt", "csv"])

    if uploaded:
        # Parse
        if uploaded.name.endswith(".csv"):
            df_in = pd.read_csv(uploaded)
            if "text" not in df_in.columns:
                st.error("CSV must contain a `text` column.")
                st.stop()
            texts = df_in["text"].fillna("").tolist()
            langs = df_in["language"].tolist() if "language" in df_in.columns else [None] * len(texts)
        else:
            content = uploaded.read().decode("utf-8", errors="ignore")
            texts = [line.strip() for line in content.splitlines() if line.strip()]
            langs = [None] * len(texts)

        st.info(f"Loaded **{len(texts)}** text samples.")

        if st.button("🔍 Analyze All", type="primary"):
            analyzer = load_model()
            progress = st.progress(0)
            results = []
            for i, (text, lang) in enumerate(zip(texts, langs)):
                results.append(analyzer.predict(text, lang))
                progress.progress((i + 1) / len(texts))
            progress.empty()

            # Build result dataframe
            df_out = pd.DataFrame({
                "text": texts,
                "language": [r["language"] for r in results],
                "sentiment": [r["sentiment"] for r in results],
                "confidence": [r["confidence"] for r in results],
                "score_positive": [r["scores"]["positive"] for r in results],
                "score_neutral": [r["scores"]["neutral"] for r in results],
                "score_negative": [r["scores"]["negative"] for r in results],
            })

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            vc = df_out["sentiment"].value_counts()
            col1.metric("Positive", vc.get("positive", 0))
            col2.metric("Neutral", vc.get("neutral", 0))
            col3.metric("Negative", vc.get("negative", 0))

            # Distribution chart
            fig = go.Figure(go.Bar(
                x=vc.index.tolist(), y=vc.values.tolist(),
                marker_color=["#27ae60" if l == "positive"
                               else "#e74c3c" if l == "negative"
                               else "#2980b9" for l in vc.index],
                text=vc.values.tolist(), textposition="outside",
            ))
            fig.update_layout(title="Sentiment Distribution",
                              yaxis_title="Count", height=300,
                              plot_bgcolor="white",
                              margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig)

            # Table
            st.dataframe(df_out.style.applymap(
                lambda v: ("background-color:#d4efdf" if v == "positive"
                           else "background-color:#fadbd8" if v == "negative"
                           else "background-color:#d6eaf8"),
                subset=["sentiment"],
            ))

            # Download
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", csv_bytes,
                               "sentiment_results.csv", "text/csv")


# ─── Batch Demo Mode ─────────────────────────────────────────────────────────
elif mode == "Batch Demo":
    st.subheader("Batch Demo: Multi-Language Samples")
    st.markdown("Quick demonstration across all four languages.")

    demo_texts = [
        ("english", "This product is absolutely amazing and exceeded all my expectations."),
        ("english", "Terrible product, broke after just two days of use."),
        ("english", "The product arrived on time and matches the description."),
        ("hindi", "यह उत्पाद बिल्कुल अद्भुत है और मेरी सभी उम्मीदों से बेहतर निकला।"),
        ("hindi", "बेकार उत्पाद, सामान्य उपयोग के केवल दो दिनों में ही टूट गया।"),
        ("hindi", "उत्पाद समय पर आया और विवरण से मेल खाता है।"),
        ("spanish", "Este producto es absolutamente increíble y superó todas mis expectativas."),
        ("spanish", "Producto terrible, se rompió después de solo dos días de uso."),
        ("spanish", "El producto llegó a tiempo y coincide con la descripción exactamente."),
        ("french", "Ce produit est absolument incroyable et a dépassé toutes mes attentes."),
        ("french", "Produit terrible, cassé après seulement deux jours d'utilisation."),
        ("french", "Le produit est arrivé à temps et correspond exactement à la description."),
    ]

    if st.button("▶️ Run Demo", type="primary"):
        analyzer = load_model()
        results = []
        with st.spinner("Analyzing 12 samples…"):
            for lang, text in demo_texts:
                r = analyzer.predict(text, lang)
                results.append((lang, text, r))

        for lang, text, result in results:
            with st.expander(
                f"[{lang.upper()}] {text[:60]}… → "
                f"{EMOJI.get(result['sentiment'],'')} {result['sentiment'].upper()} "
                f"({result['confidence']:.1%})"
            ):
                render_sentiment_card(result, text)
                render_prob_chart(result["scores"])


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<small>Assignment 1 PS-45 · Model: cardiffnlp/twitter-xlm-roberta-base-sentiment</small>",
    unsafe_allow_html=True,
)
