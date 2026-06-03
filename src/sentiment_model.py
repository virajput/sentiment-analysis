"""
==============================================================================
SENTIMENT ANALYSIS MODEL - Task 2: LLM Model Integration
==============================================================================

This module provides the core sentiment analysis functionality using pre-trained
transformer models from Hugging Face. It combines two specialized models:

1. XLM-RoBERTa (Multilingual): cardiffnlp/twitter-xlm-roberta-base-sentiment
   - Best for Spanish, French, Hindi, and other non-English languages
   - Trained on multilingual Twitter data
   - Supports 100+ languages

2. RoBERTa-English (English-only): cardiffnlp/twitter-roberta-base-sentiment-latest
   - Optimized specifically for English sentiment analysis
   - Better accuracy for English texts (optional)
   - Also trained on Twitter data for informal text understanding

Key Features:
  - Automatic language detection using langdetect library
  - Batch prediction for efficient processing
  - Support for confidence scores and per-class probability distributions
  - Graceful fallback to neutral sentiment for empty inputs
  - GPU/CPU agnostic with device parameter

Author: NLP Assignment Team
Date: 2025
"""

import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, LangDetectException

# ─────────────────────────────────────────────────────────────────────────
# MODEL IDENTIFIERS
# ─────────────────────────────────────────────────────────────────────────
# These are Hugging Face model IDs. Models are automatically downloaded on first use.

MULTILINGUAL_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
ENGLISH_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"



# ─────────────────────────────────────────────────────────────────────────
# LABEL MAPPINGS
# ─────────────────────────────────────────────────────────────────────────
# Different models output labels in different formats. This mapping normalizes
# all output to our standard three-class schema: {positive, negative, neutral}

LABEL_MAP = {
    # Standard format (lowercase)
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    
    # Uppercase variants
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    
    # Numeric format (as some models output)
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

# Map language codes (from langdetect) to readable language names
LANG_NAME_MAP = {
    "en": "english",
    "hi": "hindi",
    "es": "spanish",
    "fr": "french",
}





# ─────────────────────────────────────────────────────────────────────────
# FUNCTION: detect_language(text)
# ─────────────────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """
    Automatically detect the language of a given text using langdetect.
    
    The function uses a probabilistic language detection model trained on
    common language patterns. It maps ISO 639-1 language codes to our
    supported language names.
    
    Args:
        text (str): Text to detect language for
    
    Returns:
        str: Language name (e.g., 'english', 'hindi', 'spanish', 'french')
             Defaults to 'english' if detection fails or language not recognized
    
    Note:
        - The detection might be unreliable for very short texts (< 5 words)
        - Mixed-language text will be detected based on the dominant language
        - Unknown languages default to 'english'
    
    Example:
        >>> detect_language("This is an English sentence")
        'english'
        >>> detect_language("यह एक हिंदी वाक्य है")
        'hindi'
    """
    try:
        code = detect(text)
        return LANG_NAME_MAP.get(code, code)
    except LangDetectException:
        # If detection fails (e.g., too short text), default to English
        return "english"





# ─────────────────────────────────────────────────────────────────────────
# CLASS: SentimentAnalyzer
# ─────────────────────────────────────────────────────────────────────────
class SentimentAnalyzer:
    """
    Main class for multilingual sentiment analysis using pre-trained transformers.
    
    This class wraps the Hugging Face transformers library and provides a simple
    interface for sentiment prediction. It can optionally use an English-specific
    model for improved accuracy on English text.
    
    Attributes:
        device (int): CUDA device ID (-1 for CPU, 0+ for GPU)
        use_english_model_for_en (bool): Whether to use English-specific model
        multilingual_pipe: Transformers pipeline for multilingual model
        english_pipe: Optional transformers pipeline for English model
    
    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> result = analyzer.predict("I love this product!")
        >>> print(result)
        {
            'sentiment': 'positive',
            'confidence': 0.98,
            'scores': {'positive': 0.98, 'neutral': 0.01, 'negative': 0.01},
            'language': 'english'
        }
    """

    def __init__(self, use_english_model_for_en: bool = False, device: int = -1):
        """
        Initialize the sentiment analyzer with language models.
        
        This constructor loads the multilingual model and optionally the English-specific
        model. Model loading happens once and is cached in memory for subsequent predictions.
        
        Args:
            use_english_model_for_en (bool): If True, uses English-specific model for
                                            English text (better accuracy but slower).
                                            Default: False (use multilingual for all).
            device (int): PyTorch device ID. -1 for CPU, 0+ for GPU.
                         Default: -1 (CPU mode)
        
        Note:
            - First run will download models (~500 MB total) to ~/.cache/huggingface/
            - Subsequent runs load from cache (very fast)
            - GPU usage requires torch with CUDA support installed
        """
        self.device = device
        self.use_english_model_for_en = use_english_model_for_en
        print(f"Loading multilingual model: {MULTILINGUAL_MODEL}")
        self.multilingual_pipe = pipeline(
            "sentiment-analysis",
            model=MULTILINGUAL_MODEL,
            tokenizer=MULTILINGUAL_MODEL,
            device=device,
            truncation=True,
            max_length=512,
            top_k=None,
        )
        self.english_pipe = None
        if use_english_model_for_en:
            print(f"Loading English model: {ENGLISH_MODEL}")
            self.english_pipe = pipeline(
                "sentiment-analysis",
                model=ENGLISH_MODEL,
                tokenizer=ENGLISH_MODEL,
                device=device,
                truncation=True,
                max_length=512,
                top_k=None,
            )

    def _normalize_scores(self, raw_scores) -> dict:
        """
        Normalize raw model output to a standardized score dictionary.
        
        The transformers library can return scores in different formats depending
        on the version and configuration. This method normalizes all variants to:
        {'positive': float, 'negative': float, 'neutral': float}
        
        Each score is rounded to 4 decimal places for consistency.
        
        Args:
            raw_scores: Raw output from transformers pipeline (list of lists/dicts)
        
        Returns:
            dict: Normalized scores with keys 'positive', 'negative', 'neutral'
                  Each value is a float between 0 and 1, summing to 1.0
        
        Implementation Note:
            - Handles nested list format: [[{label, score}, ...]]
            - Handles flat format: [{label, score}, ...]
            - Ensures all three sentiment keys exist (fills missing with 0.0)
        """
        # Flatten nested list if needed (transformers >= 4.x returns [[{...},...]])
        if raw_scores and isinstance(raw_scores[0], list):
            raw_scores = raw_scores[0]
        result = {}
        for item in raw_scores:
            if isinstance(item, dict):
                label = LABEL_MAP.get(item["label"], item["label"].lower())
                result[label] = round(float(item["score"]), 4)
        for k in ("positive", "negative", "neutral"):
            result.setdefault(k, 0.0)
        return result

    def predict(self, text: str, language: str = None) -> dict:
        """
        Predict sentiment for a single piece of text.
        
        This is the main inference method. It automatically:
        1. Detects language if not provided
        2. Selects appropriate model (English-specific or multilingual)
        3. Runs inference
        4. Returns normalized results with all metadata
        
        Args:
            text (str): Input text to analyze (any length up to 512 tokens)
            language (str): Optional language override. If None, auto-detects.
        
        Returns:
            dict with keys:
                'sentiment' (str): 'positive', 'negative', or 'neutral'
                'confidence' (float): Confidence score (0.0 to 1.0)
                'scores' (dict): Per-class probabilities {positive, negative, neutral}
                'language' (str): Detected or provided language
        
        Example:
            >>> result = analyzer.predict("Amazing quality!")
            >>> result['sentiment']
            'positive'
            >>> result['confidence']
            0.95
            >>> result['scores']
            {'positive': 0.95, 'neutral': 0.04, 'negative': 0.01}
        
        Note:
            - Empty input defaults to neutral sentiment with 100% confidence
            - Text is automatically truncated to 512 tokens by the model
            - Results are deterministic (same input = same output)
        """
        if not text or not text.strip():
            return {"sentiment": "neutral", "confidence": 1.0,
                    "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                    "language": "unknown"}

        detected_lang = language or detect_language(text)

        if self.use_english_model_for_en and detected_lang == "english" and self.english_pipe:
            pipe = self.english_pipe
        else:
            pipe = self.multilingual_pipe

        raw = pipe(text)
        # top_k=None returns [[{label,score},...]] for single input
        scores = self._normalize_scores(raw[0] if raw else [])
        best_label = max(scores, key=scores.get)

        return {
            "sentiment": best_label,
            "confidence": round(scores[best_label], 4),
            "scores": scores,
            "language": detected_lang,
        }

    def predict_batch(self, texts: list, languages: list = None, batch_size: int = 32) -> list:
        """
        Predict sentiment for multiple texts efficiently.
        
        Batch processing is more memory-efficient than individual predictions,
        especially for large datasets. Predictions are still run individually
        but grouped for any shared setup.
        
        Args:
            texts (list): List of text strings to analyze
            languages (list): Optional list of languages (one per text).
                            If None, auto-detects each text individually.
            batch_size (int): Number of samples per progress update (default: 32)
        
        Returns:
            list: List of result dictionaries (same format as predict())
        
        Example:
            >>> texts = ["Great!", "Terrible.", "It's okay."]
            >>> results = analyzer.predict_batch(texts)
            >>> [r['sentiment'] for r in results]
            ['positive', 'negative', 'neutral']
        
        Note:
            - Progress printed every batch_size*5 samples
            - Languages auto-detected if not provided
            - Individual timeouts can occur on GPU for very long texts
        """
        if languages is None:
            languages = [None] * len(texts)

        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            batch_langs = languages[i: i + batch_size]
            for text, lang in zip(batch_texts, batch_langs):
                results.append(self.predict(text, lang))
            if i % (batch_size * 5) == 0:
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} samples")
        return results


_analyzer = None


def get_analyzer(use_english_model_for_en: bool = False) -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer(use_english_model_for_en=use_english_model_for_en)
    return _analyzer


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    test_cases = [
        ("This product is absolutely amazing!", "english"),
        ("यह उत्पाद बिल्कुल बेकार है।", "hindi"),
        ("Este producto es fantástico y superó mis expectativas.", "spanish"),
        ("Le produit est correct, rien de spécial.", "french"),
        ("Terrible product, broke after two days.", "english"),
    ]
    print("\nSample predictions:")
    for text, lang in test_cases:
        result = analyzer.predict(text, lang)
        print(f"\nText    : {text}")
        print(f"Language: {result['language']}")
        print(f"Sentiment: {result['sentiment']} ({result['confidence']:.2%})")
        print(f"Scores  : {result['scores']}")
