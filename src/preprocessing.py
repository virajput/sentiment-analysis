"""
==============================================================================
PREPROCESSING MODULE - Task 1: Data Loading & Preprocessing
==============================================================================

This module handles all text preprocessing operations for the multilingual
sentiment analysis pipeline. The primary goal is to clean and normalize text
data before it goes into the ML model.

Key Operations:
  - Text normalization (lowercase conversion)
  - URL and email removal
  - Special character and digit filtering
  - Tokenization (word-level splitting)
  - Language-specific stopword removal
  - Token length filtering

The module supports four languages: English, Hindi, Spanish, and French.
For Hindi, we use a custom stopword list since NLTK's support is limited.
"""

import re
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download essential NLTK data packages on import
# These are needed for tokenization and stopword lookup
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────
# Language Code to NLTK Language Name Mapping
# ─────────────────────────────────────────────────────────────────────────
# This maps both language names and ISO-639-1 codes to NLTK's stopword library.
# Note: NLTK's Hindi support is minimal, so we use a custom list below.

STOPWORD_MAP = {
    "english": "english",
    "en": "english",
    "spanish": "spanish",
    "es": "spanish",
    "french": "french",
    "fr": "french",
    "hindi": None,      # Custom Hindi list (see below)
    "hi": None,
}


# ─────────────────────────────────────────────────────────────────────────
# Hindi Stopwords List (Custom)
# ─────────────────────────────────────────────────────────────────────────
# Since NLTK doesn't have comprehensive Hindi stopword support, we maintain
# our own curated list covering common Hindi function words, pronouns, and
# prepositions. These words are typically low in semantic value and can be
# safely removed without significant information loss.

HINDI_STOPWORDS = {
    # Conjunctions and particles
    "और", "या", "लेकिन", "अगर", "क्योंकि", "तो", "तब", "जब", "फिर",
    
    # Prepositions and postpositions
    "का", "की", "के", "में", "से", "को", "पर", "तक", "साथ", "बाद", "पहले",
    
    # Verbs (auxiliary and common)
    "है", "हैं", "हो", "होता", "होती", "होते", "था", "थी", "थे", "होगा",
    "किया", "किए", "कर", "करने", "करता", "करती", "करते", "जा", "रहा", "रही", "रहे",
    
    # Pronouns
    "यह", "वह", "यहाँ", "वहाँ", "कहाँ", "इस", "उस", "इन", "उन", "आप", "हम",
    "मैं", "तुम", "वो", "उसे", "उन्हें", "मुझे", "हमें", "तुम्हें",
    
    # Possessive pronouns
    "उनका", "उनकी", "उनके", "मेरा", "मेरी", "मेरे", "आपका", "आपकी", "आपके",
    "हमारा", "हमारी", "हमारे", "तुम्हारा", "तुम्हारी", "तुम्हारे",
    
    # Articles and quantifiers
    "एक", "कोई", "कुछ", "सब", "सभी", "बहुत", "कभी", "कहीं", "हर", "ही",
    
    # Question words
    "क्या", "क्यों", "कितना", "कितनी", "कितने", "कैसे", "कब", "किसका",
}



# ─────────────────────────────────────────────────────────────────────────
# FUNCTION: get_stopwords(language)
# ─────────────────────────────────────────────────────────────────────────
def get_stopwords(language: str) -> set:
    """
    Retrieve the appropriate stopword list for a given language.
    
    This function abstracts away the language-specific stopword sources. It handles
    both NLTK's built-in stopwords (for English, Spanish, French) and our custom
    Hindi stopword list.
    
    Args:
        language (str): Language name or code (e.g., 'english', 'en', 'hindi', 'hi')
    
    Returns:
        set: Set of stopwords for the specified language. Empty set if language
             is not recognized.
    
    Example:
        >>> sw_en = get_stopwords('english')
        >>> sw_hi = get_stopwords('hindi')
        >>> 'the' in sw_en
        True
        >>> 'और' in sw_hi
        True
    """
    lang_key = language.lower()
    nltk_lang = STOPWORD_MAP.get(lang_key)
    
    # Handle Hindi separately (custom list)
    if lang_key in ("hindi", "hi"):
        return HINDI_STOPWORDS
    
    # Try to get NLTK stopwords for other languages
    if nltk_lang:
        try:
            return set(stopwords.words(nltk_lang))
        except Exception:
            return set()
    
    # Unknown language - return empty set (no filtering)
    return set()




# ─────────────────────────────────────────────────────────────────────────
# FUNCTION: clean_text(text, language)
# ─────────────────────────────────────────────────────────────────────────
def clean_text(text: str, language: str = "english") -> str:
    """
    Perform comprehensive text cleaning and preprocessing on a single sample.
    
    This function applies a multi-step preprocessing pipeline:
    1. Converts text to lowercase
    2. Removes URLs and email addresses
    3. Removes special characters and digits (preserves letters in all supported languages)
    4. Tokenizes the text into words
    5. Removes language-specific stopwords
    6. Filters out very short tokens (length <= 1)
    
    The function is designed to work with multilingual text, supporting Unicode
    characters for Hindi (Devanagari) and Latin-based languages.
    
    Args:
        text (str): Raw input text to clean
        language (str): Language code or name (default: 'english')
    
    Returns:
        str: Cleaned, tokenized text with tokens separated by spaces
    
    Example:
        >>> raw = "Check this out! Visit https://example.com or email me@test.com"
        >>> cleaned = clean_text(raw, 'english')
        >>> print(cleaned)
        # Output: "check out visit email"
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Lowercase (helps with consistency)
    text = text.lower()

    # Step 2: Remove URLs (both http:// and www.)
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Step 3: Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Step 4: Remove special characters and digits
    # This regex preserves: English letters (a-z, A-Z), Hindi Unicode range (ऀ-ॿ),
    # and Latin extended characters for Spanish and French (À-ɏ), plus spaces
    text = re.sub(r"[^a-zA-Zऀ-ॿÀ-ɏ\s]", " ", text)

    # Step 5: Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Step 6: Tokenize into words
    try:
        tokens = word_tokenize(text)
    except Exception:
        # Fallback to simple split if tokenizer fails
        tokens = text.split()

    # Step 7: Remove stopwords and short tokens
    sw = get_stopwords(language)
    tokens = [t for t in tokens if t not in sw and len(t) > 1]

    return " ".join(tokens)





# ─────────────────────────────────────────────────────────────────────────
# FUNCTION: preprocess_dataframe(df)
# ─────────────────────────────────────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to an entire dataset.
    
    This function processes a DataFrame by:
    1. Creating a new 'cleaned_text' column with preprocessed text
    2. Removing rows where preprocessing resulted in empty/whitespace-only text
    
    The function expects the input DataFrame to have 'text' and optionally
    'language' columns. If 'language' is missing, English is assumed.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'text' and optionally 'language' columns
    
    Returns:
        pd.DataFrame: Processed DataFrame with added 'cleaned_text' column and
                     rows with empty cleaned text removed. Index is reset.
    
    Example:
        >>> df = pd.DataFrame({
        ...     'text': ['Check this!', 'Great product!!!'],
        ...     'language': ['english', 'english'],
        ...     'sentiment': ['positive', 'positive']
        ... })
        >>> cleaned_df = preprocess_dataframe(df)
        >>> print(cleaned_df['cleaned_text'].tolist())
        # Output: ['check', 'great product']
    """
    df = df.copy()
    
    # Apply cleaning to each row, using the language column if available
    df["cleaned_text"] = df.apply(
        lambda row: clean_text(row["text"], row.get("language", "english")),
        axis=1,
    )
    
    # Remove rows where cleaning resulted in empty or whitespace-only text
    # This can happen if a text was entirely composed of stopwords or special chars
    df = df[df["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)
    
    return df



def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """Load CSV dataset and apply preprocessing."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    df = preprocess_dataframe(df)
    print(f"After preprocessing: {len(df)} samples")
    return df


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.create_dataset import generate_dataset

    df_raw = generate_dataset(output_path="data/multilingual_sentiment_dataset.csv")
    df = preprocess_dataframe(df_raw)

    #store processed data for later use
    df.to_csv("data/preprocessed_dataset.csv", index=False) 
    print("\nSample cleaned texts:")
    for lang in df["language"].unique():
        sample = df[df["language"] == lang].iloc[0]
        print(f"\n[{lang}] Original : {sample['text'][:80]}...")
        print(f"[{lang}] Cleaned  : {sample['cleaned_text'][:80]}...")
