"""
MBTI Personality Prediction Module
----------------------------------

This package provides:

1. Data preprocessing utilities
2. Functions to load the trained MBTI BERT model bundle
3. Convenience prediction functions for:
    - Predicting MBTI for a single text
    - Predicting MBTI for all dialogues of a character
4. Optional baseline (Naive Bayes) classifier

The main entry point for CLI usage is:

    python -m mbti_prediction

(or equivalently, mbti_prediction.__main__)

"""

from .mbti_prediction import (
    load_bundle,
    load_all_dialogues,
    predict_mbti_for_character,
    score_quotes_for_character,
    SHOW_FILES,
    SHOW_DISPLAY_NAMES,
    SHOW_ALIASES,
)

from .data_processing import preprocess_text
