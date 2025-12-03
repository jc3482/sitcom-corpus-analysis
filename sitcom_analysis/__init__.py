"""
Sitcom Analysis - Unified Integration Layer

This package provides a unified interface to both information retrieval
and MBTI personality prediction functionality, plus combined features
that leverage both packages together.

Basic imports from core packages:
    from sitcom_analysis import search_episodes, load_show_data
    from sitcom_analysis import load_bundle, predict_mbti_for_character

Combined features:
    from sitcom_analysis import search_with_character_info
    from sitcom_analysis import analyze_character_moments
"""

# Import core functionality from both packages
from information_retrieval import (
    search_episodes,
    load_show_data,
    parse_query,
    DATASETS,
    SHOW_NAMES
)

from mbti_prediction import (
    load_bundle,
    load_all_dialogues,
    predict_mbti_for_character,
    score_quotes_for_character
)

# Import combined features
from .combined_features import (
    search_with_character_info,
    analyze_character_moments,
    get_character_mbti
)

__version__ = '1.0.0'

__all__ = [
    # Information Retrieval
    'search_episodes',
    'load_show_data',
    'parse_query',
    'DATASETS',
    'SHOW_NAMES',
    
    # MBTI Prediction
    'load_bundle',
    'load_all_dialogues',
    'predict_mbti_for_character',
    'score_quotes_for_character',
    
    # Combined Features
    'search_with_character_info',
    'analyze_character_moments',
    'get_character_mbti',
]

