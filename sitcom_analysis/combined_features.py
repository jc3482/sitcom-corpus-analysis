"""
Combined Features - Cross-package functionality

This module provides features that use both information retrieval
and MBTI prediction together.
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Tuple


def get_character_mbti(show_key: str, character_name: str) -> Optional[Dict]:
    """
    Get MBTI prediction for a specific character.
    
    Args:
        show_key: Show identifier (e.g., 'friends', 'tbbt')
        character_name: Character's name (case-insensitive)
    
    Returns:
        Dictionary with mbti, probabilities, and line count, or None if not found
    """
    from mbti_prediction import (
        load_bundle,
        load_all_dialogues,
        predict_mbti_for_character
    )
    
    tokenizer, model, label_mapping, preproc_info, max_len, device = load_bundle()
    df_all = load_all_dialogues()
    
    mbti_str, dim_probs, df_char = predict_mbti_for_character(
    tokenizer, model, label_mapping, df_all, show_key, character_name, max_len, device)
    
    if mbti_str is None or df_char is None or df_char.empty:
        return None
    
    return {
        'mbti': mbti_str,
        'probabilities': dim_probs,
        'line_count': len(df_char),
        'character': character_name,
        'show': show_key
    }


def search_with_character_info(
    show: str,
    query: str,
    top_k: int = 5,
    title_weight: int = 2,
    analyze_characters: bool = True
) -> Tuple[pd.DataFrame, Optional[Dict[str, str]]]:
    """
    Search episodes and optionally analyze characters mentioned in results.
    
    Args:
        show: Show key (e.g., 'friends', 'tbbt')
        query: Search query
        top_k: Number of results to return
        title_weight: Weight for title boosting
        analyze_characters: Whether to predict MBTI for detected characters
    
    Returns:
        Tuple of (search_results_df, character_mbti_dict)
    """
    from information_retrieval import load_show_data, search_episodes
    
    # Perform search
    episodes = load_show_data(show)
    results = search_episodes(episodes, query, top_k, title_weight)
    
    if not analyze_characters or results.empty:
        return results, None
    
    # Try to detect characters mentioned in snippets
    # This is a simple heuristic - could be enhanced
    character_mbti = {}
    
    try:
        from mbti_prediction import (
            load_bundle,
            load_all_dialogues,
            predict_mbti_for_character
        )
        
        tokenizer, model, label_mapping, preproc_info, max_len, device = load_bundle()
        df_all = load_all_dialogues()
        
        # Get list of characters from the show
        show_chars = df_all[df_all['show_key'] == show]['character'].unique()
        
        # Analyze top characters (limit to avoid slowdown)
        for char in list(show_chars)[:10]:
            mbti_str, _, df_char = predict_mbti_for_character(
                        tokenizer, model, label_mapping, df_all, show, char, max_len, device
                        )
            if mbti_str and df_char is not None and not df_char.empty:
                character_mbti[char] = mbti_str
    
    except Exception as e:
        print(f"[WARN] Could not analyze characters: {e}")
        return results, None
    
    return results, character_mbti


def analyze_character_moments(
    show: str,
    character: str,
    query: Optional[str] = None,
    top_k: int = 5
) -> Dict:
    """
    Analyze a character's personality and optionally search their dialogue.
    
    Args:
        show: Show key (e.g., 'friends', 'tbbt')
        character: Character name
        query: Optional search query to filter character's dialogue
        top_k: Number of quotes to return
    
    Returns:
        Dictionary containing:
            - mbti: Predicted MBTI type
            - probabilities: Per-dimension probabilities
            - top_quotes: Representative quotes
            - matching_moments: (if query provided) Moments matching the query
    """
    from mbti_prediction import (
        load_bundle,
        load_all_dialogues,
        predict_mbti_for_character,
        score_quotes_for_character
    )
    
    # Load MBTI components
    tokenizer, model, label_mapping, preproc_info, max_len, device = load_bundle()
    df_all = load_all_dialogues()
    
    # Predict MBTI
    mbti_str, dim_probs, df_char = predict_mbti_for_character(
        tokenizer, model, label_mapping, df_all, show, character, max_len, device
        )
    
    if mbti_str is None or df_char is None or df_char.empty:
        return {
            'success': False,
            'error': f"No dialogue found for {character} in {show}"
        }
    
    # Get representative quotes
    top_quotes = score_quotes_for_character(
        tokenizer, model, label_mapping, df_char, mbti_str, max_len, device, top_k=top_k
    )
    
    result = {
        'success': True,
        'character': character,
        'show': show,
        'mbti': mbti_str,
        'probabilities': dim_probs,
        'top_quotes': top_quotes,
        'total_lines': len(df_char)
    }
    
    # If query provided, search character's dialogue
    if query:
        from information_retrieval.search import extract_best_snippet, highlight
        
        # Simple search through character's dialogue
        query_lower = query.lower()
        matching_lines = []
        
        for _, row in df_char.iterrows():
            dialogue = str(row.get('dialogue_raw', ''))
            if query_lower in dialogue.lower():
                snippet = extract_best_snippet(dialogue, query)
                highlighted = highlight(snippet, query)
                matching_lines.append({
                    'season': row.get('season'),
                    'episode': row.get('episode'),
                    'episode_title': row.get('episode_title', 'Unknown'),
                    'snippet': highlighted,
                    'full_dialogue': dialogue
                })
        
        result['matching_moments'] = matching_lines[:top_k]
        result['total_matches'] = len(matching_lines)
    
    return result


def get_show_character_personalities(show: str, limit: int = 10) -> Dict[str, str]:
    """
    Get MBTI predictions for all major characters in a show.
    
    Args:
        show: Show key (e.g., 'friends', 'tbbt')
        limit: Maximum number of characters to analyze
    
    Returns:
        Dictionary mapping character names to MBTI types
    """
    from mbti_prediction import (
        load_bundle,
        load_all_dialogues,
        predict_mbti_for_character
    )
    
    tokenizer, model, label_mapping, preproc_info, max_len, device = load_bundle()
    df_all = load_all_dialogues()
    
    # Get characters from the show, sorted by number of lines
    show_df = df_all[df_all['show_key'] == show]
    char_counts = show_df['character'].value_counts().head(limit)
    
    results = {}
    
    for char in char_counts.index:
        mbti_str, _, df_char = predict_mbti_for_character(
            tokenizer, model, label_mapping, df_all, show, char, max_len, device,
        )
        if mbti_str and df_char is not None and not df_char.empty:
            results[char] = mbti_str
    
    return results

