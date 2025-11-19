"""
Core search functionality for TV show episode retrieval
"""

import pandas as pd
from rank_bm25 import BM25Okapi

import os

# Dataset configuration
DATASETS = {
    'friends': 'raw_data/friends_dialogues.csv',
    'tbbt': 'raw_data/tbbt_dialogues.csv',
    'seinfeld': 'raw_data/seinfeld_scripts.csv',
    'office': 'raw_data/the_office.csv',
    'modern_family': 'raw_data/modern_family_scripts.csv',
}

SHOW_NAMES = {
    'friends': 'Friends',
    'tbbt': 'The Big Bang Theory',
    'seinfeld': 'Seinfeld',
    'office': 'The Office',
    'modern_family': 'Modern Family',
}


def parse_query(query):
    """
    Parse query into structured format
    
    Args:
        query: Search query string
        
    Returns:
        Dict with 'type' (phrase/and/or) and 'value'
    """
    query = query.strip()

    # CASE 1: exact phrase (space-separated, no & or /)
    if "&" not in query and "/" not in query:
        return {"type": "phrase", "value": query.lower()}

    # CASE 2: AND: keyword1&keyword2
    if "&" in query:
        parts = [p.strip().lower() for p in query.split("&") if p.strip()]
        return {"type": "and", "value": parts}

    # CASE 3: OR: keyword1/keyword2
    if "/" in query:
        parts = [p.strip().lower() for p in query.split("/") if p.strip()]
        return {"type": "or", "value": parts}


def load_show_data(show):
    """
    Load and prepare episode data for a show
    
    Args:
        show: Show key (e.g., 'friends', 'tbbt')
        
    Returns:
        DataFrame with episodes grouped by season/episode
        
    Raises:
        ValueError: If show is unknown
        FileNotFoundError: If dataset file doesn't exist
    """
    if show not in DATASETS:
        raise ValueError(f"Unknown show: {show}. Choose from: {list(DATASETS.keys())}")
    
    filepath = DATASETS[show]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Check for required columns
    if 'dialogue' not in df.columns:
        raise ValueError(f"Dataset must have 'dialogue' column. Found: {list(df.columns)}")
    
    # Combine dialogue by episode
    group_cols = ['season', 'episode']
    if 'episode_title' in df.columns:
        group_cols.append('episode_title')
    
    episodes = (
        df.groupby(group_cols)["dialogue"]
          .apply(lambda x: " ".join(x.astype(str)))
          .reset_index()
    )
    
    return episodes


def search_episodes(df, query, top_k=5):
    """
    Search episodes using query with TF-IDF ranking
    
    Args:
        df: Episode dataframe with 'dialogue' column
        query: Search query string
        top_k: Number of top results to return (default: 5)
        
    Returns:
        DataFrame with top matching episodes, sorted by relevance
        
    Examples:
        >>> episodes = load_show_data('friends')
        >>> results = search_episodes(episodes, "coffee&shop", top_k=3)
    """
    parsed = parse_query(query)
    qtype = parsed["type"]
    value = parsed["value"]

    text = df["dialogue"].str.lower()

    # Filter based on query type
    if qtype == "phrase":
        mask = text.str.contains(value, regex=False, na=False)
        query_for_tfidf = value

    elif qtype == "and":
        mask = text.apply(lambda t: all(v in str(t) for v in value))
        query_for_tfidf = " ".join(value)

    elif qtype == "or":
        mask = text.apply(lambda t: any(v in str(t) for v in value))
        query_for_tfidf = " ".join(value)

    else:
        raise ValueError(f"Unknown query type: {qtype}")

    # Get matching episodes
    sub = df[mask].copy()
    
    if sub.empty:
        return sub
    
    # Rank by BM25 similarity
    try:
        # Tokenization for BM25
        corpus = [doc.lower().split() for doc in sub["dialogue"]]
        bm25 = BM25Okapi(corpus)

        # Tokenize query
        q_tokens = query_for_tfidf.lower().split()

        # Compute BM25 scores
        scores = bm25.get_scores(q_tokens)

        # Add scores to DataFrame
        sub['relevance_score'] = scores

        if 'episode_title' in sub.columns:
            kw = query_for_tfidf.lower()
            title_mask = sub['episode_title'].str.lower().str.contains(rf'\b{kw}\b', na=False)
            
            # give a small bonus to episodes with keyword-in-title
            sub.loc[title_mask, 'relevance_score'] += 0.30

        # Sort results
        sub = sub.sort_values('relevance_score', ascending=False)

    except Exception:
        pass


    
    return sub.head(top_k)


def search_all_shows(query, top_k=5):
    """
    Search across all available TV shows
    
    Args:
        query: Search query string
        top_k: Number of top results per show (default: 5)
        
    Returns:
        List of tuples: (show_key, show_name, results_df)
    """
    all_results = []
    
    for show_key in DATASETS.keys():
        show_name = SHOW_NAMES[show_key]
        
        try:
            episodes = load_show_data(show_key)
            results = search_episodes(episodes, query, top_k)
            
            if not results.empty:
                results['show'] = show_name
                results['show_key'] = show_key
                all_results.append((show_key, show_name, results))
                
        except Exception:
            # Skip shows that fail to load or search
            pass
    
    return all_results

