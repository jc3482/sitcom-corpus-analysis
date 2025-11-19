"""
Information Retrieval Package for TV Show Episode Search

This package provides tools to search across TV show dialogue datasets
using keyword queries with AND (&) / OR (/) operators.

Usage:
    python -m information_retrieval search <show> "<query>"
    python -m information_retrieval search-all "<query>"
"""

from .search import (
    search_episodes,
    load_show_data,
    parse_query,
    DATASETS,
    SHOW_NAMES
)

__version__ = '1.0.0'
__all__ = [
    'search_episodes',
    'load_show_data', 
    'parse_query',
    'DATASETS',
    'SHOW_NAMES'
]

