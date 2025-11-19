"""
Command-line interface for Information Retrieval package

USAGE:
    python -m information_retrieval search <show> "<query>" [--top N]
    python -m information_retrieval search-all "<query>" [--top N]
    python -m information_retrieval list
    
SHOWS:
    friends, tbbt, seinfeld, office, modern_family
    
QUERY SYNTAX:
    "wedding cake"      - Exact phrase (both words together)
    "wedding&cake"      - AND (both words must appear)
    "wedding/cake"      - OR (either word must appear)
    
EXAMPLES:
    python -m information_retrieval search friends "coffee shop"
    python -m information_retrieval search tbbt "physics&quantum" --top 10
    python -m information_retrieval search-all "wedding" --top 5
    python -m information_retrieval list
"""

import sys
import pandas as pd
from rank_bm25 import BM25Okapi

from .search import (
    load_show_data,
    search_episodes,
    search_all_shows,
    DATASETS,
    SHOW_NAMES
)


def display_results(results, show, query):
    """Display search results for a single show"""
    show_name = SHOW_NAMES.get(show, show)
    
    print(f"\n{'='*70}")
    print(f"Search Results: {show_name}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    if results.empty:
        print("No episodes found matching your query.")
        print("\nTry:")
        print("   - Using OR (/) for more results: 'word1/word2'")
        print("   - Checking spelling")
        print("   - Using more general terms")
        return
    
    print(f"Found {len(results)} matching episode(s)\n")
    
    # Display results
    for idx, row in results.iterrows():
        season = row.get('season', '?')
        episode = row.get('episode', '?')
        title = row.get('episode_title', 'Unknown')
        
        print(f"Season {season}, Episode {episode}: {title}")
        
        if 'relevance_score' in row:
            score = row['relevance_score']
            print(f"   Relevance: {score:.3f}")
        
        # Show snippet of dialogue
        dialogue = str(row.get('dialogue', ''))
        if len(dialogue) > 200:
            dialogue = dialogue[:200] + "..."
        print(f"   Preview: {dialogue}\n")
    
    print(f"{'='*70}")


def display_all_results(all_results, query, top_k):
    """Display combined results from all shows"""
    if not all_results:
        print(f"\n{'='*70}")
        print("No episodes found in any show")
        print(f"{'='*70}")
        return
    
    print(f"\n{'='*70}")
    print(f"RESULTS ACROSS ALL SHOWS")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    # Combine all results
    combined = pd.concat([results for _, _, results in all_results], ignore_index=True)
    
    # Sort by relevance if available
    if 'relevance_score' in combined.columns:
        combined = combined.sort_values('relevance_score', ascending=False)
    
    # Display top results
    print(f"Top {min(len(combined), top_k * len(DATASETS))} results:\n")
    
    for idx, row in combined.head(top_k * len(DATASETS)).iterrows():
        show = row.get('show', 'Unknown')
        season = row.get('season', '?')
        episode = row.get('episode', '?')
        title = row.get('episode_title', 'Unknown')
        
        print(f"{show}")
        print(f"   Season {season}, Episode {episode}: {title}")
        
        if 'relevance_score' in row:
            score = row['relevance_score']
            print(f"   Relevance: {score:.3f}")
        
        # Show snippet
        dialogue = str(row.get('dialogue', ''))
        if len(dialogue) > 150:
            dialogue = dialogue[:150] + "..."
        print(f"   Preview: {dialogue}\n")
    
    # Summary by show
    print(f"{'='*70}")
    print("SUMMARY BY SHOW")
    print(f"{'='*70}\n")
    
    show_counts = combined['show'].value_counts()
    for show, count in show_counts.items():
        print(f"  {show}: {count} matching episode(s)")
    
    print(f"\n{'='*70}")


def cmd_search(args):
    """Handle 'search' command"""
    if len(args) < 2:
        print("Error: 'search' requires <show> and <query>")
        print("Usage: python -m information_retrieval search <show> \"<query>\" [--top N]")
        print(f"\nAvailable shows: {', '.join(DATASETS.keys())}")
        sys.exit(1)
    
    show = args[0].lower()
    query = args[1]
    
    # Parse --top argument
    top_k = 5
    if '--top' in args:
        try:
            idx = args.index('--top')
            top_k = int(args[idx + 1])
        except (IndexError, ValueError):
            print("Error: --top requires a number")
            sys.exit(1)
    
    # Validate show
    if show not in DATASETS:
        print(f"Unknown show: {show}")
        print(f"Available shows: {', '.join(DATASETS.keys())}")
        sys.exit(1)
    
    # Load and search
    print(f"Loading {SHOW_NAMES[show]} dataset...")
    try:
        episodes = load_show_data(show)
        print(f"Loaded {len(episodes)} episodes")
        print(f"Searching for: {query}")
        
        results = search_episodes(episodes, query, top_k)
        display_results(results, show, query)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_search_all(args):
    """Handle 'search-all' command"""
    if len(args) < 1:
        print("Error: 'search-all' requires <query>")
        print("Usage: python -m information_retrieval search-all \"<query>\" [--top N]")
        sys.exit(1)
    
    query = args[0]
    
    # Parse --top argument
    top_k = 5
    if '--top' in args:
        try:
            idx = args.index('--top')
            top_k = int(args[idx + 1])
        except (IndexError, ValueError):
            print("Error: --top requires a number")
            sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"SEARCHING ALL SHOWS")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    # Search each show
    for show_key in DATASETS.keys():
        show_name = SHOW_NAMES[show_key]
        print(f"Searching {show_name}...", end=" ")
        try:
            episodes = load_show_data(show_key)
            results = search_episodes(episodes, query, top_k)
            if not results.empty:
                print(f"Found {len(results)} episodes")
            else:
                print("No matches")
        except Exception as e:
            print(f"Error: {e}")
    
    # Get and display all results
    all_results = search_all_shows(query, top_k)
    display_all_results(all_results, query, top_k)


def cmd_list(args):
    """Handle 'list' command - list available shows"""
    print("\n" + "="*70)
    print("AVAILABLE TV SHOWS")
    print("="*70 + "\n")
    
    for key, name in SHOW_NAMES.items():
        filepath = DATASETS[key]
        status = "[OK]" if __import__('os').path.exists(filepath) else "[MISSING]"
        print(f"  {status} {key:<15} - {name}")
        print(f"        File: {filepath}")
    
    print("\n" + "="*70)


def main():
    """Main entry point for CLI"""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    if command == 'search':
        cmd_search(args)
    elif command == 'search-all':
        cmd_search_all(args)
    elif command == 'list':
        cmd_list(args)
    else:
        print(f"Unknown command: {command}")
        print("\nAvailable commands: search, search-all, list")
        print("Run 'python -m information_retrieval' for help")
        sys.exit(1)


if __name__ == '__main__':
    main()

