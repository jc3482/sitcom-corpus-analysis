"""
Main.py for TV Show Information Retrieval System
"""

import sys
import os
import pandas as pd

from .search import (
    load_show_data,
    search_episodes,
    search_all_shows,
    DATASETS,
    SHOW_NAMES
)


# =========================================================
# Display functions
# =========================================================
def display_results(results, show, query):
    show_name = SHOW_NAMES.get(show, show)
    
    print(f"\n{'='*70}")
    print(f"Search Results: {show_name}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    if results.empty:
        print("No episodes found matching your query.")
        return
    
    for idx, row in results.iterrows():
        season = row.get('season', '?')
        episode = row.get('episode', '?')
        title = row.get('episode_title', 'Unknown')
        
        print(f"Season {season}, Episode {episode}: {title}")

        # print(f"   Relevance: {row['relevance_score']:.3f}")
        print(f"   Snippet:   {row['snippet_highlight']}\n")
    
    print(f"{'='*70}")


def display_all_results(all_results, query, top_k):
    if not all_results:
        print("\nNo episodes found across any show.")
        return
    
    print(f"\n{'='*70}")
    print(f"RESULTS ACROSS ALL SHOWS")
    print(f"Query: {query}")
    print(f"{'='*70}\n")

    combined = pd.concat([r for _, _, r in all_results], ignore_index=True)

    if 'relevance_score' in combined.columns:
        combined = combined.sort_values('relevance_score', ascending=False)

    for idx, row in combined.head(top_k * len(DATASETS)).iterrows():
        print(f"{row['show']}: S{row['season']}, E{row['episode']}  — {row['episode_title']}")
        # print(f"   Relevance: {row['relevance_score']:.3f}")
        print(f"   Snippet:   {row['snippet_highlight']}\n")
    
    print(f"{'='*70}")


# =========================================================
# Parse CLI options
# =========================================================
def parse_optional_params(args):
    top_k = 5
    title_weight = 2

    if "--top" in args:
        idx = args.index("--top")
        top_k = int(args[idx + 1])

    if "--title-weight" in args:
        idx = args.index("--title-weight")
        title_weight = float(args[idx + 1])

    return top_k, title_weight


# =========================================================
# Commands
# =========================================================
def cmd_search(args):
    if len(args) < 2:
        print("Error: 'search' requires <show> and <query>")
        sys.exit(1)
    
    show = args[0].lower()
    query = args[1]
    top_k, title_weight = parse_optional_params(args)
    
    print(f"Loading {SHOW_NAMES[show]}...")
    episodes = load_show_data(show)
    
    results = search_episodes(episodes, query, top_k, title_weight)
    display_results(results, show, query)


def cmd_search_all(args):
    if len(args) < 1:
        print("Error: 'search-all' requires <query>")
        sys.exit(1)
    
    query = args[0]
    top_k, title_weight = parse_optional_params(args)

    print(f"\n{'='*70}")
    print(f"SEARCHING ALL SHOWS…")
    print(f"{'='*70}\n")

    all_results = search_all_shows(query, top_k, title_weight)
    display_all_results(all_results, query, top_k)


def cmd_list(args):
    print("\nAVAILABLE TV SHOWS:\n")
    for key, name in SHOW_NAMES.items():
        print(f"  {key:<15}  {name}")
    print()


# =========================================================
# Main entry point
# =========================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python -m information_retrieval <command>")
        sys.exit(1)
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    if command == "search":
        cmd_search(args)
    elif command == "search-all":
        cmd_search_all(args)
    elif command == "list":
        cmd_list(args)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
