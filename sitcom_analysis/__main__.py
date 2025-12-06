"""
Unified CLI for Sitcom Analysis

This provides a single entry point that delegates to both packages
and provides combined functionality.
"""

import sys
import os


def print_usage():
    """Print usage information."""
    print("""
Sitcom Analysis - Unified CLI

USAGE:
    sitcom <command> [args...]

COMMANDS:
    
    Information Retrieval:
        search <show> "<query>"          Search a specific show
        search-all "<query>"             Search across all shows
        list                             List available shows
        
    MBTI Prediction:
        mbti                             Interactive character MBTI analysis
        
    Combined Features:
        analyze <show> "<query>"         Search + show character personalities
        character <show> <name>          Get character's MBTI profile
        character <show> <name> "<query>" Character MBTI + search their dialogue
        personalities <show>             List MBTI for all major characters

    Style Transfer:
        style <character> "<text>"       Rewrite text to match character's speaking style

OPTIONS:
    --top N              Number of results (default: 5)
    --title-weight N     Title boost weight (default: 2)

EXAMPLES:
    sitcom search friends "coffee shop"
    sitcom search-all "wedding"
    sitcom mbti
    sitcom analyze friends "thanksgiving"
    sitcom character friends "Ross Geller"
    sitcom character friends Ross "dinosaurs"
    sitcom personalities friends
    sitcom style Joey "How are you doing today?"
""")


def cmd_search(args):
    """Delegate to information_retrieval package."""
    from information_retrieval.__main__ import cmd_search as ir_search
    ir_search(args)


def cmd_search_all(args):
    """Delegate to information_retrieval package."""
    from information_retrieval.__main__ import cmd_search_all as ir_search_all
    ir_search_all(args)


def cmd_list(args):
    """Delegate to information_retrieval package."""
    from information_retrieval.__main__ import cmd_list as ir_list
    ir_list(args)


def cmd_mbti(args):
    """Delegate to mbti_prediction package."""
    from mbti_prediction.__main__ import main as mbti_main
    mbti_main()


def cmd_analyze(args):
    """Combined feature: Search + show character MBTI."""
    if len(args) < 2:
        print("Error: 'analyze' requires <show> and <query>")
        print("Example: sitcom analyze friends 'wedding'")
        sys.exit(1)
    
    show = args[0].lower()
    query = args[1]
    
    # Parse optional params
    top_k = 5
    title_weight = 2
    
    if "--top" in args:
        idx = args.index("--top")
        top_k = int(args[idx + 1])
    
    if "--title-weight" in args:
        idx = args.index("--title-weight")
        title_weight = float(args[idx + 1])
    
    from .combined_features import search_with_character_info
    from information_retrieval import SHOW_NAMES
    
    print(f"\n{'='*70}")
    print(f"COMBINED ANALYSIS: {SHOW_NAMES.get(show, show)}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    print("Searching episodes...")
    results, character_mbti = search_with_character_info(
        show, query, top_k, title_weight, analyze_characters=True
    )
    
    # Display search results
    if results.empty:
        print("No episodes found matching your query.")
    else:
        print(f"\n--- Top {len(results)} Episodes ---\n")
        for idx, row in results.iterrows():
            season = row.get('season', '?')
            episode = row.get('episode', '?')
            title = row.get('episode_title', 'Unknown')
            
            print(f"Season {season}, Episode {episode}: {title}")
            print(f"   Snippet:   {row['snippet_highlight']}\n")
    
    # Display character MBTI if available
    if character_mbti:
        print(f"\n--- Character Personalities ({show.upper()}) ---\n")
        for char, mbti in sorted(character_mbti.items())[:10]:
            print(f"  {char:<25} {mbti}")
        print()
    
    print(f"{'='*70}\n")


def cmd_character(args):
    """Analyze character's MBTI and optionally search their dialogue."""
    if len(args) < 2:
        print("Error: 'character' requires <show> and <character_name>")
        print("Example: sitcom character friends 'Ross Geller'")
        print("         sitcom character friends Ross 'dinosaurs'")
        sys.exit(1)
    
    show = args[0].lower()
    character = args[1]
    query = args[2] if len(args) > 2 else None
    
    top_k = 5
    if "--top" in args:
        idx = args.index("--top")
        top_k = int(args[idx + 1])
    
    from .combined_features import analyze_character_moments
    from information_retrieval import SHOW_NAMES
    from mbti_prediction import SHOW_DISPLAY_NAMES
    
    show_name = SHOW_DISPLAY_NAMES.get(show, SHOW_NAMES.get(show, show))
    
    print(f"\n{'='*70}")
    print(f"CHARACTER ANALYSIS: {character} ({show_name})")
    print(f"{'='*70}\n")
    
    result = analyze_character_moments(show, character, query, top_k)
    
    if not result.get('success'):
        print(f"[ERROR] {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    # Display MBTI
    print(f"Predicted MBTI: {result['mbti']}")
    print(f"Total lines: {result['total_lines']}")
    
    print(f"\nPer-dimension probabilities:")
    for dim, probs in result['probabilities'].items():
        items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        print(f"  {dim}: {items[0][0]} ({items[0][1]:.2%}) vs {items[1][0]} ({items[1][1]:.2%})")
    
    # Display representative quotes
    if result['top_quotes']:
        print(f"\n--- Top {len(result['top_quotes'])} Representative Quotes ---\n")
        for i, q in enumerate(result['top_quotes'], start=1):
            print(f"[{i}] S{q['season']}E{q['episode']} - {q['episode_title']}")
            snippet = q['dialogue_raw']
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            print(f"    \"{snippet}\"\n")
    
    # Display matching moments if query was provided
    if query and 'matching_moments' in result:
        matches = result['matching_moments']
        total = result.get('total_matches', len(matches))
        
        print(f"\n--- Dialogue Matching '{query}' ({total} total) ---\n")
        if not matches:
            print("No matching dialogue found.\n")
        else:
            for i, m in enumerate(matches, start=1):
                print(f"[{i}] S{m['season']}E{m['episode']} - {m['episode_title']}")
                print(f"    {m['snippet']}\n")
    
    print(f"{'='*70}\n")


def cmd_personalities(args):
    """Show MBTI for all major characters in a show."""
    if len(args) < 1:
        print("Error: 'personalities' requires <show>")
        print("Example: sitcom personalities friends")
        sys.exit(1)
    
    show = args[0].lower()
    limit = 10
    
    if "--limit" in args:
        idx = args.index("--limit")
        limit = int(args[idx + 1])
    
    from .combined_features import get_show_character_personalities
    from information_retrieval import SHOW_NAMES
    from mbti_prediction import SHOW_DISPLAY_NAMES
    
    show_name = SHOW_DISPLAY_NAMES.get(show, SHOW_NAMES.get(show, show))
    
    print(f"\n{'='*70}")
    print(f"CHARACTER PERSONALITIES: {show_name}")
    print(f"{'='*70}\n")
    
    print(f"Analyzing top {limit} characters (by number of lines)...\n")
    
    results = get_show_character_personalities(show, limit)
    
    if not results:
        print("No characters found or analyzed.")
    else:
        print(f"{'Character':<30} {'MBTI'}")
        print("-" * 40)
        for char, mbti in results.items():
            print(f"{char:<30} {mbti}")
    
    print(f"\n{'='*70}\n")


# ----------------------------------------------------------------------
# NEW STYLE TRANSFER COMMAND
# ----------------------------------------------------------------------
def cmd_style(args):
    """Rewrite text in the speaking style of a sitcom character."""
    if len(args) < 2:
        print("Error: 'style' requires <character> \"<text>\"")
        print("Example: sitcom style Joey \"How are you doing?\"")
        sys.exit(1)

    character = args[0]
    text = args[1]

    from .style_transfer_module import run_style_transfer

    print(f"\n{'='*70}")
    print(f"STYLE TRANSFER: {character}")
    print(f"{'='*70}\n")

    try:
        output = run_style_transfer(text, character)
        print(output)
    except Exception as e:
        print(f"[ERROR] {e}")

    print(f"\n{'='*70}\n")


# ----------------------------------------------------------------------


def main():
    """Main entry point for unified CLI."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    # Information Retrieval commands (delegate)
    if command == "search":
        cmd_search(args)
    elif command == "search-all":
        cmd_search_all(args)
    elif command == "list":
        cmd_list(args)
    
    # MBTI commands (delegate)
    elif command == "mbti":
        cmd_mbti(args)
    
    # Combined commands
    elif command == "analyze":
        cmd_analyze(args)
    elif command == "character":
        cmd_character(args)
    elif command == "personalities":
        cmd_personalities(args)

    # NEW STYLE COMMAND
    elif command == "style":
        cmd_style(args)
    
    # Help and unknown
    elif command in ["help", "--help", "-h"]:
        print_usage()
    else:
        print(f"Unknown command: {command}")
        print("Run 'sitcom help' for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
