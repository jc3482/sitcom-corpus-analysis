import os
import pickle
from pathlib import Path
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# Global lazy-loaded resources
_embedder = None
_character_dialogues = None
_character_embeddings = None
_client = None

DATA_DIR = Path(__file__).parent.parent / "raw_data"
CACHE_PATH = Path(__file__).parent / "embedding_cache.pkl"


# ============================================================
# Lazy loading
# ============================================================
def load_resources():
    global _embedder, _character_dialogues, _character_embeddings, _client

    if _embedder is None:
        print("Loading embedding model...")
        _embedder = SentenceTransformer("all-mpnet-base-v2")

    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if key is None:
            raise ValueError("OPENAI_API_KEY is not set.")
        _client = OpenAI(api_key=key)

    if _character_dialogues is None:
        _character_dialogues = load_all_dialogues()

    if _character_embeddings is None:
        _character_embeddings = load_embeddings()

    return _embedder, _client, _character_dialogues, _character_embeddings


# ============================================================
# Load CSV datasets
# ============================================================
def load_csv(name):
    return pd.read_csv(DATA_DIR / name)[["character", "dialogue"]].dropna()


def load_all_dialogues():
    print("Loading sitcom datasets...")

    df = pd.concat([
        load_csv("friends_dialogues.csv"),
        load_csv("tbbt_dialogues.csv"),
        load_csv("the_office.csv"),
        load_csv("modern_family_scripts.csv"),
        load_csv("seinfeld_scripts.csv"),
    ])

    character_dialogues = {}
    for char, group in df.groupby("character"):
        if isinstance(char, str) and len(char) >= 3:
            lines = [x for x in group["dialogue"].astype(str).tolist() if len(x) > 5]
            if len(lines) >= 120:
                character_dialogues[char] = lines

    return character_dialogues


# ============================================================
# Load or compute embeddings
# ============================================================
def load_embeddings():
    global _embedder, _character_dialogues

    if CACHE_PATH.exists():
        print("Loading cached embeddings...")
        return pickle.load(open(CACHE_PATH, "rb"))

    print("Computing embeddings (first run only)...")
    embeddings = {
        char: _embedder.encode(lines, convert_to_tensor=True)
        for char, lines in _character_dialogues.items()
    }
    pickle.dump(embeddings, open(CACHE_PATH, "wb"))
    return embeddings


# ============================================================
# Style Transfer Function
# ============================================================
def run_style_transfer(input_sentence, character):
    embedder, client, character_dialogues, character_embeddings = load_resources()

    if character not in character_dialogues:
        raise ValueError(f"Character '{character}' not found.")

    corpus = character_dialogues[character]
    embeddings = character_embeddings[character]

    # Find top-5 similar lines
    input_emb = embedder.encode(input_sentence, convert_to_tensor=True)
    scores = util.cos_sim(input_emb, embeddings)[0]
    top_idx = torch.topk(scores, 5).indices.tolist()

    examples = [corpus[i] for i in top_idx]

    prompt = f"""
Rewrite the following sentence in the style of **{character}**.

Examples:
1. {examples[0]}
2. {examples[1]}
3. {examples[2]}
4. {examples[3]}
5. {examples[4]}

Sentence: "{input_sentence}"
Output:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    return response.choices[0].message.content.strip()


# ============================================================
# CLI ENTRYPOINT (for pyproject.toml)
# ============================================================
def main():
    """
    CLI command:
        sitcom-style CHARACTER "sentence"
        sitcom-style --list
    """
    import argparse

    parser = argparse.ArgumentParser(description="Sitcom Style Transfer CLI")
    parser.add_argument("character", type=str, nargs="?", help="Character name")
    parser.add_argument("text", type=str, nargs="?", help="Text to rewrite")
    parser.add_argument("--list", action="store_true", help="List all available characters")
    args = parser.parse_args()

    # Load resources to access character list
    _, _, character_dialogues, _ = load_resources()

    # Handle: sitcom-style --list
    if args.list:
        chars = sorted(character_dialogues.keys())
        print(f"Available characters ({len(chars)}):")
        for c in chars:
            print(" -", c)
        return

    # Missing arguments
    if not args.character or not args.text:
        print("Error: You must provide both CHARACTER and TEXT.")
        print('Example: sitcom-style Joey "How you doin?"')
        return

    # Run style transfer
    output = run_style_transfer(args.text, args.character)
    print(output)

