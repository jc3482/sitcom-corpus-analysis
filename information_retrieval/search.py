"""
BM25 Search System with preprocessing + caching + snippet + highlight
"""

import os
import re
import pandas as pd
from rank_bm25 import BM25Okapi

from .cache_utils import save_pickle, load_pickle, exists
from .data_processing import preprocess_for_ir, combine_dialogue_and_title


# =========================================================
# Query Parsing
# =========================================================
def parse_query(query):
    query = query.strip()

    if "&" not in query and "/" not in query:
        return {"type": "phrase", "value": query.lower()}

    if "&" in query:
        parts = [p.strip().lower() for p in query.split("&") if p.strip()]
        return {"type": "and", "value": parts}

    if "/" in query:
        parts = [p.strip().lower() for p in query.split("/") if p.strip()]
        return {"type": "or", "value": parts}


# =========================================================
# Highlight Utility
# =========================================================
def highlight(text, query):
    """Highlight all query terms using ANSI yellow."""
    if not isinstance(text, str):
        return ""

    # Support AND/OR multi word highlight
    parts = re.split(r"[&/]", query.lower())
    parts = [p.strip() for p in parts if p.strip()]
    parts = list(dict.fromkeys(parts))  # dedupe

    for p in parts:
        pattern = re.escape(p)
        text = re.sub(
            f"({pattern})",
            r"\033[1;33m\1\033[0m",
            text,
            flags=re.IGNORECASE
        )
    return text


# =========================================================
# Snippet Extraction
# =========================================================
def extract_best_snippet(dialogue, query, max_len=240):
    if not isinstance(dialogue, str):
        return ""

    # Normalize input for matching
    q = query.lower()
    q_parts = re.split(r"[&/]", q)
    q_parts = [p.strip() for p in q_parts if p.strip()]

    # -------------------------
    # 1. Line-level split 
    # -------------------------
    lines = re.split(r'[\r\n]+', dialogue)
    lines = [ln.strip() for ln in lines if ln.strip()]

    # find exact hit lines
    hit_idx = [
        i for i, ln in enumerate(lines)
        if any(qp in ln.lower() for qp in q_parts)
    ]

    # -------------------------
    # 2. extract context snippet
    # -------------------------
    if hit_idx:
        i = hit_idx[0]

        prev_line = lines[i-1] if i > 0 else None
        hit_line = lines[i]
        next_line = lines[i+1] if i < len(lines)-1 else None

        snippet_parts = []
        if prev_line:
            snippet_parts.append(prev_line)
        snippet_parts.append(hit_line)
        if next_line:
            snippet_parts.append(next_line)

        snippet = " ".join(snippet_parts).strip()

        # truncate to max_len
        if len(snippet) > max_len:
            snippet = snippet[:max_len] + "..."

        return snippet

    # -------------------------
    # 3. Fallback Step: punctuation-based sentence split
    # -------------------------
    sentences = re.split(r'(?<=[.!?])\s+', dialogue)
    sentences = [s.strip() for s in sentences if s.strip()]

    hit_sentences = [
        s for s in sentences
        if any(qp in s.lower() for qp in q_parts)
    ]

    if hit_sentences:
        best = min(hit_sentences, key=len)
        if len(best) > max_len:
            best = best[:max_len] + "..."
        return best

    # -------------------------
    # 4. Fallback 2: return first line
    # -------------------------
    fallback = lines[0] if lines else dialogue
    if len(fallback) > max_len:
        fallback = fallback[:max_len] + "..."
    return fallback


# =========================================================
# Loading Dataset 
# =========================================================
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


# =========================================================
# Load dataset + group by episode
# =========================================================
def load_show_data(show):
    if show not in DATASETS:
        raise ValueError(f"Unknown show: {show}")

    path = DATASETS[show]
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "dialogue" not in df.columns:
        raise ValueError("Dataset missing 'dialogue' column")

    group_cols = ["season", "episode"]
    if "episode_title" in df.columns:
        group_cols.append("episode_title")

    # NOTE: use "\n".join so each original line remains a line → snippet works
    episodes = (
        df.groupby(group_cols)["dialogue"]
          .apply(lambda x: "\n".join(x.astype(str)))
          .reset_index()
    )

    episodes["dialogue_raw"] = episodes["dialogue"]
    episodes["show_key"] = show

    return episodes


# =========================================================
# BUILD BM25 
# =========================================================
def build_bm25_corpus(full_episodes, show_key, title_weight=2):

    cache_tokens = f"{show_key}_tokens.pkl"
    cache_bm25   = f"{show_key}_bm25.pkl"
    cache_meta   = f"{show_key}_meta.pkl"

    # ----- loading cache -----
    if exists(cache_bm25) and exists(cache_tokens) and exists(cache_meta):
        meta = load_pickle(cache_meta)
        if (
            meta.get("num_episodes") == len(full_episodes)
            and meta.get("title_weight") == title_weight
        ):
            tokens = load_pickle(cache_tokens)
            bm25   = load_pickle(cache_bm25)
            full_episodes["tokens"] = tokens
            return bm25, full_episodes

    # ----- Build BM25 -----
    if "episode_title" in full_episodes.columns:
        full_episodes["combined_text"] = full_episodes.apply(
            lambda row: combine_dialogue_and_title(
                row["dialogue_raw"],
                row.get("episode_title", ""),
                weight=title_weight
            ),
            axis=1
        )
    else:
        full_episodes["combined_text"] = full_episodes["dialogue_raw"]

    full_episodes["tokens"] = full_episodes["combined_text"].apply(preprocess_for_ir)
    corpus = full_episodes["tokens"].tolist()

    bm25 = BM25Okapi(corpus)

    # save cache (now with title_weight)
    save_pickle(cache_tokens, full_episodes["tokens"])
    save_pickle(cache_bm25, bm25)
    save_pickle(cache_meta, {
        "num_episodes": len(full_episodes),
        "title_weight": title_weight,
    })

    return bm25, full_episodes


# =========================================================
# Search episodes  (scoring logic)
# =========================================================
def search_episodes(df, query, top_k=5, title_weight=2):

    show_key = df["show_key"].iloc[0]

    bm25, full_df = build_bm25_corpus(
        df.copy(),
        show_key=show_key,
        title_weight=title_weight
    )

    parsed = parse_query(query)
    qtype = parsed["type"]
    value = parsed["value"]

    # ---- Boolean filtering on both dialogue & title ----
    raw_text = (
        full_df["dialogue_raw"].fillna("") + " " +
        full_df.get("episode_title", "").fillna("")
    ).str.lower()

    if qtype == "phrase":
        mask = raw_text.str.contains(value, regex=False, na=False)
    elif qtype == "and":
        mask = raw_text.apply(lambda t: all(v in t for v in value))
    elif qtype == "or":
        mask = raw_text.apply(lambda t: any(v in t for v in value))
    else:
        raise ValueError("Unknown query type")

    sub = full_df[mask].copy()
    if sub.empty:
        return sub

    # ---- BM25 scores over full corpus ----
    q_tokens = preprocess_for_ir(query)
    if not q_tokens:
        q_tokens = query.lower().split()

    all_scores = bm25.get_scores(q_tokens)
    sub["bm25_score"] = all_scores[sub.index]

    # ---- Enhance the weight of the title ----
    kw = query.lower()
    title_series = sub.get("episode_title", "").fillna("").str.lower()

    # Supporting multi-word query e.g."wedding&dress"
    q_parts = re.split(r"[&/]", kw)
    q_parts = [p.strip() for p in q_parts if p.strip()]

    def count_title_hits(title: str) -> int:
        if not title:
            return 0
        return sum(1 for qp in q_parts if qp in title)

    sub["title_hit_count"] = title_series.apply(count_title_hits)

    # Give title word a big boost
    sub["title_boost"] = sub["title_hit_count"] * (2.0 * title_weight)

    # FINAL SCORE:BM25 + title_boost
    sub["relevance_score"] = sub["bm25_score"] + sub["title_boost"]

    # ----- HIGHLIGHTING query terms in the snippet -----
    sub["snippet"] = sub["dialogue_raw"].apply(
        lambda t: extract_best_snippet(t, query)
    )
    sub["snippet_highlight"] = sub["snippet"].apply(
        lambda t: highlight(t, query)
    )

    sub = sub.sort_values("relevance_score", ascending=False)

    keep_cols = [
        "season", "episode", "episode_title",
        "relevance_score",
        "snippet_highlight"
    ]

    return sub[keep_cols].head(top_k)


# =========================================================
# Search all shows
# =========================================================
def search_all_shows(query, top_k=5, title_weight=2):
    results = []

    for show_key in DATASETS:
        show_name = SHOW_NAMES[show_key]

        try:
            episodes = load_show_data(show_key)
            res = search_episodes(episodes, query, top_k, title_weight)

            if not res.empty:
                res["show"] = show_name
                res["show_key"] = show_key
                results.append((show_key, show_name, res))

        except Exception as e:
            print(f"Error in {show_key}: {e}")

    return results
