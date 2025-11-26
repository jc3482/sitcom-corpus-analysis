import os
import sys
import pandas as pd
import joblib
import numpy as np

from .data_processing import preprocess_text

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "mbti_bundle.pkl")
RAW_DATA_DIR = "../raw_data"

SHOW_FILES = {
    "friends": os.path.join(RAW_DATA_DIR, "friends_dialogues.csv"),
    "modern_family": os.path.join(RAW_DATA_DIR, "modern_family_scripts.csv"),
    "seinfeld": os.path.join(RAW_DATA_DIR, "seinfeld_scripts.csv"),
    "tbbt": os.path.join(RAW_DATA_DIR, "tbbt_dialogues.csv"),
    "the_office": os.path.join(RAW_DATA_DIR, "the_office.csv"),
}

SHOW_DISPLAY_NAMES = {
    "friends": "Friends",
    "modern_family": "Modern Family",
    "seinfeld": "Seinfeld",
    "tbbt": "The Big Bang Theory",
    "the_office": "The Office (US)",
}

SHOW_ALIASES = {
    "friends": "friends",
    "friend": "friends",

    "modern family": "modern_family",
    "modern_family": "modern_family",
    "mf": "modern_family",

    "seinfeld": "seinfeld",

    "tbbt": "tbbt",
    "the big bang theory": "tbbt",
    "big bang theory": "tbbt",

    "the office": "the_office",
    "office": "the_office",
    "the_office": "the_office",
}


# Load bundle
def load_bundle(bundle_path: str = BUNDLE_PATH):
    """
    Load the trained MBTI bundle (tfidf + 4 models + metadata).

    By default, it loads 'mbti_bundle.pkl' from the same directory
    as this file (the mbti_prediction package folder).
    """
    if not os.path.exists(bundle_path):
        print(f"[ERROR] Model bundle not found at: {bundle_path}")
        sys.exit(1)

    bundle = joblib.load(bundle_path)
    tfidf = bundle["tfidf"]
    models = bundle["models"]
    label_mapping = bundle["label_mapping"]
    preproc_info = bundle.get("preprocessing_info", {})

    return tfidf, models, label_mapping, preproc_info



# Load & preprocess dialogues
def load_all_dialogues() -> pd.DataFrame:
    """
    Read all show CSVs, unify columns, and preprocess 'dialogue' using
    the SAME pipeline as training (no MBTI removal).
    """
    frames = []

    for show_key, path in SHOW_FILES.items():
        if not os.path.exists(path):
            print(f"[WARNING] File not found for {show_key}: {path}")
            continue

        df = pd.read_csv(path)

        # Expect columns: episode_title, season, episode, character, dialogue
        expected_cols = {"episode_title", "season", "episode", "character", "dialogue"}
        missing = expected_cols - set(df.columns)
        if missing:
            print(f"[WARNING] {show_key}: missing columns {missing}, skipping this show.")
            continue

        df["show_key"] = show_key
        df["show"] = SHOW_DISPLAY_NAMES.get(show_key, show_key)
        df["dialogue_raw"] = df["dialogue"].astype(str)

        frames.append(df)

    if not frames:
        print("[ERROR] No dialogue data loaded. Check your CSV paths.")
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)

    print("[INFO] Preprocessing dialogues (this may take a moment)...")
    all_df = preprocess_text(
        all_df,
        column_name="dialogue",
        remove_mbti_words=False
    )

    return all_df


# Prediction helpers
def predict_mbti_for_character(tfidf, models, label_mapping, df_all,
                               show_key: str, character_name: str):
    """
    Aggregate all cleaned dialogue for (show_key, character_name) and predict MBTI.
    Returns (mbti_str, per_dim_probs_dict, df_char).
    """
    mask = (
        (df_all["show_key"] == show_key) &
        (df_all["character"].str.lower() == character_name.lower())
    )
    df_char = df_all[mask].copy()

    if df_char.empty:
        return None, None, None

    text_all = " ".join(df_char["dialogue"].astype(str).tolist()).strip()
    if not text_all:
        return None, None, df_char

    X_char = tfidf.transform([text_all])

    dims = ["EI", "SN", "TF", "JP"]
    mbti_letters = []
    dim_probs = {}

    for dim in dims:
        model = models[dim]
        probs = model.predict_proba(X_char)[0]  # [p0, p1]

        mapping = label_mapping[dim]  # e.g. {1: 'E', 0: 'I'}
        idx = probs.argmax()
        letter = mapping[int(idx)]
        mbti_letters.append(letter)

        dim_probs[dim] = {
            mapping[0]: float(probs[0]),
            mapping[1]: float(probs[1]),
        }

    mbti_str = "".join(mbti_letters)
    return mbti_str, dim_probs, df_char


def score_quotes_for_character(tfidf, models, label_mapping,
                               df_char: pd.DataFrame, mbti_str: str,
                               top_k: int = 5):
    """
    For each dialogue line of the character, compute how strongly it supports
    the final MBTI letters; return top_k quotes with highest scores.
    """
    if df_char.empty:
        return []

    texts = df_char["dialogue"].astype(str).tolist()
    X_lines = tfidf.transform(texts)

    dims = ["EI", "SN", "TF", "JP"]
    per_dim_scores = {}

    for dim, letter in zip(dims, mbti_str):
        model = models[dim]
        probs = model.predict_proba(X_lines)  # (n_lines, 2)
        mapping = label_mapping[dim]

        idx_for_letter = None
        for idx_int, ltr in mapping.items():
            if ltr == letter:
                idx_for_letter = idx_int
                break
        if idx_for_letter is None:
            idx_for_letter = 1

        per_dim_scores[dim] = probs[:, idx_for_letter]

    scores_matrix = np.vstack([per_dim_scores[d] for d in dims])  # (4, n_lines)
    mean_scores = scores_matrix.mean(axis=0)

    df_char = df_char.copy()
    df_char["mbti_match_score"] = mean_scores

    df_top = df_char.sort_values("mbti_match_score", ascending=False).head(top_k)

    results = []
    for _, row in df_top.iterrows():
        results.append({
            "episode_title": row["episode_title"],
            "season": row["season"],
            "episode": row["episode"],
            "dialogue_raw": row["dialogue_raw"],
            "score": float(row["mbti_match_score"]),
        })

    return results