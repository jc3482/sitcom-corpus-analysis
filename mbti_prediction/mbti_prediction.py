import os
import sys
import pandas as pd
import joblib
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from .data_processing import preprocess_text

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)

BUNDLE_PATH = os.path.join(BASE_DIR, "mbti_bundle.pkl")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "raw_data")

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

DIMENSION_COLS = ["EI", "SN", "TF", "JP"]


class MBTIMultiBertModel(nn.Module):

    def __init__(self, model_name: str, num_labels: int = 4, dropout: float = 0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

def load_bundle(bundle_path: str = BUNDLE_PATH):

    if not os.path.exists(bundle_path):
        print(f"[ERROR] Model bundle not found at: {bundle_path}")
        sys.exit(1)

    bundle = joblib.load(bundle_path)

    tokenizer_name = bundle["tokenizer_name"]
    max_len        = bundle["max_len"]
    state_dict     = bundle["state_dict"]
    label_mapping  = bundle["label_mapping"]
    preproc_info   = bundle.get("preprocessing_info", {})

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MBTIMultiBertModel(model_name=tokenizer_name, num_labels=4)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return tokenizer, model, label_mapping, preproc_info, max_len, device


def load_all_dialogues() -> pd.DataFrame:

    frames = []

    for show_key, path in SHOW_FILES.items():
        if not os.path.exists(path):
            print(f"[WARNING] File not found for {show_key}: {path}")
            continue

        df = pd.read_csv(path)

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

    df_all = pd.concat(frames, ignore_index=True)

    print("[INFO] Preprocessing dialogues (this may take a moment)...")
    df_all = preprocess_text(
        df_all,
        column_name="dialogue",
        remove_mbti_words=False
    )

    return df_all


def _text_to_logits(tokenizer, model, device, text: str, max_len: int):

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)  # (1, 4)

    return logits.squeeze(0)  # (4,)


def predict_mbti_for_character(
    tokenizer,
    model,
    label_mapping,
    df_all: pd.DataFrame,
    show_key: str,
    character_name: str,
    max_len: int,
    device,
):

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

    logits = _text_to_logits(tokenizer, model, device, text_all, max_len)
    probs  = torch.sigmoid(logits).cpu().numpy()

    dims = DIMENSION_COLS
    mbti_letters = []
    dim_probs = {}

    for i, dim in enumerate(dims):
        mapping = label_mapping[dim] 
        if 1 in mapping:
            map_1 = mapping[1]
            map_0 = mapping[0]
        else:
            map_1 = mapping["1"]
            map_0 = mapping["0"]

        p1 = float(probs[i])
        p0 = 1.0 - p1

        if p1 >= 0.5:
            chosen_letter = map_1
        else:
            chosen_letter = map_0

        mbti_letters.append(chosen_letter)
        dim_probs[dim] = {
            map_0: p0,
            map_1: p1,
        }


    mbti_str = "".join(mbti_letters)
    return mbti_str, dim_probs, df_char

def score_quotes_for_character(
    tokenizer,
    model,
    label_mapping,
    df_char: pd.DataFrame,
    mbti_str: str,
    max_len: int,
    device,
    top_k: int = 5,
    batch_size: int = 16,
):

    if df_char.empty:
        return []

    texts = df_char["dialogue"].astype(str).tolist()
    n = len(texts)

    dims = DIMENSION_COLS
    all_scores = []

    for start in range(0, n, batch_size):
        batch_texts = texts[start:start + batch_size]

        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)  # (b,4)
            probs  = torch.sigmoid(logits).cpu().numpy()  # (b,4)

        for row_probs in probs:
            per_dim_match = []
            for i, dim in enumerate(dims):
                mapping = label_mapping[dim]
                if 1 in mapping:
                    map_1 = mapping[1]
                    map_0 = mapping[0]
                else:
                    map_1 = mapping["1"]
                    map_0 = mapping["0"]

                p1 = float(row_probs[i])
                p0 = 1.0 - p1

                target_letter = mbti_str[i]

                if target_letter == map_1:
                    per_dim_match.append(p1)
                else:
                    per_dim_match.append(p0)

            all_scores.append(float(np.mean(per_dim_match)))

    df_char = df_char.copy()
    df_char["mbti_match_score"] = all_scores

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
