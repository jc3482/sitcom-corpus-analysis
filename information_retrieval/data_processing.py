# information_retrieval/data_processing.py

import re
import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
STOP_WORDS = set(stopwords.words("english"))

SITCOM_NAMES = {
    n.lower() for n in [
        "Sheldon","Cooper","Leonard","Hofstadter","Penny","Howard","Wolowitz","Rajesh","Koothrappali",
        "Amy","Farrah","Fowler","Bernadette","Rostenkowski","Wolowitz",
        "Rachel","Green","Monica","Geller","Phoebe","Buffay","Ross",
        "Chandler","Bing","Joey","Tribbiani",
        "Jay","Pritchett","Gloria","Delgado","Phil","Dunphy","Claire",
        "Mitchell","Cameron","Lily","Manny",
        "Jerry","Seinfeld","George","Costanza","Elaine","Benes","Kramer",
        "Michael","Scott","Dwight","Schrute","Jim","Halpert","Pam",
        "Beesly","Kevin","Malone","Stanley","Hudson","Kelly","Kapoor"
    ]
}

# ---------------------------------------------------------
# 1. normalize
# ---------------------------------------------------------
def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'https?:\/\/\S+', '', text)
    text = text.replace("|", " ")
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([a-z])\1{2,}', r'\1\1', text)
    text = re.sub(r'\b\w{30,}\b', ' ', text)
    return text

# ---------------------------------------------------------
# 2. tokenize for BM25
# ---------------------------------------------------------
def tokenize(text):
    doc = nlp(text)
    tokens = []

    for tok in doc:
        if not tok.is_alpha:
            continue

        lemma = tok.lemma_.lower()

        if lemma in STOP_WORDS:
            continue
        if lemma in SITCOM_NAMES:
            continue

        tokens.append(lemma)

    return tokens

# ---------------------------------------------------------
# 3. pipeline
# ---------------------------------------------------------
def preprocess_for_ir(text):
    norm = normalize_text(text)
    return tokenize(norm)

# ---------------------------------------------------------
# 4. title boosting
# ---------------------------------------------------------
def combine_dialogue_and_title(dialogue, title, weight=2):
    if not isinstance(dialogue, str):
        dialogue = ""
    if not isinstance(title, str):
        title = ""

    boosted = (" " + title) * int(weight)
    return dialogue + boosted
