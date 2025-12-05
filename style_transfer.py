import pandas as pd
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import torch
import os
import pickle

# ============================================================
# 1. Load embedding model
# ============================================================
print("Loading embedding model...")
embedder = SentenceTransformer("all-mpnet-base-v2")

# ============================================================
# 2. Initialize OpenAI Client
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your_key'")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# 3. Load sitcom datasets
# ============================================================
def load_dialogues(csv_path):
    df = pd.read_csv(csv_path)
    df = df[['character', 'dialogue']].dropna()
    return df

print("Loading sitcom datasets...")

df_friends = load_dialogues("raw_data/friends_dialogues.csv")
df_tbbt = load_dialogues("raw_data/tbbt_dialogues.csv")
df_office = load_dialogues("raw_data/the_office.csv")
df_modern = load_dialogues("raw_data/modern_family_scripts.csv")
df_seinfeld = load_dialogues("raw_data/seinfeld_scripts.csv")

all_data = pd.concat([df_friends, df_tbbt, df_office, df_modern, df_seinfeld])

# ============================================================
# 4. Build character dictionaries
# ============================================================
character_dialogues = {}

for char, group in all_data.groupby("character"):

    if not isinstance(char, str):
        continue
    if len(char.strip()) < 3:
        continue

    lines = [str(x) for x in group["dialogue"].values if isinstance(x, str) and len(x) > 5]

    if len(lines) >= 120:  # Only main characters
        character_dialogues[char] = lines

character_list = sorted(character_dialogues.keys())

# ============================================================
# 5. Precompute OR Load Cached Embeddings  (FAST MODE)
# ============================================================
CACHE_PATH = "embedding_cache.pkl"

if os.path.exists(CACHE_PATH):
    print("Loading cached embeddings...")
    with open(CACHE_PATH, "rb") as f:
        character_embeddings = pickle.load(f)
else:
    print("Computing embeddings (first run only, takes ~3 min)...")
    character_embeddings = {
        char: embedder.encode(lines, convert_to_tensor=True)
        for char, lines in character_dialogues.items()
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(character_embeddings, f)
    print("Embeddings cached!")

# ============================================================
# 6. Style Transfer using OpenAI GPT-4o-mini
# ============================================================
def rewrite_in_style(input_sentence, character):

    corpus = character_dialogues[character]
    embeddings = character_embeddings[character]

    # Find top-5 similar lines
    input_emb = embedder.encode(input_sentence, convert_to_tensor=True)
    scores = util.cos_sim(input_emb, embeddings)[0]
    top_idx = torch.topk(scores, 5).indices.tolist()

    examples = [corpus[i] for i in top_idx]

    prompt = f"""
Rewrite the following sentence in the speaking style of **{character}**.

Here are 5 authentic examples of {character}'s speech patterns:
1. {examples[0]}
2. {examples[1]}
3. {examples[2]}
4. {examples[3]}
5. {examples[4]}

Guidelines:
- Keep the meaning.
- Match tone, vocabulary, pacing, and quirks.
- Output must sound like a genuine sitcom line spoken by {character}.

Sentence: "{input_sentence}"

Rewritten in {character}'s style:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )

    return response.choices[0].message.content.strip()

# ============================================================
# 7. Interactive CLI
# ============================================================
if __name__ == "__main__":

    print("\n========== Sitcom Character Style Transfer (OpenAI Version, FAST) ==========\n")

    sentence = input("Enter a sentence you want rewritten:\n> ")

    print("\nChoose a character:\n")
    for idx, c in enumerate(character_list):
        print(f"{idx+1}. {c}")

    choice = int(input("\nEnter number: ")) - 1

    if choice < 0 or choice >= len(character_list):
        print("Invalid selection.")
        exit()

    character = character_list[choice]

    print(f"\nRewriting in the style of **{character}**...\n")

    output = rewrite_in_style(sentence, character)

    print("===== Styled Output =====\n")
    print(output)
    print("\n==========================\n")
