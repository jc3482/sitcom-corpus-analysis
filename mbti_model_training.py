import re
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import os
import joblib

lemmatizer = WordNetLemmatizer()

# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_sentence(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return ""
    tagged = pos_tag(tokens)
    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
    ]
    return " ".join(lemmas)


def preprocess_text(
    df,
    column_name,
    remove_mbti_words=False,
    searching = True
):
    df = df.copy()

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    df[column_name] = df[column_name].astype(str)

    # 1. Remove URLs
    df[column_name] = df[column_name].apply(
        lambda x: re.sub(r'https?:\/\/\S+', '', x.replace("|", " "))
    )
    # 2. Keep the End Of Sentence characters
    if searching == False:
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'\.', ' EOSTokenDot ', x + " "))
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'\?', ' EOSTokenQuest ', x + " "))
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'!', ' EOSTokenExs ', x + " "))

    # 3. Lowercase EARLY
    df[column_name] = df[column_name].str.lower()

    # 4. Remove punctuation
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))

    # 5. Remove non-letter characters
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^a-z\s]', ' ', x))

    # 6. Lemmatization
    if searching == True:
        df[column_name] = df[column_name].apply(lemmatize_sentence)

    # 7. Reduce repeated letters ("soooo" → "soo")
    df[column_name] = df[column_name].apply(
        lambda x: re.sub(r'([a-z])\1{2,}', r'\1\1', x)
    )

    # 8. Remove extremely long nonsense words
    df[column_name] = df[column_name].apply(
        lambda x: re.sub(r'\b\w{30,1000}\b', ' ', x)
    )

    # 9. Remove MBTI labels (optional)
    if remove_mbti_words:
        mbti_types = [
            'infp','infj','intp','intj','entp','enfp','istp','isfp',
            'entj','istj','enfj','isfj','estp','esfp','esfj','estj'
        ]
        pattern = re.compile(r'\b(' + "|".join(mbti_types) + r')\b')
        df[column_name] = df[column_name].apply(lambda x: pattern.sub(' ', x))

    # 10. Remove stopwords
    STOP_WORDS = set(stopwords.words("english"))
    df[column_name] = df[column_name].apply(
        lambda text: " ".join([w for w in text.split() if w not in STOP_WORDS])
    )

    # 11. Remove character names
    names = [
        "Sheldon", "Cooper",
        "Leonard", "Hofstadter",
        "Penny", "Hofstadter",
        "Howard", "Wolowitz",
        "Rajesh", "Koothrappali",
        "Amy", "Farrah", "Fowler",
        "Bernadette", "Rostenkowski", "Wolowitz",

        "Rachel", "Green",
        "Monica", "Geller",
        "Phoebe", "Buffay",
        "Ross", "Geller",
        "Chandler", "Bing",
        "Joey", "Tribbiani",

        "Jay", "Pritchett",
        "Gloria", "Pritchett",
        "Manny", "Delgado",
        "Joe", "Pritchett",
        "Phil", "Dunphy",
        "Claire", "Dunphy",
        "Haley", "Dunphy",
        "Alex", "Dunphy",
        "Luke", "Dunphy",
        "Mitchell", "Pritchett",
        "Cameron", "Tucker",
        "Lily", "Tucker", "Pritchett",

        "Jerry", "Seinfeld",
        "George", "Costanza",
        "Elaine", "Benes",
        "Cosmo", "Kramer",

        "Michael", "Scott",
        "Dwight", "Schrute",
        "Jim", "Halpert",
        "Pam", "Beesly",
        "Ryan", "Howard",
        "Kelly", "Kapoor",
        "Angela", "Martin",
        "Oscar", "Martinez",
        "Kevin", "Malone",
        "Stanley", "Hudson",
        "Phyllis", "Vance",
        "Meredith", "Palmer",
        "Creed", "Bratton",
        "Toby", "Flenderson",
        "Darryl", "Philbin",
        "Andy", "Bernard",
        "Erin", "Hannon",
        "Robert", "California"
    ]
    names = set([n.lower() for n in names])

    if searching == False:
        df[column_name] = df[column_name].apply(
            lambda text: " ".join([w for w in text.split() if w not in names])
        )

    # 12. Final whitespace normalize
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    return df


def add_mbti_binary_columns(df, type_col="type"):
    """
    Add 4 binary columns:
        EI: 1 = E, 0 = I
        SN: 1 = S, 0 = N
        TF: 1 = T, 0 = F
        JP: 1 = J, 0 = P
    """
    df = df.copy()
    mbti_str = df[type_col].astype(str).str.upper()

    df["EI"] = mbti_str.str[0].map({'E': 1, 'I': 0})
    df["SN"] = mbti_str.str[1].map({'S': 1, 'N': 0})
    df["TF"] = mbti_str.str[2].map({'T': 1, 'F': 0})
    df["JP"] = mbti_str.str[3].map({'J': 1, 'P': 0})

    return df

data = pd.read_csv("raw_data/mbti_data.csv")

data = preprocess_text(data, column_name="posts", remove_mbti_words=True, searching= False)
df_encoded = add_mbti_binary_columns(data, type_col="type")

tfidf = TfidfVectorizer(max_features=1500)
X = tfidf.fit_transform(df_encoded["posts"])

dimension_cols = ["EI", "SN", "TF", "JP"]


param_distributions = {
    "n_estimators": randint(100, 400),      # integer between 100 and 399
    "max_depth": randint(3, 8),             # 3–7
    "learning_rate": uniform(0.01, 0.15),   # 0.01–0.16
    "subsample": uniform(0.6, 0.4),         # 0.6–1.0
    "colsample_bytree": uniform(0.6, 0.4),  # 0.6–1.0
}

best_params_per_dim = {}
best_scores_per_dim = {}

for dim in dimension_cols:
    print(f"\n=== Random search for {dim} ===")
    y = df_encoded[dim].values

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=7
    )

    rand_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="accuracy",  # or "f1"
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    rand_search.fit(X, y)

    print(f"Best score for {dim}: {rand_search.best_score_:.4f}")
    print("Best params:", rand_search.best_params_)

    best_params_per_dim[dim] = rand_search.best_params_
    best_scores_per_dim[dim] = rand_search.best_score_

print("\n" + "="*60)
print(" BEST PARAMETER SUMMARY FOR ALL DIMENSIONS ")
print("="*60)

for dim in dimension_cols:
    print(f"\n>>> {dim}")
    print("Best CV score:", round(best_scores_per_dim[dim], 4))
    print("Best parameters:")
    for k, v in best_params_per_dim[dim].items():
        print(f"  - {k}: {v}")

print("="*60 + "\n")

final_models = {}

for dim in dimension_cols:
    print(f"\n=== Training FINAL model for {dim} with best params ===")
    y = df_encoded[dim].values
    best_params = best_params_per_dim[dim]

    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=7,
        **best_params
    )

    final_model.fit(X, y)
    final_models[dim] = final_model


bundle = {
    "tfidf": tfidf,
    "models": final_models,                 # dict: {"EI": model, "SN": model, ...}
    "best_params": best_params_per_dim,     # best params found by RandomizedSearchCV
    "cv_scores": best_scores_per_dim,       # mean CV scores from tuning
    "label_mapping": {
        "EI": {1: "E", 0: "I"},
        "SN": {1: "S", 0: "N"},
        "TF": {1: "T", 0: "F"},
        "JP": {1: "J", 0: "P"},
    },
    "preprocessing_info": {
        "column_name": "posts",
        "remove_mbti_words": True,
        "searching": False,
        "max_features_tfidf": 1500,
    }
}


os.makedirs("saved_models", exist_ok=True)
save_path = "saved_models/mbti_bundle.pkl"

joblib.dump(bundle, save_path)
print("Bundle keys:", list(bundle.keys()))

