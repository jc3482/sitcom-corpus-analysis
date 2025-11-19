import re
import pandas as pd
from nltk.corpus import stopwords
import pandas as pd

def preprocess_text(
    df,
    column_name,
    remove_mbti_words=False,
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
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\.', ' EOSTokenDot ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\?', ' EOSTokenQuest ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'!', ' EOSTokenExs ', x + " "))

    # 3. Lowercase EARLY
    df[column_name] = df[column_name].str.lower()

    # 4. Remove punctuation
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))

    # 5. Remove non-letter characters
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^a-z\s]', ' ', x))

    # 6. Normalize whitespace (so split() is clean)
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

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

    df[column_name] = df[column_name].apply(
        lambda text: " ".join([w for w in text.split() if w not in names])
    )

    # Final whitespace normalize
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

data = preprocess_text(data,column_name="posts",remove_mbti_words=True)
df_encoded = add_mbti_binary_columns(data, type_col="type")

from sklearn.feature_extraction.text import TfidfVectorizer

# Fit TF–IDF on the cleaned posts
tfidf = TfidfVectorizer(max_features=1500)   # no stop_words argument
X = tfidf.fit_transform(df_encoded["posts"]) # Sparse matrix (n_samples × 1500)


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Columns for each dimension
dimension_cols = ["EI", "SN", "TF", "JP"]

# XGBoost hyperparameters (you can tune these)
xgb_params = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 7,
}

models = {}
scores = {}

for dim in dimension_cols:
    print(f"\n=== Training XGBoost for {dim} ===")
    y = df_encoded[dim].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=7, stratify=y
    )

    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy for {dim}: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    models[dim] = model
    scores[dim] = acc


def predict_mbti_for_text(
    raw_text,
    tfidf_vectorizer,
    models,
    text_col_name="posts",
    remove_mbti_words=True,
):
    # Put text into a one-row DataFrame
    tmp_df = pd.DataFrame({text_col_name: [raw_text], "type": ["INTP"]})  
    # dummy type just to satisfy preprocess/add_mbti if needed

    # Apply same preprocessing (this will also remove stopwords, names, etc.)
    tmp_df = preprocess_text(tmp_df, text_col_name, remove_mbti_words=remove_mbti_words)

    # TF–IDF transform using the pre-fitted vectorizer
    X_new = tfidf_vectorizer.transform(tmp_df[text_col_name])

    # Predict each dimension
    letter_map = {
        "EI": {1: "E", 0: "I"},
        "SN": {1: "S", 0: "N"},
        "TF": {1: "T", 0: "F"},
        "JP": {1: "J", 0: "P"},
    }

    result_letters = []
    for dim in ["EI", "SN", "TF", "JP"]:
        model = models[dim]
        y_pred = model.predict(X_new)[0]
        result_letters.append(letter_map[dim][int(y_pred)])

    return "".join(result_letters)


example_text = "I love solving abstract problems and discussing theories with my friends."
pred_type = predict_mbti_for_text(example_text, tfidf, models)
print("Predicted MBTI:", pred_type)
