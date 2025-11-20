import re
import pandas as pd
from nltk.corpus import stopwords
import pandas as pd
import preprocess_text from data_processing  # this is a self written python function


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

data = preprocess_text(data,column_name="posts", remove_mbti_words=True, searching = False)
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
