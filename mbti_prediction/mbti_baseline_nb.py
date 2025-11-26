import re
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

def preprocess_text(
    df,
    column_name,
    remove_mbti_words=False
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
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\.', ' EOSTokenDot ', x + " "))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\?', ' EOSTokenQuest ', x + " "))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'!', ' EOSTokenExs ', x + " "))

    # 3. Lowercase EARLY
    df[column_name] = df[column_name].str.lower()

    # 4. Remove punctuation
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))

    # 5. Remove non-letter characters
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^a-z\s]', ' ', x))

    # 6. Reduce repeated letters ("soooo" → "soo")
    df[column_name] = df[column_name].apply(
        lambda x: re.sub(r'([a-z])\1{2,}', r'\1\1', x)
    )

    # 7. Remove extremely long nonsense words
    df[column_name] = df[column_name].apply(
        lambda x: re.sub(r'\b\w{30,1000}\b', ' ', x)
    )

    # 8. Remove MBTI labels (optional)
    if remove_mbti_words:
        mbti_types = [
            'infp','infj','intp','intj','entp','enfp','istp','isfp',
            'entj','istj','enfj','isfj','estp','esfp','esfj','estj'
        ]
        pattern = re.compile(r'\b(' + "|".join(mbti_types) + r')\b')
        df[column_name] = df[column_name].apply(lambda x: pattern.sub(' ', x))

    # 9. Remove stopwords
    STOP_WORDS = set(stopwords.words("english"))
    df[column_name] = df[column_name].apply(
        lambda text: " ".join([w for w in text.split() if w not in STOP_WORDS])
    )

    # 10. Remove character names
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

    # 11. Final whitespace normalize
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


data = pd.read_csv("../raw_data/mbti_data.csv")

# Preprocess
data = preprocess_text(
    data,
    column_name="posts",
    remove_mbti_words=True
)

# Add EI, SN, TF, JP columns
df_encoded = add_mbti_binary_columns(data, type_col="type")

# TF–IDF
print("Fitting TF-IDF (1500 features)...")
tfidf = TfidfVectorizer(max_features=1500)
X = tfidf.fit_transform(df_encoded["posts"])

dimension_cols = ["EI", "SN", "TF", "JP"]

print("\n================= Naive Bayes Baseline =================")

for dim in dimension_cols:
    print(f"\n=== Baseline Naive Bayes for {dim} ===")

    y = df_encoded[dim].values
    nb = MultinomialNB(alpha=1.0)

    scores = cross_val_score(
        nb, X, y,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    print(f"Accuracy (mean): {scores.mean():.4f}")
    print(f"Std Dev:          {scores.std():.4f}")

print("\n========================================================\n")
