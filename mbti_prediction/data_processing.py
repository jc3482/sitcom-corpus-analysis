import re
from nltk.corpus import stopwords


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

    # 2. Lowercase EARLY
    df[column_name] = df[column_name].str.lower()

    # 3. Remove extremely long nonsense words
    df[column_name] = df[column_name].apply(
        lambda x: re.sub(r'\b\w{30,1000}\b', ' ', x)
    )

    # 4. Remove MBTI labels (optional)
    if remove_mbti_words:
        mbti_types = [
            'infp','infj','intp','intj','entp','enfp','istp','isfp',
            'entj','istj','enfj','isfj','estp','esfp','esfj','estj'
        ]
        pattern = re.compile(r'\b(' + "|".join(mbti_types) + r')\b')
        df[column_name] = df[column_name].apply(lambda x: pattern.sub(' ', x))

    # 5. Remove character names
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

    # 6. Final whitespace normalize
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    return df