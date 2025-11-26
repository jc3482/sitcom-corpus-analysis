# Sitcom Corpus Analysis

A comprehensive toolkit for analyzing sitcom dialogues with **Information Retrieval** and **MBTI Personality Prediction** capabilities.

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate sitcom-env

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### 2. Usage Examples

#### Information Retrieval

```bash
# Search a single show
python -m information_retrieval search friends "coffee shop"

# AND operator (both terms must appear)
python -m information_retrieval search tbbt "physics&quantum" --top 10

# OR operator (either term can appear)
python -m information_retrieval search office "meeting/conference"

# Search across all shows
python -m information_retrieval search-all "thanksgiving"

# List available shows
python -m information_retrieval list
```

#### MBTI Personality Prediction

```bash
# Train the model
python mbti_model_training.py

# Predict character personality
python mbti_prediction/predict_characters.py raw_data/tbbt_dialogues.csv Sheldon

# Predict multiple characters
python mbti_prediction/predict_characters.py raw_data/friends_dialogues.csv Rachel Ross Monica

# Predict all major characters
python mbti_prediction/predict_characters.py raw_data/tbbt_dialogues.csv --all
```

## Project Structure

```
sitcom-corpus-analysis/
├── pyproject.toml              # Project configuration
├── environment.yml             # Conda environment spec
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── information_retrieval/      # Information retrieval module
│   ├── __main__.py            # CLI entry point
│   ├── search.py              # BM25 search engine
│   ├── data_processing.py     # Text preprocessing
│   └── cache_utils.py         # Caching utilities
│
├── mbti_prediction/            # MBTI prediction module
│   ├── predict_characters.py  # Character prediction script
│   └── run_mbti_model.py      # Model training
│
├── raw_data/                   # Data files
│   ├── friends_dialogues.csv
│   ├── tbbt_dialogues.csv
│   ├── seinfeld_scripts.csv
│   ├── the_office.csv
│   ├── modern_family_scripts.csv
│   └── mbti_data.csv
│
└── mbti_model_training.py      # Model training pipeline
```

## Supported Shows

- Friends
- The Big Bang Theory (TBBT)
- Seinfeld
- The Office
- Modern Family

## Query Syntax

| Syntax | Description | Example |
|--------|-------------|---------|
| `"word"` | Phrase search | `"coffee shop"` |
| `"word1&word2"` | AND (both must appear) | `"wedding&cake"` |
| `"word1/word2"` | OR (either can appear) | `"wedding/party"` |
| `--top N` | Limit results to top N | `--top 10` |

## Technical Stack

- **BM25**: Information retrieval ranking algorithm
- **spaCy**: NLP text processing
- **NLTK**: Natural language toolkit
- **XGBoost**: MBTI personality prediction model
- **pandas**: Data processing

## Installation Options

### Option 1: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate sitcom-env
```

### Option 2: Using pip
```bash
pip install -e .
```

### Option 3: From requirements.txt
```bash
pip install -r requirements.txt
```

## Dependencies

Main packages (see `environment.yml` or `requirements.txt` for complete list):
- pandas, numpy
- rank-bm25
- spacy, nltk
- scikit-learn, xgboost
- joblib, scipy

## Building and Distribution

To build the package:
```bash
pip install build
python -m build
```

To install in development mode:
```bash
pip install -e .
```

After installation, you can use the CLI command:
```bash
sitcom-search list
sitcom-search search friends "coffee"
```

---

**Python Version**: 3.9 - 3.11  
**Environment Name**: `sitcom-env`
