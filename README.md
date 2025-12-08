# Sitcom Corpus Analysis
## Author: Stacy Che, Qiyue Gu, Herry Wei

A comprehensive toolkit for analyzing sitcom dialogues with **Information Retrieval** **MBTI Personality Prediction**, and **Character Style Transfer** capabilities.

## Quick Start

### 1. Environment Setup

To use the features of our **Character Style Transfer**, you must obtain your own API key from the **OpenAI website**.

1. Visit https://platform.openai.com/ to generate your personal API key.

2. Add the key to your environment variables by running the following command in your terminal:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Download our models for **MBTI Personality Prediction**, and **Character Style Transfer** from this Google Drive link: https://drive.google.com/drive/folders/1oxtXMy6T3tFbxBAVx9tuY_NeP3ahCzF-?usp=sharing, and add them to directed folder(`embedding_cache.pkl`in root folder, `mbti_bundle.pkl` in mbti_prediction folder)

Run the following code:

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment (always activate before using)
conda activate sitcom-env

# Install package in editable mode
pip install -e .

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Verify installation
python verify_installation.py

# Install testing dependencies
pip install pytest pytest-cov
```

### 2. Usage Examples

#### Unified Interface 

```bash
# Information Retrieval
sitcom list                                       # list of shows
sitcom search-all "thanksgiving"                  # search all shows
sitcom search friends "coffee shop"               # search one specific show
sitcom search friends "wedding&dress"             # search one specific show with two keywords

# MBTI Personality Prediction
sitcom analyze friends "wedding"                  # Search + character personalities
sitcom character friends Ross                     # Character MBTI profile
sitcom personalities friends                      # Character MBTI profiles from an entire show

# Style Transfer

```

#### Individual Package Interfaces

```bash
# Information Retrieval
python -m information_retrieval search friends "coffee shop"
sitcom-search search friends "coffee shop"
sitcom-search search friends "coffee shop" 

# MBTI Prediction
python -m mbti_prediction
sitcom-mbti
```

#### Python API

```python
# Information Retrieval
from information_retrieval import search_episodes, load_show_data
episodes = load_show_data('friends')
results = search_episodes(episodes, "coffee", 5)

# MBTI Prediction
from mbti_prediction import load_bundle, predict_mbti_for_character, load_all_dialogues
tfidf, models, labels, _ = load_bundle()
df = load_all_dialogues()
mbti, probs, _ = predict_mbti_for_character(tfidf, models, labels, df, 'friends', 'Ross')

# Combined Features (NEW)
from sitcom_analysis import (
    search_with_character_info, 
    analyze_character_moments,
    get_character_mbti
)

# Search with character info
results, char_mbti = search_with_character_info('friends', 'wedding')

# Character analysis with dialogue search
analysis = analyze_character_moments('friends', 'Ross', 'dinosaurs')
print(analysis['mbti'])
print(analysis['top_quotes'])

# Just get MBTI quickly
mbti_info = get_character_mbti('friends', 'Ross')
print(mbti_info['mbti'])
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
│   ├── __main__.py            # CLI entry point
│   ├── mbti_prediction.py     # Core prediction logic
│   ├── data_processing.py     # Text preprocessing
│   ├── mbti_model_training.py # Model training
│   └── mbti_bundle.pkl        # Trained models
│
├── sitcom_analysis/            # Unified integration layer 
│   ├── __init__.py            # Exports from both packages
│   ├── __main__.py            # Unified CLI
│   ├── style_transfer_module.py     
│   └── combined_features.py   # Cross-package features
│
└── raw_data/                   # Data files
    ├── friends_dialogues.csv
    ├── tbbt_dialogues.csv
    ├── seinfeld_scripts.csv
    ├── the_office.csv
    ├── modern_family_scripts.csv
    └── mbti_data.csv
```

## Supported Shows

| Show Key | Show Name |
|----------|-----------|
| `friends` | Friends |
| `tbbt` | The Big Bang Theory |
| `seinfeld` | Seinfeld |
| `office` | The Office |
| `modern_family` | Modern Family |

## Features

### Information Retrieval
- BM25-based episode search with boolean operators
- Query syntax: phrase, AND (`&`), OR (`/`)
- Episode snippet extraction with highlighting
- Title boosting for relevance ranking

### MBTI Personality Prediction
- 4-dimensional personality classification (E/I, S/N, T/F, J/P)
- XGBoost models with TF-IDF features
- Character-level dialogue aggregation
- Representative quote extraction

### Combined Analysis (NEW)
- Search episodes with character personality insights
- Character-focused dialogue search with MBTI context
- Show-wide personality profiling

## Query Syntax

### Search Operators

| Syntax | Description | Example |
|--------|-------------|---------|
| `"word"` | Phrase search | `"coffee shop"` |
| `"word1&word2"` | AND (both must appear) | `"wedding&cake"` |
| `"word1/word2"` | OR (either can appear) | `"wedding/party"` |

### Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--top N` | Number of results to return | 5 |
| `--title-weight N` | Boost weight for title matches | 2 |
| `--limit N` | Character limit (for personalities command) | 10 |

## Technical Stack

- **BM25**: Information retrieval ranking algorithm
- **spaCy**: NLP text processing
- **NLTK**: Natural language toolkit
- **XGBoost**: MBTI personality prediction model
- **pandas**: Data processing

## Installation Options

### Option 1: Using Conda 
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

After installation, you can use the CLI commands:
```bash
# Unified interface 
sitcom list
sitcom search friends "coffee"
sitcom mbti
sitcom analyze friends "wedding"

# Independent interfaces 
sitcom-search list
sitcom-mbti

# Style Transfer Interface
sitcom-style --list
sitcom-style Sheldon "This is highly illogical."
sitcom-style Sheldon
```

### Verify Installation

Test that the integration works correctly:
```bash
python verify_installation.py
```

This will verify that all three packages (information_retrieval, mbti_prediction, and sitcom_analysis) are properly installed and can import from each other.

## Testing

This project includes comprehensive unit tests using pytest.

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_information_retrieval.py

# Run tests with coverage report
pytest --cov=information_retrieval --cov=mbti_prediction --cov=sitcom_analysis

# Run tests and generate HTML coverage report
pytest --cov=information_retrieval --cov=mbti_prediction --cov=sitcom_analysis --cov-report=html

# Run only fast tests (skip integration tests)
pytest -m "not slow"
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py                       # Shared fixtures
├── test_information_retrieval.py     # IR package tests
├── test_mbti_prediction.py           # MBTI package tests
└── test_sitcom_analysis.py           # Integration layer tests
```

### Test Coverage

The test suite covers:
- Query parsing and search functionality
- Text preprocessing for both IR and MBTI
- MBTI prediction and quote scoring
- Combined features in integration layer
- Error handling and edge cases

Run `pytest --cov` to see current coverage statistics.

## Getting Help

```bash
sitcom help                    # Show all commands and usage
sitcom list                    # List available shows
```

For command-specific help, see the documentation files in each package:
- `sitcom_analysis/USAGE.txt` - Unified CLI documentation
- `information_retrieval/USAGE.txt` - IR package details
- `mbti_prediction/USAGE.txt` - MBTI package details

## Additional Documentation

- `verify_installation.py` - Installation verification script
- `tests/` - Full unit test suite using pytest (73 tests, 52 passing)

## Package Architecture

This project uses a **thin integration layer** approach:

1. **information_retrieval** - BM25 search engine 
2. **mbti_prediction** - Personality prediction models 
3. **sitcom_analysis** - Integration layer that coordinates between both packages

All 3 packages are installed together with a single `pip install -e .` command, providing:
- Backward compatibility with original commands
- New unified CLI interface
- Combined features that leverage both packages

---

**Python Version**: 3.9 - 3.11  
**Environment Name**: `sitcom-env`  
