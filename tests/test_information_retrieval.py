"""
Unit tests for information_retrieval package

Tests for search functionality, query parsing, and data processing.
"""

import pytest
import pandas as pd
from information_retrieval.search import (
    parse_query,
    extract_best_snippet,
    highlight,
    search_episodes,
    load_show_data
)
from information_retrieval.data_processing import (
    normalize_text,
    tokenize,
    preprocess_for_ir,
    combine_dialogue_and_title
)


class TestQueryParsing:
    """Test query parsing functionality."""
    
    def test_parse_simple_query(self):
        """Test parsing a simple phrase query."""
        result = parse_query("coffee")
        assert result['type'] == 'phrase'
        assert result['value'] == 'coffee'
    
    def test_parse_and_query(self):
        """Test parsing AND query."""
        result = parse_query("coffee&shop")
        assert result['type'] == 'and'
        assert result['value'] == ['coffee', 'shop']
    
    def test_parse_or_query(self):
        """Test parsing OR query."""
        result = parse_query("coffee/shop")
        assert result['type'] == 'or'
        assert result['value'] == ['coffee', 'shop']
    
    def test_parse_complex_and(self):
        """Test parsing complex AND query with multiple terms."""
        result = parse_query("wedding&cake&beautiful")
        assert result['type'] == 'and'
        assert len(result['value']) == 3
        assert 'wedding' in result['value']
    
    def test_parse_query_strips_whitespace(self):
        """Test that query parsing removes extra whitespace."""
        result = parse_query("  coffee & shop  ")
        assert result['type'] == 'and'
        assert all(term.strip() == term for term in result['value'])


class TestTextPreprocessing:
    """Test text preprocessing functions."""
    
    def test_normalize_text_lowercase(self):
        """Test that text is converted to lowercase."""
        text = "Hello WORLD"
        result = normalize_text(text)
        assert result.islower()
    
    def test_normalize_text_removes_urls(self):
        """Test that URLs are removed."""
        text = "Check this out https://example.com cool"
        result = normalize_text(text)
        assert 'https' not in result
        assert 'example.com' not in result
    
    def test_normalize_text_removes_punctuation(self):
        """Test that punctuation is removed."""
        text = "Hello, world! How are you?"
        result = normalize_text(text)
        assert ',' not in result
        assert '!' not in result
        assert '?' not in result
    
    def test_normalize_text_removes_numbers(self):
        """Test that numbers are removed."""
        text = "I have 123 apples"
        result = normalize_text(text)
        assert '123' not in result
    
    def test_normalize_handles_empty_string(self):
        """Test normalization of empty string."""
        result = normalize_text("")
        assert result == ""
    
    def test_normalize_handles_non_string(self):
        """Test normalization of non-string input."""
        result = normalize_text(None)
        assert result == ""
        
        result = normalize_text(123)
        assert result == ""
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "the quick brown fox"
        tokens = tokenize(text)
        assert isinstance(tokens, list)
        # 'the' should be removed as stopword
        assert 'the' not in tokens
    
    def test_tokenize_removes_character_names(self):
        """Test that character names are filtered."""
        text = "ross said hello"
        tokens = tokenize(text)
        assert 'ross' not in tokens
        assert 'hello' in tokens or 'hello' in [t.lower() for t in tokens]
    
    def test_preprocess_for_ir_pipeline(self):
        """Test the full preprocessing pipeline."""
        text = "Ross said: Check https://example.com! It's AMAZING!!!"
        tokens = preprocess_for_ir(text)
        assert isinstance(tokens, list)
        # Should be lowercase, no URLs, no punctuation
        for token in tokens:
            assert token.islower()
            assert '.' not in token
            assert '!' not in token
    
    def test_combine_dialogue_and_title(self):
        """Test combining dialogue with boosted title."""
        dialogue = "This is dialogue"
        title = "Title"
        result = combine_dialogue_and_title(dialogue, title, weight=2)
        assert "This is dialogue" in result
        # Title should appear multiple times based on weight
        assert result.count("Title") >= 2
    
    def test_combine_handles_missing_values(self):
        """Test combining with None/empty values."""
        result = combine_dialogue_and_title(None, "title", weight=2)
        assert "title" in result.lower()
        
        result = combine_dialogue_and_title("dialogue", None, weight=2)
        assert "dialogue" in result


class TestSnippetExtraction:
    """Test snippet extraction and highlighting."""
    
    def test_extract_snippet_with_match(self):
        """Test snippet extraction with matching query."""
        dialogue = "Line 1\nThis is about coffee shops\nLine 3"
        query = "coffee"
        snippet = extract_best_snippet(dialogue, query)
        assert "coffee" in snippet.lower()
        assert len(snippet) > 0
    
    def test_extract_snippet_respects_max_length(self):
        """Test that snippet respects maximum length."""
        dialogue = "a " * 500  # Very long text
        query = "a"
        snippet = extract_best_snippet(dialogue, query, max_len=100)
        assert len(snippet) <= 103  # 100 + "..." = 103
    
    def test_extract_snippet_no_match(self):
        """Test snippet extraction when query doesn't match."""
        dialogue = "This is some text"
        query = "xyz"
        snippet = extract_best_snippet(dialogue, query)
        # Should return fallback snippet
        assert len(snippet) > 0
    
    def test_highlight_adds_formatting(self):
        """Test that highlighting adds ANSI codes."""
        text = "I love coffee"
        query = "coffee"
        result = highlight(text, query)
        # Should contain ANSI escape codes
        assert '\033[' in result
    
    def test_highlight_multiple_terms(self):
        """Test highlighting with multiple query terms."""
        text = "I love coffee and tea"
        query = "coffee&tea"
        result = highlight(text, query)
        # Both terms should be highlighted
        assert result.count('\033[') >= 2


class TestSearchFunctionality:
    """Test search functionality with sample data."""
    
    def test_search_episodes_basic(self, sample_episode_data):
        """Test basic episode search."""
        results = search_episodes(sample_episode_data, "coffee", top_k=2)
        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 2
    
    def test_search_episodes_with_and_query(self, sample_episode_data):
        """Test search with AND operator."""
        results = search_episodes(sample_episode_data, "wedding&beautiful", top_k=5)
        assert isinstance(results, pd.DataFrame)
        # Should find the wedding episode
        if not results.empty:
            assert any('wedding' in str(title).lower() for title in results.get('episode_title', []))
    
    def test_search_episodes_no_matches(self, sample_episode_data):
        """Test search with no matches."""
        results = search_episodes(sample_episode_data, "dinosaur&spaceship", top_k=5)
        assert isinstance(results, pd.DataFrame)
        assert results.empty
    
    def test_search_returns_required_columns(self, sample_episode_data):
        """Test that search results have required columns."""
        results = search_episodes(sample_episode_data, "coffee", top_k=5)
        if not results.empty:
            assert 'season' in results.columns
            assert 'episode' in results.columns
            assert 'relevance_score' in results.columns
            assert 'snippet_highlight' in results.columns
    
    def test_search_respects_top_k(self, sample_episode_data):
        """Test that search respects top_k parameter."""
        results = search_episodes(sample_episode_data, "the", top_k=1)
        assert len(results) <= 1


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_show_data_invalid_show(self):
        """Test loading data for invalid show."""
        with pytest.raises(ValueError):
            load_show_data("invalid_show_name")
    
    def test_load_show_data_valid_show(self, skip_if_no_data):
        """Test loading data for valid show."""
        # This test requires actual data files
        try:
            df = load_show_data('friends')
            assert isinstance(df, pd.DataFrame)
            assert 'season' in df.columns
            assert 'episode' in df.columns
            assert 'dialogue' in df.columns
        except FileNotFoundError:
            pytest.skip("Data files not available")

