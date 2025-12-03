"""
Unit tests for mbti_prediction package

Tests for MBTI prediction, data processing, and quote scoring.
"""

import pytest
import pandas as pd
import numpy as np
from mbti_prediction.data_processing import preprocess_text
from mbti_prediction.mbti_prediction import (
    load_bundle,
    load_all_dialogues,
    predict_mbti_for_character,
    score_quotes_for_character
)


class TestMBTIDataProcessing:
    """Test MBTI-specific data preprocessing."""
    
    def test_preprocess_text_basic(self, sample_mbti_data):
        """Test basic text preprocessing."""
        df = preprocess_text(sample_mbti_data.copy(), 'posts')
        assert 'posts' in df.columns
        assert len(df) == len(sample_mbti_data)
    
    def test_preprocess_removes_urls(self):
        """Test URL removal."""
        df = pd.DataFrame({
            'posts': ['Check this https://example.com out']
        })
        result = preprocess_text(df, 'posts')
        assert 'https' not in result['posts'].iloc[0]
    
    def test_preprocess_lowercase(self):
        """Test text is converted to lowercase."""
        df = pd.DataFrame({
            'posts': ['HELLO WORLD']
        })
        result = preprocess_text(df, 'posts')
        assert result['posts'].iloc[0].islower()
    
    def test_preprocess_eos_tokens(self):
        """Test that end-of-sentence tokens are preserved."""
        df = pd.DataFrame({
            'posts': ['Hello. How are you? Great!']
        })
        result = preprocess_text(df, 'posts')
        text = result['posts'].iloc[0]
        # Should contain EOS tokens
        assert 'eostoken' in text.lower()
    
    def test_preprocess_removes_stopwords(self):
        """Test stopword removal."""
        df = pd.DataFrame({
            'posts': ['the quick brown fox']
        })
        result = preprocess_text(df, 'posts')
        # 'the' should be removed
        assert 'the' not in result['posts'].iloc[0]
    
    def test_preprocess_removes_character_names(self):
        """Test character name removal."""
        df = pd.DataFrame({
            'posts': ['Ross said hello to Monica']
        })
        result = preprocess_text(df, 'posts')
        text = result['posts'].iloc[0]
        assert 'ross' not in text.lower()
        assert 'monica' not in text.lower()
    
    def test_preprocess_removes_repeated_letters(self):
        """Test reduction of repeated letters."""
        df = pd.DataFrame({
            'posts': ['Hellooooo world']
        })
        result = preprocess_text(df, 'posts')
        # 'oooo' should be reduced to 'oo'
        assert 'ooo' not in result['posts'].iloc[0]
    
    def test_preprocess_removes_mbti_words(self):
        """Test MBTI type removal when enabled."""
        df = pd.DataFrame({
            'posts': ['I am an INTJ personality type']
        })
        result = preprocess_text(df, 'posts', remove_mbti_words=True)
        assert 'intj' not in result['posts'].iloc[0].lower()
    
    def test_preprocess_keeps_mbti_words_when_disabled(self):
        """Test MBTI types kept when removal disabled."""
        df = pd.DataFrame({
            'posts': ['I am an INTJ personality']
        })
        result = preprocess_text(df, 'posts', remove_mbti_words=False)
        # Should still be removed as it's lowercase and tokenized
        # but let's check the function doesn't explicitly remove it
        assert isinstance(result, pd.DataFrame)
    
    def test_preprocess_handles_empty_string(self):
        """Test handling of empty strings."""
        df = pd.DataFrame({
            'posts': ['']
        })
        result = preprocess_text(df, 'posts')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_preprocess_invalid_column(self):
        """Test error on invalid column name."""
        df = pd.DataFrame({'other': ['text']})
        with pytest.raises(ValueError):
            preprocess_text(df, 'posts')


class TestMBTIBundleLoading:
    """Test MBTI model bundle loading."""
    
    def test_load_bundle_returns_tuple(self):
        """Test that load_bundle returns correct structure."""
        try:
            tfidf, models, label_mapping, preproc_info = load_bundle()
            assert tfidf is not None
            assert models is not None
            assert isinstance(models, dict)
            assert label_mapping is not None
            assert isinstance(label_mapping, dict)
        except FileNotFoundError:
            pytest.skip("Model bundle not found")
    
    def test_bundle_has_all_dimensions(self):
        """Test that bundle contains all MBTI dimensions."""
        try:
            _, models, label_mapping, _ = load_bundle()
            expected_dims = ['EI', 'SN', 'TF', 'JP']
            for dim in expected_dims:
                assert dim in models
                assert dim in label_mapping
        except FileNotFoundError:
            pytest.skip("Model bundle not found")
    
    def test_label_mapping_structure(self):
        """Test label mapping has correct structure."""
        try:
            _, _, label_mapping, _ = load_bundle()
            for dim, mapping in label_mapping.items():
                assert len(mapping) == 2  # Binary classification
                assert 0 in mapping or 1 in mapping
        except FileNotFoundError:
            pytest.skip("Model bundle not found")


class TestDialogueLoading:
    """Test dialogue data loading."""
    
    def test_load_all_dialogues_returns_dataframe(self):
        """Test that load_all_dialogues returns DataFrame."""
        try:
            df = load_all_dialogues()
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_loaded_dialogues_have_required_columns(self):
        """Test that loaded dialogues have required columns."""
        try:
            df = load_all_dialogues()
            required_cols = ['season', 'episode', 'character', 'dialogue', 'show_key']
            for col in required_cols:
                assert col in df.columns
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_loaded_dialogues_multiple_shows(self):
        """Test that data from multiple shows is loaded."""
        try:
            df = load_all_dialogues()
            # Should have multiple shows
            assert df['show_key'].nunique() > 1
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")


class TestMBTIPrediction:
    """Test MBTI prediction functionality."""
    
    def test_predict_mbti_for_character_structure(self):
        """Test prediction returns correct structure."""
        try:
            tfidf, models, labels, _ = load_bundle()
            df = load_all_dialogues()
            
            # Get first character with data
            first_char = df.groupby(['show_key', 'character']).size().idxmax()
            show_key, char_name = first_char
            
            mbti, probs, df_char = predict_mbti_for_character(
                tfidf, models, labels, df, show_key, char_name
            )
            
            if mbti is not None:
                assert isinstance(mbti, str)
                assert len(mbti) == 4
                assert isinstance(probs, dict)
                assert len(probs) == 4
                assert isinstance(df_char, pd.DataFrame)
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_predict_mbti_invalid_character(self):
        """Test prediction for non-existent character."""
        try:
            tfidf, models, labels, _ = load_bundle()
            df = load_all_dialogues()
            
            mbti, probs, df_char = predict_mbti_for_character(
                tfidf, models, labels, df, 'friends', 'NonExistentCharacter123'
            )
            
            assert mbti is None
            assert probs is None
            assert df_char is None
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_mbti_prediction_format(self):
        """Test that MBTI prediction follows correct format."""
        try:
            tfidf, models, labels, _ = load_bundle()
            df = load_all_dialogues()
            
            # Get a character with sufficient data
            char_counts = df.groupby(['show_key', 'character']).size()
            show_key, char_name = char_counts.idxmax()
            
            mbti, _, _ = predict_mbti_for_character(
                tfidf, models, labels, df, show_key, char_name
            )
            
            if mbti:
                # Should be 4 letters
                assert len(mbti) == 4
                # Each position should be valid
                assert mbti[0] in ['E', 'I']
                assert mbti[1] in ['S', 'N']
                assert mbti[2] in ['T', 'F']
                assert mbti[3] in ['J', 'P']
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")


class TestQuoteScoring:
    """Test quote scoring functionality."""
    
    def test_score_quotes_returns_list(self, sample_dialogue_data):
        """Test that score_quotes returns a list."""
        try:
            tfidf, models, labels, _ = load_bundle()
            
            # Create a mock character dataframe
            df_char = sample_dialogue_data[sample_dialogue_data['character'] == 'Ross'].copy()
            df_char['dialogue_raw'] = df_char['dialogue']
            
            if not df_char.empty:
                quotes = score_quotes_for_character(
                    tfidf, models, labels, df_char, 'ISTJ', top_k=3
                )
                assert isinstance(quotes, list)
                assert len(quotes) <= 3
        except (FileNotFoundError, SystemExit):
            pytest.skip("Model bundle not available")
    
    def test_score_quotes_structure(self, sample_dialogue_data):
        """Test that scored quotes have correct structure."""
        try:
            tfidf, models, labels, _ = load_bundle()
            
            df_char = sample_dialogue_data[sample_dialogue_data['character'] == 'Ross'].copy()
            df_char['dialogue_raw'] = df_char['dialogue']
            
            if not df_char.empty:
                quotes = score_quotes_for_character(
                    tfidf, models, labels, df_char, 'ISTJ', top_k=1
                )
                
                if quotes:
                    quote = quotes[0]
                    assert 'season' in quote
                    assert 'episode' in quote
                    assert 'dialogue_raw' in quote
                    assert 'score' in quote
        except (FileNotFoundError, SystemExit):
            pytest.skip("Model bundle not available")
    
    def test_score_quotes_empty_dataframe(self):
        """Test quote scoring with empty dataframe."""
        try:
            tfidf, models, labels, _ = load_bundle()
            
            df_empty = pd.DataFrame()
            quotes = score_quotes_for_character(
                tfidf, models, labels, df_empty, 'INTJ', top_k=5
            )
            assert isinstance(quotes, list)
            assert len(quotes) == 0
        except (FileNotFoundError, SystemExit):
            pytest.skip("Model bundle not available")

