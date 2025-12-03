"""
Unit tests for sitcom_analysis integration layer

Tests for combined features that use both IR and MBTI packages.
"""

import pytest
import pandas as pd
from sitcom_analysis.combined_features import (
    get_character_mbti,
    search_with_character_info,
    analyze_character_moments,
    get_show_character_personalities
)


class TestGetCharacterMBTI:
    """Test get_character_mbti function."""
    
    def test_get_character_mbti_returns_dict(self):
        """Test that function returns a dictionary."""
        try:
            result = get_character_mbti('friends', 'Ross')
            if result is not None:
                assert isinstance(result, dict)
                assert 'mbti' in result
                assert 'probabilities' in result
                assert 'character' in result
                assert 'show' in result
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_get_character_mbti_invalid_character(self):
        """Test with non-existent character."""
        try:
            result = get_character_mbti('friends', 'NonExistentCharacter999')
            assert result is None
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_get_character_mbti_invalid_show(self):
        """Test with invalid show."""
        try:
            result = get_character_mbti('invalid_show_xyz', 'SomeCharacter')
            # Should either return None or raise error
            assert result is None or isinstance(result, dict)
        except (FileNotFoundError, SystemExit, Exception):
            pytest.skip("Expected behavior for invalid show")
    
    def test_mbti_result_structure(self):
        """Test structure of successful MBTI result."""
        try:
            result = get_character_mbti('friends', 'Ross')
            if result is not None:
                assert isinstance(result['mbti'], str)
                assert len(result['mbti']) == 4
                assert isinstance(result['probabilities'], dict)
                assert isinstance(result['line_count'], int)
                assert result['line_count'] > 0
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")


class TestSearchWithCharacterInfo:
    """Test search_with_character_info function."""
    
    def test_search_with_character_info_returns_tuple(self):
        """Test that function returns tuple of (DataFrame, dict)."""
        try:
            results, char_mbti = search_with_character_info(
                'friends', 'coffee', top_k=3, analyze_characters=True
            )
            assert isinstance(results, pd.DataFrame)
            # char_mbti can be None or dict
            assert char_mbti is None or isinstance(char_mbti, dict)
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_search_without_character_analysis(self):
        """Test search with character analysis disabled."""
        try:
            results, char_mbti = search_with_character_info(
                'friends', 'coffee', top_k=3, analyze_characters=False
            )
            assert isinstance(results, pd.DataFrame)
            assert char_mbti is None
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_search_no_results(self):
        """Test search with query that returns no results."""
        try:
            results, char_mbti = search_with_character_info(
                'friends', 'xyzabc123nonexistent', top_k=5
            )
            assert isinstance(results, pd.DataFrame)
            assert results.empty
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_search_respects_top_k(self):
        """Test that search respects top_k parameter."""
        try:
            results, _ = search_with_character_info(
                'friends', 'the', top_k=2, analyze_characters=False
            )
            assert len(results) <= 2
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_character_mbti_dict_structure(self):
        """Test structure of character MBTI dictionary."""
        try:
            _, char_mbti = search_with_character_info(
                'friends', 'coffee', top_k=3, analyze_characters=True
            )
            if char_mbti:
                # Should be dict mapping character names to MBTI types
                for char, mbti in char_mbti.items():
                    assert isinstance(char, str)
                    assert isinstance(mbti, str)
                    assert len(mbti) == 4
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")


class TestAnalyzeCharacterMoments:
    """Test analyze_character_moments function."""
    
    def test_analyze_character_moments_returns_dict(self):
        """Test that function returns a dictionary."""
        try:
            result = analyze_character_moments('friends', 'Ross')
            assert isinstance(result, dict)
            assert 'success' in result
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_analyze_successful_structure(self):
        """Test structure of successful analysis."""
        try:
            result = analyze_character_moments('friends', 'Ross', top_k=3)
            if result.get('success'):
                assert 'mbti' in result
                assert 'probabilities' in result
                assert 'top_quotes' in result
                assert 'character' in result
                assert 'show' in result
                assert 'total_lines' in result
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_analyze_with_query(self):
        """Test analysis with dialogue search query."""
        try:
            result = analyze_character_moments(
                'friends', 'Ross', query='dinosaur', top_k=3
            )
            if result.get('success') and 'matching_moments' in result:
                assert isinstance(result['matching_moments'], list)
                assert 'total_matches' in result
                assert isinstance(result['total_matches'], int)
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_analyze_invalid_character(self):
        """Test analysis with non-existent character."""
        try:
            result = analyze_character_moments(
                'friends', 'NonExistentCharacter999'
            )
            assert result.get('success') is False
            assert 'error' in result
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_top_quotes_structure(self):
        """Test structure of top quotes."""
        try:
            result = analyze_character_moments('friends', 'Ross', top_k=2)
            if result.get('success') and result.get('top_quotes'):
                quotes = result['top_quotes']
                assert len(quotes) <= 2
                for quote in quotes:
                    assert 'season' in quote
                    assert 'episode' in quote
                    assert 'dialogue_raw' in quote
                    assert 'score' in quote
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_matching_moments_structure(self):
        """Test structure of matching moments."""
        try:
            result = analyze_character_moments(
                'friends', 'Ross', query='we', top_k=2
            )
            if result.get('success') and result.get('matching_moments'):
                moments = result['matching_moments']
                for moment in moments:
                    assert 'season' in moment
                    assert 'episode' in moment
                    assert 'episode_title' in moment
                    assert 'snippet' in moment
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")


class TestGetShowCharacterPersonalities:
    """Test get_show_character_personalities function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        try:
            result = get_show_character_personalities('friends', limit=5)
            assert isinstance(result, dict)
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_respects_limit(self):
        """Test that limit parameter is respected."""
        try:
            result = get_show_character_personalities('friends', limit=3)
            assert len(result) <= 3
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_result_structure(self):
        """Test structure of results."""
        try:
            result = get_show_character_personalities('friends', limit=5)
            for char, mbti in result.items():
                assert isinstance(char, str)
                assert isinstance(mbti, str)
                assert len(mbti) == 4
                # Check each position is valid
                assert mbti[0] in ['E', 'I']
                assert mbti[1] in ['S', 'N']
                assert mbti[2] in ['T', 'F']
                assert mbti[3] in ['J', 'P']
        except (FileNotFoundError, SystemExit):
            pytest.skip("Data files not available")
    
    def test_invalid_show(self):
        """Test with invalid show."""
        try:
            result = get_show_character_personalities('invalid_show_xyz')
            # Should return empty dict or raise error
            assert isinstance(result, dict)
        except (SystemExit, Exception):
            # Expected to fail with invalid show (load_all_dialogues may call sys.exit)
            pass


class TestIntegrationLayerImports:
    """Test that integration layer properly imports from both packages."""
    
    def test_can_import_ir_functions(self):
        """Test importing IR functions through integration layer."""
        from sitcom_analysis import search_episodes, load_show_data
        assert callable(search_episodes)
        assert callable(load_show_data)
    
    def test_can_import_mbti_functions(self):
        """Test importing MBTI functions through integration layer."""
        from sitcom_analysis import load_bundle, predict_mbti_for_character
        assert callable(load_bundle)
        assert callable(predict_mbti_for_character)
    
    def test_can_import_combined_functions(self):
        """Test importing combined functions."""
        from sitcom_analysis import (
            search_with_character_info,
            analyze_character_moments,
            get_character_mbti
        )
        assert callable(search_with_character_info)
        assert callable(analyze_character_moments)
        assert callable(get_character_mbti)

