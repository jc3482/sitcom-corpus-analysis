#!/usr/bin/env python3
"""
Installation verification script for sitcom_analysis package.

This script checks that the package is properly installed and all
imports work correctly. For comprehensive testing, use pytest.

Run this after: pip install -e .

For unit tests, run: pytest
"""

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test information_retrieval imports
        from information_retrieval import search_episodes, load_show_data, SHOW_NAMES
        print("  [OK] information_retrieval imports")
        
        # Test mbti_prediction imports
        from mbti_prediction import load_bundle, predict_mbti_for_character
        print("  [OK] mbti_prediction imports")
        
        # Test sitcom_analysis imports
        from sitcom_analysis import (
            search_with_character_info,
            analyze_character_moments,
            get_character_mbti
        )
        print("  [OK] sitcom_analysis imports")
        
        # Test that sitcom_analysis can import from both packages
        import sitcom_analysis
        assert hasattr(sitcom_analysis, 'search_episodes')
        assert hasattr(sitcom_analysis, 'load_bundle')
        assert hasattr(sitcom_analysis, 'search_with_character_info')
        print("  [OK] sitcom_analysis exports all expected functions")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_package_structure():
    """Test that package structure is correct."""
    print("\nTesting package structure...")
    
    import os
    
    required_files = [
        'sitcom_analysis/__init__.py',
        'sitcom_analysis/__main__.py',
        'sitcom_analysis/combined_features.py',
        'information_retrieval/__init__.py',
        'mbti_prediction/__init__.py',
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  [OK] {file}")
        else:
            print(f"  [FAIL] {file} not found")
            return False
    
    return True


def test_show_data():
    """Test that show data is accessible."""
    print("\nTesting data access...")
    
    try:
        from information_retrieval import DATASETS, SHOW_NAMES
        print(f"  [OK] Available shows: {list(SHOW_NAMES.keys())}")
        
        # Test that mbti_prediction can load data (don't need internal constants)
        from mbti_prediction.mbti_prediction import SHOW_FILES, SHOW_DISPLAY_NAMES
        print(f"  [OK] MBTI shows: {list(SHOW_DISPLAY_NAMES.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Data access error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("SITCOM ANALYSIS INTEGRATION TEST")
    print("="*60)
    
    results = []
    
    results.append(test_imports())
    results.append(test_package_structure())
    results.append(test_show_data())
    
    print("\n" + "="*60)
    if all(results):
        print("SUCCESS: Installation verified!")
        print("\nYou can now use:")
        print("  - sitcom search friends 'coffee'")
        print("  - sitcom mbti")
        print("  - sitcom analyze friends 'wedding'")
        print("  - sitcom character friends 'Ross'")
        print("\nTo run unit tests:")
        print("  - pytest")
        print("  - pytest --cov")
    else:
        print("FAILURE: Installation verification failed.")
        print("Make sure you ran: pip install -e .")
    print("="*60)


if __name__ == "__main__":
    main()

