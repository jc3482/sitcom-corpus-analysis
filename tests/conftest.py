"""
Pytest configuration and fixtures

Shared test fixtures used across all test modules.
"""

import pytest
import pandas as pd
import os


@pytest.fixture
def sample_dialogue_data():
    """Sample dialogue data for testing."""
    return pd.DataFrame({
        'season': [1, 1, 1, 2, 2],
        'episode': [1, 1, 2, 1, 1],
        'episode_title': ['Pilot', 'Pilot', 'The Second One', 'Season Two', 'Season Two'],
        'character': ['Ross', 'Rachel', 'Ross', 'Monica', 'Chandler'],
        'dialogue': [
            'I love dinosaurs and paleontology.',
            'I work at the coffee shop.',
            'We were on a break!',
            'Everything must be clean and organized.',
            'Could this BE any funnier?'
        ]
    })


@pytest.fixture
def sample_episode_data():
    """Sample episode-level data for IR testing."""
    df = pd.DataFrame({
        'season': [1, 1, 2],
        'episode': [1, 2, 1],
        'episode_title': ['Pilot', 'The One About Coffee', 'The Wedding'],
        'dialogue': [
            'Welcome to Central Perk. This is a coffee shop.',
            'Let us talk about coffee and more coffee.',
            'The wedding was beautiful. Everyone loved it.'
        ],
        'show_key': ['friends', 'friends', 'friends']
    })
    # Add dialogue_raw column that search expects
    df['dialogue_raw'] = df['dialogue']
    return df


@pytest.fixture
def sample_mbti_data():
    """Sample MBTI training data."""
    return pd.DataFrame({
        'type': ['INTJ', 'ENFP', 'ISTJ'],
        'posts': [
            'I enjoy analyzing complex systems and planning ahead.',
            'I love meeting new people and exploring creative ideas!',
            'I prefer following established procedures and routines.'
        ]
    })


@pytest.fixture
def mock_show_data(tmp_path):
    """Create temporary CSV files for testing."""
    test_data = pd.DataFrame({
        'season': [1, 1],
        'episode': [1, 1],
        'episode_title': ['Test Episode', 'Test Episode'],
        'character': ['TestChar1', 'TestChar2'],
        'dialogue': ['This is test dialogue one.', 'This is test dialogue two.']
    })
    
    csv_path = tmp_path / "test_show.csv"
    test_data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_query_strings():
    """Sample queries for testing."""
    return {
        'simple': 'coffee',
        'phrase': 'coffee shop',
        'and': 'coffee&shop',
        'or': 'coffee/shop',
        'complex': 'wedding&beautiful'
    }


@pytest.fixture
def skip_if_no_data():
    """Skip tests if raw data files don't exist."""
    data_dir = 'raw_data'
    if not os.path.exists(data_dir):
        pytest.skip(f"Data directory {data_dir} not found")
    
    required_files = [
        'friends_dialogues.csv',
        'mbti_data.csv'
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            pytest.skip(f"Required data file {file} not found")

