"""
Simple Character MBTI Predictor

USAGE:
    # Predict one character
    python3 predict_characters.py raw_data/tbbt_corpus.csv Penny
    
    # Predict multiple characters at once
    python3 predict_characters.py raw_data/tbbt_corpus.csv Penny Sheldon Leonard
    
    # Predict ALL characters with 50+ lines
    python3 predict_characters.py raw_data/tbbt_corpus.csv --all
    
    # Predict ALL characters with 100+ lines
    python3 predict_characters.py raw_data/tbbt_corpus.csv --all 100
"""

import pandas as pd
from mbti_personality_predictor import MBTIPredictor
import os
import sys

def predict_character(predictor, data, character_name):
    """Predict single character's MBTI type"""
    # Filter dialogue
    char_data = data[data['character'].str.lower() == character_name.lower()]
    
    if len(char_data) == 0:
        print(f"❌ No dialogue found for '{character_name}'")
        return None
    
    # Aggregate dialogue
    text = ' '.join(char_data['dialogue'].dropna().astype(str))
    lines = len(char_data)
    words = len(text.split())
    
    # Predict
    mbti = predictor.predict(text)
    
    print(f"\n{'='*60}")
    print(f"Character: {character_name}")
    print(f"Lines: {lines} | Words: {words:,}")
    print(f"Predicted MBTI: **{mbti}**")
    print(f"{'='*60}")
    
    return {'character': character_name, 'mbti': mbti, 'lines': lines, 'words': words}


def predict_all_characters(predictor, data, min_lines=50):
    """Predict all characters with sufficient dialogue"""
    char_counts = data['character'].value_counts()
    eligible = char_counts[char_counts >= min_lines]
    
    print(f"\n{'='*60}")
    print(f"Found {len(eligible)} characters with >={min_lines} lines")
    print(f"{'='*60}")
    
    results = []
    for idx, (character, line_count) in enumerate(eligible.items(), 1):
        print(f"\n[{idx}/{len(eligible)}] {character}...", end=" ")
        
        char_data = data[data['character'] == character]['dialogue'].dropna()
        text = ' '.join(char_data.astype(str))
        words = len(text.split())
        
        try:
            mbti = predictor.predict(text)
            print(f"✓ {mbti}")
            results.append({
                'character': character,
                'mbti': mbti,
                'lines': line_count,
                'words': words
            })
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Display summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    results_df = pd.DataFrame(results).sort_values('lines', ascending=False)
    print(f"{'Character':<20} {'MBTI':<6} {'Lines':<8} {'Words':<10}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['character']:<20} {row['mbti']:<6} {row['lines']:<8} {row['words']:<10,}")
    
    print(f"\n{'='*60}")
    
    # Save results
    output = data.attrs.get('filepath', 'output').replace('.csv', '_mbti_results.csv')
    results_df.to_csv(output, index=False)
    print(f"💾 Saved to: {output}")
    
    return results_df


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nEXAMPLES:")
        print("  python3 predict_characters.py raw_data/tbbt_corpus.csv Penny")
        print("  python3 predict_characters.py raw_data/tbbt_corpus.csv Sheldon Leonard Penny")
        print("  python3 predict_characters.py raw_data/tbbt_corpus.csv --all")
        print("  python3 predict_characters.py raw_data/friends_dialogues.csv Rachel Ross")
        print("\nNOTE: Train model first with: python3 run_mbti_model.py")
        sys.exit(1)
    
    corpus_file = sys.argv[1]
    
    # Check model
    model_path = 'trained_mbti_model.pkl'
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        print("Train the model first: python3 run_mbti_model.py")
        sys.exit(1)
    
    # Load model
    print("Loading model...")
    predictor = MBTIPredictor()
    predictor.load_models(model_path)
    print("✓ Model loaded")
    
    # Load data
    print(f"Loading {corpus_file}...")
    data = pd.read_csv(corpus_file)
    data.attrs['filepath'] = corpus_file
    print(f"✓ Loaded {len(data)} lines")
    
    # Check for --all flag
    if '--all' in sys.argv:
        min_lines = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        predict_all_characters(predictor, data, min_lines)
    else:
        # Predict specific characters
        characters = sys.argv[2:]
        results = []
        
        for char in characters:
            result = predict_character(predictor, data, char)
            if result:
                results.append(result)
        
        # Summary if multiple
        if len(results) > 1:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}\n")
            for r in results:
                print(f"{r['character']:<20} → {r['mbti']}")


if __name__ == "__main__":
    main()

