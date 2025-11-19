"""
Run MBTI Personality Predictor on your dataset
This script will train the model on raw_data/mbti_data.csv and show results
"""

import pandas as pd
from mbti_personality_predictor import MBTIPredictor
import time

print("="*70)
print("MBTI PERSONALITY PREDICTOR - TRAINING ON YOUR DATA")
print("="*70)

# Load your data
print("\n📂 Loading data from raw_data/mbti_data.csv...")
data = pd.read_csv('raw_data/mbti_data.csv')

print(f"✓ Data loaded successfully!")
print(f"   Dataset shape: {data.shape}")
print(f"   Total samples: {len(data)}")

# Show distribution
print(f"\n📊 MBTI Type Distribution:")
print("-" * 50)
type_counts = data['type'].value_counts().sort_index()
for mbti_type, count in type_counts.items():
    percentage = (count / len(data)) * 100
    bar = "█" * int(percentage / 2)  # Visual bar
    print(f"   {mbti_type}: {count:4d} ({percentage:5.2f}%) {bar}")

print(f"\n   Most common: {type_counts.idxmax()} ({type_counts.max()} samples)")
print(f"   Least common: {type_counts.idxmin()} ({type_counts.min()} samples)")

# Check data quality
print(f"\n🔍 Data Quality Check:")
print("-" * 50)
data['post_length'] = data['posts'].str.len()
data['word_count'] = data['posts'].str.split().str.len()

print(f"   Average post length: {data['post_length'].mean():.0f} characters")
print(f"   Average word count: {data['word_count'].mean():.0f} words")
print(f"   Shortest post: {data['post_length'].min()} characters")
print(f"   Longest post: {data['post_length'].max()} characters")

# Create and train predictor
print("\n" + "="*70)
print("🚀 TRAINING MODEL")
print("="*70)
print("\nThis may take a few minutes depending on your dataset size...")
print("Please be patient...\n")

start_time = time.time()

predictor = MBTIPredictor()
predictor.train(data, test_size=0.33, min_words=15)

training_time = time.time() - start_time
print(f"\n⏱️  Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Evaluate the model
print("\n" + "="*70)
print("📈 MODEL EVALUATION RESULTS")
print("="*70)

accuracies = predictor.evaluate(detailed=True)

# Summary
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

avg_accuracy = sum(accuracies.values()) / len(accuracies)
print(f"\n✓ Average Accuracy: {avg_accuracy:.2f}%")

print("\n📋 Individual Dimension Accuracies:")
for dimension, accuracy in accuracies.items():
    dim_short = dimension.split(':')[0]
    print(f"   {dim_short}: {accuracy:.2f}%")

# Compare to expectations
print("\n💭 How did the model perform?")
print("-" * 50)
if avg_accuracy >= 80:
    print("   🌟 EXCELLENT! The model performs very well on your data.")
elif avg_accuracy >= 70:
    print("   ✓ GOOD! The model performs reasonably well.")
    print("   This is typical for MBTI text-based prediction.")
elif avg_accuracy >= 60:
    print("   ⚠ FAIR. The model performs at an acceptable level.")
    print("   Consider: more data, longer posts, or balanced types.")
else:
    print("   ❌ BELOW EXPECTATIONS. The model struggles with this data.")
    print("   Possible reasons: imbalanced data, short posts, or noisy data.")

print("\n💡 Expected Performance (for reference):")
print("   I/E: 75-80%")
print("   N/S: 85-90%")
print("   T/F: 70-75%")
print("   J/P: 70-75%")

# Test predictions
print("\n" + "="*70)
print("🎯 TESTING PREDICTIONS")
print("="*70)

test_cases = [
    "I love spending time alone reading books and thinking deeply about philosophical questions. I prefer meaningful one-on-one conversations.",
    "I'm energized by being around people! I love organizing events and being the center of attention. Let's party!",
    "I focus on facts and concrete details. I'm very practical and rely on past experiences to make decisions.",
    "I enjoy exploring abstract theories and future possibilities. What-if scenarios fascinate me.",
    "I make decisions based on logic and objective analysis, not feelings.",
    "I care deeply about people's feelings and try to maintain harmony in relationships.",
    "I prefer structure and planning. I like to have everything organized and decided in advance.",
    "I prefer to keep my options open and go with the flow rather than making rigid plans."
]

print("\nTesting on sample texts:")
print("-" * 50)

for i, text in enumerate(test_cases, 1):
    prediction = predictor.predict(text)
    print(f"\n{i}. Text: {text[:60]}...")
    print(f"   Predicted MBTI: {prediction}")

# Save the model
print("\n" + "="*70)
print("💾 SAVING MODEL")
print("="*70)

model_path = 'trained_mbti_model.pkl'
predictor.save_models(model_path)
print(f"✓ Model saved to: {model_path}")
print(f"  You can load this later without retraining!")

print("\n" + "="*70)
print("✅ COMPLETE!")
print("="*70)
print("""
Next steps:
1. Review the accuracy results above
2. Use the saved model for predictions: 
   
   predictor = MBTIPredictor()
   predictor.load_models('trained_mbti_model.pkl')
   result = predictor.predict("Your text here")

3. See example_usage.py for more ways to use the model
""")
print("="*70)

