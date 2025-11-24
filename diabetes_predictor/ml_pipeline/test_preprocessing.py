"""
Simple test to show processed data
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.load_data import load_and_preprocess_data
from src.preprocessing.preprocessing import DiabetesPreprocessor

# Load raw data
data_dir = Path(__file__).parent / "data"
raw_data_path = data_dir / "raw" / "diabetes.csv"

print("=" * 60)
print("LOADING RAW DATA")
print("=" * 60)
print(f"Looking for data at: {raw_data_path}")
df = load_and_preprocess_data(raw_data_path)
print(f"Raw data shape: {df.shape}")
print(f"Raw columns: {list(df.columns)}")

# Initialize preprocessor
preprocessor = DiabetesPreprocessor()

# Fit and transform
print("\n" + "=" * 60)
print("APPLYING PREPROCESSING")
print("=" * 60)
X, y = preprocessor.fit_transform(df)

print(f"\nProcessed data shape: {X.shape}")
print(f"\nProcessed columns ({len(X.columns)}):")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

# Save
processed_df = X.copy()
processed_df['Outcome'] = y

processed_dir = data_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

output_path = processed_dir / "diabetes_processed.csv"
processed_df.to_csv(output_path, index=False)

print(f"\n✅ Saved to: {output_path}")
print(f"\nOriginal features: 8")
print(f"Total features after preprocessing: {X.shape[1]}")
print(f"New features created: {X.shape[1] - 8}")
