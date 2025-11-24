"""
Script to generate and save processed dataset
Shows the result of feature engineering and preprocessing
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.load_data import load_and_preprocess_data
from src.preprocessing.preprocessing import DiabetesPreprocessor

def main():
    # Load raw data
    data_dir = Path(__file__).parent.parent / "data"
    raw_data_path = data_dir / "raw" / "diabetes.csv"
    
    print("Loading raw data...")
    df = load_and_preprocess_data(raw_data_path)
    print(f"Raw data shape: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = DiabetesPreprocessor()
    
    # Fit and transform (without scaling to keep values readable)
    print("\nApplying feature engineering...")
    X, y = preprocessor.fit_transform(df)
    
    # Combine features and target
    processed_df = X.copy()
    processed_df['Outcome'] = y
    
    # Save to processed folder
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = processed_dir / "diabetes_processed.csv"
    processed_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Processed data saved to: {output_path}")
    print(f"Processed data shape: {processed_df.shape}")
    print(f"\nNew features created: {processed_df.shape[1] - 9}")  # Original 8 + Outcome
    print(f"\nFirst few columns: {list(processed_df.columns[:10])}")
    
    # Show summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Original features: 8")
    print(f"Engineered features: {processed_df.shape[1] - 1}")  # Excluding Outcome
    print(f"Total samples: {len(processed_df)}")
    print("="*60)

if __name__ == "__main__":
    main()
