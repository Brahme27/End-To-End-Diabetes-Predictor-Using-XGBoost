"""
Data Loading Utilities
Handles loading the Pima Indians Diabetes Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_path: Path = None) -> pd.DataFrame:
    """
    Load and perform initial preprocessing on the real diabetes dataset
    
    Args:
        data_path: Path to the diabetes.csv file (Pima Indians Diabetes Dataset)
        
    Returns:
        Preprocessed DataFrame with columns: Pregnancies, Glucose, BloodPressure, 
        SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
    """
    try:
        if data_path is None:
            # Try to find diabetes.csv in common locations
            possible_paths = [
                Path('diabetes.csv'),
                Path('ml_pipeline/data/raw/diabetes.csv'),
                Path('../diabetes.csv'),
                Path('../../diabetes.csv')
            ]
            
            for path in possible_paths:
                if path.exists():
                    data_path = path
                    break
        
        if data_path is None or not Path(data_path).exists():
            raise FileNotFoundError("diabetes.csv not found. Please provide the path to the dataset.")
            
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        
        # Verify expected columns
        expected_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"Dataset must contain columns: {expected_cols}")
        
        # Handle zero values which represent missing data in this dataset
        # Zero is not a valid value for Glucose, BloodPressure, SkinThickness, Insulin, BMI
        # We'll replace with median for now
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if col in df.columns:
                median_val = df[df[col] != 0][col].median()
                df[col] = df[col].replace(0, median_val)
        
        logger.info(f"Loaded {len(df)} records with {df.shape[1]} features")
        logger.info(f"Diabetes cases: {df['Outcome'].sum()} ({df['Outcome'].mean():.2%})")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def save_processed_data(df: pd.DataFrame, output_path: Path):
    """
    Save processed data to disk
    
    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'features': list(df.columns),
        'diabetes_cases': int(df['Outcome'].sum()) if 'Outcome' in df.columns else None,
        'diabetes_rate': df['Outcome'].mean() if 'Outcome' in df.columns else None,
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict()
    }
    
    return summary


def generate_synthetic_data(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data matching Pima Indians Diabetes Dataset schema
    Useful for testing purposes
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic diabetes data
    """
    np.random.seed(random_state)
    
    # Generate features with realistic distributions
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples).clip(0, 300),
        'BloodPressure': np.random.normal(70, 12, n_samples).clip(0, 140),
        'SkinThickness': np.random.normal(20, 15, n_samples).clip(0, 99),
        'Insulin': np.random.normal(80, 100, n_samples).clip(0, 900),
        'BMI': np.random.normal(32, 7, n_samples).clip(10, 70),
        'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate outcome based on risk factors (simplified model)
    risk_score = (
        (df['Glucose'] > 140).astype(int) * 0.3 +
        (df['BMI'] > 30).astype(int) * 0.2 +
        (df['Age'] > 45).astype(int) * 0.15 +
        (df['Pregnancies'] > 6).astype(int) * 0.15 +
        (df['BloodPressure'] > 80).astype(int) * 0.1 +
        (df['DiabetesPedigreeFunction'] > 0.5).astype(int) * 0.1 +
        np.random.uniform(-0.2, 0.2, n_samples)  # Add randomness
    )
    
    df['Outcome'] = (risk_score > 0.5).astype(int)
    
    # For compatibility with old tests that use 'diabetes' column name
    df['diabetes'] = df['Outcome']
    
    return df



if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load real dataset
    data_dir = Path(__file__).parent.parent / "data"
    raw_path = data_dir / "raw" / "diabetes.csv"
    
    if raw_path.exists():
        df = load_and_preprocess_data(raw_path)
        
        # Print summary
        summary = get_data_summary(df)
        print("\n=== Dataset Summary ===")
        print(f"Total Records: {summary['total_records']}")
        print(f"Diabetes Cases: {summary['diabetes_cases']}")
        print(f"Diabetes Rate: {summary['diabetes_rate']:.2%}")
    else:
        print(f"Dataset not found at {raw_path}")

