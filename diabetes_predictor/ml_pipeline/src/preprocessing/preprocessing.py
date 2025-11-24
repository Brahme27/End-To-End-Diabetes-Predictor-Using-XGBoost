"""
Preprocessing Pipeline
Feature engineering and data transformation for diabetes prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DiabetesPreprocessor:
    """
    Preprocessing pipeline for diabetes prediction model
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for real Pima Indians Diabetes dataset
        
        Args:
            df: Input DataFrame with columns: Pregnancies, Glucose, BloodPressure, 
                SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Age groups
        df['age_group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], 
                                 labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        # BMI categories
        df['bmi_category'] = pd.cut(df['BMI'], 
                                     bins=[0, 18.5, 25, 30, 100], 
                                     labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Blood pressure categories
        df['bp_category'] = pd.cut(df['BloodPressure'],
                                    bins=[0, 80, 120, 140, 300],
                                    labels=['Low', 'Normal', 'Elevated', 'High'])
        
        # Glucose categories
        df['glucose_category'] = pd.cut(df['Glucose'],
                                         bins=[0, 100, 126, 300],
                                         labels=['Normal', 'Prediabetes', 'Diabetes'])
        
        # Insulin categories (0 values are common in this dataset - missing data)
        df['insulin_available'] = (df['Insulin'] > 0).astype(int)
        
        # Pregnancy categories
        df['pregnancy_category'] = pd.cut(df['Pregnancies'], 
                                          bins=[-1, 0, 3, 6, 20],
                                          labels=['None', 'Low', 'Medium', 'High'])
        
        # BMI * Diabetes Pedigree interaction
        df['bmi_pedigree_interaction'] = df['BMI'] * df['DiabetesPedigreeFunction']
        
        # Age * Pregnancies interaction
        df['age_pregnancy_interaction'] = df['Age'] * df['Pregnancies']
        
        logger.info(f"Created engineered features. New shape: {df.shape}")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        
        # Categorical columns from real dataset feature engineering
        categorical_cols = ['age_group', 'bmi_category', 'bp_category', 'glucose_category', 'pregnancy_category']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    # One-hot encoding for training
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(col, axis=1)
                    
                    # Store column names for later use
                    if not hasattr(self, 'dummy_columns'):
                        self.dummy_columns = {}
                    self.dummy_columns[col] = list(dummies.columns)
                else:
                    # Use stored column names for inference
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    
                    # Ensure same columns as training
                    if hasattr(self, 'dummy_columns') and col in self.dummy_columns:
                        for dummy_col in self.dummy_columns[col]:
                            if dummy_col not in dummies.columns:
                                dummies[dummy_col] = 0
                        dummies = dummies[self.dummy_columns[col]]
                    
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(col, axis=1)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features from real dataset
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scaler (True for training, False for inference)
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        # Numerical columns to scale from real dataset
        numerical_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                         'bmi_pedigree_interaction', 'age_pregnancy_interaction', 'insulin_available']
        
        # Only scale columns that exist
        cols_to_scale = [col for col in numerical_cols if col in df.columns]
        
        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Outcome') -> tuple:
        """
        Fit preprocessor and transform training data
        
        Args:
            df: Training DataFrame with real dataset columns
            target_col: Name of target column (default: 'Outcome')
            
        Returns:
            Tuple of (X, y) - features and target
        """
        logger.info("Fitting preprocessor on training data...")
        
        # Separate features and target
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(target_col, axis=1)
        else:
            y = None
            X = df
        
        # Create features
        X = self.create_features(X)
        
        # Encode categorical
        X = self.encode_categorical(X, fit=True)
        
        # Scale features
        X = self.scale_features(X, fit=True)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        logger.info(f"Preprocessing complete. Features: {len(self.feature_names)}")
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        # Create features
        X = self.create_features(df)
        
        # Encode categorical
        X = self.encode_categorical(X, fit=False)
        
        # Scale features
        X = self.scale_features(X, fit=False)
        
        # Ensure same columns as training
        if self.feature_names is not None:
            # Add missing columns
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            
            # Remove extra columns and reorder
            X = X[self.feature_names]
        
        return X
    
    def save(self, path: Path):
        """Save preprocessor to disk"""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved preprocessor to {path}")
    
    @staticmethod
    def load(path: Path):
        """Load preprocessor from disk"""
        preprocessor = joblib.load(path)
        logger.info(f"Loaded preprocessor from {path}")
        return preprocessor


def prepare_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepare train/test split with preprocessing
    
    Args:
        df: Input DataFrame with 'Outcome' as target column
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    logger.info(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
    
    # Initialize preprocessor
    preprocessor = DiabetesPreprocessor()
    
    # Split before preprocessing to avoid data leakage
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Outcome'])
    
    # Fit and transform training data
    X_train, y_train = preprocessor.fit_transform(train_df)
    
    # Transform test data
    X_test = preprocessor.transform(test_df.drop('Outcome', axis=1))
    y_test = test_df['Outcome']
    
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Class distribution - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test preprocessing
    from ml_pipeline.src.utils.load_data import generate_synthetic_data
    
    df = generate_synthetic_data(n_samples=1000)
    
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(df)
    
    print("\n=== Preprocessing Summary ===")
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Feature names: {preprocessor.feature_names[:10]}... ({len(preprocessor.feature_names)} total)")
