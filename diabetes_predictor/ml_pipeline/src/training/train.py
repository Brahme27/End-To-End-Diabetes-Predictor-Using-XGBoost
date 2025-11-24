"""
Model Training Script
Train diabetes prediction model using XGBoost and save artifacts including SHAP explainer
"""

import sys
from pathlib import Path
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
import shap
from sklearn.model_selection import RandomizedSearchCV

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import warnings
warnings.filterwarnings('ignore')

from src.utils.load_data import load_and_preprocess_data
from src.preprocessing.preprocessing import prepare_train_test_split
from src.training.evaluator import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model with hyperparameter tuning"""
    logger.info("Training XGBoost Classifier with RandomizedSearchCV...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    
    # Randomized Search
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=20,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    logger.info(f"Best parameters: {search.best_params_}")
    
    # Evaluate
    evaluator = ModelEvaluator(best_model, X_test, y_test)
    metrics = evaluator.evaluate()
    
    logger.info(f"XGBoost - Accuracy: {metrics['accuracy']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
    
    return best_model, metrics


def save_artifacts(model, preprocessor, explainer, output_dir: Path, model_name: str = "diabetes_model"):
    """Save trained model, preprocessor, and SHAP explainer"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save preprocessor
    preprocessor_path = output_dir / f"{model_name}_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    # Save SHAP explainer
    explainer_path = output_dir / f"{model_name}_explainer.joblib"
    joblib.dump(explainer, explainer_path)
    logger.info(f"SHAP explainer saved to {explainer_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'training_date': datetime.now().isoformat(),
        'features': preprocessor.feature_names if hasattr(preprocessor, 'feature_names') else None,
        'metrics': model.best_score if hasattr(model, 'best_score') else None
    }
    
    metadata_path = output_dir / f"{model_name}_metadata.joblib"
    joblib.dump(metadata, metadata_path)
    logger.info(f"Metadata saved to {metadata_path}")
    
    return model_path


def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("DIABETES RISK PREDICTION - MODEL TRAINING (XGBoost + SHAP)")
    logger.info("=" * 60)
    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    models_dir = project_root.parent / "backend" / "models"
    
    # Load data
    logger.info("\n📊 Loading real Pima Indians Diabetes dataset...")
    raw_data_path = data_dir / "raw" / "diabetes.csv"
    
    if raw_data_path.exists():
        df = load_and_preprocess_data(raw_data_path)
        logger.info(f"Loaded {len(df)} records from {raw_data_path}")
    else:
        raise FileNotFoundError(f"Dataset not found at {raw_data_path}. Please ensure diabetes.csv is in the correct location.")
    
    logger.info(f"Dataset shape: {df.shape}")
    
    # Prepare train/test split
    logger.info("\n🔄 Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(df, test_size=0.2)
    
    # Train XGBoost model
    logger.info("\n🤖 Training XGBoost model...")
    best_model, metrics = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Initialize SHAP explainer
    logger.info("\n🧠 Initializing SHAP explainer...")
    explainer = shap.TreeExplainer(best_model)
    
    # Save artifacts
    logger.info("\n💾 Saving model artifacts...")
    model_path = save_artifacts(best_model, preprocessor, explainer, models_dir)
    
    # Final evaluation report
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
