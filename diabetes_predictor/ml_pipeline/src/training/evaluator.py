"""
Model Evaluation Utilities
Performance metrics, visualization, and reporting
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and reporting
    """
    
    def __init__(self, model, X_test, y_test):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_pred_proba = None
        
    def evaluate(self) -> dict:
        """
        Perform comprehensive evaluation
        
        Returns:
            Dictionary of metrics
        """
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        
        if hasattr(self.model, 'predict_proba'):
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        else:
            self.y_pred_proba = self.y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_test, self.y_pred, zero_division=0),
            'f1': f1_score(self.y_test, self.y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba),
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        
        return confusion_matrix(self.y_test, self.y_pred)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        
        return classification_report(
            self.y_test, 
            self.y_pred,
            target_names=['No Diabetes', 'Diabetes']
        )
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance (if available)
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
        
        importance = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def print_evaluation_report(self):
        """Print comprehensive evaluation report"""
        metrics = self.evaluate()
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)
        
        print("\n📊 Performance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        print("\n🔢 Confusion Matrix:")
        cm = self.get_confusion_matrix()
        print(f"  True Negatives:  {cm[0,0]:5d}   False Positives: {cm[0,1]:5d}")
        print(f"  False Negatives: {cm[1,0]:5d}   True Positives:  {cm[1,1]:5d}")
        
        print("\n📋 Classification Report:")
        print(self.get_classification_report())
        
        print("=" * 60)


def compare_models(models_dict: dict, X_test, y_test) -> pd.DataFrame:
    """
    Compare multiple models
    
    Args:
        models_dict: Dictionary of {model_name: model}
        X_test: Test features
        y_test: Test labels
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, model in models_dict.items():
        evaluator = ModelEvaluator(model, X_test, y_test)
        metrics = evaluator.evaluate()
        metrics['model'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df.set_index('model')
    
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator(model, X_test, y_test)
    evaluator.print_evaluation_report()
