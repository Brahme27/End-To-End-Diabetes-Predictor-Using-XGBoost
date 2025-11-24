"""
Predictor Service - Loads model and performs inference
"""

import sys
from pathlib import Path

# Add ml_pipeline to path so model can be loaded
ml_pipeline_path = Path(__file__).parent.parent.parent.parent / "ml_pipeline"
if ml_pipeline_path.exists() and str(ml_pipeline_path) not in sys.path:
    sys.path.insert(0, str(ml_pipeline_path))

import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any
import shap

from ..schemas import PredictionRequest

logger = logging.getLogger(__name__)

class DiabetesPredictor:
    """
    ML Model service for diabetes prediction
    Handles model loading, preprocessing, and inference
    """
    
    def __init__(self):
        """Initialize predictor and load model"""
        self.model = None
        self.scaler = None
        self.preprocessor = None
        self.explainer = None
        self.feature_names = None
        self.model_version = "v1.0"
        self.model_type = "XGBoost Classifier"
        self.metadata = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk"""
        try:
            models_dir = Path(__file__).parent.parent.parent / "models"
            model_path = models_dir / "diabetes_model.joblib"
            preprocessor_path = models_dir / "diabetes_model_preprocessor.joblib"
            explainer_path = models_dir / "diabetes_model_explainer.joblib"
            metadata_path = models_dir / "diabetes_model_metadata.joblib"
            
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"✅ Model loaded successfully from {model_path}")
                
                # Load preprocessor (critical for feature engineering)
                if preprocessor_path.exists():
                    self.preprocessor = joblib.load(preprocessor_path)
                    logger.info(f"✅ Preprocessor loaded")
                
                # Load SHAP explainer
                if explainer_path.exists():
                    self.explainer = joblib.load(explainer_path)
                    logger.info(f"✅ SHAP Explainer loaded")
                
                # Load metadata
                if metadata_path.exists():
                    self.metadata = joblib.load(metadata_path)
                    self.model_type = self.metadata.get('model_type', 'XGBoostClassifier')
                    self.feature_names = self.metadata.get('features', [])
                    logger.info(f"✅ Metadata loaded")
            else:
                logger.warning(f"⚠️ Model file not found at {model_path}")
                logger.warning("⚠️ Using dummy model for development/testing")
                self._create_dummy_model()
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            logger.warning("⚠️ Falling back to dummy model")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a simple rule-based model for testing"""
        logger.info("Creating dummy model for development...")
        self.model = "dummy"
        self.model_type = "Rule-Based (Development)"
    
    def _preprocess_input(self, request: PredictionRequest) -> pd.DataFrame:
        """
        Convert request to feature vector matching Pima Indians dataset
        Apply same preprocessing as training
        """
        # Create feature dictionary with dataset column names
        features = {
            'Pregnancies': request.pregnancies,
            'Glucose': request.glucose,
            'BloodPressure': request.blood_pressure,
            'SkinThickness': request.skin_thickness,
            'Insulin': request.insulin,
            'BMI': request.bmi,
            'DiabetesPedigreeFunction': request.diabetes_pedigree_function,
            'Age': request.age,
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Apply preprocessing pipeline if available
        if self.preprocessor is not None:
            try:
                # Use the trained preprocessor for feature engineering and scaling
                df_processed = self.preprocessor.transform(df)
                logger.debug(f"Applied preprocessor: {df.shape} -> {df_processed.shape}")
                return df_processed
            except Exception as e:
                logger.warning(f"Preprocessor failed: {e}. Using raw features.")
                return df
        
        # If no preprocessor, return raw features
        return df
    
    def _calculate_risk_score(self, request: PredictionRequest) -> float:
        """
        Rule-based risk calculation for dummy model
        Based on Pima Indians dataset clinical factors
        """
        risk_score = 0.0
        
        # Age risk (increases with age)
        if request.age >= 60:
            risk_score += 0.20
        elif request.age >= 45:
            risk_score += 0.15
        elif request.age >= 30:
            risk_score += 0.10
        
        # BMI risk (obesity indicator)
        if request.bmi >= 40:
            risk_score += 0.25
        elif request.bmi >= 35:
            risk_score += 0.20
        elif request.bmi >= 30:
            risk_score += 0.15
        
        # Blood pressure (diastolic)
        if request.blood_pressure >= 90:
            risk_score += 0.15
        elif request.blood_pressure >= 80:
            risk_score += 0.10
        
        # Glucose (most important factor)
        if request.glucose >= 140:
            risk_score += 0.35
        elif request.glucose >= 126:
            risk_score += 0.25
        elif request.glucose >= 100:
            risk_score += 0.15
        
        # Insulin levels
        if request.insulin > 200:
            risk_score += 0.15
        elif request.insulin == 0:
            # Missing insulin data is common in this dataset
            risk_score += 0.05
        
        # Pregnancies (higher number increases risk)
        if request.pregnancies >= 6:
            risk_score += 0.15
        elif request.pregnancies >= 3:
            risk_score += 0.10
        
        # Diabetes Pedigree Function (genetic factor)
        if request.diabetes_pedigree_function > 0.8:
            risk_score += 0.20
        elif request.diabetes_pedigree_function > 0.5:
            risk_score += 0.15
        elif request.diabetes_pedigree_function > 0.3:
            risk_score += 0.10
        
        # Skin thickness (indirect measure of body fat)
        if request.skin_thickness > 40:
            risk_score += 0.10
        elif request.skin_thickness == 0:
            # Missing data
            risk_score += 0.02
        
        # Normalize to 0-1 range
        risk_score = min(max(risk_score, 0.0), 1.0)
        
        return risk_score
    
    def predict(self, request: PredictionRequest) -> Dict[str, Any]:
        """
        Make prediction for a single patient
        
        Args:
            request: PredictionRequest with patient data
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess input
            features = self._preprocess_input(request)
            
            explanation = None
            
            # Make prediction
            if self.model == "dummy":
                # Use rule-based model
                probability = self._calculate_risk_score(request)
                prediction = 1 if probability >= 0.5 else 0
            else:
                # Use trained ML model
                prediction = self.model.predict(features)[0]
                probability = self.model.predict_proba(features)[0][1]
                
                # Generate SHAP explanation
                if self.explainer is not None:
                    try:
                        logger.info("Generating SHAP explanation...")
                        shap_values = self.explainer.shap_values(features)
                        logger.info(f"SHAP values type: {type(shap_values)}")
                        
                        # Handle different SHAP output formats (some return list for classes)
                        if isinstance(shap_values, list):
                            logger.info(f"SHAP returned list with {len(shap_values)} elements")
                            shap_values = shap_values[1] # Positive class
                        
                        # Create explanation dictionary
                        # Map feature names to shap values
                        feature_names = features.columns.tolist()
                        
                        # Handle expected_value format
                        expected_val = self.explainer.expected_value
                        if isinstance(expected_val, (list, np.ndarray)):
                            expected_val = float(expected_val[1])  # Positive class
                        else:
                            expected_val = float(expected_val)
                        
                        explanation = {
                            "feature_names": feature_names,
                            "shap_values": shap_values[0].tolist(), # First (and only) sample
                            "base_value": expected_val
                        }
                        logger.info(f"✅ SHAP explanation generated with {len(feature_names)} features")
                    except Exception as e:
                        logger.error(f"❌ SHAP explanation failed: {e}", exc_info=True)
                else:
                    logger.warning("⚠️ SHAP explainer is not loaded!")

            
            # Determine risk level
            if probability >= 0.7:
                risk_level = "High"
            elif probability >= 0.4:
                risk_level = "Moderate"
            else:
                risk_level = "Low"
            
            # Prepare response
            result = {
                "prediction": "Positive" if prediction == 1 else "Negative",
                "probability": float(probability),
                "risk_level": risk_level,
                "confidence": float(probability * 100),
                "explanation": explanation,
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Failed to make prediction: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_version": self.model_version,
            "model_type": self.model_type,
            "status": "loaded" if self.model is not None else "not loaded",
            "features": self.feature_names if self.feature_names else [
                "pregnancies", "glucose", "blood_pressure", "skin_thickness",
                "insulin", "bmi", "diabetes_pedigree_function", "age"
            ],
            "description": "Diabetes risk prediction model trained on Pima Indians Diabetes Dataset"
        }
