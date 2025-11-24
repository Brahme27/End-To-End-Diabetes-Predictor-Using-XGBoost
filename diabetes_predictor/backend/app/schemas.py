"""
Pydantic Schemas for Request/Response Validation
Based on Pima Indians Diabetes Dataset
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from datetime import datetime

class PredictionRequest(BaseModel):
    """Input schema for diabetes prediction - Pima Indians Dataset features"""
    
    # Number of times pregnant
    pregnancies: int = Field(..., ge=0, le=20, description="Number of times pregnant")
    
    # Plasma glucose concentration
    glucose: float = Field(..., ge=0, le=300, description="Plasma glucose concentration (mg/dL) - 2 hours in oral glucose tolerance test")
    
    # Blood pressure
    blood_pressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    
    # Skin thickness
    skin_thickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    
    # Insulin
    insulin: float = Field(..., ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    
    # BMI
    bmi: float = Field(..., ge=0.0, le=70.0, description="Body mass index (weight in kg/(height in m)^2)")
    
    # Diabetes pedigree function
    diabetes_pedigree_function: float = Field(..., ge=0.0, le=2.5, description="Diabetes pedigree function (genetic influence)")
    
    # Age
    age: int = Field(..., ge=21, le=100, description="Age in years")
    
    @validator('glucose')
    def validate_glucose(cls, v):
        """Check glucose levels"""
        if v < 40:
            raise ValueError('Glucose level is critically low')
        if v > 250:
            raise ValueError('Glucose level is critically high')
        return v
    
    @validator('bmi')
    def validate_bmi(cls, v):
        """Ensure BMI is reasonable"""
        if v < 10 or v > 70:
            raise ValueError('BMI seems unusual. Please verify the value.')
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "pregnancies": 6,
                "glucose": 148.0,
                "blood_pressure": 72.0,
                "skin_thickness": 35.0,
                "insulin": 79.8,
                "bmi": 33.6,
                "diabetes_pedigree_function": 0.627,
                "age": 50
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for diabetes prediction"""
    
    prediction: Literal["Positive", "Negative"] = Field(..., description="Diabetes risk prediction")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of positive prediction")
    risk_level: str = Field(..., description="Risk level category")
    confidence: float = Field(..., description="Model confidence percentage")
    explanation: Optional[dict] = Field(None, description="SHAP explanation of the prediction")
    model_version: str = Field(default="v1.0", description="Model version used")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Prediction timestamp")
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "prediction": "Positive",
                "probability": 0.75,
                "risk_level": "High",
                "confidence": 75.0,
                "model_version": "v1.0",
                "timestamp": "2025-11-22T10:30:00"
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Schema for batch predictions"""
    
    patients: list[PredictionRequest] = Field(..., description="List of patient records")
    
    @validator('patients')
    def validate_batch_size(cls, v):
        """Limit batch size"""
        if len(v) > 100:
            raise ValueError('Batch size cannot exceed 100 records')
        if len(v) == 0:
            raise ValueError('Batch must contain at least one record')
        return v


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction results"""
    
    results: list[PredictionResponse] = Field(..., description="Prediction results")
    total_records: int = Field(..., description="Total number of records processed")
    processing_time: float = Field(..., description="Time taken to process batch (seconds)")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ErrorResponse(BaseModel):
    """Error response schema"""
    
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "detail": "Invalid input data",
                "timestamp": "2025-11-22T10:30:00"
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    components: dict = Field(..., description="Component statuses")
