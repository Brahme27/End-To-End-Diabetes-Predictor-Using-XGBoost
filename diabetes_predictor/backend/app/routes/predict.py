"""
Prediction Routes - /predict endpoint
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

from ..schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse
)
from ..services.predictor import DiabetesPredictor

# Initialize router
router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize predictor (singleton pattern)
predictor = DiabetesPredictor()

@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction error"}
    }
)
async def predict_diabetes(request: PredictionRequest):
    """
    Predict diabetes risk for a single patient
    
    **Input:**
    - Patient demographics, physical measurements, lifestyle factors, and lab values
    
    **Output:**
    - Prediction (Positive/Negative)
    - Probability score
    - Risk level classification
    - Model confidence
    """
    try:
        logger.info(f"Received prediction request for patient: age={request.age}, glucose={request.glucose}, bmi={request.bmi}")
        
        # Make prediction
        result = predictor.predict(request)
        
        logger.info(f"Prediction completed: {result['prediction']} (probability: {result['probability']:.3f})")
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Batch prediction error"}
    }
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict diabetes risk for multiple patients
    
    **Input:**
    - List of patient records (max 100)
    
    **Output:**
    - Array of predictions
    - Processing statistics
    """
    try:
        start_time = datetime.now()
        logger.info(f"Received batch prediction request for {len(request.patients)} patients")
        
        # Process each patient
        results = []
        for idx, patient in enumerate(request.patients):
            try:
                result = predictor.predict(patient)
                results.append(PredictionResponse(**result))
            except Exception as e:
                logger.error(f"Error processing patient {idx}: {str(e)}")
                # Add error result
                results.append({
                    "prediction": "Error",
                    "probability": 0.0,
                    "risk_level": "Unknown",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Batch prediction completed: {len(results)} results in {processing_time:.2f}s")
        
        return BatchPredictionResponse(
            results=results,
            total_records=len(results),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/predict/info")
async def model_info():
    """
    Get information about the prediction model
    
    **Returns:**
    - Model version
    - Model type
    - Features used
    - Performance metrics
    """
    try:
        info = predictor.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
