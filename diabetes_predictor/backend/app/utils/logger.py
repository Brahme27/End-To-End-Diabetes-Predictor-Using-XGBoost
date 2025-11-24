"""
Logger Utility - Centralized logging configuration
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "diabetes_api", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Optional: File handler for production
    # log_dir = Path("logs")
    # log_dir.mkdir(exist_ok=True)
    # file_handler = logging.FileHandler(
    #     log_dir / f"api_{datetime.now().strftime('%Y%m%d')}.log"
    # )
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    
    return logger


def log_prediction(patient_data: dict, prediction: dict):
    """
    Log prediction details for monitoring
    
    Args:
        patient_data: Input patient data
        prediction: Prediction results
    """
    logger = logging.getLogger("diabetes_api")
    logger.info(
        f"PREDICTION | Age: {patient_data.get('age')} | "
        f"BMI: {patient_data.get('bmi')} | "
        f"Result: {prediction.get('prediction')} | "
        f"Probability: {prediction.get('probability'):.3f}"
    )
