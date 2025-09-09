#!/usr/bin/env python3
"""
Example API server demonstrating real-time weight processing.
This shows how the Kalman filter would be deployed in production.

To run:
    pip install fastapi uvicorn
    uvicorn example_api_server:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import logging

from src.processing.state_manager import RealTimeKalmanProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Weight Processing API", version="1.0.0")

# Initialize processor with configuration
processor_config = {
    'kalman': {
        'process_noise_weight': 0.5,
        'process_noise_trend': 0.01
    },
    'validation_gamma': 3.0,
    'state_storage_path': 'output/api_states'
}

processor = RealTimeKalmanProcessor(processor_config)


# Request/Response models
class WeightMeasurement(BaseModel):
    user_id: str
    weight: float
    timestamp: datetime
    source: str = "unknown"


class ProcessingResponse(BaseModel):
    accepted: bool
    confidence: float
    filtered_weight: Optional[float]
    trend_kg_per_week: Optional[float]
    prediction_error: Optional[float]
    needs_initialization: bool
    message: Optional[str] = None


class InitializationMeasurement(BaseModel):
    weight: float
    timestamp: datetime
    source: str = "unknown"


class InitializationRequest(BaseModel):
    user_id: str
    measurements: List[InitializationMeasurement]


class InitializationResponse(BaseModel):
    success: bool
    baseline_weight: Optional[float]
    baseline_confidence: Optional[str]
    message: Optional[str] = None


# API Endpoints
@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Weight Processing API"}


@app.post("/api/weight/process", response_model=ProcessingResponse)
def process_weight(measurement: WeightMeasurement):
    """
    Process a single weight measurement in real-time.
    
    This is the main production endpoint that processes each measurement
    independently using only saved state.
    """
    try:
        result = processor.process_single_measurement(
            user_id=measurement.user_id,
            weight=measurement.weight,
            timestamp=measurement.timestamp,
            source=measurement.source
        )
        
        return ProcessingResponse(
            accepted=result['accepted'],
            confidence=result['confidence'],
            filtered_weight=result.get('filtered_weight'),
            trend_kg_per_week=result.get('trend_kg_per_week'),
            prediction_error=result.get('prediction_error'),
            needs_initialization=result['needs_initialization'],
            message=result.get('message')
        )
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/weight/initialize", response_model=InitializationResponse)
def initialize_user(request: InitializationRequest):
    """
    Initialize a new user with baseline measurements.
    
    Should be called when a user has collected 3+ measurements over 7 days.
    """
    try:
        # Convert to format expected by processor
        measurements = [
            {
                'weight': m.weight,
                'timestamp': m.timestamp,
                'source': m.source
            }
            for m in request.measurements
        ]
        
        if len(measurements) < 3:
            return InitializationResponse(
                success=False,
                baseline_weight=None,
                baseline_confidence=None,
                message="At least 3 measurements required for initialization"
            )
        
        success = processor.initialize_user(request.user_id, measurements)
        
        if success:
            # Calculate baseline for response
            weights = [m['weight'] for m in measurements]
            baseline_weight = sum(weights) / len(weights)
            confidence = "high" if len(measurements) >= 5 else "medium"
            
            return InitializationResponse(
                success=True,
                baseline_weight=baseline_weight,
                baseline_confidence=confidence,
                message="User initialized successfully"
            )
        else:
            return InitializationResponse(
                success=False,
                baseline_weight=None,
                baseline_confidence=None,
                message="Initialization failed"
            )
            
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weight/state/{user_id}")
def get_user_state(user_id: str):
    """
    Get the current Kalman filter state for a user.
    
    Useful for debugging and monitoring.
    """
    try:
        state = processor.state_manager.load_state(user_id)
        
        if state is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Convert to JSON-serializable format
        return {
            "user_id": user_id,
            "kalman_state": {
                "weight": state['kalman_state'].weight,
                "trend": state['kalman_state'].trend,
                "covariance": state['kalman_state'].covariance,
                "measurement_count": state['kalman_state'].measurement_count,
                "last_timestamp": state['kalman_state'].timestamp.isoformat()
            },
            "baseline": state['baseline'],
            "last_processed": state['last_processed'].isoformat(),
            "measurement_count": state['measurement_count']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"State retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/weight/state/{user_id}")
def delete_user_state(user_id: str):
    """
    Delete the saved state for a user.
    
    Used to trigger re-initialization.
    """
    try:
        success = processor.state_manager.delete_state(user_id)
        
        if success:
            return {"message": f"State deleted for user {user_id}"}
        else:
            raise HTTPException(status_code=404, detail="User not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"State deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
def get_stats():
    """
    Get API statistics (in production, this would query actual metrics).
    """
    return {
        "total_users": "N/A",  # Would query from database
        "measurements_today": "N/A",
        "average_confidence": "N/A",
        "acceptance_rate": "N/A"
    }


# Example usage documentation
@app.get("/api/docs/examples")
def get_examples():
    """
    Get example API calls for documentation.
    """
    return {
        "process_measurement": {
            "endpoint": "POST /api/weight/process",
            "body": {
                "user_id": "user_123",
                "weight": 80.5,
                "timestamp": "2025-01-15T10:30:00Z",
                "source": "care-team-upload"
            },
            "response": {
                "accepted": True,
                "confidence": 0.95,
                "filtered_weight": 80.4,
                "trend_kg_per_week": 0.15,
                "prediction_error": 0.3,
                "needs_initialization": False
            }
        },
        "initialize_user": {
            "endpoint": "POST /api/weight/initialize",
            "body": {
                "user_id": "user_123",
                "measurements": [
                    {"weight": 80.0, "timestamp": "2025-01-01T08:00:00Z", "source": "care-team"},
                    {"weight": 80.2, "timestamp": "2025-01-02T08:00:00Z", "source": "care-team"},
                    {"weight": 79.8, "timestamp": "2025-01-03T08:00:00Z", "source": "patient"}
                ]
            },
            "response": {
                "success": True,
                "baseline_weight": 80.0,
                "baseline_confidence": "medium"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)