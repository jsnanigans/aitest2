from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class OutlierType(Enum):
    ADDITIVE = "additive_outlier"
    INNOVATIONAL = "innovational_outlier"  
    LEVEL_SHIFT = "level_shift"
    TEMPORARY_CHANGE = "temporary_change"
    PHYSIOLOGICAL_IMPOSSIBLE = "physiological_impossible"
    RATE_VIOLATION = "rate_violation"
    DEVICE_ERROR = "device_error"


@dataclass
class WeightMeasurement:
    weight: float
    timestamp: datetime
    source_type: Optional[str] = None
    user_id: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    

@dataclass
class ProcessedMeasurement:
    measurement: WeightMeasurement
    is_valid: bool
    confidence: float
    filtered_weight: Optional[float] = None
    predicted_weight: Optional[float] = None
    trend_kg_per_day: Optional[float] = None
    outlier_type: Optional[OutlierType] = None
    rejection_reason: Optional[str] = None
    processing_metadata: Optional[Dict[str, Any]] = None


@dataclass
class BaselineResult:
    success: bool
    baseline_weight: Optional[float] = None
    measurement_variance: Optional[float] = None
    measurement_noise_std: Optional[float] = None
    confidence: Optional[str] = None  # 'high', 'medium', 'low'
    readings_used: Optional[int] = None
    method: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass  
class KalmanState:
    weight: float
    trend: float
    covariance: List[List[float]]
    timestamp: datetime
    measurement_count: int