"""API request and response models with consistent naming and standardized structure."""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ============= Enums =============

class WeightUnit(str, Enum):
    """Supported weight units."""
    KG = "kg"
    LBS = "lbs"
    LB = "lb"
    G = "g"
    OZ = "oz"
    ST = "st"  # stones - now supported

    @classmethod
    def normalize(cls, unit: str) -> str:
        """Normalize unit string."""
        unit = unit.lower().strip()
        if unit in ["lb", "lbs", "pound", "pounds"]:
            return cls.LBS
        return unit


class MeasurementSource(str, Enum):
    """Measurement source types with reliability scores."""
    CARE_TEAM_UPLOAD = "care-team-upload"  # 1.0 reliability
    PATIENT_DEVICE = "patient-device"  # 0.8 reliability
    PATIENT_UPLOAD = "patient-upload"  # 0.92 reliability
    QUESTIONNAIRE = "questionnaire"  # 0.8 reliability
    CONNECTIVE_HEALTH = "connectivehealth.io"  # 0.8 reliability
    IGLUCOSE = "iglucose.com"  # 0.8 reliability


# ============= Request Models =============

class Measurement(BaseModel):
    """Weight measurement model with improved validation."""

    measurement_id: str = Field(alias="uuid", description="Unique measurement ID")
    weight_value: float = Field(gt=0, alias="weight", description="Weight value")
    weight_unit: str = Field(alias="unit", description="Unit of measurement")
    measured_at: datetime = Field(alias="effectiveDateTime", description="When measurement was taken")
    source: str = Field(description="Data source")
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase
        use_enum_values = True

    @field_validator("weight_unit")
    @classmethod
    def validate_unit(cls, v: str) -> str:
        """Validate and normalize weight unit."""
        try:
            normalized = WeightUnit.normalize(v)
            # Check if it's a valid unit
            if normalized not in [u.value for u in WeightUnit]:
                raise ValueError(f"Unsupported weight unit: {v}")
            return normalized
        except Exception:
            raise ValueError(f"Invalid weight unit: {v}. Supported: kg, lbs, lb, g, oz, st")

    @model_validator(mode='after')
    def validate_weight_range(self) -> 'Measurement':
        """Validate weight is within physiological bounds."""
        # Convert to kg for validation
        weight_kg = self.convert_to_kg(self.weight_value, self.weight_unit)

        if weight_kg < 10:
            raise ValueError(f"Weight {weight_kg:.1f}kg is below minimum (10kg)")
        if weight_kg > 500:
            raise ValueError(f"Weight {weight_kg:.1f}kg exceeds maximum (500kg)")

        return self

    @staticmethod
    def convert_to_kg(value: float, unit: str) -> float:
        """Convert weight to kilograms."""
        unit = WeightUnit.normalize(unit)

        if unit == WeightUnit.KG:
            return value
        elif unit in [WeightUnit.LBS, WeightUnit.LB]:
            return value * 0.453592
        elif unit == WeightUnit.G:
            return value / 1000
        elif unit == WeightUnit.OZ:
            return value * 0.0283495
        elif unit == WeightUnit.ST:
            return value * 6.35029  # 1 stone = 6.35029 kg
        else:
            raise ValueError(f"Unknown unit for conversion: {unit}")


class ProcessOptions(BaseModel):
    """Options for processing with clearer names."""

    force_replay: bool = Field(False, description="Force replay mode")
    fail_on_conflict: bool = Field(True, description="Fail on historical conflict")
    include_debug_info: bool = Field(False, description="Include debug information")


class ProcessRequest(BaseModel):
    """Request to process measurements."""

    measurements: List[Measurement] = Field(..., min_items=1, description="Measurements to process")
    options: ProcessOptions = Field(default_factory=ProcessOptions)
    user_height_m: Optional[float] = Field(None, gt=0, le=3.0, description="User height in meters (optional)")


class CleanupOptions(BaseModel):
    """Options for cleanup operation - simplified."""

    cleanup_type: str = Field("reset_adaptive", description="Type of cleanup")
    preserve_buffer: bool = Field(False, description="Preserve buffer state")
    preserve_kalman: bool = Field(False, description="Preserve Kalman state")


class CleanupRequest(BaseModel):
    """Request for cleanup operation - no measurements required."""

    cleanup_type: str = Field("reset_adaptive", description="Type of cleanup operation")
    options: CleanupOptions = Field(default_factory=CleanupOptions)


class ReplayOptions(BaseModel):
    """Options for replay operation."""

    validate_order: bool = Field(True, description="Validate temporal order")
    stop_on_error: bool = Field(False, description="Stop on first error")
    use_snapshot: bool = Field(True, description="Use state snapshot if available")


# ============= Replay Models (must be before Request models that use them) =============

class ReplayWindowInfo(BaseModel):
    """Information about a replay window."""

    window_start: datetime = Field(description="Start time of replay window")
    window_end: datetime = Field(description="End time of replay window")
    measurements_in_window: int = Field(description="Number of measurements in window")
    measurement_ids: List[str] = Field(default_factory=list, description="IDs of measurements to re-evaluate")


class ReplayRequest(BaseModel):
    """Request for replay operation with consistent field naming."""

    replay_from_timestamp: datetime = Field(..., description="Timestamp to replay from")
    measurements: List[Measurement] = Field(..., min_items=1, description="Measurements to replay")
    options: ReplayOptions = Field(default_factory=ReplayOptions)
    user_height_m: Optional[float] = Field(None, gt=0, le=3.0, description="User height in meters (optional)")


class ReplayCheckRequest(BaseModel):
    """Request to check if replay should trigger."""

    user_id: str = Field(..., description="User identifier")
    current_timestamp: datetime = Field(..., description="Timestamp of last processed measurement")
    buffer_hours: Optional[int] = Field(None, ge=1, le=168, description="Replay window in hours (default: 72)")


class ReplayExecuteRequest(BaseModel):
    """Request to execute replay for a window."""

    user_id: str = Field(..., description="User identifier")
    window_info: ReplayWindowInfo = Field(..., description="Window information from replay check")


# ============= Response Models =============

class ApiMeta(BaseModel):
    """Metadata for API responses."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="2.0.0")
    request_id: Optional[str] = None


class MeasurementResult(BaseModel):
    """Result of processing a single measurement with consistent naming."""

    measurement_id: str
    accepted: bool
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    kalman_estimate: Optional[float] = None
    kalman_uncertainty: Optional[float] = None
    rejection_reason: Optional[str] = None
    processing_stage: Optional[str] = None
    reset_triggered: bool = False
    quality_components: Optional[Dict[str, float]] = None


class StateInfo(BaseModel):
    """User state information with proper user_id."""

    user_id: str
    current_weight: Optional[float] = None
    previous_weight: Optional[float] = None
    last_processed_at: Optional[datetime] = None
    measurements_count: int = 0
    last_source: Optional[str] = None
    adaptation_state: Optional[str] = None
    kalman_state: Optional[Dict[str, Any]] = None


class ProcessResponseData(BaseModel):
    """Data for process response."""

    user_id: str
    measurements_processed: int = Field(ge=0)
    measurements_accepted: int = Field(ge=0)
    measurements_rejected: int = Field(ge=0)
    results: List[MeasurementResult]
    state_update: Optional[StateInfo] = None
    replay_metadata: Optional[List[Dict[str, Any]]] = None


class StandardResponse(BaseModel):
    """Standard API response format."""

    success: bool
    data: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    meta: ApiMeta = Field(default_factory=ApiMeta)


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    documentation: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = False
    error: ErrorDetail
    meta: ApiMeta = Field(default_factory=ApiMeta)


class HealthStatus(BaseModel):
    """Health check status."""

    status: str  # "healthy", "degraded", "unhealthy"
    environment: str
    components: Dict[str, Dict[str, Any]]
    runtime: Dict[str, Any]


class CleanupResponseData(BaseModel):
    """Data for cleanup response."""

    user_id: str
    cleanup_type: str
    measurements_processed: int = 0
    state_cleared: bool = False
    message: str


class ReplayResponseData(BaseModel):
    """Data for replay response."""

    user_id: str
    replay_from: datetime
    replay_status: str = "completed"  # Status of replay operation
    measurements_processed: int
    measurements_replayed: int  # Alias for measurements_processed
    measurements_accepted: int
    measurements_rejected: int
    results: List[MeasurementResult]
    final_state: Optional[StateInfo] = None


# ============= Historical Conflict Models =============

class HistoricalConflictDetails(BaseModel):
    """Details about a historical conflict."""

    earliest_measurement_timestamp: datetime
    last_processed_timestamp: datetime
    replay_from_timestamp: datetime
    snapshot_available: Optional[datetime] = None
    conflicting_measurement_ids: List[str]


class HistoricalConflictResponse(BaseModel):
    """Response for historical conflict detection."""

    error: str
    details: HistoricalConflictDetails


class ReplayResultData(BaseModel):
    """Results from replay execution that caller must process."""

    user_id: str
    success: bool
    window_start: datetime
    window_end: datetime

    # NEW acceptance results - caller must update tracking
    measurement_results: List[MeasurementResult] = Field(
        description="One result per measurement in window with NEW acceptance status"
    )

    # Metadata
    outliers_detected: List[str] = Field(
        default_factory=list,
        description="measurement_ids marked as outliers"
    )
    outliers_count: int = Field(default=0)
    corrections_made: int = Field(default=0, description="Number of acceptance changes")
    state_restored_to: Optional[datetime] = Field(default=None)

    error: Optional[str] = None


class ReplayTriggerCheckResponse(BaseModel):
    """Response from checking if replay should trigger."""

    should_trigger: bool = Field(description="Whether replay is recommended")
    window_info: Optional[ReplayWindowInfo] = Field(
        default=None,
        description="Window information if replay should trigger"
    )


# ============= Helper Functions =============

def create_success_response(data: Any, meta: Optional[ApiMeta] = None) -> StandardResponse:
    """Create a standard success response."""
    return StandardResponse(
        success=True,
        data=data,
        meta=meta or ApiMeta()
    )


def create_error_response(
    code: str,
    message: str,
    field: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    suggestion: Optional[str] = None
) -> ErrorResponse:
    """Create a standard error response."""
    return ErrorResponse(
        success=False,
        error=ErrorDetail(
            code=code,
            message=message,
            field=field,
            details=details,
            suggestion=suggestion,
            documentation=f"https://api.docs/errors#{code.lower()}"
        ),
        meta=ApiMeta()
    )