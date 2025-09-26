"""API request and response models."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class Measurement(BaseModel):
    """Weight measurement model."""

    uuid: UUID
    weight: float = Field(gt=0, le=1000, description="Weight value")
    unit: str = Field(pattern="^(kg|lbs?|g|oz)$", description="Unit of measurement")
    effective_date_time: datetime = Field(alias="effectiveDateTime")
    source: str = Field(description="Data source")
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float, info) -> float:
        """Validate weight is within physiological bounds."""
        unit = info.data.get("unit", "kg")

        # Convert to kg for validation
        weight_kg = v
        if unit in ["lb", "lbs"]:
            weight_kg = v * 0.453592
        elif unit == "g":
            weight_kg = v / 1000
        elif unit == "oz":
            weight_kg = v * 0.0283495

        if weight_kg < 10 or weight_kg > 500:
            raise ValueError(f"Weight {weight_kg}kg outside valid range (10-500kg)")

        return v


class UserProfile(BaseModel):
    """User profile for validation."""

    height: Optional[float] = None
    height_unit: Optional[str] = "cm"
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None


class ProcessOptions(BaseModel):
    """Options for processing."""

    fail_on_historical_conflict: bool = True


class ProcessRequest(BaseModel):
    """Request to process measurements."""

    measurements: List[Measurement]
    options: Optional[ProcessOptions] = ProcessOptions()


class CleanupOptions(BaseModel):
    """Options for cleanup operation."""

    reset_state: bool = True
    include_quality_scores: bool = True
    include_debug_info: bool = False


class CleanupRequest(BaseModel):
    """Request for cleanup operation."""

    measurements: List[Measurement]
    user_profile: Optional[UserProfile] = None
    options: Optional[CleanupOptions] = CleanupOptions()


class ReplayOptions(BaseModel):
    """Options for replay operation."""

    use_snapshot: bool = True
    create_new_snapshot: bool = True


class ReplayRequest(BaseModel):
    """Request for replay operation."""

    replay_from_timestamp: datetime
    measurements: List[Measurement]
    options: Optional[ReplayOptions] = ReplayOptions()


class MeasurementResult(BaseModel):
    """Result of processing a single measurement."""

    uuid: UUID
    accepted: bool
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    kalman_estimate: Optional[float] = None
    kalman_uncertainty: Optional[float] = None
    rejection_reason: Optional[str] = None
    stage: Optional[str] = None
    reset_triggered: bool = False
    components: Optional[Dict[str, float]] = None


class StateUpdate(BaseModel):
    """State update information."""

    previous_weight: Optional[float] = None
    current_weight: Optional[float] = None
    last_processed_timestamp: datetime


class ProcessResponse(BaseModel):
    """Response from processing measurements."""

    status: str
    processed_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    measurements: List[MeasurementResult]
    state_update: Optional[StateUpdate] = None


class FinalState(BaseModel):
    """Final state after processing."""

    current_weight: float
    uncertainty: float
    last_processed_timestamp: datetime
    total_measurements: int
    adaptation_state: str


class CleanupResponse(BaseModel):
    """Response from cleanup operation."""

    user_id: str
    processed_count: int
    accepted_count: int
    rejected_count: int
    measurements: List[MeasurementResult]
    final_state: Optional[FinalState] = None


class HistoricalConflictDetails(BaseModel):
    """Details about historical conflict."""

    earliest_measurement_timestamp: datetime
    last_processed_timestamp: datetime
    replay_required: bool = True
    replay_from_timestamp: datetime
    snapshot_available: Optional[datetime] = None
    conflicting_measurements: List[str]


class HistoricalConflictResponse(BaseModel):
    """Response when historical conflict is detected."""

    status: str = "historical_conflict"
    error: str
    details: HistoricalConflictDetails
