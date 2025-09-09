
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timedelta


@dataclass
class WeightData:
    """Represents a single weight data point."""
    id: str
    user_id: str
    timestamp: str
    weight: float
    unit: str
    database_entry_time: datetime
    effectivDateTime: datetime
    source_type: str

@dataclass
class TimeOfDayAnalysis:
    """Analysis of time-of-day patterns for weight readings."""
    hour_of_day: int
    time_since_last: Optional[timedelta] = None
    typical_time_pattern: str = 'unknown'  # 'morning', 'afternoon', 'evening', 'irregular'
    deviation_from_usual: Optional[float] = None  # hours different from user's typical time


@dataclass
class EnrichedWeightData:
    """Weight data with extensible enrichment capabilities."""
    # Core fields (immutable)
    user_id: str
    weight: float
    database_entry_time: datetime
    effective_date_time: datetime
    source_type: str

    # Enrichment fields (added by pipeline stages)
    time_of_day_analysis: Optional[TimeOfDayAnalysis] = None
    relative_change_analysis: Optional[Any] = None # Using Any for now, will import specific type later if needed
    _extensions: Dict[str, Any] = field(default_factory=dict)

    def add_extension(self, key: str, value: Any) -> 'EnrichedWeightData':
        """Add a custom extension to this data point."""
        if self._extensions is None:
            self._extensions = {}
        self._extensions[key] = value
        return self

    def get_extension(self, key: str, default: Any = None) -> Any:
        """Retrieve a custom extension value."""
        return self._extensions.get(key, default)

    @classmethod
    def from_weight_data(cls, weight_data) -> 'EnrichedWeightData':
        """Create EnrichedWeightData from existing WeightData object."""
        return cls(
            user_id=weight_data.user_id,
            weight=weight_data.weight,
            database_entry_time=weight_data.database_entry_time,
            effective_date_time=weight_data.effectivDateTime,
            source_type=weight_data.source_type
        )
