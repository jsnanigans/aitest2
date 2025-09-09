"""
Layer 2: Time-Series Modeling for Contextual Anomaly Detection
Framework Part III, Section 3.2
"""

import numpy as np
from typing import List, Optional, Tuple, Deque
from collections import deque
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import logging

from src.core.types import WeightMeasurement, OutlierType

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


class ARIMAOutlierDetector:
    """
    ARIMA-based outlier detection with classification.
    Framework: "powerful statistical method" that can classify outliers into 4 types.
    """
    
    def __init__(self, 
                 window_size: int = 30,
                 arima_order: Tuple[int, int, int] = (1, 0, 1),
                 residual_threshold: float = 3.0,
                 min_data_points: int = 10):
        self.window_size = window_size
        self.arima_order = arima_order
        self.residual_threshold = residual_threshold
        self.min_data_points = min_data_points
        self.history = deque(maxlen=window_size)
        self.model = None
        self.last_residuals = []
        
    def add_measurement(self, weight: float):
        """Add validated measurement to history."""
        self.history.append(weight)
        
    def _check_stationarity(self, series: np.ndarray) -> bool:
        """Check if series is stationary using ADF test."""
        try:
            result = adfuller(series, autolag='AIC')
            return result[1] < 0.05  # p-value < 0.05 indicates stationarity
        except:
            return False
            
    def _difference_series(self, series: np.ndarray) -> Tuple[np.ndarray, int]:
        """Difference series until stationary."""
        diff_order = 0
        current_series = series.copy()
        
        while not self._check_stationarity(current_series) and diff_order < 2:
            current_series = np.diff(current_series)
            diff_order += 1
            if len(current_series) < self.min_data_points:
                break
                
        return current_series, diff_order
        
    def _classify_outlier(self, 
                         residual: float, 
                         residual_std: float,
                         recent_residuals: List[float]) -> OutlierType:
        """
        Classify outlier type based on residual patterns.
        Framework specifies 4 types: AO, IO, LS, TC
        """
        normalized_residual = abs(residual) / residual_std if residual_std > 0 else 0
        
        # Additive Outlier: Single point anomaly
        if len(recent_residuals) < 3:
            return OutlierType.ADDITIVE
            
        # Check if previous residuals were normal
        prev_normal = all(abs(r) < residual_std * 2 for r in recent_residuals[-3:-1])
        
        if prev_normal:
            # Check next measurements to distinguish types
            if normalized_residual > 5:
                # Level Shift: Permanent change in mean
                return OutlierType.LEVEL_SHIFT
            elif normalized_residual > 3:
                # Temporary Change: Shock that decays
                return OutlierType.TEMPORARY_CHANGE
            else:
                # Additive Outlier: Isolated spike
                return OutlierType.ADDITIVE
        else:
            # Innovational Outlier: Propagating effect
            return OutlierType.INNOVATIONAL
            
    def validate(self, measurement: WeightMeasurement) -> Tuple[bool, Optional[OutlierType], dict]:
        """
        Validate measurement using ARIMA model.
        Returns: (is_valid, outlier_type, metadata)
        """
        metadata = {}
        
        # Need minimum data for ARIMA
        if len(self.history) < self.min_data_points:
            self.add_measurement(measurement.weight)
            return True, None, metadata
            
        try:
            # Prepare data
            series = np.array(self.history)
            
            # Check for stationarity and difference if needed
            working_series, diff_order = self._difference_series(series)
            
            # Adjust ARIMA order if differencing was applied
            order = (self.arima_order[0], diff_order, self.arima_order[2])
            
            # Fit ARIMA model
            model = ARIMA(series, order=order)
            fitted = model.fit()  # Remove deprecated disp parameter
            self.model = fitted
            
            # Get one-step-ahead forecast
            forecast = fitted.forecast(steps=1)[0]
            
            # Calculate residual
            residual = measurement.weight - forecast
            
            # Get residual standard deviation from model
            residual_std = np.std(fitted.resid)
            
            # Normalize residual
            if residual_std > 0:
                normalized_residual = abs(residual) / residual_std
            else:
                normalized_residual = 0
                
            metadata['forecast'] = forecast
            metadata['residual'] = residual
            metadata['normalized_residual'] = normalized_residual
            metadata['residual_std'] = residual_std
            
            # Store residuals for classification
            self.last_residuals.append(residual)
            if len(self.last_residuals) > 10:
                self.last_residuals.pop(0)
                
            # Check if outlier
            if normalized_residual > self.residual_threshold:
                outlier_type = self._classify_outlier(
                    residual, residual_std, self.last_residuals
                )
                logger.debug(f"Layer2: ARIMA outlier detected: {measurement.weight}kg, "
                           f"forecast={forecast:.1f}, residual={residual:.1f}, "
                           f"type={outlier_type.value}")
                return False, outlier_type, metadata
                
            # Measurement is valid, add to history
            self.add_measurement(measurement.weight)
            return True, None, metadata
            
        except Exception as e:
            logger.warning(f"ARIMA validation failed: {e}, accepting measurement")
            self.add_measurement(measurement.weight)
            return True, None, metadata


class Layer2Pipeline:
    """
    Orchestrates Layer 2 time-series modeling.
    More computationally intensive but catches contextual anomalies.
    """
    
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        
        self.arima_detector = ARIMAOutlierDetector(
            window_size=config.get('arima_window_size', 30),
            arima_order=config.get('arima_order', (1, 0, 1)),
            residual_threshold=config.get('residual_threshold', 3.0),
            min_data_points=config.get('min_data_points', 10)
        )
        
    def process(self, measurement: WeightMeasurement) -> Tuple[bool, Optional[OutlierType], dict]:
        """
        Process measurement through Layer 2 filters.
        Only called if Layer 1 passed.
        """
        return self.arima_detector.validate(measurement)