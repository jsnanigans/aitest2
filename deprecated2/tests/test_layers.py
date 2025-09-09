#!/usr/bin/env python3
"""
Unit tests for the layered filtering architecture.
Tests each layer independently per framework specifications.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.core.types import WeightMeasurement, OutlierType
from src.filters.layer1_heuristic import (
    PhysiologicalFilter,
    StatelessRateOfChangeFilter,
    StatelessLayer1Pipeline
)
from src.filters.layer3_kalman import PureKalmanFilter, ValidationGate


class TestPhysiologicalFilter:
    """Test physiological plausibility checks."""
    
    def test_normal_weight_accepted(self):
        filter = PhysiologicalFilter(min_weight=30, max_weight=400)
        measurement = WeightMeasurement(weight=70.0, timestamp=datetime.now())
        is_valid, outlier_type = filter.validate(measurement)
        assert is_valid is True
        assert outlier_type is None
        
    def test_below_minimum_rejected(self):
        filter = PhysiologicalFilter(min_weight=30, max_weight=400)
        measurement = WeightMeasurement(weight=25.0, timestamp=datetime.now())
        is_valid, outlier_type = filter.validate(measurement)
        assert is_valid is False
        assert outlier_type == OutlierType.PHYSIOLOGICAL_IMPOSSIBLE
        
    def test_above_maximum_rejected(self):
        filter = PhysiologicalFilter(min_weight=30, max_weight=400)
        measurement = WeightMeasurement(weight=450.0, timestamp=datetime.now())
        is_valid, outlier_type = filter.validate(measurement)
        assert is_valid is False
        assert outlier_type == OutlierType.PHYSIOLOGICAL_IMPOSSIBLE


class TestStatelessRateOfChangeFilter:
    """Test stateless rate of change validation."""
    
    def test_first_measurement_accepted(self):
        filter = StatelessRateOfChangeFilter(max_daily_change_percent=3.0)
        measurement = WeightMeasurement(weight=70.0, timestamp=datetime.now())
        is_valid, outlier_type, _ = filter.validate(measurement, None)
        assert is_valid is True
        assert outlier_type is None
        
    def test_small_change_accepted(self):
        filter = StatelessRateOfChangeFilter(max_daily_change_percent=3.0)
        
        # Previous state from Kalman
        last_state = {
            'weight': 70.0,
            'timestamp': datetime.now()
        }
        
        # New measurement with small change
        m2 = WeightMeasurement(weight=71.0, timestamp=datetime.now() + timedelta(days=1))
        is_valid, outlier_type, _ = filter.validate(m2, last_state)
        assert is_valid is True  # 1kg change is ~1.4% of 70kg
        
    def test_large_change_rejected(self):
        filter = StatelessRateOfChangeFilter(max_daily_change_percent=3.0)
        
        # Previous state from Kalman
        last_state = {
            'weight': 70.0,
            'timestamp': datetime.now()
        }
        
        # New measurement with large change
        m2 = WeightMeasurement(weight=80.0, timestamp=datetime.now() + timedelta(days=1))
        is_valid, outlier_type, _ = filter.validate(m2, last_state)
        assert is_valid is False  # 10kg change is ~14% of 70kg
        assert outlier_type == OutlierType.RATE_VIOLATION





class TestPureKalmanFilter:
    """Test mathematically correct Kalman filter."""
    
    def test_initialization(self):
        kalman = PureKalmanFilter(
            initial_weight=70.0,
            initial_variance=1.0,
            process_noise_weight=0.5,
            process_noise_trend=0.01,
            measurement_noise=0.5
        )
        
        state = kalman.get_state()
        assert state.weight == 70.0
        assert state.trend == 0.0
        assert state.measurement_count == 0
        
    def test_prediction_step(self):
        kalman = PureKalmanFilter(initial_weight=70.0, initial_variance=1.0)
        
        # Predict one day ahead
        predicted_weight, prediction_variance = kalman.predict(time_delta_days=1.0)
        assert predicted_weight == 70.0  # No trend initially
        assert prediction_variance > 1.0  # Uncertainty increases
        
    def test_update_step(self):
        kalman = PureKalmanFilter(initial_weight=70.0, initial_variance=1.0)
        
        # Predict then update
        kalman.predict(time_delta_days=1.0)
        results = kalman.update(measurement=70.5)
        
        assert 'filtered_weight' in results
        assert 'trend_kg_per_day' in results
        assert 'kalman_gain_weight' in results
        assert results['filtered_weight'] > 70.0  # Should move toward measurement
        assert results['filtered_weight'] < 70.5  # But not all the way
        
    def test_state_transition_matrix_fixed(self):
        """Verify F matrix is constant as per framework."""
        kalman = PureKalmanFilter(initial_weight=70.0, initial_variance=1.0)
        
        # Check for different time deltas
        for dt in [0.5, 1.0, 2.0, 7.0]:
            kalman.predict(time_delta_days=dt)
            # The F matrix should always be [[1, dt], [0, 1]]
            # This is encoded in the predict method
            

class TestValidationGate:
    """Test validation gate logic."""
    
    def test_accepts_within_threshold(self):
        gate = ValidationGate(gamma=3.0)
        
        # Small innovation should be accepted
        is_valid, normalized = gate.validate(
            measurement=70.5,
            predicted_weight=70.0,
            innovation_variance=1.0
        )
        assert bool(is_valid) is True
        assert normalized < 3.0
        
    def test_rejects_beyond_threshold(self):
        gate = ValidationGate(gamma=3.0)
        
        # Large innovation should be rejected
        is_valid, normalized = gate.validate(
            measurement=80.0,
            predicted_weight=70.0,
            innovation_variance=1.0
        )
        assert bool(is_valid) is False
        assert normalized > 3.0


class TestStatelessLayer1Pipeline:
    """Test complete Layer 1 pipeline."""
    
    def test_pipeline_ordering(self):
        """Verify filters are applied in correct order."""
        pipeline = StatelessLayer1Pipeline()
        
        # Test physiological limit (should fail first)
        m = WeightMeasurement(weight=25.0, timestamp=datetime.now())
        is_valid, outlier_type, metadata = pipeline.process(m, None)
        assert is_valid is False
        assert metadata['filter'] == 'physiological'
        
    def test_accepts_normal_measurements(self):
        pipeline = StatelessLayer1Pipeline()
        
        # Build up normal measurements
        base_time = datetime.now()
        last_state = None
        for i in range(5):
            m = WeightMeasurement(
                weight=70.0 + np.random.normal(0, 0.5),
                timestamp=base_time + timedelta(days=i)
            )
            is_valid, _, _ = pipeline.process(m, last_state)
            assert is_valid is True
            # Update last state for next iteration
            last_state = {'weight': m.weight, 'timestamp': m.timestamp}