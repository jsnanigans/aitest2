#!/usr/bin/env python3
"""
Unit tests for robust baseline establishment.
Tests framework Part II implementation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.core.types import WeightMeasurement, BaselineResult
from src.processing.robust_baseline import RobustBaselineEstimator


class TestRobustBaselineEstimator:
    """Test baseline establishment per framework specifications."""
    
    def setup_method(self):
        """Create test data for each test."""
        self.estimator = RobustBaselineEstimator()
        
        # Generate clean test measurements
        base_time = datetime.now()
        self.clean_measurements = [
            WeightMeasurement(
                weight=70.0 + np.random.normal(0, 0.2),
                timestamp=base_time + timedelta(days=i)
            )
            for i in range(14)
        ]
        
        # Generate measurements with outliers
        self.noisy_measurements = self.clean_measurements.copy()
        # Add outliers
        self.noisy_measurements[2] = WeightMeasurement(
            weight=85.0,  # Outlier
            timestamp=base_time + timedelta(days=2)
        )
        self.noisy_measurements[7] = WeightMeasurement(
            weight=55.0,  # Outlier
            timestamp=base_time + timedelta(days=7)
        )
        
    def test_successful_baseline_establishment(self):
        """Test baseline with clean data."""
        result = self.estimator.establish_baseline(self.clean_measurements[:7])
        
        assert result.success is True
        assert result.baseline_weight is not None
        assert 69.5 < result.baseline_weight < 70.5  # Should be close to 70
        assert result.measurement_variance > 0
        assert result.confidence in ['high', 'medium', 'low']
        assert result.method == "IQR→Median→MAD"
        
    def test_iqr_outlier_removal(self):
        """Test that IQR removes outliers correctly."""
        result = self.estimator.establish_baseline(self.noisy_measurements[:10])
        
        assert result.success is True
        # Baseline should be close to 70, not affected by outliers
        assert 69.5 < result.baseline_weight < 70.5
        assert result.metadata['outliers_removed'] >= 1  # At least one outlier removed
        
    def test_insufficient_data(self):
        """Test with too few measurements."""
        result = self.estimator.establish_baseline(self.clean_measurements[:2])
        
        assert result.success is False
        assert "Insufficient readings" in result.error
        
    def test_window_filtering(self):
        """Test that baseline window is correctly applied."""
        # Set specific start date
        start_date = datetime.now() - timedelta(days=20)
        
        # Create measurements spread over time
        measurements = []
        for i in range(30):
            measurements.append(WeightMeasurement(
                weight=70.0 + i * 0.1,  # Increasing weight
                timestamp=start_date + timedelta(days=i)
            ))
            
        result = self.estimator.establish_baseline(measurements, start_date)
        
        assert result.success is True
        # Should use only first 7 days (default window)
        assert result.baseline_weight < 71.0  # Should be around 70.3
        assert result.readings_used <= 7
        
    def test_mad_variance_calculation(self):
        """Test MAD-based variance estimation."""
        # Create measurements with known variance
        measurements = [
            WeightMeasurement(weight=70.0, timestamp=datetime.now()),
            WeightMeasurement(weight=70.5, timestamp=datetime.now()),
            WeightMeasurement(weight=69.5, timestamp=datetime.now()),
            WeightMeasurement(weight=70.0, timestamp=datetime.now()),
        ]
        
        result = self.estimator.establish_baseline(measurements)
        
        assert result.success is True
        assert result.measurement_variance > 0
        # MAD should capture the spread
        assert result.metadata['mad'] > 0
        
    def test_confidence_assessment(self):
        """Test confidence level assignment."""
        # High confidence: many clean readings
        result_high = self.estimator.establish_baseline(self.clean_measurements[:12])
        assert result_high.confidence == 'high'
        
        # Low confidence: few readings
        result_low = self.estimator.establish_baseline(self.clean_measurements[:3])
        assert result_low.confidence == 'low'
        
    def test_percentile_calculation(self):
        """Test that percentiles are calculated for sufficient data."""
        result = self.estimator.establish_baseline(self.clean_measurements[:10])
        
        assert result.success is True
        assert 'percentiles' in result.metadata
        percentiles = result.metadata['percentiles']
        
        # Check ordering
        assert percentiles['p5'] <= percentiles['p25']
        assert percentiles['p25'] <= percentiles['p50']
        assert percentiles['p50'] <= percentiles['p75']
        assert percentiles['p75'] <= percentiles['p95']
        
    def test_trimmed_mean_fallback(self):
        """Test fallback to trimmed mean when IQR too aggressive."""
        # Create data where IQR would remove too many points
        measurements = [
            WeightMeasurement(weight=60.0, timestamp=datetime.now()),
            WeightMeasurement(weight=70.0, timestamp=datetime.now()),
            WeightMeasurement(weight=80.0, timestamp=datetime.now()),
            WeightMeasurement(weight=90.0, timestamp=datetime.now()),
        ]
        
        # This should trigger fallback logic
        result = self.estimator.establish_baseline(measurements)
        # Should still succeed with fallback method
        assert result.success is True