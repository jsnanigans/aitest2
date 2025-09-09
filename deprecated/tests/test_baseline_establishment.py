#!/usr/bin/env python3

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.processing.baseline_establishment import RobustBaselineEstimator


class TestRobustBaselineEstimator:
    
    def setup_method(self):
        self.estimator = RobustBaselineEstimator()
        self.base_date = datetime(2024, 1, 1)
    
    def generate_readings(self, weights, start_date=None):
        if start_date is None:
            start_date = self.base_date
        
        readings = []
        for i, weight in enumerate(weights):
            readings.append({
                'weight': weight,
                'date': start_date + timedelta(days=i),
                'source_type': 'internal-questionnaire' if i == 0 else 'patient-upload',
                'confidence': 0.8
            })
        return readings
    
    def test_simple_baseline_establishment(self):
        weights = [75.0, 75.2, 74.8, 75.1, 75.3, 74.9, 75.0]
        readings = self.generate_readings(weights)
        
        result = self.estimator.establish_baseline(readings)
        
        assert result['success'] == True
        assert 74.5 <= result['baseline_weight'] <= 75.5
        assert result['readings_used'] == 7
        assert result['outliers_removed'] == 0
        assert result['confidence'] == 'medium'
        assert result['method'] == 'IQR→Median→MAD'
    
    def test_baseline_with_outliers(self):
        weights = [75.0, 75.2, 74.8, 85.0, 75.1, 65.0, 75.3, 74.9, 75.0]
        readings = self.generate_readings(weights)
        
        result = self.estimator.establish_baseline(readings)
        
        assert result['success'] == True
        assert 74.5 <= result['baseline_weight'] <= 75.5
        assert result['outliers_removed'] >= 2
        assert result['readings_used'] <= 7
        assert result['outlier_ratio'] > 0
    
    def test_insufficient_readings(self):
        weights = [75.0, 75.2]
        readings = self.generate_readings(weights)
        
        result = self.estimator.establish_baseline(readings)
        
        assert result['success'] == False
        assert 'Insufficient readings' in result['error']
    
    def test_extreme_outliers_rejection(self):
        weights = [75.0, 85.0, 65.0, 75.2, 74.8, 90.0, 75.1]
        readings = self.generate_readings(weights)
        
        result = self.estimator.establish_baseline(readings)
        
        assert result['success'] == True
        assert 74.0 <= result['baseline_weight'] <= 76.0
        assert result['outliers_removed'] >= 2
    
    def test_mad_variance_calculation(self):
        weights = [75.0] * 10
        readings = self.generate_readings(weights)
        
        result = self.estimator.establish_baseline(readings)
        
        assert result['success'] == True
        assert result['baseline_weight'] == 75.0
        assert result['mad'] < 0.001
        assert result['measurement_variance'] >= self.estimator.min_variance
    
    def test_baseline_window_filtering(self):
        weights = list(range(70, 90))
        readings = self.generate_readings(weights)
        signup_date = self.base_date
        
        result = self.estimator.establish_baseline(readings, signup_date)
        
        assert result['success'] == True
        assert result['window_count'] <= 8
    
    def test_high_variability_detection(self):
        np.random.seed(42)
        weights = 75.0 + np.random.normal(0, 5, 20)
        readings = self.generate_readings(weights.tolist())
        
        result = self.estimator.establish_baseline(readings)
        
        assert result['success'] == True
        assert result['measurement_noise_std'] > 1.0
    
    def test_quality_validation(self):
        weights = [75.0, 75.2, 74.8, 75.1, 75.3]
        readings = self.generate_readings(weights)
        
        result = self.estimator.establish_baseline(readings)
        quality = self.estimator.validate_baseline_quality(result)
        
        assert quality['valid'] == True
        assert quality['quality_score'] >= 0.75
        assert len(quality['issues']) == 0
    
    def test_quality_validation_with_issues(self):
        weights = [75.0, 85.0, 65.0]
        readings = self.generate_readings(weights)
        
        result = self.estimator.establish_baseline(readings)
        quality = self.estimator.validate_baseline_quality(result)
        
        assert quality['valid'] == False
        assert 'Few readings' in quality['issues']
        assert len(quality['recommendations']) > 0
    
    def test_percentile_calculation(self):
        weights = list(range(70, 80))
        readings = self.generate_readings(weights)
        
        result = self.estimator.establish_baseline(readings)
        
        assert result['success'] == True
        assert 'percentiles' in result
        assert result['percentiles']['p5'] < result['percentiles']['p50']
        assert result['percentiles']['p50'] < result['percentiles']['p95']
    
    def test_iqr_fence_calculation(self):
        weights = [70, 72, 74, 75, 76, 78, 80, 100]
        readings = self.generate_readings(weights)
        
        result = self.estimator.establish_baseline(readings)
        
        assert result['success'] == True
        assert 'iqr_fences' in result
        assert result['iqr_fences']['lower'] < result['iqr_fences']['q1']
        assert result['iqr_fences']['q3'] < result['iqr_fences']['upper']
        assert result['outliers_removed'] >= 1
    
    def test_empty_readings(self):
        result = self.estimator.establish_baseline([])
        
        assert result['success'] == False
        assert result['error'] == 'No readings provided'
    
    def test_all_outliers_scenario(self):
        weights = [10, 20, 300, 400]
        readings = self.generate_readings(weights)
        
        result = self.estimator.establish_baseline(readings)
        
        if result['success']:
            assert result['confidence'] == 'low'
            assert result['outlier_ratio'] > 0.4
    
    def test_multimodal_detection(self):
        weights = [60]*5 + [90]*5
        readings = self.generate_readings(weights)
        
        result = self.estimator.establish_baseline(readings)
        
        assert result['success'] == True
        assert result['measurement_noise_std'] > 5.0
    
    @pytest.mark.parametrize("weight_range,expected_valid", [
        ([40, 41, 42], True),
        ([30, 31, 32], True),
        ([15, 16, 17], False),
        ([350, 351, 352], False)
    ])
    def test_weight_range_validation(self, weight_range, expected_valid):
        readings = self.generate_readings(weight_range)
        result = self.estimator.establish_baseline(readings)
        
        if expected_valid:
            assert result.get('success', False) == True
        else:
            assert result['success'] == False
            assert 'outside reasonable range' in result.get('error', '')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])