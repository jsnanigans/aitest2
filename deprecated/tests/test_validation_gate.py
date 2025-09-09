import pytest
import numpy as np
from datetime import datetime, timedelta
from src.filters.validation_gate import ValidationGate
from src.filters.custom_kalman_filter import CustomKalmanFilter


class TestValidationGate:
    
    def test_initialization(self):
        gate = ValidationGate(gamma=3.0, enable_adaptive=True)
        assert gate.gamma == 3.0
        assert gate.base_gamma == 3.0
        assert gate.enable_adaptive == True
        assert gate.stats['accepted'] == 0
        assert gate.stats['rejected'] == 0
    
    def test_valid_measurement(self):
        gate = ValidationGate(gamma=3.0)
        
        measurement = 70.0
        prediction = 70.5
        innovation_covariance = 1.0
        
        is_valid, confidence, reason = gate.validate(
            measurement=measurement,
            prediction=prediction,
            innovation_covariance=innovation_covariance
        )
        
        assert is_valid == True
        assert confidence > 0.5
        assert reason is None
        assert gate.stats['accepted'] == 1
        assert gate.stats['rejected'] == 0
    
    def test_outlier_rejection(self):
        gate = ValidationGate(gamma=3.0)
        
        measurement = 70.0
        prediction = 80.0  # 10kg difference
        innovation_covariance = 1.0  # Small uncertainty -> high normalized innovation
        
        is_valid, confidence, reason = gate.validate(
            measurement=measurement,
            prediction=prediction,
            innovation_covariance=innovation_covariance
        )
        
        assert is_valid == False
        assert confidence < 0.1
        assert reason in ['extreme_outlier', 'severe_deviation', 'major_deviation', 'exceeds_threshold']
        assert gate.stats['accepted'] == 0
        assert gate.stats['rejected'] == 1
    
    def test_edge_case_zero_uncertainty(self):
        gate = ValidationGate(gamma=3.0)
        
        measurement = 70.0
        prediction = 70.5
        innovation_covariance = 0.0  # Zero uncertainty
        
        is_valid, confidence, reason = gate.validate(
            measurement=measurement,
            prediction=prediction,
            innovation_covariance=innovation_covariance
        )
        
        # Should handle gracefully
        assert is_valid == True
        assert confidence == 0.5
        assert reason is None
    
    def test_multi_level_feedback(self):
        gate = ValidationGate(gamma=3.0)
        
        test_cases = [
            (70.0, 70.5, 1.0, 'normal'),           # 0.5σ deviation
            (70.0, 72.2, 1.0, 'marginal'),         # 2.2σ deviation  
            (70.0, 72.8, 1.0, 'suspicious'),       # 2.8σ deviation
            (70.0, 73.5, 1.0, 'likely_error'),     # 3.5σ deviation
            (70.0, 76.0, 1.0, 'rejected'),         # 6σ deviation
        ]
        
        for measurement, prediction, cov, expected_status in test_cases:
            status, message, normalized = gate.get_multi_level_feedback(
                measurement=measurement,
                prediction=prediction,
                innovation_covariance=cov
            )
            assert status == expected_status
    
    def test_adaptive_gamma_calculation(self):
        gate = ValidationGate(gamma=3.0, enable_adaptive=True)
        
        # Create history with low variability (std < 0.5)
        stable_history = [
            {'normalized_innovation': 0.2 + 0.1 * (i % 3)} for i in range(35)
        ]
        
        # First validate with stable history
        is_valid, _, _ = gate.validate(
            measurement=70.0,
            prediction=70.5,
            innovation_covariance=1.0,
            user_history=stable_history
        )
        
        # Gamma should be reduced for stable user
        assert gate.gamma == gate.base_gamma - 0.5
        
        # Reset gate
        gate.gamma = gate.base_gamma
        
        # Create history with high variability (std > 2.0)
        # Use values that will give std > 2.0
        variable_history = []
        for i in range(35):
            # Create high variance: alternating between very low and very high values
            value = 0.1 if i % 2 == 0 else 5.0
            variable_history.append({'normalized_innovation': value})
        
        # Validate with variable history
        is_valid, _, _ = gate.validate(
            measurement=70.0,
            prediction=70.5,
            innovation_covariance=1.0,
            user_history=variable_history
        )
        
        # Gamma should be increased for variable user
        assert gate.gamma == gate.base_gamma + 0.5
    
    def test_metrics_calculation(self):
        gate = ValidationGate(gamma=3.0)
        
        # Process several measurements
        for i in range(10):
            if i < 8:
                # Accept most measurements
                gate.validate(70.0, 70.5, 1.0)
            else:
                # Reject some
                gate.validate(70.0, 80.0, 0.5)
        
        metrics = gate.get_metrics()
        
        assert metrics['total_processed'] == 10
        assert metrics['total_accepted'] == 8
        assert metrics['total_rejected'] == 2
        assert metrics['acceptance_rate'] == 0.8
        assert metrics['rejection_rate'] == 0.2
        assert 0.5 <= metrics['health_score'] <= 1.0
    
    def test_should_rebaseline(self):
        gate = ValidationGate(gamma=3.0)
        
        # Initially should not suggest rebaselining
        assert gate.should_rebaseline() == False
        
        # Add several rejections
        for _ in range(4):
            gate.validate(70.0, 80.0, 0.5)  # Will be rejected
        
        # Should suggest rebaselining after many rejections
        assert gate.should_rebaseline() == True
    
    def test_reset(self):
        gate = ValidationGate(gamma=3.5)
        
        # Process some measurements
        gate.validate(70.0, 70.5, 1.0)
        gate.validate(70.0, 80.0, 0.5)
        
        assert gate.stats['accepted'] > 0
        assert len(gate.innovation_history) > 0
        
        # Reset
        gate.reset()
        
        assert gate.gamma == 3.5  # Back to base gamma
        assert gate.stats['accepted'] == 0
        assert gate.stats['rejected'] == 0
        assert len(gate.innovation_history) == 0
        assert len(gate.rejection_history) == 0


class TestKalmanWithValidation:
    
    def test_kalman_accepts_valid_measurement(self):
        kf = CustomKalmanFilter(
            initial_weight=70.0,
            validation_gamma=3.0,
            enable_validation=True
        )
        
        # Initialize filter
        result1 = kf.process_measurement(70.0, datetime.now())
        assert result1['measurement_accepted'] == True
        
        # Process normal measurement
        result2 = kf.process_measurement(70.5, datetime.now() + timedelta(days=1))
        assert result2['measurement_accepted'] == True
        assert 'rejection_reason' not in result2 or result2.get('rejection_reason') is None
    
    def test_kalman_rejects_outlier(self):
        kf = CustomKalmanFilter(
            initial_weight=70.0,
            validation_gamma=3.0,
            enable_validation=True
        )
        
        # Initialize filter
        kf.process_measurement(70.0, datetime.now())
        kf.process_measurement(70.2, datetime.now() + timedelta(days=1))
        kf.process_measurement(70.1, datetime.now() + timedelta(days=2))
        
        # Process extreme outlier
        result = kf.process_measurement(85.0, datetime.now() + timedelta(days=3))
        
        assert result['measurement_accepted'] == False
        assert result['rejection_reason'] is not None
        assert result['filtered_weight'] < 75.0  # Should return prediction, not measurement
        assert len(kf.rejected_measurements) == 1
    
    def test_kalman_without_validation(self):
        kf = CustomKalmanFilter(
            initial_weight=70.0,
            enable_validation=False
        )
        
        # Initialize filter
        kf.process_measurement(70.0, datetime.now())
        
        # Process extreme outlier - should still be accepted
        result = kf.process_measurement(85.0, datetime.now() + timedelta(days=1))
        
        assert result['measurement_accepted'] == True
        assert result['filtered_weight'] > 70.0  # Will be influenced by outlier
        assert len(kf.rejected_measurements) == 0
    
    def test_validation_summary(self):
        kf = CustomKalmanFilter(
            initial_weight=70.0,
            validation_gamma=3.0,
            enable_validation=True
        )
        
        # Process several measurements
        timestamps = [datetime.now() + timedelta(days=i) for i in range(10)]
        weights = [70.0, 70.2, 70.1, 85.0, 70.3, 70.0, 90.0, 70.2, 70.1, 70.3]
        
        for ts, weight in zip(timestamps, weights):
            kf.process_measurement(weight, ts)
        
        summary = kf.get_validation_summary()
        
        assert summary is not None
        assert 'metrics' in summary
        assert 'rejected_measurements' in summary
        assert 'total_rejected' in summary
        assert summary['total_rejected'] >= 2  # At least the two extreme outliers
        assert 'should_rebaseline' in summary
    
    def test_state_not_corrupted_by_outlier(self):
        kf = CustomKalmanFilter(
            initial_weight=70.0,
            validation_gamma=3.0,
            enable_validation=True
        )
        
        # Establish stable state
        for i in range(5):
            kf.process_measurement(70.0 + i * 0.1, datetime.now() + timedelta(days=i))
        
        state_before = kf.get_state()
        weight_before = state_before['weight']
        
        # Process extreme outlier
        result = kf.process_measurement(100.0, datetime.now() + timedelta(days=6))
        assert result['measurement_accepted'] == False
        
        state_after = kf.get_state()
        weight_after = state_after['weight']
        
        # State should not be significantly affected by rejected measurement
        assert abs(weight_after - weight_before) < 2.0  # Allow for normal prediction drift


if __name__ == "__main__":
    pytest.main([__file__, "-v"])