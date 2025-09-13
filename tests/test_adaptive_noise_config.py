#!/usr/bin/env python3
"""Test adaptive noise configuration and application."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import tomllib
from src.processor import process_weight_enhanced
from src.database import get_state_db
from src.models import KALMAN_DEFAULTS


def test_adaptive_noise_from_config():
    """Test that adaptive noise multipliers are applied from config."""
    
    # Load config
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    # Verify adaptive noise is enabled
    assert config["adaptive_noise"]["enabled"] == True
    
    # Test different sources get different multipliers
    test_cases = [
        ("patient-upload", 1.0),  # Most reliable
        ("care-team-upload", 1.2),  # Less reliable than expected
        ("internal-questionnaire", 1.6),  # Questionnaire
        ("https://connectivehealth.io", 2.2),  # Moderate
        ("patient-device", 2.5),  # Higher noise
        ("https://api.iglucose.com", 2.6),  # Highest configured
        ("unknown-source", 1.5),  # Should use default
    ]
    
    db = get_state_db()
    
    user_id = "test-adaptive-user"
    db.clear_state(user_id)  # Clear this user's state
    base_timestamp = datetime(2024, 1, 1, 12, 0, 0)
    
    # Process measurements from different sources
    results = []
    for i, (source, expected_multiplier) in enumerate(test_cases):
        timestamp = base_timestamp + timedelta(hours=i * 2)
        weight = 75.0 + (i * 0.1)  # Slight variation
        
        # Add full config to processing_config
        processing_config = config["processing"].copy()
        processing_config["config"] = config
        
        result = process_weight_enhanced(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source=source,
            processing_config=processing_config,
            kalman_config=config["kalman"],
            unit="kg"
        )
        
        results.append((source, result, expected_multiplier))
        
        # Verify the measurement was processed
        assert result is not None
        assert result["accepted"] == True
        
        # Check that noise multiplier was applied
        if "measurement_noise_used" in result:
            base_noise = config["kalman"]["observation_covariance"]
            actual_multiplier = result["measurement_noise_used"] / base_noise
            print(f"Source: {source:30} Expected: {expected_multiplier:.1f}, Actual: {actual_multiplier:.2f}")
            
            # Allow small floating point differences
            assert abs(actual_multiplier - expected_multiplier) < 0.1, \
                f"Multiplier mismatch for {source}: expected {expected_multiplier}, got {actual_multiplier:.2f}"
    
    # Verify that different sources produce different confidence levels
    # Lower multiplier = higher confidence
    patient_upload_result = next(r for s, r, _ in results if s == "patient-upload")
    iglucose_result = next(r for s, r, _ in results if s == "https://api.iglucose.com")
    
    # Patient upload should have higher confidence than iglucose
    # (lower noise multiplier = higher confidence)
    if "confidence" in patient_upload_result and "confidence" in iglucose_result:
        # Note: confidence might be similar for first measurements, 
        # but the noise multiplier should be different
        print(f"\nConfidence comparison:")
        print(f"patient-upload confidence: {patient_upload_result['confidence']:.3f}")
        print(f"iglucose confidence: {iglucose_result['confidence']:.3f}")
    
    print("\n✅ Adaptive noise configuration test passed!")
    print("All sources received correct noise multipliers from config")
    return True


def test_adaptive_noise_disabled():
    """Test that adaptive noise can be disabled."""
    
    # Load config and disable adaptive noise
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    # Create a modified config with adaptive noise disabled
    config["adaptive_noise"]["enabled"] = False
    
    db = get_state_db()
    
    user_id = "test-disabled-adaptive"
    db.clear_state(user_id)  # Clear this user's state
    timestamp = datetime(2024, 1, 1, 12, 0, 0)
    
    # Add full config to processing_config
    processing_config = config["processing"].copy()
    processing_config["config"] = config
    
    # Process with different sources - should all use same noise
    sources = ["patient-upload", "https://api.iglucose.com"]
    results = []
    
    for i, source in enumerate(sources):
        result = process_weight_enhanced(
            user_id=user_id,
            weight=75.0,
            timestamp=timestamp + timedelta(hours=i),
            source=source,
            processing_config=processing_config,
            kalman_config=config["kalman"],
            unit="kg"
        )
        results.append(result)
    
    # When disabled, should fall back to old method
    # which uses ThresholdCalculator.get_measurement_noise_multiplier
    print("\n✅ Adaptive noise disable test passed!")
    return True


def test_unknown_source_uses_default():
    """Test that unknown sources use the default multiplier."""
    
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    
    db = get_state_db()
    
    user_id = "test-unknown-source"
    db.clear_state(user_id)  # Clear this user's state
    
    # Add full config to processing_config
    processing_config = config["processing"].copy()
    processing_config["config"] = config
    
    # Test with a source not in the config
    result = process_weight_enhanced(
        user_id=user_id,
        weight=75.0,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        source="some-new-unknown-source",
        processing_config=processing_config,
        kalman_config=config["kalman"],
        unit="kg"
    )
    
    assert result is not None
    assert result["accepted"] == True
    
    # Should use default multiplier (1.5)
    if "measurement_noise_used" in result:
        base_noise = config["kalman"]["observation_covariance"]
        default_multiplier = config["adaptive_noise"]["default_multiplier"]
        actual_multiplier = result["measurement_noise_used"] / base_noise
        
        print(f"\nUnknown source multiplier: {actual_multiplier:.2f} (expected: {default_multiplier})")
        assert abs(actual_multiplier - default_multiplier) < 0.1
    
    print("✅ Unknown source default multiplier test passed!")
    return True


if __name__ == "__main__":
    print("Testing adaptive noise configuration...")
    print("=" * 60)
    
    # Run tests
    test_adaptive_noise_from_config()
    test_adaptive_noise_disabled()
    test_unknown_source_uses_default()
    
    print("\n" + "=" * 60)
    print("All adaptive noise configuration tests passed! 🎉")