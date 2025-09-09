#!/usr/bin/env python3
"""Test immediate initialization without buffering."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.processor_database import ProcessorStateDB
import toml

# Load config
config = toml.load('config.toml')
processing_config = config['processing']
kalman_config = config['kalman']

def test_immediate_initialization():
    """Test that first measurement initializes immediately without buffering."""
    print("Testing immediate initialization...")
    
    # Create a fresh database
    db = ProcessorStateDB()
    user_id = "test_immediate_user"
    
    # Clear any existing state
    db.delete_state(user_id)
    
    # Process first measurement
    timestamp1 = datetime(2025, 1, 1, 10, 0, 0)
    weight1 = 75.0
    
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=weight1,
        timestamp=timestamp1,
        source="test",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    # Verify we get a result immediately (not None)
    assert result1 is not None, "First measurement should return a result, not None"
    assert result1['accepted'] == True, "First measurement should be accepted"
    assert result1['raw_weight'] == weight1, "Raw weight should match input"
    assert 'filtered_weight' in result1, "Should have filtered weight"
    
    print(f"  ✓ First measurement processed immediately")
    print(f"    Raw: {result1['raw_weight']:.2f} kg")
    print(f"    Filtered: {result1['filtered_weight']:.2f} kg")
    
    # Process second measurement
    timestamp2 = timestamp1 + timedelta(days=1)
    weight2 = 74.8
    
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=weight2,
        timestamp=timestamp2,
        source="test",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    assert result2 is not None, "Second measurement should return a result"
    assert result2['accepted'] == True, "Second measurement should be accepted"
    
    print(f"  ✓ Second measurement processed")
    print(f"    Raw: {result2['raw_weight']:.2f} kg")
    print(f"    Filtered: {result2['filtered_weight']:.2f} kg")
    
    # Check state
    state = db.get_state(user_id)
    assert state['measurement_count'] == 2, f"Should have 2 measurements, got {state.get('measurement_count')}"
    assert len(state.get('measurement_history', [])) == 2, "Should have 2 measurements in history"
    
    print(f"  ✓ State tracking works: {state['measurement_count']} measurements")
    
    return True

def test_parameter_adaptation():
    """Test that parameters adapt after 10 measurements."""
    print("\nTesting parameter adaptation at measurement 10...")
    
    # Create a fresh database
    db = ProcessorStateDB()
    user_id = "test_adaptation_user"
    
    # Clear any existing state
    db.delete_state(user_id)
    
    # Process 10 measurements with some noise
    base_weight = 80.0
    timestamp = datetime(2025, 1, 1, 10, 0, 0)
    
    for i in range(10):
        # Add some variation
        if i % 3 == 0:
            weight = base_weight + 0.5  # Small increase
        elif i % 3 == 1:
            weight = base_weight - 0.3  # Small decrease
        else:
            weight = base_weight
        
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp + timedelta(days=i),
            source="test",
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        
        assert result is not None, f"Measurement {i+1} should return a result (no buffering!)"
        
    # Check state after 10th measurement
    state = db.get_state(user_id)
    
    # Verify adaptation occurred
    assert state['measurement_count'] == 10, "Should have 10 measurements"
    assert 'adapted_params' in state, "Should have adapted parameters after 10 measurements"
    assert len(state['adapted_params']) > 0, "Adapted parameters should not be empty"
    assert state.get('measurement_history', []) == [], "History should be cleared after adaptation"
    
    print(f"  ✓ Parameters adapted after 10 measurements")
    print(f"    Observation covariance: {state['adapted_params'].get('observation_covariance', 'N/A')}")
    print(f"    Extreme threshold: {state['adapted_params'].get('extreme_threshold', 'N/A')}")
    
    return True

def test_no_buffering_for_user_0093a653():
    """Test that user 0093a653 processes immediately without buffering."""
    print("\nTesting user 0093a653 (no buffering)...")
    
    # Create a fresh database
    db = ProcessorStateDB()
    user_id = "0093a653-476b-4401-bbec-33a89abc2b18"
    
    # Clear any existing state
    db.delete_state(user_id)
    
    # Test data from the actual user
    measurements = [
        (datetime(2023, 8, 10), 114.76),
        (datetime(2024, 11, 15), 107.86),
        (datetime(2025, 6, 10), 107.32),
    ]
    
    results = []
    for i, (timestamp, weight) in enumerate(measurements):
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=timestamp,
            source="test",
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        
        assert result is not None, f"Measurement {i+1} should return a result, not None (no buffering!)"
        assert result['accepted'] == True, f"Measurement {i+1} should be accepted"
        results.append(result)
        
        print(f"  Measurement {i+1}: {weight:.2f} kg -> Filtered: {result['filtered_weight']:.2f} kg")
    
    print(f"  ✓ All {len(measurements)} measurements processed immediately (no buffering)")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("IMMEDIATE INITIALIZATION TEST - NO BUFFERING")
    print("=" * 60)
    
    try:
        # Run tests
        test_immediate_initialization()
        test_parameter_adaptation()
        test_no_buffering_for_user_0093a653()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - NO BUFFERING DELAYS!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())