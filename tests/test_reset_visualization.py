"""
Test that reset indicators appear correctly in visualizations
"""

from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processor import WeightProcessor
from src.visualization import create_dashboard
from src.database import ProcessorStateDB


def test_reset_flag_in_results():
    """Test that processor adds reset flag to results after gap."""
    
    db = ProcessorStateDB()
    user_id = "test_reset_viz"
    
    processing_config = {
        "min_weight": 30,
        "max_weight": 400,
        "extreme_threshold": 0.5,
    }
    
    kalman_config = {
        "initial_variance": 0.5,
        "transition_covariance_weight": 0.05,
        "transition_covariance_trend": 0.0005,
        "observation_covariance": 1.5,
        "reset_gap_days": 30,
    }
    
    # First measurement
    result1 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=80.0,
        timestamp=datetime(2024, 1, 1),
        source="test",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    assert result1 is not None
    assert result1.get('was_reset', False) == False
    
    # Measurement after 35-day gap
    result2 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=85.0,
        timestamp=datetime(2024, 2, 5),
        source="test",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    assert result2 is not None
    assert result2.get('was_reset', False) == True
    assert result2.get('gap_days', 0) == 35.0
    
    # Next measurement without gap
    result3 = WeightProcessor.process_weight(
        user_id=user_id,
        weight=84.5,
        timestamp=datetime(2024, 2, 6),
        source="test",
        processing_config=processing_config,
        kalman_config=kalman_config,
        db=db
    )
    
    assert result3 is not None
    assert result3.get('was_reset', False) == False
    
    print("✓ Reset flags correctly added to results")
    print(f"  Day 1: No reset flag")
    print(f"  Day 36: Reset flag with gap_days=35")
    print(f"  Day 37: No reset flag")


def test_visualization_with_resets():
    """Test that visualization handles reset indicators."""
    
    db = ProcessorStateDB()
    user_id = "test_viz_resets"
    
    processing_config = {
        "min_weight": 30,
        "max_weight": 400,
        "extreme_threshold": 0.5,
    }
    
    kalman_config = {
        "initial_variance": 0.5,
        "transition_covariance_weight": 0.05,
        "transition_covariance_trend": 0.0005,
        "observation_covariance": 1.5,
        "reset_gap_days": 30,
    }
    
    viz_config = {
        "cropped_months": 3,
        "dashboard_dpi": 100,
    }
    
    results = []
    
    # Create data with multiple gaps
    dates_weights = [
        (datetime(2023, 1, 1), 80.0),
        (datetime(2023, 1, 15), 80.5),
        (datetime(2023, 1, 30), 81.0),
        # 40-day gap
        (datetime(2023, 3, 11), 82.0),  # Should have reset
        (datetime(2023, 3, 12), 82.2),
        (datetime(2023, 3, 20), 82.5),
        # 35-day gap
        (datetime(2023, 4, 24), 83.0),  # Should have reset
        (datetime(2023, 4, 25), 83.1),
        (datetime(2023, 4, 30), 83.3),
    ]
    
    for date, weight in dates_weights:
        result = WeightProcessor.process_weight(
            user_id=user_id,
            weight=weight,
            timestamp=date,
            source="test",
            processing_config=processing_config,
            kalman_config=kalman_config,
            db=db
        )
        if result:
            results.append(result)
    
    # Count resets
    reset_count = sum(1 for r in results if r.get('was_reset', False))
    assert reset_count == 2, f"Expected 2 resets, got {reset_count}"
    
    # Create visualization
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dashboard(user_id, results, tmpdir, viz_config)
        
        # Check that file was created
        expected_file = Path(tmpdir) / f"{user_id}.png"
        assert expected_file.exists(), f"Dashboard file not created: {expected_file}"
        
        print(f"✓ Visualization created with {reset_count} reset indicators")
        print(f"  Output: {expected_file}")
    
    # Verify reset details
    for i, r in enumerate(results):
        if r.get('was_reset'):
            print(f"  Reset at {r['timestamp'].date()}: {r['gap_days']:.0f}-day gap")


if __name__ == "__main__":
    print("Testing reset visualization...")
    print()
    
    test_reset_flag_in_results()
    print()
    
    test_visualization_with_resets()
    print()
    
    print("All reset visualization tests passed!")