"""Test that rejection reasons are properly captured."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from datetime import datetime
from src.processor import WeightProcessor

# Test configuration
config = {
    'min_weight': 30.0,
    'max_weight': 400.0,
    'max_daily_change': 0.05,
    'extreme_threshold': 0.20,
    'kalman_cleanup_threshold': 4.0,
    'physiological': {
        'enable_physiological_limits': True,
        'max_change_1h_percent': 0.02,
        'max_change_6h_percent': 0.025,
        'max_change_24h_percent': 0.035,
        'max_change_1h_absolute': 3.0,
        'max_change_6h_absolute': 4.0,
        'max_change_24h_absolute': 5.0,
        'max_sustained_daily': 1.5,
        'session_timeout_minutes': 5.0,
        'session_variance_threshold': 5.0
    }
}

kalman_config = {
    'initial_variance': 0.5,
    'transition_covariance_weight': 0.05,
    'transition_covariance_trend': 0.0005,
    'observation_covariance': 1.5,
    'reset_gap_days': 30
}

print("Testing Rejection Reason Capture")
print("=" * 70)

# Test cases that should be rejected with reasons
test_cases = [
    ("Normal weight", "2025-01-01 08:00:00", 75.0, True),
    ("Child weight", "2025-01-01 08:05:00", 35.0, False),  # Should reject
    ("Back to adult", "2025-01-01 08:06:00", 85.0, False),  # Should reject
    ("Outside bounds", "2025-01-01 09:00:00", 25.0, False),  # Below min_weight
]

for description, timestamp, weight, should_accept in test_cases:
    result = WeightProcessor.process_weight(
        user_id="test_rejection",
        weight=weight,
        timestamp=datetime.fromisoformat(timestamp),
        source="test",
        processing_config=config,
        kalman_config=kalman_config
    )
    
    accepted = result.get('accepted', False)
    reason = result.get('reason', 'No reason provided')
    
    status = "✓" if accepted == should_accept else "✗"
    
    print(f"\n{status} {description}: {weight}kg")
    print(f"  Expected: {'Accept' if should_accept else 'Reject'}")
    print(f"  Got: {'Accepted' if accepted else f'Rejected - {reason}'}")
    
    # Check that rejected measurements have a reason
    if not accepted and not result.get('reason'):
        print("  ⚠️  WARNING: Rejection without reason!")

print("\n" + "=" * 70)
print("Summary: All rejections should have clear reasons")
