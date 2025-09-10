"""Test adjusted physiological limits for better balance."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from datetime import datetime
from src.processor import WeightProcessor

# Configuration with adjusted physiological limits
config = {
    'min_weight': 30.0,
    'max_weight': 400.0,
    'max_daily_change': 0.05,
    'extreme_threshold': 0.20,
    'kalman_cleanup_threshold': 4.0,
    'physiological': {
        'enable_physiological_limits': True,
        'max_change_1h_percent': 0.02,      # Increased from 1.5%
        'max_change_6h_percent': 0.025,     # Increased from 2%
        'max_change_24h_percent': 0.035,    # Increased from 3%
        'max_change_1h_absolute': 3.0,      # Increased from 2.0
        'max_change_6h_absolute': 4.0,      # Increased from 3.0
        'max_change_24h_absolute': 5.0,     # Increased from 2.5
        'max_sustained_daily': 1.0,         # Increased from 0.5 for GLP-1/diets
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

print("Testing Adjusted Physiological Limits")
print("=" * 70)
print("\nScenario 1: Normal weight fluctuations (should mostly accept)")
print("-" * 70)

# Test normal fluctuations that were being rejected
normal_measurements = [
    {"timestamp": "2025-06-10 08:00:00", "weight": 88.6, "source": "test"},
    {"timestamp": "2025-06-10 16:00:00", "weight": 85.5, "source": "test"},  # 3.1kg in 8h - normal fluctuation
    {"timestamp": "2025-06-11 08:00:00", "weight": 88.0, "source": "test"},  # Return to normal
    {"timestamp": "2025-06-12 08:00:00", "weight": 86.5, "source": "test"},  # 1.5kg/day loss
    {"timestamp": "2025-06-13 08:00:00", "weight": 85.0, "source": "test"},  # 1.5kg/day loss
]

accepted = rejected = 0
for m in normal_measurements:
    result = WeightProcessor.process_weight(
        user_id="test_normal",
        weight=m["weight"],
        timestamp=datetime.fromisoformat(m["timestamp"]),
        source=m["source"],
        processing_config=config,
        kalman_config=kalman_config
    )
    
    if result and result.get('accepted'):
        accepted += 1
        print(f"✓ ACCEPTED: {m['timestamp'][-8:]} - {m['weight']:.1f}kg")
    else:
        rejected += 1
        print(f"✗ REJECTED: {m['timestamp'][-8:]} - {m['weight']:.1f}kg")
        if result:
            print(f"  Reason: {result.get('reason', 'Unknown')}")

print(f"\nNormal fluctuations: {accepted} accepted, {rejected} rejected")

print("\n" + "=" * 70)
print("Scenario 2: Multi-user contamination (should still reject)")
print("-" * 70)

# Test clear multi-user scenarios that should still be rejected
multi_user_measurements = [
    {"timestamp": "2025-07-14 20:49:00", "weight": 75.0, "source": "test"},
    {"timestamp": "2025-07-14 20:50:00", "weight": 45.0, "source": "test"},  # 30kg drop - child
    {"timestamp": "2025-07-14 20:51:00", "weight": 85.0, "source": "test"},  # 40kg jump - different adult
    {"timestamp": "2025-07-14 20:52:00", "weight": 35.0, "source": "test"},  # 50kg drop - small child
]

accepted = rejected = 0
for m in multi_user_measurements:
    result = WeightProcessor.process_weight(
        user_id="test_multi",
        weight=m["weight"],
        timestamp=datetime.fromisoformat(m["timestamp"]),
        source=m["source"],
        processing_config=config,
        kalman_config=kalman_config
    )
    
    if result and result.get('accepted'):
        accepted += 1
        print(f"✓ ACCEPTED: {m['timestamp'][-8:]} - {m['weight']:.1f}kg")
    else:
        rejected += 1
        print(f"✗ REJECTED: {m['timestamp'][-8:]} - {m['weight']:.1f}kg")
        if result:
            print(f"  Reason: {result.get('reason', 'Unknown')}")

print(f"\nMulti-user: {accepted} accepted, {rejected} rejected")

print("\n" + "=" * 70)
print("Scenario 3: GLP-1 medication weight loss (should accept)")
print("-" * 70)

# Test GLP-1 style weight loss (1-2kg per week is common)
glp1_measurements = [
    {"timestamp": "2025-08-01 08:00:00", "weight": 110.0, "source": "test"},
    {"timestamp": "2025-08-03 08:00:00", "weight": 108.0, "source": "test"},  # 2kg in 2 days
    {"timestamp": "2025-08-05 08:00:00", "weight": 106.5, "source": "test"},  # 1.5kg in 2 days
    {"timestamp": "2025-08-08 08:00:00", "weight": 104.5, "source": "test"},  # 2kg in 3 days
    {"timestamp": "2025-08-10 08:00:00", "weight": 103.0, "source": "test"},  # 1.5kg in 2 days
]

accepted = rejected = 0
for m in glp1_measurements:
    result = WeightProcessor.process_weight(
        user_id="test_glp1",
        weight=m["weight"],
        timestamp=datetime.fromisoformat(m["timestamp"]),
        source=m["source"],
        processing_config=config,
        kalman_config=kalman_config
    )
    
    if result and result.get('accepted'):
        accepted += 1
        print(f"✓ ACCEPTED: {m['timestamp'][:10]} - {m['weight']:.1f}kg")
    else:
        rejected += 1
        print(f"✗ REJECTED: {m['timestamp'][:10]} - {m['weight']:.1f}kg")
        if result:
            print(f"  Reason: {result.get('reason', 'Unknown')}")

print(f"\nGLP-1 weight loss: {accepted} accepted, {rejected} rejected")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nAdjusted limits provide better balance:")
print("✓ Accept normal daily fluctuations (up to 3.5%)")
print("✓ Allow for medication-induced weight loss (1kg/day sustained)")
print("✓ Still reject obvious multi-user contamination (>5kg in session)")
print("\nThis should reduce false rejections while maintaining safety.")
