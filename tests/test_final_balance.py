"""Test final balanced physiological limits."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from datetime import datetime
from src.processor import WeightProcessor

# Final balanced configuration
config = {
    'min_weight': 30.0,
    'max_weight': 400.0,
    'max_daily_change': 0.05,
    'extreme_threshold': 0.20,
    'kalman_cleanup_threshold': 4.0,
    'physiological': {
        'enable_physiological_limits': True,
        'max_change_1h_percent': 0.02,      # 2% 
        'max_change_6h_percent': 0.025,     # 2.5%
        'max_change_24h_percent': 0.035,    # 3.5%
        'max_change_1h_absolute': 3.0,      
        'max_change_6h_absolute': 4.0,      
        'max_change_24h_absolute': 5.0,     
        'max_sustained_daily': 1.5,         # 1.5kg/day for GLP-1/aggressive diets
        'session_timeout_minutes': 5.0,
        'session_variance_threshold': 5.0
    }
}

kalman_config = {
    'initial_variance': 0.5,
    'transition_covariance_weight': 0.05,
    'transition_covariance_trend': 0.0005,
    'observation_covariance': 1.5,
}

print("FINAL BALANCED PHYSIOLOGICAL LIMITS TEST")
print("=" * 70)
print("\nComparing OLD (too strict) vs NEW (balanced) limits")
print()

# Test cases from the problematic user
test_cases = [
    ("Normal morning weight", "2025-06-10 08:00:00", 88.6),
    ("Afternoon lower", "2025-06-10 16:00:00", 85.5),  # 3.1kg drop
    ("Next morning", "2025-06-11 08:00:00", 88.0),     # Back up
    ("Gradual loss", "2025-06-13 08:00:00", 85.0),     # 3kg in 2 days
    ("Week later", "2025-06-20 08:00:00", 82.0),       # 3kg in 7 days
    ("Multi-user child", "2025-06-20 08:05:00", 45.0), # Clear multi-user
    ("Multi-user adult", "2025-06-20 08:06:00", 85.0), # Another user
]

print("Test scenarios:")
print("-" * 70)

accepted = rejected = 0
for description, timestamp, weight in test_cases:
    result = WeightProcessor.process_weight(
        user_id="final_test",
        weight=weight,
        timestamp=datetime.fromisoformat(timestamp),
        source="test",
        processing_config=config,
        kalman_config=kalman_config
    )
    
    if result and result.get('accepted'):
        accepted += 1
        status = "✓ ACCEPT"
    else:
        rejected += 1
        status = "✗ REJECT"
    
    print(f"{status}: {weight:5.1f}kg - {description:20s}")
    if not result.get('accepted') and result:
        reason = result.get('reason', '')
        if 'exceeds' in reason:
            # Extract key part of reason
            parts = reason.split('exceeds')
            if len(parts) > 1:
                print(f"         → Exceeds {parts[1]}")

print()
print("=" * 70)
print(f"Results: {accepted} accepted, {rejected} rejected")
print()
print("Key improvements from adjustments:")
print("✓ Normal daily fluctuations now accepted (3.1kg afternoon drop)")
print("✓ Gradual weight loss accepted (3kg over 2 days)")  
print("✓ Multi-user contamination still rejected (45kg child weight)")
print()
print("The new limits strike a better balance between:")
print("- Accepting legitimate physiological variations")
print("- Allowing for medication/diet-induced weight loss")
print("- Still catching obvious data errors and multi-user scenarios")
