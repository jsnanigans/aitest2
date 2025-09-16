#!/usr/bin/env python3
"""
Implement adaptive quality scoring during the initial/post-reset period.
"""

def implement_adaptive_quality():
    # Read processor.py
    with open('/Users/brendanmullins/Projects/aitest/strem_process_anchor/src/processor.py', 'r') as f:
        lines = f.readlines()
    
    # Find the quality scoring section (around line 240)
    for i in range(len(lines)):
        if 'quality_score = PhysiologicalValidator.calculate_quality_score(' in lines[i]:
            # Insert adaptive logic before quality scoring
            indent = '        '
            
            # Check if we're in adaptive period
            adaptive_check = [
                f'{indent}# Check if we\'re in adaptive period (initial or post-reset)\n',
                f'{indent}in_adaptive_period = False\n',
                f'{indent}if state:\n',
                f'{indent}    measurements_since_reset = state.get("measurements_since_reset", 100)\n',
                f'{indent}    if measurements_since_reset < 10:  # First 10 measurements\n',
                f'{indent}        in_adaptive_period = True\n',
                f'{indent}    else:\n',
                f'{indent}        # Also check time-based (7 days)\n',
                f'{indent}        reset_timestamp = get_reset_timestamp(state)\n',
                f'{indent}        if not reset_timestamp and not state.get("kalman_params"):\n',
                f'{indent}            # Initial measurement\n',
                f'{indent}            reset_timestamp = timestamp\n',
                f'{indent}        if reset_timestamp:\n',
                f'{indent}            days_since = (timestamp - reset_timestamp).total_seconds() / 86400.0\n',
                f'{indent}            if days_since < 7:\n',
                f'{indent}                in_adaptive_period = True\n',
                f'{indent}\n',
                f'{indent}# Adjust quality config if in adaptive period\n',
                f'{indent}adaptive_quality_config = quality_config.copy()\n',
                f'{indent}if in_adaptive_period:\n',
                f'{indent}    # Lower threshold during adaptation\n',
                f'{indent}    adaptive_quality_config["threshold"] = 0.4  # vs normal 0.6\n',
                f'{indent}    # Adjust component weights to be more forgiving\n',
                f'{indent}    if "component_weights" in adaptive_quality_config:\n',
                f'{indent}        weights = adaptive_quality_config["component_weights"].copy()\n',
                f'{indent}        # Reduce plausibility weight (often fails during adaptation)\n',
                f'{indent}        weights["plausibility"] = 0.1  # vs normal 0.25\n',
                f'{indent}        weights["consistency"] = 0.15  # vs normal 0.25\n',
                f'{indent}        weights["safety"] = 0.45  # vs normal 0.35 (keep safety high)\n',
                f'{indent}        weights["reliability"] = 0.30  # vs normal 0.15\n',
                f'{indent}        adaptive_quality_config["component_weights"] = weights\n',
                f'{indent}\n',
            ]
            
            # Insert before the quality score calculation
            lines[i:i] = adaptive_check
            
            # Now update the quality_score call to use adaptive_quality_config
            # Find the config parameter in the call
            for j in range(i + len(adaptive_check), min(i + len(adaptive_check) + 10, len(lines))):
                if 'config=quality_config' in lines[j]:
                    lines[j] = lines[j].replace('config=quality_config', 'config=adaptive_quality_config')
                    break
            break
    
    # Write back
    with open('/Users/brendanmullins/Projects/aitest/strem_process_anchor/src/processor.py', 'w') as f:
        f.writelines(lines)
    
    print("✓ Implemented adaptive quality scoring")
    print("\nDuring adaptive period (first 10 measurements or 7 days):")
    print("- Quality threshold: 0.4 (vs 0.6 normal)")
    print("- Plausibility weight: 0.1 (vs 0.25 normal)")
    print("- Consistency weight: 0.15 (vs 0.25 normal)")
    print("- Safety weight: 0.45 (vs 0.35 normal)")
    print("- Reliability weight: 0.30 (vs 0.15 normal)")
    print("\nThis prevents rejections during the learning period.")

if __name__ == "__main__":
    implement_adaptive_quality()
