#!/usr/bin/env python3
"""
Fix the processor to properly use reset parameters for adaptive period.
"""

def fix_processor():
    """Fix processor.py to properly handle adaptive period."""
    
    with open('src/processor.py', 'r') as f:
        lines = f.readlines()
    
    # Find and fix the incomplete line
    for i, line in enumerate(lines):
        if "if days_since < adaptive_days:" in line and i == 268:
            # This line is missing the rest
            lines[i] = "                    if days_since < adaptive_days:\n"
            lines.insert(i+1, "                        in_adaptive_period = True\n")
    
    # Now fix the quality config section to use proper parameter names
    for i, line in enumerate(lines):
        if "adaptive_threshold = reset_params.get('quality_threshold'" in line:
            # Update to use new parameter name
            lines[i] = "            adaptive_threshold = reset_params.get('quality_acceptance_threshold', reset_params.get('quality_threshold', 0.4))\n"
        
        # Fix the component weights to use reset parameters
        elif "weights['plausibility'] = 0.1" in line:
            # Replace hardcoded weights with values from reset parameters
            new_lines = [
                "                # Use weights from reset parameters if available\n",
                "                weights['plausibility'] = reset_params.get('quality_plausibility_weight', 0.1)\n"
            ]
            lines[i] = ''.join(new_lines)
        
        elif "weights['consistency'] = 0.15" in line:
            lines[i] = "                weights['consistency'] = reset_params.get('quality_consistency_weight', 0.15)\n"
        
        elif "weights['safety'] = 0.45" in line:
            lines[i] = "                weights['safety'] = reset_params.get('quality_safety_weight', 0.45)\n"
        
        elif "weights['reliability'] = 0.30" in line:
            lines[i] = "                weights['reliability'] = reset_params.get('quality_reliability_weight', 0.30)\n"
    
    # Also update the parameter name references for new naming
    for i, line in enumerate(lines):
        if "warmup_measurements = reset_params.get('warmup_measurements'" in line:
            lines[i] = "            warmup_measurements = reset_params.get('adaptation_measurements', reset_params.get('warmup_measurements', 10))\n"
        elif "adaptive_days = reset_params.get('adaptive_days'" in line:
            lines[i] = "                    adaptive_days = reset_params.get('adaptation_days', reset_params.get('adaptive_days', 7))\n"
    
    with open('src/processor.py', 'w') as f:
        f.writelines(lines)
    
    print("Fixed src/processor.py to properly use reset parameters")

if __name__ == "__main__":
    fix_processor()
