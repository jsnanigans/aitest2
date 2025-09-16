#!/usr/bin/env python3
"""
Fix the extreme threshold to be adaptive during the adaptation phase
"""

def fix_processor():
    # Read the current processor.py
    with open('src/processor.py', 'r') as f:
        lines = f.readlines()
    
    # Find the extreme threshold line (around line 355)
    for i, line in enumerate(lines):
        if 'extreme_threshold = processing_config.get' in line:
            # Insert adaptation logic before this line
            indent = '        '
            
            # Add adaptation check
            adaptation_lines = [
                f'{indent}# Check if we\'re in adaptation phase for more lenient threshold\n',
                f'{indent}extreme_threshold = processing_config.get(\'extreme_threshold\', 0.20)\n',
                f'{indent}if state:\n',
                f'{indent}    measurements_since_reset = state.get("measurements_since_reset", 100)\n',
                f'{indent}    reset_params = state.get(\'reset_parameters\', {{}})\n',
                f'{indent}    adaptation_measurements = reset_params.get(\'adaptation_measurements\', 10)\n',
                f'{indent}    if measurements_since_reset < adaptation_measurements:\n',
                f'{indent}        # During adaptation, use a much more lenient threshold\n',
                f'{indent}        # or skip the check entirely if quality_acceptance_threshold is 0\n',
                f'{indent}        quality_threshold = reset_params.get(\'quality_acceptance_threshold\', 0.4)\n',
                f'{indent}        if quality_threshold == 0:\n',
                f'{indent}            # Skip extreme deviation check during initial adaptation\n',
                f'{indent}            extreme_threshold = float(\'inf\')  # Effectively disable the check\n',
                f'{indent}        else:\n',
                f'{indent}            # Use a more lenient threshold\n',
                f'{indent}            extreme_threshold = 0.5  # 50% deviation allowed during adaptation\n',
            ]
            
            # Replace the original line with our new logic
            lines[i:i+1] = adaptation_lines
            break
    
    # Write the fixed content
    with open('src/processor.py', 'w') as f:
        f.writelines(lines)
    
    print("Fixed processor.py:")
    print("1. Added adaptation check for extreme threshold")
    print("2. During adaptation with quality_acceptance_threshold=0, extreme deviation check is disabled")
    print("3. Otherwise, uses 50% threshold during adaptation vs 20% normally")

if __name__ == "__main__":
    fix_processor()
