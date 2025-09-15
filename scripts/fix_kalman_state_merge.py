#!/usr/bin/env python3
"""
Fix the Kalman state initialization to preserve reset parameters
"""

def fix_processor():
    # Read the current processor.py
    with open('src/processor.py', 'r') as f:
        lines = f.readlines()
    
    # Find and fix the line where we overwrite state with Kalman initialization
    # Around line 183: state = KalmanFilterManager.initialize_immediate(...)
    for i, line in enumerate(lines):
        if 'state = KalmanFilterManager.initialize_immediate(' in line:
            # Replace the assignment to merge instead of overwrite
            lines[i] = line.replace(
                'state = KalmanFilterManager.initialize_immediate(',
                'kalman_state = KalmanFilterManager.initialize_immediate('
            )
            # Add merge logic after this line
            # Find the closing parenthesis
            j = i
            while ')' not in lines[j]:
                j += 1
            # Insert merge logic after the function call
            indent = '        '  # Match the indentation
            merge_lines = [
                f'{indent}# Merge Kalman state with existing state to preserve reset parameters\n',
                f'{indent}state.update(kalman_state)\n'
            ]
            lines[j+1:j+1] = merge_lines
            break
    
    # Write the fixed content
    with open('src/processor.py', 'w') as f:
        f.writelines(lines)
    
    print("Fixed processor.py:")
    print("1. Changed Kalman initialization to merge with existing state")
    print("2. This preserves reset_type and reset_parameters")

if __name__ == "__main__":
    fix_processor()
