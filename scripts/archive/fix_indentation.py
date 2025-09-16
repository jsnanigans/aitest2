"""Fix indentation in quality_scorer.py"""

with open('src/quality_scorer.py', 'r') as f:
    lines = f.readlines()

# Fix line 246 - should have 4 spaces, not 8
if '        def calculate_consistency_score(' in lines[245]:
    lines[245] = '    def calculate_consistency_score(\n'
    print("Fixed indentation on line 246")

with open('src/quality_scorer.py', 'w') as f:
    f.writelines(lines)
