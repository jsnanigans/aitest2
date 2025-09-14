# Fix for the viz_diagnostic.py timestamp issue
import sys
from pathlib import Path

# Read the file
file_path = Path(__file__).parent / "viz_diagnostic.py"
content = file_path.read_text()

# Fix the add_vline call to convert timestamp to string
old_line = """                fig.add_vline(
                    x=reset['timestamp'],"""

new_line = """                fig.add_vline(
                    x=reset['timestamp'].isoformat() if hasattr(reset['timestamp'], 'isoformat') else str(reset['timestamp']),"""

content = content.replace(old_line, new_line)

# Write back
file_path.write_text(content)
print("Fixed timestamp issue in viz_diagnostic.py")
