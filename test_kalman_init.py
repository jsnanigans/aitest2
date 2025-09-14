#!/usr/bin/env python3
"""Test if kalman_params can ever be None/False."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import ProcessorStateDB

# Create a fresh database
db = ProcessorStateDB()

# Create a new user state
state = db.create_initial_state()
print(f"Initial state kalman_params: {state.get('kalman_params')}")
print(f"Has kalman_params? {state.get('kalman_params') is not None}")

# Save the state
db.save_state("test_user", state)

# Export to CSV to see what happens
db.export_to_csv("test_kalman_check.csv")

# Read the CSV
with open("test_kalman_check.csv", "r") as f:
    lines = f.readlines()
    print("\nCSV Output:")
    print(lines[0].strip())  # Header
    if len(lines) > 1:
        print(lines[1].strip())  # Data

# Clean up
Path("test_kalman_check.csv").unlink()
