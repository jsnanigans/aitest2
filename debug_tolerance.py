#!/usr/bin/env python3
"""Debug why tolerance isn't working."""

# Check the calculation
time_delta_hours = 25.3
last_weight = 161.1
change = 1.9

# Sustained limit calculation
days = time_delta_hours / 24
sustained_rate = 1.5
base_limit = days * sustained_rate
tolerance = 0.10
effective_limit = base_limit * (1 + tolerance)

print(f"Time: {time_delta_hours}h = {days:.2f} days")
print(f"Base limit: {base_limit:.2f}kg")
print(f"With 10% tolerance: {effective_limit:.2f}kg")
print(f"Actual change: {change:.1f}kg")
print(f"Should accept: {change <= effective_limit}")
print()
print(f"The issue: {change:.1f} > {effective_limit:.2f} is {change > effective_limit}")
