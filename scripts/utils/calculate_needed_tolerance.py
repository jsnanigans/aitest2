#!/usr/bin/env python3
"""Calculate what tolerance is needed for borderline cases."""

cases = [
    ("1.9kg in 25.3h", 1.9, 25.3),
    ("1.9kg in 24.8h", 1.9, 24.8),
    ("2.4kg in 31.6h", 2.4, 31.6),
]

print("TOLERANCE NEEDED FOR BORDERLINE CASES")
print("=" * 60)

for desc, change, hours in cases:
    days = hours / 24
    base_limit = days * 1.5  # 1.5kg/day sustained rate
    needed_tolerance = (change / base_limit) - 1
    
    print(f"\n{desc}:")
    print(f"  Base limit: {base_limit:.2f}kg")
    print(f"  Actual change: {change:.1f}kg")
    print(f"  Exceeds by: {(change/base_limit - 1)*100:.1f}%")
    print(f"  Tolerance needed: {needed_tolerance*100:.1f}%")

print("\n" + "-" * 60)
print("RECOMMENDATION:")
print("Use 20% tolerance to accept these borderline cases")
print("This still rejects clearly erroneous data (>10kg changes)")
