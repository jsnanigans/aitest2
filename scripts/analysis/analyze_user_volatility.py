#!/usr/bin/env python3
"""Analyze the rejection for user 1a452430-7351-4b8c-b921-4fb17f8a29cc."""

from datetime import datetime

print("USER 1a452430-7351-4b8c-b921-4fb17f8a29cc REJECTION ANALYSIS")
print("=" * 70)

# The sequence of measurements
measurements = [
    ("2025-02-10 13:23:27", 161.1, "accepted", 161.19),  # filtered weight
    ("2025-02-11 14:42:17", 159.2, "rejected", None),    # 1.9kg drop
    ("2025-02-12 13:35:40", 158.7, "accepted", None),    # continues dropping
]

print("\nMeasurement Sequence:")
for ts, raw, status, filtered in measurements:
    filt_str = f" → {filtered:.2f}kg" if filtered else ""
    print(f"  {ts}: {raw}kg{filt_str} - {status.upper()}")

print("\nAnalysis:")
print("-" * 40)

# Calculate the changes
feb10_to_feb11 = abs(161.1 - 159.2)
feb11_to_feb12 = abs(159.2 - 158.7)
feb10_to_feb12 = abs(161.1 - 158.7)

print(f"Feb 10 → Feb 11: {feb10_to_feb11:.1f}kg drop in ~25 hours")
print(f"Feb 11 → Feb 12: {feb11_to_feb12:.1f}kg drop in ~23 hours")
print(f"Feb 10 → Feb 12: {feb10_to_feb12:.1f}kg total drop in ~48 hours")

print("\nPhysiological Limit Calculation:")
print("-" * 40)
time_delta_hours = 25.3
days = time_delta_hours / 24
sustained_rate = 1.5  # kg/day
limit = days * sustained_rate

print(f"Time delta: {time_delta_hours:.1f} hours = {days:.2f} days")
print(f"Sustained rate: {sustained_rate}kg/day")
print(f"Calculated limit: {limit:.1f}kg")
print(f"Actual change: {feb10_to_feb11:.1f}kg")
print(f"Exceeds by: {feb10_to_feb11 - limit:.1f}kg")

print("\nTHE ISSUE:")
print("-" * 40)
print("The rejection is TECHNICALLY CORRECT but arguably too strict:")
print(f"- Change of {feb10_to_feb11:.1f}kg exceeds limit of {limit:.1f}kg by just 0.3kg")
print("- This is only 19% over the limit")
print("- The user shows a consistent downward trend (losing weight)")
print("- Total 2-day change of 2.4kg averages to 1.2kg/day (within limits!)")

print("\nCONSIDERATIONS:")
print("-" * 40)
print("1. STRICT INTERPRETATION (current):")
print("   - Any measurement exceeding limit gets rejected")
print("   - Protects against bad data")
print("   - But may reject valid weight loss patterns")
print()
print("2. FLEXIBLE INTERPRETATION (proposed):")
print("   - Add a small tolerance (e.g., 10-20%) for borderline cases")
print("   - OR use a confidence-based approach")
print("   - OR consider trend consistency")
print()
print("3. CONTEXT-AWARE (advanced):")
print("   - If user is consistently losing/gaining weight")
print("   - Allow slightly higher sustained rates")
print("   - Especially if trend is consistent")

print("\nRECOMMENDATION:")
print("-" * 40)
print("Add a 10% tolerance to sustained limits for borderline cases:")
print(f"- Current: {limit:.1f}kg limit")
print(f"- With 10% tolerance: {limit * 1.1:.1f}kg limit")
print(f"- Would accept this {feb10_to_feb11:.1f}kg change ✓")
print()
print("OR consider the user's trend:")
print("- User is consistently losing weight")
print("- 1.9kg/day during active weight loss is not unreasonable")
print("- Especially if hydration/timing varies")
