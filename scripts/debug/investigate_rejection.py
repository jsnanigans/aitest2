from datetime import datetime

# Data from the rejection
prev_timestamp = datetime.strptime("2025-09-02 19:08:00", "%Y-%m-%d %H:%M:%S")
curr_timestamp = datetime.strptime("2025-09-03 20:36:02", "%Y-%m-%d %H:%M:%S")
prev_weight = 103.4
curr_weight = 103.1

# Calculate time delta
time_delta = curr_timestamp - prev_timestamp
time_delta_hours = time_delta.total_seconds() / 3600

# Calculate change
change = abs(curr_weight - prev_weight)

print(f"Previous: {prev_timestamp} - {prev_weight}kg")
print(f"Current:  {curr_timestamp} - {curr_weight}kg")
print(f"Time delta: {time_delta_hours:.1f} hours")
print(f"Weight change: {change:.1f}kg")
print()

# Check against sustained limit (>24h)
if time_delta_hours > 24:
    daily_rate = 1.5  # kg/day
    days = time_delta_hours / 24
    limit = days * daily_rate
    print(f"Sustained limit calculation:")
    print(f"  Days: {days:.2f}")
    print(f"  Daily rate: {daily_rate}kg/day")
    print(f"  Limit: {limit:.1f}kg")
    print(f"  Actual change: {change:.1f}kg")
    print(f"  Would reject: {change > limit}")
    print()

# But wait - let's check the actual filtered weight from previous measurement
prev_filtered = 104.95974564199044  # From line 518 in the debug file
print(f"Previous FILTERED weight: {prev_filtered:.2f}kg")
print(f"Change from filtered: {abs(curr_weight - prev_filtered):.1f}kg")
print()

# The rejection message says 1.9kg change
print("Rejection message says: 'Change of 1.9kg in 25.5h'")
print(f"This matches: {abs(curr_weight - prev_filtered):.1f}kg ≈ 1.9kg")
