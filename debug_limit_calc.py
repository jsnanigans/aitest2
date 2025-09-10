"""Debug limit calculation."""

# Test the calculation
time_delta_hours = 48.0
last_weight = 108.0
daily_rate = 1.0

days = time_delta_hours / 24
absolute_limit = days * daily_rate
print(f"Time delta: {time_delta_hours} hours = {days} days")
print(f"Daily rate: {daily_rate} kg/day")
print(f"Absolute limit: {absolute_limit} kg")
print(f"Actual change: 3.0 kg")
print(f"Should accept? {3.0 <= absolute_limit}")
