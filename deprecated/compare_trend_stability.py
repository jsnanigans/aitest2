#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

user_id = "0040872d-333a-4ace-8c5a-b2fcd056e65a"

with open("output/results_yes.json", "r") as f:
    data = json.load(f)

if "users" not in data or user_id not in data["users"]:
    print(f"User {user_id} not found")
    exit(1)

user_data = data["users"][user_id]
dates = [datetime.fromisoformat(r["date"]) for r in user_data["time_series"]]
weights = [r["weight"] for r in user_data["time_series"]]
kalman_weights = [r.get("kalman_filtered", r["weight"]) for r in user_data["time_series"]]
kalman_trends = [r.get("kalman_trend_kg_per_day", 0) * 7 for r in user_data["time_series"]]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.scatter(dates, weights, alpha=0.3, s=20, color='red', label='Raw measurements')
ax1.plot(dates, kalman_weights, 'b-', linewidth=2, label='Kalman filtered')
ax1.set_ylabel('Weight (kg)')
ax1.set_title(f'Weight Trajectory with Stable Kalman Filter - User {user_id[:8]}...')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(dates, kalman_trends, 'g-', linewidth=2, label='Trend (kg/week)')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_ylabel('Trend (kg/week)')
ax2.set_xlabel('Date')
ax2.set_title('Kalman Filter Trend - Now More Stable')
ax2.legend()
ax2.grid(True, alpha=0.3)

for i in range(1, len(kalman_trends)):
    if abs(kalman_trends[i] - kalman_trends[i-1]) > 0.5:
        ax2.axvline(x=dates[i], color='orange', alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig('output/trend_stability_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved comparison to output/trend_stability_comparison.png")

trend_changes = []
for i in range(1, len(kalman_trends)):
    trend_changes.append(abs(kalman_trends[i] - kalman_trends[i-1]))

print(f"\nTrend stability metrics:")
print(f"Mean trend change: {np.mean(trend_changes):.4f} kg/week")
print(f"Std trend change: {np.std(trend_changes):.4f} kg/week")
print(f"Max trend change: {np.max(trend_changes):.4f} kg/week")
print(f"Trend changes > 0.5 kg/week: {sum(1 for c in trend_changes if c > 0.5)}/{len(trend_changes)}")