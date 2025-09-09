#!/usr/bin/env python3
"""Visualize trend decay behavior."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load the user data
with open('output/users/02E11B7DC40F41638D3FA6169A147156.json', 'r') as f:
    data = json.load(f)

time_series = data['time_series']

# Extract trends and gaps
trends = []
gaps = []
dates = []
weights = []

for i, reading in enumerate(time_series):
    trend = reading.get('kalman_trend_kg_per_day', 0)
    gap = reading.get('days_since_last', 0)
    trends.append(trend)
    gaps.append(gap)
    dates.append(i)  # Use index for x-axis
    weights.append(reading['weight'])

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Trend over time with gaps highlighted
ax1 = axes[0]
ax1.plot(dates, trends, 'b-', linewidth=1, alpha=0.7, label='Trend')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Highlight large gaps
for i, gap in enumerate(gaps):
    if gap > 5:
        ax1.axvline(x=i, color='r', alpha=0.2, linewidth=gap/2)
        ax1.text(i, max(trends)*0.8, f'{gap:.1f}d', rotation=90, fontsize=8, alpha=0.5)

ax1.set_ylabel('Trend (kg/day)')
ax1.set_title('Trend Evolution with Time Gaps (Red bars = gaps > 5 days)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Trend decay visualization
ax2 = axes[1]

# Find sequences of repeated measurements (same timestamp)
sequences = []
current_seq = [0]

for i in range(1, len(gaps)):
    if gaps[i] < 0.01:  # Same timestamp
        current_seq.append(i)
    else:
        if len(current_seq) > 1:
            sequences.append(current_seq)
        current_seq = [i]

if len(current_seq) > 1:
    sequences.append(current_seq)

# Plot decay sequences
colors = plt.cm.rainbow(np.linspace(0, 1, min(len(sequences), 10)))
for seq_idx, seq in enumerate(sequences[:10]):  # Show first 10 sequences
    seq_trends = [trends[i] for i in seq]
    seq_x = list(range(len(seq_trends)))
    if len(seq_trends) > 3:  # Only show significant sequences
        ax2.plot(seq_x, seq_trends, 'o-', color=colors[seq_idx % len(colors)],
                label=f'Seq {seq_idx+1} (n={len(seq)})', markersize=4)

ax2.set_xlabel('Measurements in Sequence')
ax2.set_ylabel('Trend (kg/day)')
ax2.set_title('Trend Decay Within Measurement Sequences (Same Timestamp)')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Weight and trend correlation
ax3 = axes[2]
ax3_twin = ax3.twinx()

ax3.plot(dates, weights, 'g-', alpha=0.5, linewidth=1, label='Weight')
ax3_twin.plot(dates, trends, 'b-', alpha=0.7, linewidth=1, label='Trend')
ax3_twin.axhline(y=0, color='b', linestyle='--', alpha=0.3)

ax3.set_xlabel('Reading Index')
ax3.set_ylabel('Weight (kg)', color='g')
ax3_twin.set_ylabel('Trend (kg/day)', color='b')
ax3.set_title('Weight vs Trend Over Time')
ax3.tick_params(axis='y', labelcolor='g')
ax3_twin.tick_params(axis='y', labelcolor='b')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/trend_decay_analysis.png', dpi=150, bbox_inches='tight')
print(f"Visualization saved to output/trend_decay_analysis.png")

# Print summary statistics
print("\nDecay Analysis Summary:")
print("=" * 60)

# Analyze decay effectiveness
large_gaps = [(i, gap) for i, gap in enumerate(gaps) if gap > 5]
print(f"Number of large gaps (>5 days): {len(large_gaps)}")

for i, gap in large_gaps[:5]:
    if i > 0:
        prev_trend = trends[i-1]
        curr_trend = trends[i]
        expected_decay = 1 - (min(gap/20, 1.0) ** 3)
        actual_retention = curr_trend / prev_trend if prev_trend != 0 else 0
        print(f"\nGap at index {i}: {gap:.1f} days")
        print(f"  Previous trend: {prev_trend:.5f}")
        print(f"  Current trend:  {curr_trend:.5f}")
        print(f"  Expected retention: {expected_decay:.1%}")
        print(f"  Actual change: {actual_retention:.1%}")

print(f"\nTotal measurement sequences found: {len(sequences)}")
print(f"Average sequence length: {np.mean([len(s) for s in sequences]):.1f}")