import json
import numpy as np
from datetime import datetime
import pandas as pd

with open('output/debug_test_no_date/user_0040872d-333a-4ace-8c5a-b2fcd056e65a.json', 'r') as f:
    data = json.load(f)

measurements = []
for log in data['processing_logs']:
    if log['timestamp'] and log['weight']:
        measurements.append({
            'timestamp': datetime.fromisoformat(log['timestamp']),
            'weight': log['weight'],
            'result': log['result'],
            'source': log['source']
        })

df = pd.DataFrame(measurements)
df['hour'] = df['timestamp'].dt.hour
df['date'] = df['timestamp'].dt.date

print('=== WEIGHT DISTRIBUTION ANALYSIS ===')
accepted = df[df['result'] == 'accepted']
rejected = df[df['result'] == 'rejected']

print(f'Accepted weights: mean={accepted.weight.mean():.1f}, std={accepted.weight.std():.1f}')
print(f'Rejected weights: mean={rejected.weight.mean():.1f}, std={rejected.weight.std():.1f}')

print('\n=== WEIGHT RANGES (Potential Different Users) ===')
ranges = [
    ('Child/Teen (<50kg)', df[df['weight'] < 50]),
    ('Light Adult (50-70kg)', df[(df['weight'] >= 50) & (df['weight'] < 70)]),
    ('Average Adult (70-90kg)', df[(df['weight'] >= 70) & (df['weight'] < 90)]),
    ('Heavy Adult (>=90kg)', df[df['weight'] >= 90])
]

for name, subset in ranges:
    if len(subset) > 0:
        acc = subset[subset['result'] == 'accepted']
        rej = subset[subset['result'] == 'rejected']
        print(f'\n{name}:')
        print(f'  Total: {len(subset)}, Accepted: {len(acc)}, Rejected: {len(rej)}')
        print(f'  Weight: mean={subset.weight.mean():.1f}kg, range={subset.weight.min():.1f}-{subset.weight.max():.1f}kg')
        top_hours = subset.groupby('hour').size().sort_values(ascending=False).head(3)
        hour_str = ', '.join([f'{h}:00({c})' for h, c in zip(top_hours.index, top_hours.values)])
        print(f'  Top hours: {hour_str}')

print('\n=== SEQUENTIAL WEIGHT CHANGES (Physiological Plausibility) ===')
df_sorted = df.sort_values('timestamp')
df_sorted['weight_change'] = df_sorted['weight'].diff()
df_sorted['time_diff_hours'] = df_sorted['timestamp'].diff().dt.total_seconds() / 3600

print('\nLarge changes (>15kg) within 24 hours:')
same_day = df_sorted[df_sorted['time_diff_hours'] < 24].copy()
same_day['abs_change'] = same_day['weight_change'].abs()
large = same_day[same_day['abs_change'] > 15].head(10)

for _, row in large.iterrows():
    prev_idx = df_sorted.index.get_loc(row.name) - 1
    prev_weight = df_sorted.iloc[prev_idx]['weight']
    print(f'  {row.timestamp}: {prev_weight:.1f}kg → {row.weight:.1f}kg ({row.weight_change:+.1f}kg in {row.time_diff_hours:.1f}h)')

print('\n=== TIME INTERVAL PATTERNS ===')
df_sorted['time_since_last'] = df_sorted['timestamp'].diff()
regular_users = df_sorted[df_sorted['time_since_last'] < pd.Timedelta(hours=1)]
print(f'Measurements within 1 hour of each other: {len(regular_users)}')

df_sorted['session'] = (df_sorted['time_since_last'] > pd.Timedelta(hours=2)).cumsum()
sessions = df_sorted.groupby('session').agg({
    'weight': ['count', 'mean', 'std', 'min', 'max'],
    'timestamp': ['min', 'max']
})

multi_measure_sessions = sessions[sessions[('weight', 'count')] > 1]
print(f'\nSessions with multiple measurements: {len(multi_measure_sessions)}')
if len(multi_measure_sessions) > 0:
    print('Sample multi-measurement sessions (showing weight variance):')
    for idx in multi_measure_sessions.head(5).index:
        session_data = df_sorted[df_sorted['session'] == idx]
        weights = list(session_data['weight'].values)
        weight_str = ', '.join([f'{w:.1f}' for w in weights])
        print(f'  Session {idx}: weights=[{weight_str}], range={max(weights)-min(weights):.1f}kg')

print('\n=== DAILY WEIGHT CHANGE LIMITS ===')
print('Maximum physiologically possible weight changes:')
print('  - Water weight: ±2-3kg/day')
print('  - Extreme dehydration: up to -5kg/day')
print('  - Actual tissue change: ±0.5kg/day max')
print('\nCurrent acceptance thresholds need adjustment for:')
rapid_changes = same_day[same_day['abs_change'] > 5]
print(f'  - Rapid changes >5kg within a day: {len(rapid_changes)} occurrences')
print(f'  - These are likely different users on the same scale')
