import json
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict

# Load the user data
with open('output/debug_test_no_date/user_0040872d-333a-4ace-8c5a-b2fcd056e65a.json', 'r') as f:
    data = json.load(f)

# Extract all measurements
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
df = df.sort_values('timestamp')
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['time_of_day'] = df['hour'] + df['minute']/60
df['date'] = df['timestamp'].dt.date

print("=" * 80)
print("INVESTIGATION REPORT: Multi-User Scale Detection")
print("User ID: 0040872d-333a-4ace-8c5a-b2fcd056e65a")
print("=" * 80)

print("\n1. EVIDENCE OF MULTIPLE USERS")
print("-" * 40)

# Simple clustering by weight ranges
def classify_user(weight):
    if weight < 50:
        return 0  # Child/Teen
    elif weight < 70:
        return 1  # Light Adult
    elif weight < 90:
        return 2  # Average Adult
    else:
        return 3  # Heavy Adult

df['user_cluster'] = df['weight'].apply(classify_user)

# Get cluster statistics
clusters = []
cluster_names = ['Child/Teen (<50kg)', 'Light Adult (50-70kg)', 'Average Adult (70-90kg)', 'Heavy Adult (>90kg)']

for i in range(4):
    cluster_data = df[df['user_cluster'] == i]
    if len(cluster_data) > 0:
        clusters.append({
            'id': i,
            'name': cluster_names[i],
            'mean': cluster_data['weight'].mean(),
            'std': cluster_data['weight'].std(),
            'count': len(cluster_data),
            'min': cluster_data['weight'].min(),
            'max': cluster_data['weight'].max(),
            'accepted': len(cluster_data[cluster_data['result'] == 'accepted'])
        })

print("\nIdentified Weight Clusters (Likely Different Users):")
for c in clusters:
    print(f"\n{c['name']}:")
    print(f"  Weight range: {c['min']:.1f} - {c['max']:.1f} kg")
    print(f"  Mean weight: {c['mean']:.1f} ± {c['std']:.1f} kg")
    print(f"  Measurements: {c['count']} (accepted: {c['accepted']})")
    
    # Time patterns for this cluster
    cluster_df = df[df['user_cluster'] == c['id']]
    top_hours = cluster_df.groupby('hour').size().sort_values(ascending=False).head(3)
    if len(top_hours) > 0:
        print(f"  Preferred times: {', '.join([f'{h:02d}:00' for h in top_hours.index])}")

print("\n2. PHYSIOLOGICALLY IMPOSSIBLE CHANGES")
print("-" * 40)

# Calculate sequential changes
df['prev_weight'] = df['weight'].shift(1)
df['weight_change'] = df['weight'] - df['prev_weight']
df['time_diff_hours'] = df['timestamp'].diff().dt.total_seconds() / 3600
df['change_rate_kg_per_day'] = (df['weight_change'] / df['time_diff_hours']) * 24

# Filter for rapid changes
rapid_changes = df[(df['time_diff_hours'] < 24) & (df['weight_change'].abs() > 5)].copy()

print(f"\nFound {len(rapid_changes)} physiologically impossible changes (>5kg in <24h)")
print("\nTop 10 Most Extreme Changes:")
rapid_changes['abs_change'] = rapid_changes['weight_change'].abs()
for _, row in rapid_changes.nlargest(10, 'abs_change').iterrows():
    print(f"  {row['timestamp'].strftime('%Y-%m-%d %H:%M')}: "
          f"{row['prev_weight']:.1f} → {row['weight']:.1f} kg "
          f"({row['weight_change']:+.1f} kg in {row['time_diff_hours']:.1f}h)")

print("\n3. TIME-BASED USER SEPARATION PATTERNS")
print("-" * 40)

# Analyze measurement sessions
df['session_gap'] = df['timestamp'].diff() > pd.Timedelta(hours=1)
df['session_id'] = df['session_gap'].cumsum()

# Find sessions with multiple measurements
session_stats = df.groupby('session_id').agg({
    'weight': ['count', 'std', 'min', 'max'],
    'timestamp': ['min', 'max']
})

multi_sessions = session_stats[session_stats[('weight', 'count')] > 1]
high_variance_sessions = multi_sessions[multi_sessions[('weight', 'std')] > 10]

print(f"\nSessions with multiple measurements: {len(multi_sessions)}")
print(f"Sessions with high weight variance (>10kg std): {len(high_variance_sessions)}")

if len(high_variance_sessions) > 0:
    print("\nExample High-Variance Sessions (Multiple Users):")
    for idx in list(high_variance_sessions.index)[:5]:
        session_data = df[df['session_id'] == idx]
        weights = session_data['weight'].values
        times = session_data['timestamp'].dt.strftime('%H:%M').values
        weight_str = ', '.join([f'{w:.1f}' for w in weights])
        print(f"  Session at {session_data['timestamp'].iloc[0].strftime('%Y-%m-%d')}: "
              f"weights=[{weight_str}], range={weights.max()-weights.min():.1f}kg")

print("\n4. PATTERN ANALYSIS")
print("-" * 40)

# Check for consistent user patterns
morning_weights = df[(df['hour'] >= 5) & (df['hour'] <= 9)]
evening_weights = df[(df['hour'] >= 18) & (df['hour'] <= 22)]

print(f"\nMorning measurements (5-9am): {len(morning_weights)}")
if len(morning_weights) > 0:
    print(f"  Mean: {morning_weights['weight'].mean():.1f}kg, Std: {morning_weights['weight'].std():.1f}kg")

print(f"\nEvening measurements (6-10pm): {len(evening_weights)}")
if len(evening_weights) > 0:
    print(f"  Mean: {evening_weights['weight'].mean():.1f}kg, Std: {evening_weights['weight'].std():.1f}kg")

# Look for rapid succession measurements (family weighing together)
rapid_succession = df[df['time_diff_hours'] < 0.25]  # Within 15 minutes
print(f"\nMeasurements within 15 minutes of previous: {len(rapid_succession)}")
if len(rapid_succession) > 0:
    weight_jumps = rapid_succession[rapid_succession['weight_change'].abs() > 10]
    print(f"  With >10kg difference (different person): {len(weight_jumps)}")

print("\n5. RECOMMENDATIONS FOR IMPROVEMENT")
print("-" * 40)

print("\n5.1 Physiological Rate Limits:")
print("  Current: 50% change rejection (line 208 in processor.py)")
print("  PROBLEM: Too permissive - allows 40kg person to jump to 60kg")
print("  ")
print("  RECOMMENDED: Implement time-based absolute limits:")
print("    - < 1 hour: max 2kg change (drinking/bathroom)")
print("    - < 6 hours: max 3kg change (meals + hydration)")
print("    - < 24 hours: max 5kg change (extreme dehydration)")
print("    - > 24 hours: max 0.5kg/day sustained change")

print("\n5.2 Multi-User Detection:")
print("  RECOMMENDED: Add user profiling:")
print("    - Track typical weight range per user")
print("    - Detect when weight jumps between known clusters")
print("    - Maintain separate Kalman states for each detected user")
print("    - Use time-of-day patterns to help identify users")

print("\n5.3 Session-Based Processing:")
print("  RECOMMENDED: Group rapid measurements:")
print("    - Measurements within 5 minutes = same weighing session")
print("    - Take median of consistent weights in session")
print("    - Reject sessions with >5kg variance as multi-user")

print("\n6. SPECIFIC CODE CHANGES NEEDED")
print("-" * 40)

print("\nIn processor.py, replace line 207-209 with:")
print("""
    # Calculate time since last measurement
    time_delta_hours = 24  # Default for first measurement
    if state and state.get('last_timestamp'):
        last_ts = state['last_timestamp']
        if isinstance(last_ts, str):
            last_ts = datetime.fromisoformat(last_ts)
        time_delta_hours = (timestamp - last_ts).total_seconds() / 3600
    
    # Apply physiological limits based on time
    change = abs(weight - last_weight)
    
    if time_delta_hours < 1:
        max_change = 2.0  # Quick bathroom/drinking
    elif time_delta_hours < 6:
        max_change = 3.0  # Meals + hydration  
    elif time_delta_hours < 24:
        max_change = 5.0  # Extreme dehydration
    else:
        # Long-term: 0.5kg/day max sustained
        max_change = min(5.0, time_delta_hours / 24 * 0.5)
    
    if change > max_change:
        return False  # Physiologically impossible
""")

print("\n7. VALIDATION METRICS")
print("-" * 40)

# Calculate what would be rejected with new rules
new_rejections = []
for _, row in df.iterrows():
    if pd.notna(row.get('prev_weight')) and pd.notna(row.get('time_diff_hours')):
        change = abs(row['weight'] - row['prev_weight'])
        hours = row['time_diff_hours']
        
        if hours < 1:
            max_change = 2.0
        elif hours < 6:
            max_change = 3.0
        elif hours < 24:
            max_change = 5.0
        else:
            max_change = min(5.0, hours / 24 * 0.5)
        
        if change > max_change and row['result'] == 'accepted':
            new_rejections.append({
                'timestamp': row['timestamp'],
                'change': change,
                'hours': hours,
                'prev': row['prev_weight'],
                'curr': row['weight']
            })

currently_rejected = len(df[df['result'] == 'rejected'])
currently_accepted = len(df[df['result'] == 'accepted'])

print(f"\nCurrent performance:")
print(f"  Accepted: {currently_accepted}/{len(df)} ({currently_accepted/len(df)*100:.1f}%)")
print(f"  Rejected: {currently_rejected}/{len(df)} ({currently_rejected/len(df)*100:.1f}%)")

print(f"\nWith new physiological limits:")
print(f"  Additional rejections: {len(new_rejections)}")
print(f"  New acceptance rate: {(currently_accepted - len(new_rejections))/len(df)*100:.1f}%")

if len(new_rejections) > 0:
    print(f"\nExamples of newly rejected changes:")
    for rej in new_rejections[:5]:
        print(f"  {rej['timestamp'].strftime('%Y-%m-%d')}: "
              f"{rej['prev']:.1f} → {rej['curr']:.1f}kg "
              f"({rej['change']:.1f}kg in {rej['hours']:.1f}h)")

print("\n" + "=" * 80)
print("END OF INVESTIGATION REPORT")
print("=" * 80)
