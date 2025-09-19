import json
from datetime import datetime
import sys

with open('/Users/brendanmullins/Projects/aitest/strem_process_anchor/output/results_test_no_date.json', 'r') as f:
    data = json.load(f)

problematic_changes = []

for user_id, measurements in data['users'].items():
    if len(measurements) < 2:
        continue
    
    for i in range(1, len(measurements)):
        curr = measurements[i]
        prev = measurements[i-1]
        
        if not curr.get('accepted') or not prev.get('accepted'):
            continue
            
        curr_time = datetime.fromisoformat(curr['timestamp'].replace(' ', 'T'))
        prev_time = datetime.fromisoformat(prev['timestamp'].replace(' ', 'T'))
        
        time_diff = (curr_time - prev_time).total_seconds()
        time_hours = time_diff / 3600
        time_days = time_diff / 86400
        
        weight_change = abs(curr['raw_weight'] - prev['raw_weight'])
        
        # Check for rapid changes
        if time_hours < 1 and weight_change > 2:
            problematic_changes.append({
                'user_id': user_id,
                'time_hours': time_hours,
                'time_minutes': time_hours * 60,
                'weight_change': weight_change,
                'from': prev['raw_weight'],
                'to': curr['raw_weight'],
                'timestamp': curr['timestamp']
            })
        elif time_days < 30 and weight_change > prev['raw_weight'] * 0.5:  # 50% change in a month
            problematic_changes.append({
                'user_id': user_id,
                'time_days': time_days,
                'weight_change': weight_change,
                'percent_change': (weight_change / prev['raw_weight']) * 100,
                'from': prev['raw_weight'],
                'to': curr['raw_weight'],
                'timestamp': curr['timestamp']
            })

# Sort by weight change
problematic_changes.sort(key=lambda x: x['weight_change'], reverse=True)

print("Most problematic accepted weight changes:")
print("=" * 80)
for issue in problematic_changes[:20]:
    if 'time_minutes' in issue:
        print(f"User: {issue['user_id'][:8]}... | {issue['weight_change']:.1f}kg in {issue['time_minutes']:.1f} minutes")
        print(f"  {issue['from']:.1f}kg -> {issue['to']:.1f}kg at {issue['timestamp']}")
    else:
        print(f"User: {issue['user_id'][:8]}... | {issue['weight_change']:.1f}kg ({issue['percent_change']:.0f}%) in {issue['time_days']:.1f} days")
        print(f"  {issue['from']:.1f}kg -> {issue['to']:.1f}kg at {issue['timestamp']}")
