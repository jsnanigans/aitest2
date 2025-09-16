import json

# Load results
with open('output_test_user/results_test_no_date.json', 'r') as f:
    results = json.load(f)

# Get first user
user_id = list(results['users'].keys())[0]
measurements = results['users'][user_id]

# Get rejected measurements
rejected = []
accepted = []
for m in measurements:
    if m['status'] == 'rejected':
        rejected.append(m)
    else:
        accepted.append(m)

print(f"User: {user_id}")
print(f"Total measurements: {len(measurements)}")
print(f"Accepted: {len(accepted)}")
print(f"Rejected: {len(rejected)}")
print(f"\nRejected measurements:")
for r in rejected[:30]:  # Show first 30
    print(f"  {r['date']}: {r['weight']:.2f}kg - Score: {r.get('quality_score', 0):.3f}")
    if 'quality_components' in r and r['quality_components']:
        comps = r['quality_components']
        print(f"    Safety: {comps.get('safety', 0):.3f}, Plausibility: {comps.get('plausibility', 0):.3f}, "
              f"Consistency: {comps.get('consistency', 0):.3f}, Reliability: {comps.get('reliability', 0):.3f}")
    if 'rejection_reasons' in r and r['rejection_reasons']:
        print(f"    Reasons: {', '.join(r['rejection_reasons'])}")
