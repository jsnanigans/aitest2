#!/usr/bin/env python3
"""
Analyze real user data to identify potential replay mechanism issues.

Looks for:
1. Large gaps followed by dramatic weight changes
2. Multiple resets in short periods
3. Oscillating patterns that might confuse the replay
4. Source reliability issues
5. Edge cases that could break the replay logic
"""

import csv
import json
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys

def parse_timestamp(date_str):
    """Parse various timestamp formats."""
    if not date_str:
        return None
    try:
        if "T" in date_str:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        elif " " in date_str:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        else:
            return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None

def analyze_user_data(csv_path):
    """Analyze user data for potential replay issues."""

    # Read all data
    users_data = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row.get('user_id')
            if not user_id:
                continue

            weight_str = row.get('weight', '').strip()
            if not weight_str or weight_str.upper() == 'NULL':
                continue

            try:
                weight = float(weight_str)
                if weight <= 0 or weight > 1000:
                    continue
            except:
                continue

            timestamp = parse_timestamp(row.get('effectiveDateTime'))
            if not timestamp:
                continue

            source = row.get('source_type', 'unknown')
            unit = row.get('unit', 'kg')

            users_data[user_id].append({
                'weight': weight,
                'timestamp': timestamp,
                'source': source,
                'unit': unit
            })

    # Sort each user's data by timestamp
    for user_id in users_data:
        users_data[user_id].sort(key=lambda x: x['timestamp'])

    print(f"Loaded data for {len(users_data)} users")

    # Analyze for potential issues
    issues = {
        'large_gaps_with_changes': [],
        'rapid_resets': [],
        'oscillating_patterns': [],
        'source_confusion': [],
        'extreme_variations': [],
        'problematic_sequences': []
    }

    for user_id, measurements in users_data.items():
        if len(measurements) < 5:
            continue

        # 1. Check for large gaps with dramatic changes
        for i in range(1, len(measurements)):
            prev = measurements[i-1]
            curr = measurements[i]

            gap_days = (curr['timestamp'] - prev['timestamp']).total_seconds() / 86400

            if gap_days > 20:  # Large gap
                weight_change = abs(curr['weight'] - prev['weight']) / prev['weight']

                if weight_change > 0.15:  # >15% change
                    issues['large_gaps_with_changes'].append({
                        'user_id': user_id,
                        'gap_days': gap_days,
                        'weight_before': prev['weight'],
                        'weight_after': curr['weight'],
                        'change_percent': weight_change * 100,
                        'timestamp': curr['timestamp'],
                        'source_before': prev['source'],
                        'source_after': curr['source']
                    })

        # 2. Check for rapid resets (multiple questionnaires in short period)
        questionnaire_timestamps = [
            m['timestamp'] for m in measurements
            if 'questionnaire' in m['source'].lower()
        ]

        if len(questionnaire_timestamps) > 2:
            for i in range(1, len(questionnaire_timestamps)):
                gap_days = (questionnaire_timestamps[i] - questionnaire_timestamps[i-1]).total_seconds() / 86400
                if gap_days < 7:  # Multiple resets within a week
                    issues['rapid_resets'].append({
                        'user_id': user_id,
                        'gap_days': gap_days,
                        'timestamp1': questionnaire_timestamps[i-1],
                        'timestamp2': questionnaire_timestamps[i]
                    })

        # 3. Check for oscillating patterns
        if len(measurements) >= 10:
            weights = [m['weight'] for m in measurements[-10:]]  # Last 10 measurements

            # Count direction changes
            direction_changes = 0
            for i in range(2, len(weights)):
                if (weights[i] - weights[i-1]) * (weights[i-1] - weights[i-2]) < 0:
                    direction_changes += 1

            if direction_changes >= 5:  # Highly oscillating
                weight_range = max(weights) - min(weights)
                avg_weight = sum(weights) / len(weights)

                if weight_range / avg_weight > 0.1:  # Significant oscillation
                    issues['oscillating_patterns'].append({
                        'user_id': user_id,
                        'direction_changes': direction_changes,
                        'weight_range': weight_range,
                        'weights': weights,
                        'timestamps': [m['timestamp'] for m in measurements[-10:]]
                    })

        # 4. Check for source confusion (rapid source changes with different values)
        for i in range(1, min(len(measurements), len(measurements)-5)):
            window = measurements[i:i+5]
            sources = [m['source'] for m in window]
            weights = [m['weight'] for m in window]

            if len(set(sources)) >= 3:  # Multiple different sources
                weight_std = np.std(weights)
                weight_mean = np.mean(weights)

                if weight_std / weight_mean > 0.1:  # High variation between sources
                    issues['source_confusion'].append({
                        'user_id': user_id,
                        'sources': sources,
                        'weights': weights,
                        'std_percent': (weight_std / weight_mean) * 100,
                        'timestamp': window[0]['timestamp']
                    })
                    break  # Only record once per user

        # 5. Check for extreme variations in short periods
        for i in range(len(measurements) - 3):
            window = measurements[i:i+4]
            time_span = (window[-1]['timestamp'] - window[0]['timestamp']).total_seconds() / 86400

            if time_span <= 7:  # Within a week
                weights = [m['weight'] for m in window]
                weight_range = max(weights) - min(weights)
                avg_weight = sum(weights) / len(weights)

                if weight_range / avg_weight > 0.2:  # >20% variation
                    issues['extreme_variations'].append({
                        'user_id': user_id,
                        'time_span_days': time_span,
                        'weights': weights,
                        'range_percent': (weight_range / avg_weight) * 100,
                        'timestamp': window[0]['timestamp']
                    })
                    break  # Only record once per user

        # 6. Check for problematic sequences (questionnaire → large drop → recovery)
        for i in range(len(measurements) - 2):
            if 'questionnaire' in measurements[i]['source'].lower():
                reset_weight = measurements[i]['weight']

                # Check next measurement
                if i + 1 < len(measurements):
                    next_weight = measurements[i + 1]['weight']
                    drop = (reset_weight - next_weight) / reset_weight

                    if drop > 0.1:  # >10% drop after reset
                        # Check if it recovers
                        if i + 2 < len(measurements):
                            recovery_weight = measurements[i + 2]['weight']
                            if abs(recovery_weight - reset_weight) < abs(next_weight - reset_weight):
                                issues['problematic_sequences'].append({
                                    'user_id': user_id,
                                    'reset_weight': reset_weight,
                                    'drop_weight': next_weight,
                                    'recovery_weight': recovery_weight,
                                    'drop_percent': drop * 100,
                                    'timestamp': measurements[i]['timestamp']
                                })

    return issues, users_data

def find_interesting_cases(issues):
    """Find the most interesting cases for detailed analysis."""

    interesting_cases = []

    # Case 1: Largest gap with change
    if issues['large_gaps_with_changes']:
        largest_gap = max(issues['large_gaps_with_changes'], key=lambda x: x['gap_days'])
        largest_change = max(issues['large_gaps_with_changes'], key=lambda x: x['change_percent'])

        interesting_cases.append({
            'type': 'largest_gap',
            'case': largest_gap
        })

        if largest_change['user_id'] != largest_gap['user_id']:
            interesting_cases.append({
                'type': 'largest_change_after_gap',
                'case': largest_change
            })

    # Case 2: Most rapid resets
    if issues['rapid_resets']:
        most_rapid = min(issues['rapid_resets'], key=lambda x: x['gap_days'])
        interesting_cases.append({
            'type': 'rapid_resets',
            'case': most_rapid
        })

    # Case 3: Most extreme oscillation
    if issues['oscillating_patterns']:
        most_oscillating = max(issues['oscillating_patterns'], key=lambda x: x['direction_changes'])
        interesting_cases.append({
            'type': 'oscillating',
            'case': most_oscillating
        })

    # Case 4: Most problematic sequence
    if issues['problematic_sequences']:
        worst_sequence = max(issues['problematic_sequences'], key=lambda x: x['drop_percent'])
        interesting_cases.append({
            'type': 'problematic_sequence',
            'case': worst_sequence
        })

    return interesting_cases

def main():
    csv_path = "./data/2025-09-05_nocon.csv"

    print("Analyzing user data for replay mechanism issues...")
    print("=" * 60)

    issues, users_data = analyze_user_data(csv_path)

    # Print summary statistics
    print("\n=== ISSUE SUMMARY ===")
    print(f"Large gaps with changes: {len(issues['large_gaps_with_changes'])} cases")
    print(f"Rapid resets: {len(issues['rapid_resets'])} cases")
    print(f"Oscillating patterns: {len(issues['oscillating_patterns'])} users")
    print(f"Source confusion: {len(issues['source_confusion'])} users")
    print(f"Extreme variations: {len(issues['extreme_variations'])} users")
    print(f"Problematic sequences: {len(issues['problematic_sequences'])} cases")

    # Find interesting cases
    interesting = find_interesting_cases(issues)

    print("\n=== INTERESTING CASES FOR TESTING ===")
    for case in interesting:
        print(f"\n{case['type'].upper()}:")
        if case['type'] == 'largest_gap':
            c = case['case']
            print(f"  User: {c['user_id']}")
            print(f"  Gap: {c['gap_days']:.1f} days")
            print(f"  Weight change: {c['weight_before']:.1f}kg → {c['weight_after']:.1f}kg ({c['change_percent']:.1f}%)")

        elif case['type'] == 'largest_change_after_gap':
            c = case['case']
            print(f"  User: {c['user_id']}")
            print(f"  Gap: {c['gap_days']:.1f} days")
            print(f"  Weight change: {c['weight_before']:.1f}kg → {c['weight_after']:.1f}kg ({c['change_percent']:.1f}%)")

        elif case['type'] == 'rapid_resets':
            c = case['case']
            print(f"  User: {c['user_id']}")
            print(f"  Resets within {c['gap_days']:.1f} days")

        elif case['type'] == 'oscillating':
            c = case['case']
            print(f"  User: {c['user_id']}")
            print(f"  Direction changes: {c['direction_changes']}")
            print(f"  Weight range: {c['weight_range']:.1f}kg")

        elif case['type'] == 'problematic_sequence':
            c = case['case']
            print(f"  User: {c['user_id']}")
            print(f"  Reset: {c['reset_weight']:.1f}kg → Drop: {c['drop_weight']:.1f}kg → Recovery: {c['recovery_weight']:.1f}kg")
            print(f"  Drop: {c['drop_percent']:.1f}%")

    # Save detailed results
    output_file = "replay_issues_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_users': len(users_data),
                'large_gaps_with_changes': len(issues['large_gaps_with_changes']),
                'rapid_resets': len(issues['rapid_resets']),
                'oscillating_patterns': len(issues['oscillating_patterns']),
                'source_confusion': len(issues['source_confusion']),
                'extreme_variations': len(issues['extreme_variations']),
                'problematic_sequences': len(issues['problematic_sequences'])
            },
            'interesting_cases': interesting,
            'sample_issues': {
                'large_gaps': issues['large_gaps_with_changes'][:5],
                'rapid_resets': issues['rapid_resets'][:5],
                'oscillating': issues['oscillating_patterns'][:5],
                'problematic_sequences': issues['problematic_sequences'][:5]
            }
        }, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_file}")

    # Export specific user data for testing
    test_users = set()
    for case in interesting:
        test_users.add(case['case']['user_id'])

    if test_users:
        print(f"\n=== EXPORTING TEST USER DATA ===")
        for user_id in list(test_users)[:5]:  # Export up to 5 users
            if user_id in users_data:
                user_file = f"test_user_{user_id[:8]}.json"
                with open(user_file, 'w') as f:
                    json.dump(users_data[user_id], f, indent=2, default=str)
                print(f"  Exported: {user_file}")

if __name__ == "__main__":
    main()