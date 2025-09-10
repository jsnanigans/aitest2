"""
Generate comprehensive multi-user investigation report.
"""

import csv
from datetime import datetime
from collections import defaultdict
from src.visualization import normalize_source_type


def analyze_dataset():
    """Analyze the complete dataset."""
    
    # Load all data
    user_data = defaultdict(list)
    
    with open('data/test_sample.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['weight'] and row['effectiveDateTime']:
                user_data[row['user_id']].append({
                    'timestamp': datetime.fromisoformat(row['effectiveDateTime']),
                    'weight': float(row['weight']),
                    'source': row['source_type'],
                    'normalized': normalize_source_type(row['source_type'])
                })
    
    print("=" * 60)
    print("MULTI-USER DATASET ANALYSIS")
    print("=" * 60)
    print()
    
    print(f"Total Users: {len(user_data)}")
    print(f"Total Measurements: {sum(len(data) for data in user_data.values())}")
    print()
    
    print("USER BREAKDOWN:")
    print("-" * 40)
    
    # Analyze each user
    user_summaries = []
    for uid, measurements in user_data.items():
        # Sort by timestamp
        measurements.sort(key=lambda x: x['timestamp'])
        
        # Get source distribution
        sources = defaultdict(int)
        for m in measurements:
            sources[m['normalized']] += 1
        
        # Calculate time span
        if len(measurements) > 1:
            time_span = (measurements[-1]['timestamp'] - measurements[0]['timestamp']).days
        else:
            time_span = 0
        
        # Get weight range
        weights = [m['weight'] for m in measurements]
        weight_range = max(weights) - min(weights) if weights else 0
        
        summary = {
            'user_id': uid,
            'count': len(measurements),
            'sources': dict(sources),
            'source_diversity': len(sources),
            'time_span_days': time_span,
            'weight_range_kg': weight_range,
            'avg_weight': sum(weights) / len(weights) if weights else 0
        }
        user_summaries.append(summary)
    
    # Sort by measurement count
    user_summaries.sort(key=lambda x: x['count'], reverse=True)
    
    for i, summary in enumerate(user_summaries, 1):
        print(f"\n{i}. User {summary['user_id'][:8]}...")
        print(f"   Measurements: {summary['count']}")
        print(f"   Time Span: {summary['time_span_days']} days")
        print(f"   Weight Range: {summary['weight_range_kg']:.1f} kg")
        print(f"   Avg Weight: {summary['avg_weight']:.1f} kg")
        print(f"   Source Types: {summary['source_diversity']}")
        for source, count in summary['sources'].items():
            print(f"     - {source}: {count} ({count/summary['count']*100:.1f}%)")
    
    print()
    print("=" * 60)
    print("SOURCE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Overall source distribution
    all_sources = defaultdict(int)
    for data in user_data.values():
        for m in data:
            all_sources[m['normalized']] += 1
    
    total_measurements = sum(all_sources.values())
    print(f"\nOverall Source Distribution ({total_measurements} total):")
    for source, count in sorted(all_sources.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_measurements * 100
        print(f"  {source:20s}: {count:3d} ({percentage:5.1f}%)")
    
    print()
    print("=" * 60)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 60)
    
    # Categorize users
    sufficient_data = [s for s in user_summaries if s['count'] >= 30]
    moderate_data = [s for s in user_summaries if 10 <= s['count'] < 30]
    insufficient_data = [s for s in user_summaries if s['count'] < 10]
    
    print(f"\nUsers with sufficient data (30+ measurements): {len(sufficient_data)}")
    print(f"Users with moderate data (10-29 measurements): {len(moderate_data)}")
    print(f"Users with insufficient data (<10 measurements): {len(insufficient_data)}")
    
    # Source diversity analysis
    single_source = [s for s in user_summaries if s['source_diversity'] == 1]
    multi_source = [s for s in user_summaries if s['source_diversity'] >= 2]
    
    print(f"\nSingle-source users: {len(single_source)}")
    print(f"Multi-source users: {len(multi_source)}")
    
    print()
    print("=" * 60)
    print("ANALYSIS LIMITATIONS")
    print("=" * 60)
    
    print("\n⚠️ CRITICAL LIMITATIONS:")
    print(f"  • Only {len(sufficient_data)} user(s) have enough data for robust analysis")
    print(f"  • {len(single_source)} users have single source (no comparison possible)")
    print(f"  • Limited source diversity across users")
    print(f"  • Total dataset size: {total_measurements} measurements")
    
    print("\n📊 STATISTICAL VALIDITY:")
    if len(sufficient_data) < 10:
        print("  ❌ Insufficient sample size for statistical significance")
        print("  ❌ Results may not generalize to broader population")
        print("  ❌ Source impact analysis unreliable with current data")
    else:
        print("  ✅ Adequate sample size for preliminary analysis")
        print("  ⚠️ More diverse users needed for robust conclusions")
    
    print()
    print("=" * 60)
    print("RECOMMENDATIONS FOR ANALYSIS")
    print("=" * 60)
    
    print("\n1. DATA COLLECTION:")
    print("   • Need minimum 30 users with 30+ measurements each")
    print("   • Ensure source diversity within users")
    print("   • Target mix of device, API, manual, questionnaire sources")
    
    print("\n2. CURRENT ANALYSIS VALIDITY:")
    if len(sufficient_data) == 1:
        print("   • Results based on single user - NOT GENERALIZABLE")
        print("   • Treat as case study, not population analysis")
        print("   • Cannot draw conclusions about source impact")
    elif len(sufficient_data) < 5:
        print("   • Very limited sample - results preliminary only")
        print("   • High risk of user-specific bias")
        print("   • Need more data before implementation decisions")
    
    print("\n3. NEXT STEPS:")
    print("   • Collect more user data before source differentiation")
    print("   • Focus on baseline processor optimization")
    print("   • Implement source logging for future analysis")
    
    # Generate summary report
    with open('output/dataset_investigation.txt', 'w') as f:
        f.write("DATASET INVESTIGATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("EXECUTIVE SUMMARY:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Users: {len(user_data)}\n")
        f.write(f"Total Measurements: {total_measurements}\n")
        f.write(f"Users with Sufficient Data: {len(sufficient_data)}\n")
        f.write(f"Multi-Source Users: {len(multi_source)}\n\n")
        
        f.write("KEY FINDING:\n")
        if len(sufficient_data) <= 1:
            f.write("⚠️ INSUFFICIENT DATA FOR SOURCE IMPACT ANALYSIS\n")
            f.write("Only 1 user has enough measurements for analysis.\n")
            f.write("Results cannot be generalized to other users.\n\n")
        
        f.write("RECOMMENDATION:\n")
        f.write("DO NOT implement source-based processing changes.\n")
        f.write("Current data insufficient to justify modifications.\n")
        f.write("Continue with baseline processor.\n")
    
    print("\nReport saved to output/dataset_investigation.txt")
    
    return user_summaries


if __name__ == "__main__":
    analyze_dataset()
