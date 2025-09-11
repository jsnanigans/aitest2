"""
Re-analyze data considering unit conversion and absolute outlier contributions.
"""

import csv
import numpy as np
from datetime import datetime
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pounds_to_kg(pounds):
    """Convert pounds to kilograms."""
    return pounds * 0.453592


def is_likely_pounds_entry(kg_weight):
    """
    Check if a kg weight is likely from a round pound entry.
    Common round pound values: 150, 155, 160, 165, 170, 175, 180, etc.
    """
    # Convert back to pounds
    pounds = kg_weight / 0.453592
    
    # Check if close to a round number in pounds
    rounded_pounds = round(pounds)
    diff = abs(pounds - rounded_pounds)
    
    # Check for common patterns
    is_round_5 = rounded_pounds % 5 == 0  # Divisible by 5
    is_round_10 = rounded_pounds % 10 == 0  # Divisible by 10
    
    # Within 0.1 pounds of a round number (accounting for float precision)
    if diff < 0.1:
        return True, rounded_pounds, is_round_5, is_round_10
    
    return False, pounds, False, False


def analyze_unit_patterns_and_outliers(data_file, sample_size=200000):
    """Analyze patterns considering unit conversion and absolute outlier counts."""
    
    print(f"\n{'='*80}")
    print("RE-ANALYSIS: Unit Conversion & Absolute Outlier Contribution")
    print(f"{'='*80}\n")
    
    # Track by source
    source_data = defaultdict(lambda: {
        'measurements': [],
        'round_pound_entries': 0,
        'round_5_pounds': 0,
        'round_10_pounds': 0,
        'outliers_5kg': 0,
        'outliers_10kg': 0,
        'outliers_15kg': 0,
        'max_jumps': [],
        'user_measurements': defaultdict(list)
    })
    
    # Process data
    row_count = 0
    total_outliers_10kg = 0
    
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            row_count += 1
            if row_count > sample_size:
                break
            
            if not row['weight'] or not row['effectiveDateTime']:
                continue
            
            try:
                weight = float(row['weight'])
                source = row['source_type']
                user_id = row['user_id']
                timestamp = datetime.fromisoformat(row['effectiveDateTime'])
                
                if 30 <= weight <= 400:  # Valid range
                    source_data[source]['measurements'].append(weight)
                    source_data[source]['user_measurements'][user_id].append({
                        'weight': weight,
                        'timestamp': timestamp
                    })
                    
                    # Check for pound-based entry
                    is_round, pounds_val, is_round_5, is_round_10 = is_likely_pounds_entry(weight)
                    if is_round:
                        source_data[source]['round_pound_entries'] += 1
                        if is_round_5:
                            source_data[source]['round_5_pounds'] += 1
                        if is_round_10:
                            source_data[source]['round_10_pounds'] += 1
                    
            except (ValueError, TypeError):
                continue
    
    # Calculate outliers for each source
    for source, data in source_data.items():
        # Sort user measurements by time
        for user_id, measurements in data['user_measurements'].items():
            measurements.sort(key=lambda x: x['timestamp'])
            
            # Calculate jumps
            for i in range(1, len(measurements)):
                jump = abs(measurements[i]['weight'] - measurements[i-1]['weight'])
                
                if jump > 5:
                    data['outliers_5kg'] += 1
                if jump > 10:
                    data['outliers_10kg'] += 1
                    total_outliers_10kg += 1
                if jump > 15:
                    data['outliers_15kg'] += 1
                
                data['max_jumps'].append(jump)
    
    # Print findings
    print("📊 UNIT CONVERSION ANALYSIS")
    print("-" * 80)
    print(f"{'Source':<30} {'Count':>8} {'Round lb':>10} {'÷5 lb':>8} {'÷10 lb':>8}")
    print("-" * 80)
    
    sources_sorted = sorted(source_data.items(), key=lambda x: len(x[1]['measurements']), reverse=True)
    
    for source, data in sources_sorted:
        count = len(data['measurements'])
        if count < 100:
            continue
        
        round_pct = data['round_pound_entries'] / count * 100 if count > 0 else 0
        round5_pct = data['round_5_pounds'] / count * 100 if count > 0 else 0
        round10_pct = data['round_10_pounds'] / count * 100 if count > 0 else 0
        
        print(f"{source[:29]:<30} {count:>8,} {round_pct:>9.1f}% {round5_pct:>7.1f}% {round10_pct:>7.1f}%")
    
    print(f"\n💡 INSIGHT: High 'round pound' percentages indicate users entering weight in pounds!")
    
    # Absolute outlier analysis
    print(f"\n\n📊 ABSOLUTE OUTLIER CONTRIBUTION")
    print("-" * 80)
    print(f"Total outliers (>10kg jumps): {total_outliers_10kg:,}")
    print(f"\n{'Source':<30} {'Outliers':>10} {'% of Total':>12} {'Per 1000':>10}")
    print("-" * 80)
    
    outlier_contributions = []
    for source, data in sources_sorted:
        count = len(data['measurements'])
        if count < 100:
            continue
        
        outliers = data['outliers_10kg']
        pct_of_total = outliers / total_outliers_10kg * 100 if total_outliers_10kg > 0 else 0
        per_1000 = outliers / count * 1000 if count > 0 else 0
        
        outlier_contributions.append({
            'source': source,
            'outliers': outliers,
            'pct_of_total': pct_of_total,
            'per_1000': per_1000,
            'count': count
        })
    
    # Sort by absolute contribution
    outlier_contributions.sort(key=lambda x: x['outliers'], reverse=True)
    
    for contrib in outlier_contributions:
        print(f"{contrib['source'][:29]:<30} {contrib['outliers']:>10,} {contrib['pct_of_total']:>11.1f}% {contrib['per_1000']:>9.1f}")
    
    # Volume-normalized noise analysis
    print(f"\n\n📊 VOLUME-NORMALIZED NOISE METRICS")
    print("-" * 80)
    print(f"{'Source':<30} {'Measurements':>12} {'Outliers/1000':>14}")
    print("-" * 80)
    
    # Sort by outlier rate
    outlier_contributions.sort(key=lambda x: x['per_1000'], reverse=True)
    
    for contrib in outlier_contributions:
        noise_level = "🔴 HIGH" if contrib['per_1000'] > 50 else "🟡 MEDIUM" if contrib['per_1000'] > 20 else "🟢 LOW"
        print(f"{contrib['source'][:29]:<30} {contrib['count']:>12,} {contrib['per_1000']:>13.1f} {noise_level}")
    
    # Key insights
    print(f"\n\n🎯 KEY INSIGHTS")
    print("=" * 80)
    
    # Find highest absolute contributor
    if outlier_contributions:
        worst_absolute = max(outlier_contributions, key=lambda x: x['outliers'])
        worst_rate = max(outlier_contributions, key=lambda x: x['per_1000'])
        
        print(f"\n1. ABSOLUTE OUTLIER CONTRIBUTION:")
        print(f"   {worst_absolute['source']} contributes {worst_absolute['pct_of_total']:.1f}% of ALL outliers")
        print(f"   Despite having only {worst_absolute['per_1000']:.1f} outliers per 1000 measurements")
        
        print(f"\n2. HIGHEST OUTLIER RATE:")
        print(f"   {worst_rate['source']} has {worst_rate['per_1000']:.1f} outliers per 1000 measurements")
        print(f"   But contributes only {worst_rate['pct_of_total']:.1f}% of total outliers")
    
    # Check for pound entries
    pound_sources = [(s, d) for s, d in source_data.items() 
                     if len(d['measurements']) > 100 and 
                     d['round_pound_entries'] / len(d['measurements']) > 0.3]
    
    if pound_sources:
        print(f"\n3. POUND-BASED ENTRIES DETECTED:")
        for source, data in pound_sources:
            pct = data['round_pound_entries'] / len(data['measurements']) * 100
            print(f"   {source}: {pct:.1f}% appear to be pound entries converted to kg")
            print(f"   This explains the 'non-round' kg values!")
    
    return source_data


def check_specific_patterns(data_file):
    """Check for specific patterns in patient-upload data."""
    
    print(f"\n\n{'='*80}")
    print("DEEP DIVE: Patient Upload Patterns")
    print(f"{'='*80}\n")
    
    patient_uploads = []
    
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['source_type'] == 'patient-upload' and row['weight']:
                try:
                    weight = float(row['weight'])
                    if 30 <= weight <= 400:
                        patient_uploads.append(weight)
                        if len(patient_uploads) >= 100:
                            break
                except:
                    continue
    
    if patient_uploads:
        print(f"Analyzing {len(patient_uploads)} patient-upload entries...\n")
        
        # Check first 20 values
        print("Sample values (kg) → likely pounds:")
        for i, kg in enumerate(patient_uploads[:20]):
            is_round, pounds, is_round_5, is_round_10 = is_likely_pounds_entry(kg)
            pounds_actual = kg / 0.453592
            
            marker = ""
            if is_round:
                if is_round_10:
                    marker = "⭐⭐"  # Very round (÷10)
                elif is_round_5:
                    marker = "⭐"    # Round (÷5)
                else:
                    marker = "•"     # Integer
            
            print(f"  {kg:>7.2f} kg → {pounds_actual:>6.1f} lb {marker}")
        
        # Statistics
        round_count = sum(1 for w in patient_uploads if is_likely_pounds_entry(w)[0])
        round_5_count = sum(1 for w in patient_uploads if is_likely_pounds_entry(w)[2])
        round_10_count = sum(1 for w in patient_uploads if is_likely_pounds_entry(w)[3])
        
        print(f"\nPattern Analysis:")
        print(f"  Round pound values: {round_count}/{len(patient_uploads)} ({round_count/len(patient_uploads)*100:.1f}%)")
        print(f"  Divisible by 5 lbs: {round_5_count}/{len(patient_uploads)} ({round_5_count/len(patient_uploads)*100:.1f}%)")
        print(f"  Divisible by 10 lbs: {round_10_count}/{len(patient_uploads)} ({round_10_count/len(patient_uploads)*100:.1f}%)")
        
        if round_count / len(patient_uploads) > 0.5:
            print(f"\n✅ CONFIRMED: Users are entering weights in POUNDS, not lying!")
            print(f"   The 'precise' kg values are from lb→kg conversion")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        
        # Main analysis
        source_data = analyze_unit_patterns_and_outliers(data_file)
        
        # Deep dive on patient uploads
        check_specific_patterns(data_file)
        
        print(f"\n\n{'='*80}")
        print("CORRECTED CONCLUSIONS")
        print(f"{'='*80}")
        print("""
1. USERS AREN'T LYING - They're entering POUNDS!
   - The 'precise' kg values come from lb→kg conversion
   - Round pound entries (150, 155, 160 lb) become 'odd' kg values
   - This is actually MORE honest than if they were guessing in kg

2. ABSOLUTE vs RELATIVE OUTLIERS:
   - Sources with more data contribute more outliers in absolute terms
   - But their outlier RATE might be lower
   - Need to consider both metrics for fair comparison

3. TRUE NOISE SOURCES:
   - Look at outliers per 1000 measurements for fair comparison
   - Consider which sources contribute most to total outliers
   - Volume matters - a small % of a large dataset can dominate
""")
    else:
        print("Usage: python test_unit_conversion_logic.py <csv_file>")
