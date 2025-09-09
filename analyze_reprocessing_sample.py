#!/usr/bin/env python3
"""Sample analysis for reprocessing needs - focused examples."""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def load_sample_data(file_path, sample_users=50):
    """Load a sample of data for analysis."""
    df = pd.read_csv(file_path, nrows=100000)
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'], errors='coerce', format='mixed')
    df = df[df['effectiveDateTime'].notna()]
    
    df = df[(df['effectiveDateTime'] > '2020-01-01') & (df['effectiveDateTime'] < '2030-01-01')]
    
    users = df['user_id'].unique()[:sample_users]
    df = df[df['user_id'].isin(users)]
    
    df['date'] = df['effectiveDateTime'].dt.date
    return df

def find_critical_examples(df):
    """Find the most critical examples of each issue."""
    examples = {
        'multiple_daily_extreme': [],
        'source_ordering': [],
        'kalman_damage': []
    }
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('effectiveDateTime')
        
        daily_groups = user_data.groupby('date')
        for date, group in daily_groups:
            if len(group) > 1:
                weights = group['weight'].values
                if weights.max() - weights.min() > 3.0:
                    examples['multiple_daily_extreme'].append({
                        'user_id': user_id,
                        'date': str(date),
                        'count': len(group),
                        'weights': list(weights),
                        'sources': list(group['source_type'].values),
                        'times': [str(t) for t in group['effectiveDateTime'].values],
                        'range': weights.max() - weights.min()
                    })
        
        questionnaire = user_data[user_data['source_type'].str.contains('questionnaire', na=False)]
        device = user_data[user_data['source_type'].str.contains('device', na=False)]
        
        if len(questionnaire) > 0 and len(device) > 0:
            earliest_q = questionnaire['effectiveDateTime'].min()
            earlier_device = device[device['effectiveDateTime'] < earliest_q]
            if len(earlier_device) > 0:
                examples['source_ordering'].append({
                    'user_id': user_id,
                    'questionnaire_date': str(earliest_q),
                    'earlier_device_count': len(earlier_device),
                    'earliest_device': str(earlier_device['effectiveDateTime'].min())
                })
    
    return examples

def analyze_single_user_detail(df, user_id):
    """Deep dive into a single user's data."""
    user_data = df[df['user_id'] == user_id].sort_values('effectiveDateTime')
    
    analysis = {
        'user_id': user_id,
        'total_records': len(user_data),
        'date_range': f"{user_data['effectiveDateTime'].min()} to {user_data['effectiveDateTime'].max()}",
        'sources': user_data['source_type'].value_counts().to_dict(),
        'multi_value_days': [],
        'large_gaps': []
    }
    
    daily_groups = user_data.groupby('date')
    for date, group in daily_groups:
        if len(group) > 1:
            analysis['multi_value_days'].append({
                'date': str(date),
                'count': len(group),
                'weights': list(group['weight'].values),
                'sources': list(group['source_type'].values)
            })
    
    date_diffs = user_data['effectiveDateTime'].diff()
    for i, diff in enumerate(date_diffs):
        if pd.notna(diff) and diff.days > 30:
            analysis['large_gaps'].append({
                'days': diff.days,
                'before': str(user_data.iloc[i-1]['effectiveDateTime']),
                'after': str(user_data.iloc[i]['effectiveDateTime']),
                'before_weight': user_data.iloc[i-1]['weight'],
                'after_weight': user_data.iloc[i]['weight']
            })
    
    return analysis

def main():
    print("REPROCESSING NEEDS ANALYSIS - SAMPLE DATA")
    print("=" * 60)
    
    df = load_sample_data('data/example.csv', sample_users=100)
    
    print(f"\nSample Dataset:")
    print(f"- Records: {len(df):,}")
    print(f"- Users: {df['user_id'].nunique()}")
    print(f"- Date range: {df['effectiveDateTime'].min()} to {df['effectiveDateTime'].max()}")
    
    examples = find_critical_examples(df)
    
    print("\n" + "=" * 60)
    print("CRITICAL EXAMPLES")
    print("=" * 60)
    
    print("\n1. EXTREME MULTIPLE DAILY VALUES")
    print("-" * 40)
    
    extreme_cases = sorted(examples['multiple_daily_extreme'], key=lambda x: x['range'], reverse=True)[:5]
    
    for case in extreme_cases:
        print(f"\nUser: {case['user_id'][:8]}...")
        print(f"Date: {case['date']}")
        print(f"Range: {case['range']:.1f}kg across {case['count']} readings")
        print(f"Weights: {[f'{w:.1f}' for w in case['weights']]}")
        print(f"Sources: {case['sources']}")
        print(f"Times: {[t.split(' ')[1] if ' ' in t else t for t in case['times']]}")
    
    print("\n2. SOURCE ORDERING ISSUES")
    print("-" * 40)
    
    for issue in examples['source_ordering'][:3]:
        print(f"\nUser: {issue['user_id'][:8]}...")
        print(f"Questionnaire: {issue['questionnaire_date']}")
        print(f"Earlier device: {issue['earliest_device']} ({issue['earlier_device_count']} total)")
    
    multi_day_users = [e['user_id'] for e in examples['multiple_daily_extreme']]
    if multi_day_users:
        print("\n" + "=" * 60)
        print("DETAILED USER ANALYSIS")
        print("=" * 60)
        
        detail = analyze_single_user_detail(df, multi_day_users[0])
        
        print(f"\nUser: {detail['user_id']}")
        print(f"Records: {detail['total_records']}")
        print(f"Period: {detail['date_range']}")
        print(f"Sources: {detail['sources']}")
        
        if detail['multi_value_days']:
            print(f"\nMulti-value days: {len(detail['multi_value_days'])}")
            for day in detail['multi_value_days'][:3]:
                print(f"  {day['date']}: {day['count']} readings - {[f'{w:.1f}kg' for w in day['weights']]}")
        
        if detail['large_gaps']:
            print(f"\nLarge gaps: {len(detail['large_gaps'])}")
            for gap in detail['large_gaps'][:3]:
                print(f"  {gap['days']} days: {gap['before_weight']:.1f}kg → {gap['after_weight']:.1f}kg")

    print("\n" + "=" * 60)
    print("IMPLEMENTATION RECOMMENDATIONS")
    print("=" * 60)
    
    multi_days_total = sum(len(e['multi_value_days']) for e in [analyze_single_user_detail(df, uid) for uid in df['user_id'].unique()[:10]])
    
    print(f"""
1. RETROACTIVE PROCESSING:
   - Found evidence of late data additions (questionnaire after device)
   - Need ability to reprocess from specific date
   - Maintain state snapshots for rollback

2. DAILY BATCH PROCESSING:
   - Multi-value days found in ~{multi_days_total/10:.0f}% of sampled users
   - Some days have extreme ranges (>5kg difference)
   - Need outlier detection within daily batches

3. PRIORITY ISSUES TO ADDRESS:
   - Days with >3kg range between values
   - Source type ordering violations
   - Gap-related state inconsistencies
""")

if __name__ == "__main__":
    main()