#!/usr/bin/env python3
"""Analyze data for reprocessing needs: retroactive changes and multiple daily values."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json

def load_data(file_path):
    """Load CSV data with proper datetime parsing."""
    df = pd.read_csv(file_path)
    df['effectivDateTime'] = pd.to_datetime(df['effectivDateTime'], errors='coerce', format='mixed')
    df = df[df['effectivDateTime'].notna()]
    df['date'] = df['effectivDateTime'].dt.date
    return df

def find_source_ordering_issues(df):
    """Find cases where source order suggests retroactive additions."""
    issues = []
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('effectivDateTime')
        
        questionnaire_dates = user_data[
            user_data['source_type'].str.contains('questionnaire', na=False)
        ]['effectivDateTime'].values
        
        device_dates = user_data[
            user_data['source_type'].str.contains('device', na=False)
        ]['effectivDateTime'].values
        
        if len(questionnaire_dates) > 0 and len(device_dates) > 0:
            earliest_questionnaire = pd.to_datetime(questionnaire_dates[0])
            
            earlier_devices = [d for d in device_dates 
                             if pd.to_datetime(d) < earliest_questionnaire]
            
            if earlier_devices:
                issues.append({
                    'user_id': user_id,
                    'issue': 'device_before_questionnaire',
                    'earliest_questionnaire': str(earliest_questionnaire),
                    'device_dates_before': [str(pd.to_datetime(d)) for d in earlier_devices[:5]],
                    'count': len(earlier_devices)
                })
    
    return issues

def find_multiple_daily_values(df):
    """Find days with multiple weight readings that might damage Kalman filter."""
    issues = []
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('effectivDateTime')
        
        daily_groups = user_data.groupby('date')
        
        for date, group in daily_groups:
            if len(group) > 1:
                weights = group['weight'].values
                sources = group['source_type'].values
                times = group['effectivDateTime'].values
                
                weight_range = weights.max() - weights.min()
                weight_std = weights.std()
                
                if weight_range > 2.0 or (len(weights) > 2 and weight_std > 1.0):
                    median_weight = np.median(weights)
                    outliers = []
                    good_values = []
                    
                    for i, w in enumerate(weights):
                        deviation = abs(w - median_weight)
                        if deviation > 2.0:
                            outliers.append({
                                'weight': w,
                                'source': sources[i],
                                'time': str(times[i]),
                                'deviation': deviation
                            })
                        else:
                            good_values.append({
                                'weight': w,
                                'source': sources[i],
                                'time': str(times[i])
                            })
                    
                    if outliers:
                        issues.append({
                            'user_id': user_id,
                            'date': str(date),
                            'total_readings': len(group),
                            'weight_range': weight_range,
                            'weight_std': weight_std,
                            'median_weight': median_weight,
                            'outliers': outliers,
                            'good_values': good_values,
                            'all_weights': list(weights),
                            'all_sources': list(sources)
                        })
    
    return issues

def analyze_kalman_impact(df):
    """Analyze how multiple daily values could impact Kalman filter."""
    impacts = []
    
    problematic_days = find_multiple_daily_values(df)
    
    for issue in problematic_days[:10]:  
        user_data = df[df['user_id'] == issue['user_id']].sort_values('effectivDateTime')
        
        day_data = user_data[user_data['date'] == pd.to_datetime(issue['date']).date()]
        
        if len(day_data) > 1:
            first_weight = day_data.iloc[0]['weight']
            best_weight = issue['median_weight']
            
            prev_days = user_data[user_data['date'] < pd.to_datetime(issue['date']).date()]
            if len(prev_days) > 0:
                recent_weights = prev_days.tail(5)['weight'].values
                expected_weight = recent_weights.mean() if len(recent_weights) > 0 else best_weight
                
                first_error = abs(first_weight - expected_weight)
                best_error = abs(best_weight - expected_weight)
                
                if first_error > best_error + 1.0:
                    impacts.append({
                        'user_id': issue['user_id'],
                        'date': issue['date'],
                        'first_weight': first_weight,
                        'best_weight': best_weight,
                        'expected_weight': expected_weight,
                        'first_error': first_error,
                        'best_error': best_error,
                        'improvement_potential': first_error - best_error,
                        'readings': issue['total_readings']
                    })
    
    return impacts

def find_retroactive_patterns(df):
    """Find patterns suggesting retroactive data entry."""
    patterns = []
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('effectivDateTime')
        
        if len(user_data) < 10:
            continue
            
        date_diffs = user_data['effectivDateTime'].diff()
        
        gaps = []
        for i, diff in enumerate(date_diffs):
            if pd.notna(diff) and diff.days > 30:
                gaps.append({
                    'index': i,
                    'gap_days': diff.days,
                    'before': str(user_data.iloc[i-1]['effectivDateTime']),
                    'after': str(user_data.iloc[i]['effectivDateTime'])
                })
        
        for i in range(1, len(user_data) - 1):
            current = user_data.iloc[i]
            prev = user_data.iloc[i-1]
            next_val = user_data.iloc[i+1]
            
            if (current['effectivDateTime'] - prev['effectivDateTime']).days > 7:
                if (next_val['effectivDateTime'] - current['effectivDateTime']).days < 2:
                    if current['source_type'] != prev['source_type']:
                        patterns.append({
                            'user_id': user_id,
                            'pattern': 'late_insertion',
                            'date': str(current['effectivDateTime']),
                            'source': current['source_type'],
                            'gap_before': (current['effectivDateTime'] - prev['effectivDateTime']).days,
                            'gap_after': (next_val['effectivDateTime'] - current['effectivDateTime']).days
                        })
    
    return patterns

def main():
    print("Analyzing Reprocessing Needs for Weight Stream Processor")
    print("=" * 60)
    
    df = load_data('data/example.csv')
    
    print(f"\nDataset Overview:")
    print(f"- Total records: {len(df):,}")
    print(f"- Unique users: {df['user_id'].nunique():,}")
    print(f"- Date range: {df['effectivDateTime'].min()} to {df['effectivDateTime'].max()}")
    print(f"- Source types: {df['source_type'].value_counts().to_dict()}")
    
    print("\n" + "=" * 60)
    print("ISSUE 1: Source Ordering (Retroactive Additions)")
    print("=" * 60)
    
    source_issues = find_source_ordering_issues(df)
    print(f"\nFound {len(source_issues)} users with device data before questionnaire")
    
    if source_issues:
        for issue in source_issues[:3]:
            print(f"\nUser: {issue['user_id']}")
            print(f"  Questionnaire: {issue['earliest_questionnaire']}")
            print(f"  Earlier devices: {issue['count']} readings")
            for d in issue['device_dates_before'][:3]:
                print(f"    - {d}")
    
    print("\n" + "=" * 60)
    print("ISSUE 2: Multiple Daily Values")
    print("=" * 60)
    
    daily_issues = find_multiple_daily_values(df)
    print(f"\nFound {len(daily_issues)} instances of problematic multiple daily readings")
    
    severe_issues = [i for i in daily_issues if i['weight_range'] > 5.0]
    print(f"- Severe cases (>5kg range): {len(severe_issues)}")
    
    if severe_issues:
        for issue in severe_issues[:3]:
            print(f"\nUser: {issue['user_id']} on {issue['date']}")
            print(f"  {issue['total_readings']} readings, range: {issue['weight_range']:.1f}kg")
            print(f"  Weights: {[f'{w:.1f}' for w in issue['all_weights']]}")
            print(f"  Sources: {issue['all_sources']}")
            if issue['outliers']:
                outlier_strs = [f"{o['weight']:.1f}kg (dev: {o['deviation']:.1f})" for o in issue['outliers']]
                print(f"  Outliers: {outlier_strs}")
    
    print("\n" + "=" * 60)
    print("ISSUE 3: Kalman Filter Impact Analysis")
    print("=" * 60)
    
    impacts = analyze_kalman_impact(df)
    print(f"\nFound {len(impacts)} cases where first value damages Kalman accuracy")
    
    if impacts:
        for impact in impacts[:3]:
            print(f"\nUser: {impact['user_id']} on {impact['date']}")
            print(f"  First weight: {impact['first_weight']:.1f}kg (error: {impact['first_error']:.1f})")
            print(f"  Best weight: {impact['best_weight']:.1f}kg (error: {impact['best_error']:.1f})")
            print(f"  Improvement potential: {impact['improvement_potential']:.1f}kg")
    
    print("\n" + "=" * 60)
    print("ISSUE 4: Retroactive Data Patterns")
    print("=" * 60)
    
    patterns = find_retroactive_patterns(df)
    print(f"\nFound {len(patterns)} potential retroactive insertions")
    
    if patterns:
        for pattern in patterns[:5]:
            print(f"\nUser: {pattern['user_id']}")
            print(f"  Date: {pattern['date']}")
            print(f"  Gap before: {pattern['gap_before']} days")
            print(f"  Gap after: {pattern['gap_after']} days")
            print(f"  Source: {pattern['source']}")
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    users_with_multiple = df.groupby('user_id').apply(
        lambda x: (x.groupby('date').size() > 1).any()
    ).sum()
    
    print(f"\n- Users with at least one multi-value day: {users_with_multiple}")
    print(f"- Percentage of users affected: {users_with_multiple / df['user_id'].nunique() * 100:.1f}%")
    
    total_days_with_multiple = df.groupby(['user_id', 'date']).size().reset_index(name='count')
    multi_days = total_days_with_multiple[total_days_with_multiple['count'] > 1]
    print(f"- Total days with multiple readings: {len(multi_days)}")
    print(f"- Average readings on multi-value days: {multi_days['count'].mean():.1f}")
    print(f"- Max readings in a single day: {multi_days['count'].max()}")

if __name__ == "__main__":
    main()