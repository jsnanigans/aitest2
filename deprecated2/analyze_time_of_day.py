#!/usr/bin/env python3
"""
Phase 1: Time-of-Day Pattern Analysis
Analyzes weight measurement patterns by time of day
"""

import csv
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

def parse_datetime(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except:
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')

def get_time_bin(hour: int) -> str:
    if 5 <= hour < 11:
        return 'morning'
    elif 11 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 23:
        return 'evening'
    else:
        return 'night'

def analyze_time_patterns(csv_file: str) -> Dict:
    print(f"Analyzing time-of-day patterns in {csv_file}...")
    
    hour_weights = defaultdict(list)
    user_hours = defaultdict(set)
    user_weights_by_hour = defaultdict(lambda: defaultdict(list))
    user_weights_by_bin = defaultdict(lambda: defaultdict(list))
    total_readings = 0
    users_with_times = set()
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_readings += 1
            user_id = row['user_id']
            
            if row['weight'] == 'NULL' or not row['weight']:
                continue
                
            try:
                weight = float(row['weight'])
            except ValueError:
                continue
            
            try:
                dt = parse_datetime(row['effectivDateTime'])
                hour = dt.hour
                
                if hour != 0 or dt.minute != 0:
                    users_with_times.add(user_id)
                    hour_weights[hour].append(weight)
                    user_hours[user_id].add(hour)
                    user_weights_by_hour[user_id][hour].append(weight)
                    
                    time_bin = get_time_bin(hour)
                    user_weights_by_bin[user_id][time_bin].append(weight)
                    
            except Exception as e:
                continue
            
            if total_readings % 10000 == 0:
                print(f"  Processed {total_readings:,} readings...")
    
    print(f"\nAnalysis complete! Processed {total_readings:,} total readings")
    print(f"Found {len(users_with_times)} users with time information")
    
    user_time_variance = {}
    for user_id in user_weights_by_bin:
        bins = user_weights_by_bin[user_id]
        if len(bins) > 1:
            all_weights = []
            bin_averages = []
            for bin_name in bins:
                weights = bins[bin_name]
                if weights:
                    all_weights.extend(weights)
                    bin_averages.append(np.mean(weights))
            
            if len(bin_averages) > 1:
                user_time_variance[user_id] = {
                    'max_diff': max(bin_averages) - min(bin_averages),
                    'std_dev': np.std(bin_averages),
                    'bins_used': list(bins.keys()),
                    'total_readings': len(all_weights)
                }
    
    users_consistent_time = sum(1 for u in user_hours if len(user_hours[u]) <= 3)
    users_variable_time = len(user_hours) - users_consistent_time
    
    hourly_stats = {}
    for hour in range(24):
        if hour_weights[hour]:
            hourly_stats[hour] = {
                'count': len(hour_weights[hour]),
                'mean': np.mean(hour_weights[hour]),
                'std': np.std(hour_weights[hour]),
                'median': np.median(hour_weights[hour])
            }
    
    bin_stats = defaultdict(lambda: {'weights': [], 'users': set()})
    for user_id in user_weights_by_bin:
        for bin_name, weights in user_weights_by_bin[user_id].items():
            bin_stats[bin_name]['weights'].extend(weights)
            bin_stats[bin_name]['users'].add(user_id)
    
    bin_summary = {}
    for bin_name in ['morning', 'afternoon', 'evening', 'night']:
        if bin_stats[bin_name]['weights']:
            bin_summary[bin_name] = {
                'count': len(bin_stats[bin_name]['weights']),
                'users': len(bin_stats[bin_name]['users']),
                'mean': np.mean(bin_stats[bin_name]['weights']),
                'std': np.std(bin_stats[bin_name]['weights']),
                'median': np.median(bin_stats[bin_name]['weights'])
            }
    
    significant_variance_users = sum(1 for u, v in user_time_variance.items() 
                                   if v['max_diff'] > 1.0)
    
    return {
        'total_readings': total_readings,
        'users_with_time_data': len(users_with_times),
        'users_consistent_time': users_consistent_time,
        'users_variable_time': users_variable_time,
        'percent_variable_time': round(users_variable_time / len(user_hours) * 100, 1) if user_hours else 0,
        'hourly_distribution': hourly_stats,
        'time_bin_summary': bin_summary,
        'users_with_significant_variance': significant_variance_users,
        'top_variance_users': sorted(user_time_variance.items(), 
                                    key=lambda x: x[1]['max_diff'], 
                                    reverse=True)[:10]
    }

def create_visualizations(analysis: Dict):
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Time-of-Day Weight Pattern Analysis', fontsize=16)
    
    hourly_data = analysis['hourly_distribution']
    hours = sorted(hourly_data.keys())
    counts = [hourly_data[h]['count'] for h in hours]
    
    ax = axes[0, 0]
    ax.bar(hours, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Readings')
    ax.set_title('Reading Distribution by Hour')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    
    if len(hours) > 0:
        means = [hourly_data[h]['mean'] for h in hours]
        ax2 = axes[0, 1]
        ax2.plot(hours, means, 'o-', color='darkgreen', linewidth=2, markersize=6)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average Weight (kg)')
        ax2.set_title('Average Weight by Hour of Day')
        ax2.set_xticks(range(0, 24, 2))
        ax2.grid(True, alpha=0.3)
        
        if len(means) > 1:
            ax2.axhline(y=np.mean(means), color='red', linestyle='--', 
                       label=f'Overall Mean: {np.mean(means):.1f}kg')
            ax2.legend()
    
    bin_data = analysis['time_bin_summary']
    if bin_data:
        bins = ['morning', 'afternoon', 'evening', 'night']
        bin_counts = [bin_data.get(b, {}).get('count', 0) for b in bins]
        bin_means = [bin_data.get(b, {}).get('mean', 0) for b in bins]
        
        ax3 = axes[1, 0]
        colors = ['gold', 'orange', 'darkblue', 'purple']
        bars = ax3.bar(range(len(bins)), bin_counts, color=colors, alpha=0.7)
        ax3.set_xlabel('Time Period')
        ax3.set_ylabel('Number of Readings')
        ax3.set_title('Reading Distribution by Time Period')
        ax3.set_xticks(range(len(bins)))
        ax3.set_xticklabels(bins)
        
        for bar, count in zip(bars, bin_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom')
    
    ax4 = axes[1, 1]
    variance_data = analysis['top_variance_users'][:5]
    if variance_data:
        users = [f"User {i+1}" for i in range(len(variance_data))]
        variances = [v[1]['max_diff'] for v in variance_data]
        
        bars = ax4.barh(users, variances, color='coral')
        ax4.set_xlabel('Max Weight Difference Between Time Periods (kg)')
        ax4.set_title('Top 5 Users with Highest Time-of-Day Variance')
        ax4.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, variances):
            ax4.text(val, bar.get_y() + bar.get_height()/2, 
                    f'{val:.2f}kg', ha='left', va='center')
    else:
        ax4.text(0.5, 0.5, 'No variance data available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = output_dir / f'time_of_day_analysis_{timestamp}.png'
    plt.savefig(plot_file, dpi=100, bbox_inches='tight')
    print(f"Saved visualization to {plot_file}")
    
    plt.show()

def main():
    import sys
    import tomllib
    
    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)
    
    csv_file = config.get('source_file', './2025-03-27_optimized.csv')
    
    if not Path(csv_file).exists():
        print(f"Error: CSV file {csv_file} not found!")
        sys.exit(1)
    
    analysis = analyze_time_patterns(csv_file)
    
    print("\n" + "="*60)
    print("TIME-OF-DAY ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total readings analyzed: {analysis['total_readings']:,}")
    print(f"  Users with time data: {analysis['users_with_time_data']:,}")
    print(f"  Users with consistent timing: {analysis['users_consistent_time']:,}")
    print(f"  Users with variable timing: {analysis['users_variable_time']:,}")
    print(f"  Percentage with variable timing: {analysis['percent_variable_time']:.1f}%")
    
    print(f"\n⏰ TIME PERIOD DISTRIBUTION:")
    for period in ['morning', 'afternoon', 'evening', 'night']:
        if period in analysis['time_bin_summary']:
            stats = analysis['time_bin_summary'][period]
            print(f"  {period.capitalize():10} - {stats['count']:7,} readings, "
                  f"{stats['users']:5,} users, avg: {stats['mean']:.1f}kg")
    
    if analysis['time_bin_summary']:
        weights = [s['mean'] for s in analysis['time_bin_summary'].values()]
        max_diff = max(weights) - min(weights)
        print(f"\n  Max average difference between periods: {max_diff:.2f}kg")
    
    print(f"\n📈 VARIANCE ANALYSIS:")
    print(f"  Users with >1kg time-based variance: {analysis['users_with_significant_variance']:,}")
    
    if analysis['top_variance_users']:
        print(f"\n  Top 5 highest variance users:")
        for i, (user_id, variance) in enumerate(analysis['top_variance_users'][:5], 1):
            print(f"    {i}. User {user_id}: {variance['max_diff']:.2f}kg difference, "
                  f"{variance['total_readings']} readings")
    
    print("\n💡 KEY FINDINGS:")
    if analysis['percent_variable_time'] > 30:
        print("  ✅ Significant portion of users have variable weighing times")
        print("     → Time-of-day adjustment could be beneficial")
    else:
        print("  ⚠️  Most users have consistent weighing times")
        print("     → Time-of-day adjustment may have limited impact")
    
    if analysis['time_bin_summary']:
        weights = [s['mean'] for s in analysis['time_bin_summary'].values()]
        max_diff = max(weights) - min(weights)
        if max_diff > 0.5:
            print(f"  ✅ Significant weight difference between time periods ({max_diff:.2f}kg)")
            print("     → Time-based patterns are meaningful")
        else:
            print(f"  ⚠️  Small weight difference between time periods ({max_diff:.2f}kg)")
            print("     → Time-based patterns may not be significant")
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = output_dir / f'time_of_day_analysis_{timestamp}.json'
    
    with open(json_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\n📁 Full analysis saved to: {json_file}")
    
    create_visualizations(analysis)
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    if (analysis['percent_variable_time'] > 30 and 
        analysis['users_with_significant_variance'] > analysis['users_with_time_data'] * 0.1):
        print("✅ PROCEED TO PHASE 2: Implement simple time-bin model")
        print("   Evidence suggests time-of-day adjustment will improve accuracy")
    else:
        print("⚠️  CONSIDER STOPPING: Limited benefit expected")
        print("   Time-of-day patterns are not significant enough to justify complexity")
    print("="*60)

if __name__ == '__main__':
    main()