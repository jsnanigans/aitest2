"""
Complete analysis of ALL data with robust error handling.
Handles date issues, unit conversions, and data quality problems.
"""

import csv
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
import sys
import os
import traceback

# Progress tracking
def print_progress(current, total, prefix="Progress"):
    """Print progress bar."""
    if total == 0:
        return
    percent = current / total * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    sys.stdout.write(f'\r{prefix}: {bar} {percent:.1f}% ({current:,}/{total:,})')
    sys.stdout.flush()
    if current == total:
        print()  # New line when complete


class RobustDataAnalyzer:
    """Analyze complete dataset with comprehensive error handling."""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.errors = defaultdict(list)
        self.stats = {
            'total_rows': 0,
            'valid_rows': 0,
            'date_errors': 0,
            'weight_errors': 0,
            'unit_errors': 0,
            'out_of_range': 0,
            'units_found': defaultdict(int),
            'date_anomalies': defaultdict(int)
        }
        
        # Source profiles with comprehensive tracking
        self.source_profiles = defaultdict(lambda: {
            'measurements': [],
            'timestamps': [],
            'units': defaultdict(int),
            'users': defaultdict(list),
            'errors': defaultdict(int),
            'pound_entries': {'high': 0, 'medium': 0, 'low': 0, 'none': 0},
            'outliers': defaultdict(int),
            'date_issues': defaultdict(int),
            'quality_scores': []
        })
        
    def parse_date_robust(self, date_str: str) -> Optional[datetime]:
        """Parse date with multiple fallback strategies."""
        if not date_str:
            return None
            
        # Try standard ISO format first
        try:
            return datetime.fromisoformat(date_str)
        except:
            pass
        
        # Try without timezone
        try:
            if 'T' in date_str:
                date_part = date_str.split('T')[0] + 'T' + date_str.split('T')[1][:8]
                return datetime.fromisoformat(date_part)
        except:
            pass
        
        # Try common formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        self.stats['date_errors'] += 1
        return None
    
    def convert_weight_to_kg(self, weight_str: str, unit: str) -> Optional[float]:
        """Convert weight to kg with unit handling."""
        try:
            weight = float(weight_str)
        except (ValueError, TypeError):
            self.stats['weight_errors'] += 1
            return None
        
        # Handle different units
        unit_lower = unit.lower() if unit else 'kg'
        self.stats['units_found'][unit_lower] += 1
        
        if 'lb' in unit_lower or 'pound' in unit_lower:
            weight = weight * 0.453592  # Convert pounds to kg
        elif 'stone' in unit_lower:
            weight = weight * 6.35029  # Convert stone to kg
        elif 'g' in unit_lower and 'kg' not in unit_lower:
            weight = weight / 1000  # Convert grams to kg
        # Assume kg if not specified or already kg
        
        return weight
    
    def check_date_anomalies(self, date: datetime, source: str) -> bool:
        """Check for date anomalies."""
        current_year = 2025
        
        # Check for impossible dates
        if date.year < 1900:
            self.source_profiles[source]['date_issues']['before_1900'] += 1
            self.stats['date_anomalies']['before_1900'] += 1
            return False
        
        if date.year > current_year:
            self.source_profiles[source]['date_issues']['future'] += 1
            self.stats['date_anomalies']['future'] += 1
            return False
        
        # Check for suspiciously old dates
        if date.year < 2000:
            self.source_profiles[source]['date_issues']['before_2000'] += 1
            self.stats['date_anomalies']['before_2000'] += 1
        
        return True
    
    def analyze_pound_confidence(self, weight_kg: float) -> str:
        """Determine confidence that weight was entered in pounds."""
        pounds = weight_kg / 0.453592
        rounded_pounds = round(pounds)
        diff = abs(pounds - rounded_pounds)
        
        if diff < 0.01:
            return 'high'
        elif diff < 0.05:
            return 'medium'
        elif diff < 0.1:
            return 'low'
        else:
            return 'none'
    
    def process_dataset(self):
        """Process the complete dataset with error handling."""
        print(f"\n{'='*80}")
        print("COMPLETE DATASET ANALYSIS WITH ERROR HANDLING")
        print(f"{'='*80}\n")
        
        # First pass: count total rows
        print("Counting total rows...")
        with open(self.data_file, 'r', encoding='utf-8', errors='ignore') as f:
            total_rows = sum(1 for _ in f) - 1  # Subtract header
        
        print(f"Total rows to process: {total_rows:,}\n")
        
        # Second pass: process data
        print("Processing data...")
        
        with open(self.data_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 1):
                self.stats['total_rows'] += 1
                
                # Progress update
                if row_num % 10000 == 0:
                    print_progress(row_num, total_rows, "Processing")
                
                try:
                    # Parse essential fields
                    user_id = row.get('user_id', '')
                    source = row.get('source_type', 'unknown')
                    weight_str = row.get('weight', '')
                    unit = row.get('unit', 'kg')
                    date_str = row.get('effectiveDateTime', '')
                    
                    if not user_id or not weight_str or not date_str:
                        self.errors['missing_fields'].append(row_num)
                        continue
                    
                    # Parse date
                    timestamp = self.parse_date_robust(date_str)
                    if not timestamp:
                        self.errors['date_parse_failed'].append(f"Row {row_num}: {date_str}")
                        continue
                    
                    # Check date anomalies
                    if not self.check_date_anomalies(timestamp, source):
                        # Still process but mark as anomaly
                        pass
                    
                    # Convert weight to kg
                    weight_kg = self.convert_weight_to_kg(weight_str, unit)
                    if weight_kg is None:
                        continue
                    
                    # Check physiological range
                    if weight_kg < 20 or weight_kg > 500:
                        self.stats['out_of_range'] += 1
                        self.source_profiles[source]['errors']['out_of_range'] += 1
                        if weight_kg < 20:
                            # Might be in stones or wrong unit
                            self.source_profiles[source]['errors']['too_low'] += 1
                        continue
                    
                    # Valid row - process it
                    self.stats['valid_rows'] += 1
                    
                    # Store in source profile
                    profile = self.source_profiles[source]
                    profile['measurements'].append(weight_kg)
                    profile['timestamps'].append(timestamp)
                    profile['units'][unit] += 1
                    profile['users'][user_id].append({
                        'weight': weight_kg,
                        'timestamp': timestamp,
                        'unit': unit
                    })
                    
                    # Check pound confidence
                    confidence = self.analyze_pound_confidence(weight_kg)
                    profile['pound_entries'][confidence] += 1
                    
                except Exception as e:
                    self.errors['processing_errors'].append(f"Row {row_num}: {str(e)}")
                    if row_num < 10:  # Only print first few errors
                        print(f"\nError at row {row_num}: {e}")
        
        print_progress(total_rows, total_rows, "Processing")
        print("\nData loading complete. Analyzing patterns...\n")
        
        # Calculate derived metrics
        self._calculate_metrics()
        
    def _calculate_metrics(self):
        """Calculate derived metrics for each source."""
        print("Calculating metrics for each source...")
        
        for source, profile in self.source_profiles.items():
            if len(profile['measurements']) < 10:
                continue
            
            # Sort user data by timestamp
            for user_id, user_data in profile['users'].items():
                if len(user_data) < 2:
                    continue
                    
                user_data.sort(key=lambda x: x['timestamp'])
                
                # Calculate outliers with time normalization
                for i in range(1, len(user_data)):
                    time_gap = (user_data[i]['timestamp'] - user_data[i-1]['timestamp']).days
                    weight_change = abs(user_data[i]['weight'] - user_data[i-1]['weight'])
                    
                    # Categorize outliers
                    if time_gap == 0 and weight_change > 5:
                        profile['outliers']['same_day_5kg'] += 1
                    
                    if time_gap <= 7:
                        if weight_change > 5:
                            profile['outliers']['week_5kg'] += 1
                        if weight_change > 10:
                            profile['outliers']['week_10kg'] += 1
                    elif time_gap <= 30:
                        if weight_change > 10:
                            profile['outliers']['month_10kg'] += 1
                        if weight_change > 15:
                            profile['outliers']['month_15kg'] += 1
                    else:
                        # Long gap - different expectations
                        expected_max = 2.0 * (time_gap / 7.0)  # 2kg per week max
                        if weight_change > expected_max:
                            profile['outliers']['excessive_for_gap'] += 1
            
            # Calculate quality score
            total = len(profile['measurements'])
            if total > 0:
                error_rate = sum(profile['errors'].values()) / total
                outlier_rate = sum(profile['outliers'].values()) / total
                quality_score = max(0, 10 - (error_rate * 20) - (outlier_rate * 10))
                profile['quality_scores'].append(quality_score)
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print(f"\n{'='*80}")
        print("ANALYSIS RESULTS - COMPLETE DATASET")
        print(f"{'='*80}\n")
        
        # Overall statistics
        print("📊 DATASET OVERVIEW")
        print("-" * 60)
        print(f"Total rows processed: {self.stats['total_rows']:,}")
        print(f"Valid rows: {self.stats['valid_rows']:,} ({self.stats['valid_rows']/self.stats['total_rows']*100:.1f}%)")
        print(f"Date parsing errors: {self.stats['date_errors']:,}")
        print(f"Weight parsing errors: {self.stats['weight_errors']:,}")
        print(f"Out of range weights: {self.stats['out_of_range']:,}")
        
        # Date anomalies
        if self.stats['date_anomalies']:
            print(f"\n📅 DATE ANOMALIES FOUND")
            print("-" * 60)
            for anomaly, count in sorted(self.stats['date_anomalies'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {anomaly}: {count:,} occurrences")
        
        # Units found
        print(f"\n📏 UNITS DETECTED")
        print("-" * 60)
        for unit, count in sorted(self.stats['units_found'].items(), key=lambda x: x[1], reverse=True):
            pct = count / self.stats['total_rows'] * 100 if self.stats['total_rows'] > 0 else 0
            print(f"  {unit}: {count:,} ({pct:.1f}%)")
        
        # Source analysis
        print(f"\n\n📊 SOURCE TYPE ANALYSIS")
        print("=" * 80)
        
        # Sort sources by measurement count
        sorted_sources = sorted(self.source_profiles.items(), 
                              key=lambda x: len(x[1]['measurements']), 
                              reverse=True)
        
        # Summary table
        print(f"\n{'Source':<35} {'Count':>10} {'Users':>8} {'Quality':>8} {'Pounds%':>10}")
        print("-" * 80)
        
        for source, profile in sorted_sources:
            if len(profile['measurements']) < 100:
                continue
                
            count = len(profile['measurements'])
            users = len(profile['users'])
            
            # Calculate pound percentage
            pound_total = sum(profile['pound_entries'].values())
            if pound_total > 0:
                pound_pct = (profile['pound_entries']['high'] + profile['pound_entries']['medium']) / pound_total * 100
            else:
                pound_pct = 0
            
            # Quality score
            quality = profile['quality_scores'][0] if profile['quality_scores'] else 0
            
            print(f"{source[:34]:<35} {count:>10,} {users:>8,} {quality:>7.1f} {pound_pct:>9.1f}%")
        
        # Detailed profiles for top sources
        print(f"\n\n📋 DETAILED SOURCE PROFILES")
        print("=" * 80)
        
        for rank, (source, profile) in enumerate(sorted_sources[:10], 1):
            if len(profile['measurements']) < 100:
                continue
                
            self._print_detailed_profile(rank, source, profile)
        
        # Save JSON summary
        self._save_json_summary()
    
    def _print_detailed_profile(self, rank: int, source: str, profile: Dict):
        """Print detailed profile for a source."""
        print(f"\n{'-'*60}")
        print(f"#{rank}: {source}")
        print(f"{'-'*60}")
        
        measurements = profile['measurements']
        total = len(measurements)
        
        if total == 0:
            return
        
        # Basic stats
        print(f"\n📊 Statistics:")
        print(f"  Measurements: {total:,}")
        print(f"  Users: {len(profile['users']):,}")
        print(f"  Mean weight: {np.mean(measurements):.1f} kg")
        print(f"  Std dev: {np.std(measurements):.1f} kg")
        
        # Units used
        if len(profile['units']) > 1:
            print(f"\n📏 Units:")
            for unit, count in sorted(profile['units'].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    {unit}: {count:,} ({count/total*100:.1f}%)")
        
        # Pound confidence
        pound_total = sum(profile['pound_entries'].values())
        if pound_total > 0:
            print(f"\n🎯 Pound Entry Confidence:")
            for level in ['high', 'medium', 'low', 'none']:
                pct = profile['pound_entries'][level] / pound_total * 100
                print(f"    {level}: {pct:.1f}%")
        
        # Outliers
        if profile['outliers']:
            print(f"\n⚠️  Outliers:")
            total_outliers = sum(profile['outliers'].values())
            outlier_rate = total_outliers / total * 1000 if total > 0 else 0
            print(f"  Total: {total_outliers:,} ({outlier_rate:.1f} per 1000)")
            
            # Top outlier categories
            for category, count in sorted(profile['outliers'].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    {category}: {count:,}")
        
        # Data quality issues
        if profile['errors']:
            print(f"\n❌ Data Quality Issues:")
            for error_type, count in sorted(profile['errors'].items(), key=lambda x: x[1], reverse=True):
                print(f"    {error_type}: {count:,}")
        
        # Date issues
        if profile['date_issues']:
            print(f"\n📅 Date Anomalies:")
            for issue, count in sorted(profile['date_issues'].items(), key=lambda x: x[1], reverse=True):
                print(f"    {issue}: {count:,}")
    
    def _save_json_summary(self):
        """Save analysis summary as JSON."""
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'dataset': self.data_file,
            'statistics': self.stats,
            'sources': {}
        }
        
        for source, profile in self.source_profiles.items():
            if len(profile['measurements']) < 100:
                continue
                
            summary['sources'][source] = {
                'measurement_count': len(profile['measurements']),
                'user_count': len(profile['users']),
                'mean_weight': float(np.mean(profile['measurements'])) if profile['measurements'] else 0,
                'std_weight': float(np.std(profile['measurements'])) if profile['measurements'] else 0,
                'pound_entry_rate': (profile['pound_entries']['high'] + profile['pound_entries']['medium']) / 
                                   sum(profile['pound_entries'].values()) * 100 
                                   if sum(profile['pound_entries'].values()) > 0 else 0,
                'outlier_count': sum(profile['outliers'].values()),
                'error_count': sum(profile['errors'].values()),
                'quality_score': profile['quality_scores'][0] if profile['quality_scores'] else 0
            }
        
        output_file = 'output/complete_analysis_summary.json'
        os.makedirs('output', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Summary saved to: {output_file}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_all_users_robust.py <csv_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    # Check file exists
    if not os.path.exists(data_file):
        print(f"Error: File not found: {data_file}")
        sys.exit(1)
    
    # Run analysis
    analyzer = RobustDataAnalyzer(data_file)
    analyzer.process_dataset()
    analyzer.generate_report()
    
    # Print any errors encountered
    if analyzer.errors:
        print(f"\n\n⚠️  ERRORS ENCOUNTERED")
        print("=" * 80)
        for error_type, error_list in analyzer.errors.items():
            print(f"\n{error_type}: {len(error_list)} occurrences")
            if len(error_list) <= 5:
                for error in error_list:
                    print(f"  - {error}")
            else:
                for error in error_list[:3]:
                    print(f"  - {error}")
                print(f"  ... and {len(error_list) - 3} more")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
