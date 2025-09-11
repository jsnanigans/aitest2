#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.processor import WeightProcessor
from src.processor_database import ProcessorDatabase

def analyze_bmi_based_detection():
    """
    Analyze how BMI-based detection could prevent unrealistic weight changes.
    """
    print("=== BMI-Based Anomaly Detection Analysis ===\n")
    
    test_cases = [
        {
            'name': 'Normal variation',
            'height_m': 1.75,
            'initial_weight': 87.0,
            'new_weight': 85.0,
            'days_gap': 7
        },
        {
            'name': 'Dramatic drop (like user 0040872d)',
            'height_m': 1.75,
            'initial_weight': 87.0,
            'new_weight': 52.0,
            'days_gap': 30
        },
        {
            'name': 'Extreme gain',
            'height_m': 1.75,
            'initial_weight': 70.0,
            'new_weight': 120.0,
            'days_gap': 10
        },
        {
            'name': 'Underweight to dangerous',
            'height_m': 1.65,
            'initial_weight': 50.0,
            'new_weight': 35.0,
            'days_gap': 5
        },
        {
            'name': 'Obese to underweight',
            'height_m': 1.70,
            'initial_weight': 110.0,
            'new_weight': 45.0,
            'days_gap': 60
        }
    ]
    
    for case in test_cases:
        print(f"\nCase: {case['name']}")
        print(f"  Height: {case['height_m']:.2f}m")
        print(f"  Weight change: {case['initial_weight']:.1f} → {case['new_weight']:.1f} kg")
        print(f"  Time gap: {case['days_gap']} days")
        
        initial_bmi = case['initial_weight'] / (case['height_m'] ** 2)
        new_bmi = case['new_weight'] / (case['height_m'] ** 2)
        weight_change = case['new_weight'] - case['initial_weight']
        pct_change = (weight_change / case['initial_weight']) * 100
        daily_change = weight_change / case['days_gap'] if case['days_gap'] > 0 else weight_change
        
        print(f"  BMI: {initial_bmi:.1f} → {new_bmi:.1f}")
        print(f"  Change: {weight_change:.1f} kg ({pct_change:+.1f}%)")
        print(f"  Daily rate: {daily_change:.2f} kg/day")
        
        violations = []
        
        if new_bmi < 16:
            violations.append(f"BMI < 16 (severely underweight): {new_bmi:.1f}")
        elif new_bmi < 18.5:
            violations.append(f"BMI < 18.5 (underweight): {new_bmi:.1f}")
        
        if new_bmi > 40:
            violations.append(f"BMI > 40 (morbidly obese): {new_bmi:.1f}")
        elif new_bmi > 35:
            violations.append(f"BMI > 35 (severely obese): {new_bmi:.1f}")
        
        if abs(pct_change) > 40:
            violations.append(f"Weight change > 40%: {pct_change:+.1f}%")
        elif abs(pct_change) > 30:
            violations.append(f"Weight change > 30%: {pct_change:+.1f}%")
        elif abs(pct_change) > 20:
            violations.append(f"Weight change > 20%: {pct_change:+.1f}%")
        
        max_safe_daily = 1.5
        if abs(daily_change) > max_safe_daily:
            violations.append(f"Daily change > {max_safe_daily}kg: {daily_change:.2f} kg/day")
        
        max_monthly_pct = 10
        monthly_pct = (pct_change / case['days_gap']) * 30 if case['days_gap'] > 0 else pct_change
        if abs(monthly_pct) > max_monthly_pct:
            violations.append(f"Monthly change > {max_monthly_pct}%: {monthly_pct:+.1f}%")
        
        if violations:
            print("  ⚠️ VIOLATIONS:")
            for v in violations:
                print(f"    - {v}")
            print("  → Should trigger RESET or require confirmation")
        else:
            print("  ✅ Within reasonable limits")
    
    print("\n\n=== Proposed BMI-Based Reset Rules ===\n")
    
    print("1. EXTREME BMI THRESHOLDS:")
    print("   - BMI < 15: Auto-reset (life-threatening)")
    print("   - BMI < 16: Reset if change > 20%")
    print("   - BMI > 40: Reset if change > 20%")
    print("   - BMI > 50: Auto-reset (extreme obesity)")
    
    print("\n2. PERCENTAGE CHANGE RULES:")
    print("   - Single measurement > 30% change: Auto-reset")
    print("   - Single measurement > 20% change: Reset if BMI extreme")
    print("   - Daily rate > 2kg/day: Auto-reset")
    print("   - Weekly average > 10%: Consider reset")
    
    print("\n3. PHYSIOLOGICAL IMPOSSIBILITIES:")
    print("   - Weight < 30kg or > 300kg: Auto-reject")
    print("   - BMI < 13 or > 60: Auto-reject (data error)")
    print("   - Change > 50% in < 30 days: Auto-reset")
    
    print("\n4. TIME-BASED ADJUSTMENTS:")
    print("   - < 1 hour: Max 3kg or 3%")
    print("   - < 1 day: Max 5kg or 5%")
    print("   - < 1 week: Max 7kg or 7%")
    print("   - < 1 month: Max 15kg or 15%")
    print("   - > 1 month: Max 1.5kg/day sustained")

def test_specific_user_scenario():
    """Test the exact scenario from user 0040872d"""
    print("\n\n=== Testing User 0040872d Scenario ===\n")
    
    height_m = 1.75
    
    measurements = [
        {'date': '2024-05-21', 'weight': 86.64, 'source': 'connectivehealth'},
        {'date': '2024-07-17', 'weight': 86.18, 'source': 'connectivehealth'},
        {'date': '2024-09-18', 'weight': 86.18, 'source': 'connectivehealth'},
        {'date': '2024-11-19', 'weight': 88.45, 'source': 'connectivehealth'},
        {'date': '2025-01-02', 'weight': 87.09, 'source': 'connectivehealth'},
        {'date': '2025-03-12 20:06:23', 'weight': 99.3, 'source': 'iglucose'},
        {'date': '2025-03-12 20:08:05', 'weight': 76.2, 'source': 'iglucose'},
        {'date': '2025-03-15 17:11:55', 'weight': 59.4, 'source': 'iglucose'},
        {'date': '2025-03-20 22:23:40', 'weight': 32.0, 'source': 'iglucose'},
        {'date': '2025-04-04 01:46:03', 'weight': 77.0, 'source': 'iglucose'},
        {'date': '2025-04-13 20:31:13', 'weight': 53.3, 'source': 'iglucose'},
    ]
    
    print(f"Assumed height: {height_m:.2f}m\n")
    
    last_valid = None
    for i, m in enumerate(measurements):
        date = pd.to_datetime(m['date'])
        weight = m['weight']
        bmi = weight / (height_m ** 2)
        
        status = "✅"
        reset_reason = None
        
        if last_valid:
            time_diff = (date - last_valid['date']).days
            weight_change = weight - last_valid['weight']
            pct_change = (weight_change / last_valid['weight']) * 100
            
            if bmi < 16:
                status = "⚠️"
                reset_reason = f"BMI too low ({bmi:.1f})"
            elif bmi > 40:
                status = "⚠️"
                reset_reason = f"BMI too high ({bmi:.1f})"
            elif abs(pct_change) > 30:
                status = "🚫"
                reset_reason = f"Change > 30% ({pct_change:+.1f}%)"
            elif abs(pct_change) > 20 and time_diff < 7:
                status = "🚫"
                reset_reason = f"Rapid change ({pct_change:+.1f}% in {time_diff} days)"
            elif time_diff > 0 and abs(weight_change / time_diff) > 2:
                status = "🚫"
                reset_reason = f"Daily rate too high ({weight_change/time_diff:.1f} kg/day)"
            
            print(f"{status} {date.strftime('%Y-%m-%d %H:%M')}: {weight:6.1f}kg (BMI: {bmi:5.1f}) "
                  f"[{pct_change:+6.1f}% in {time_diff:3}d] - {m['source']}")
            
            if reset_reason:
                print(f"   → {reset_reason}")
                if status == "🚫":
                    print(f"   → RESET KALMAN (start fresh from {weight:.1f}kg)")
                    last_valid = {'date': date, 'weight': weight}
            else:
                last_valid = {'date': date, 'weight': weight}
        else:
            print(f"{status} {date.strftime('%Y-%m-%d %H:%M')}: {weight:6.1f}kg (BMI: {bmi:5.1f}) "
                  f"[initial] - {m['source']}")
            last_valid = {'date': date, 'weight': weight}

if __name__ == "__main__":
    analyze_bmi_based_detection()
    test_specific_user_scenario()