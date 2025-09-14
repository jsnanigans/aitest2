import sys
sys.path.insert(0, 'src')

from quality_scorer import QualityScorer

scorer = QualityScorer()

print("Edge Case: Measurement immediately after outlier")
print("=" * 60)

# The problematic sequence
print("\nScenario: 42.22 kg outlier followed by 92.17 kg (17 minutes later)")
print("-" * 60)

score = scorer.calculate_consistency_score(
    weight=92.17,
    previous_weight=42.22,
    time_diff_hours=0.3  # 17 minutes
)

print(f"Weight change: {92.17 - 42.22:.2f} kg")
print(f"Time gap: 0.3 hours (17 minutes)")
print(f"Consistency score: {score:.3f}")
print()

# This is actually correct behavior!
print("Analysis:")
print("- A 50 kg change in 17 minutes is physically impossible")
print("- This suggests data entry error or device malfunction")
print("- Rejecting this is the RIGHT decision")
print()
print("The processor should use the last ACCEPTED weight (92.17 kg")
print("from before the outlier, not the rejected 42.22 kg value.")
