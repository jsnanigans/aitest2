uv run python scripts/analysis/analyze_source_noise_characteristics.py
Loading data from ./data/2025-09-05_optimized.csv...
Total measurements: 709246
Unique users: 15760
Unique sources: 7

================================================================================
SOURCE NOISE CHARACTERISTICS ANALYSIS
================================================================================

internal-questionnaire:
  Measurements: 19,949
  Users: 15,511
  Avg std dev: 4.462 kg
  Short-term noise: 4.873 kg
  Outlier rate: 0.9%
  Max deviation: 6.78 kg
  Measurement frequency: 112.8 days
  Reliability score: 4.330
  Suggested multiplier: 1.61

https://connectivehealth.io:
  Measurements: 171,147
  Users: 14,004
  Avg std dev: nan kg
  Short-term noise: nan kg
  Outlier rate: 2.8%
  Max deviation: nan kg
  Measurement frequency: 156.7 days
  Reliability score: nan
  Suggested multiplier: 5.0

care-team-upload:
  Measurements: 2,258
  Users: 1,270
  Avg std dev: nan kg
  Short-term noise: nan kg
  Outlier rate: 1.4%
  Max deviation: nan kg
  Measurement frequency: 82.1 days
  Reliability score: nan
  Suggested multiplier: 5.0

patient-upload:
  Measurements: 23,418
  Users: 4,993
  Avg std dev: 2.684 kg
  Short-term noise: 1.923 kg
  Outlier rate: 1.6%
  Max deviation: 4.91 kg
  Measurement frequency: 27.5 days
  Reliability score: 2.691
  Suggested multiplier: 1.0

https://api.iglucose.com:
  Measurements: 195,098
  Users: 3,590
  Avg std dev: 5.317 kg
  Short-term noise: 5.153 kg
  Outlier rate: 3.4%
  Max deviation: 16.25 kg
  Measurement frequency: 4.5 days
  Reliability score: 7.079
  Suggested multiplier: 2.63

patient-device:
  Measurements: 297,374
  Users: 4,952
  Avg std dev: nan kg
  Short-term noise: nan kg
  Outlier rate: 3.1%
  Max deviation: nan kg
  Measurement frequency: 6.7 days
  Reliability score: nan
  Suggested multiplier: 5.0

----------------------------------------
RECOMMENDED MULTIPLIERS:
----------------------------------------
  patient-upload: 1.0
  internal-questionnaire: 1.61
  https://api.iglucose.com: 2.63
  https://connectivehealth.io: 5.0
  care-team-upload: 5.0
  patient-device: 5.0

Results saved to test_output/source_noise_analysis.json
