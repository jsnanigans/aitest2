import numpy as np

# The issue is that last_state stores FILTERED weight, not raw weight
# From the debug output:
# - Sep 2: raw=103.4, filtered=104.96
# - Sep 3: raw=103.1 (rejected)

# The validation compares:
# current_raw (103.1) vs last_filtered (104.96) = 1.9kg change

print("THE PROBLEM:")
print("=" * 50)
print("Validation compares RAW weight against FILTERED weight!")
print()
print("Sep 2: raw=103.4kg, filtered=104.96kg")
print("Sep 3: raw=103.1kg (being validated)")
print()
print("Validation sees: |103.1 - 104.96| = 1.86kg change")
print("But actual raw change: |103.1 - 103.4| = 0.3kg")
print()
print("This is INCORRECT because:")
print("1. It mixes raw and filtered values")
print("2. The Kalman filter can create artificial 'jumps'")
print("3. A perfectly valid 0.3kg change gets rejected")
print()
print("SOLUTION:")
print("Either compare raw-to-raw OR filtered-to-filtered")
print("But NEVER mix them in physiological validation!")
