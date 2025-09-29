
rocessing measurements individually (batch_size=1)...
Total users: 5
Total measurements: 99
[1/5] Processing user 00088d03-230... (2 measurements)
[2/5] Processing user 000ded70-578... (3 measurements)
[3/5] Processing user 001adb56-40a... (85 measurements)
[4/5] Processing user 001b4e0a-535... (7 measurements)
[5/5] Processing user 00236f31-103... (2 measurements)
  Progress: 5/5 users, 99/99 measurements

Individual processing complete:
  Successful users: 5
  Failed users: 0
  Total measurements processed: 99

Processing replay batches for 1 eligible users...
Replay window: 72 hours
[1/1] Replay for user 001adb56-40a...
  Replaying 43 measurements from 2023-10-09 00:00:00+00:00
  ✗ Replay failed: float division by zero

Replay processing complete:
  Successful replays: 0/1

Writing filtered CSV to my_filtered_data.csv...
Filtered CSV written: 0/99 measurements accepted (0.0%)

=== Processing Complete ===
Duration: 9.8 seconds
Results saved to: output_api/api_processing_results_20250929_112247.json
Filtered CSV saved to: my_filtered_data.csv
Individual processing: 99 processed, 91 accepted
Replay processing: 0/1 successful
Filtered output: 0 accepted measurements written
