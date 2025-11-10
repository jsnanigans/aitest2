# Cross-Language Test Report
**Test Suite**: Phase 1: Integration Tests
<<<<<<< Updated upstream
**Date**: 2025-11-10T16:03:59.381Z
=======
**Date**: 2025-11-10T16:02:02.991Z
>>>>>>> Stashed changes
**Total Tests**: 5
**Passed**: 0
**Failed**: 5
**Success Rate**: 0.0%
<<<<<<< Updated upstream
**Duration**: 3.70s
=======
**Duration**: 3.65s
>>>>>>> Stashed changes

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Passed | 0 | 0.0% |
| ❌ Failed | 5 | 100.0% |

## Performance Comparison

<<<<<<< Updated upstream
- **Python avg**: 176.74ms
- **TypeScript avg**: 561.33ms
- **Speed ratio**: Python is 0.69x faster
=======
- **Python avg**: 168.01ms
- **TypeScript avg**: 560.24ms
- **Speed ratio**: Python is 0.70x faster
>>>>>>> Stashed changes

## Failed Tests

### Test 1: Single Measurement Processing
**Description**: Process a single measurement and verify initialization

**Comparison**: ✗ Found 22 difference(s): 6 numeric, 16 structural

**Differences**:
```
Found 22 difference(s):

  root.results[0].timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.results[0].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

  root.results[0].reset_reason:
    Type: value
    Python:     initial_measurement
    TypeScript: initial reset triggered
    Value mismatch: initial_measurement !== initial reset triggered

  root.results[0].gap_days:
    Type: value
    Python:     null
    TypeScript: 0
    Python value is null, TypeScript value is 0

  root.results[0].reset_event.gap_days:
    Type: missing
    Python:     null
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_event.reason:
    Type: value
    Python:     initial_measurement
    TypeScript: initial reset triggered
    Value mismatch: initial_measurement !== initial reset triggered

  root.finalState.last_state[0][0]:
    Type: type
    Python:     70
    TypeScript: [
  70
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[0][1]:
    Type: type
    Python:     0
    TypeScript: [
  0
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][0]:
    Type: type
    Python:     70
    TypeScript: [
  70
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][1]:
    Type: type
    Python:     0
    TypeScript: [
  0
]
    Type mismatch: Python number, TypeScript object

  root.finalState.reset_parameters.quality_acceptance_threshold:
    Type: missing
    Python:     0.25
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_safety_weight:
    Type: missing
    Python:     0.5
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_plausibility_weight:
    Type: missing
    Python:     0.05
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_consistency_weight:
    Type: missing
    Python:     0.05
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_reliability_weight:
    Type: missing
    Python:     0.4
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.enabled:
    Type: extra
    Python:     undefined
    TypeScript: true
    Extra key in TypeScript output

  root.finalState.reset_timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.finalState.reset_events:
    Type: missing
    Python:     [
  {
    "timestamp": 1762770272710,
    "type": "initial",
    "source": "withings",
    "weight":
    TypeScript: []
    Array length mismatch: Python 1, TypeScript 0

  root.finalState.last_timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.finalState.last_accepted_timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.finalState.measurement_history[0].timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

  root.finalState.temporal_baseline.last_timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

```

### Test 2: Multi-Measurement Sequence
**Description**: Process 10 measurements and verify state evolution

**Comparison**: ✗ Found 181 difference(s): 156 numeric, 25 structural

**Differences**:
```
Found 181 difference(s):

  root.results[0].timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.results[0].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

  root.results[0].reset_reason:
    Type: value
    Python:     initial_measurement
    TypeScript: initial reset triggered
    Value mismatch: initial_measurement !== initial reset triggered

  root.results[0].gap_days:
    Type: value
    Python:     null
    TypeScript: 0
    Python value is null, TypeScript value is 0

  root.results[0].reset_event.gap_days:
    Type: missing
    Python:     null
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_event.reason:
    Type: value
    Python:     initial_measurement
    TypeScript: initial reset triggered
    Value mismatch: initial_measurement !== initial reset triggered

  root.results[1].timestamp:
    Type: type
    Python:     1762770359110
    TypeScript: 2025-11-10T10:25:59.110Z
    Type mismatch: Python number, TypeScript string

  root.results[1].filtered_weight:
    Type: value
    Python:     69.56233059287037
    TypeScript: 69.57903566138819
    Difference: 1.671e-2
    Numeric difference exceeds tolerance: abs=1.671e-2, rel=0.024009%

  root.results[1].trend:
    Type: value
    Python:     5.4847386926894944e-8
    TypeScript: 4.1928677223600443e-7
    Difference: 3.644e-7
    Numeric difference exceeds tolerance: abs=3.644e-7, rel=86.918885%

  root.results[1].trend_weekly:
    Type: value
    Python:     3.839317084882646e-7
    TypeScript: 0.000002935007405652031
    Difference: 2.551e-6
    Numeric difference exceeds tolerance: abs=2.551e-6, rel=86.918885%

  root.results[1].confidence:
    Type: value
    Python:     0.9999902384619538
    TypeScript: 0.9999702215956114
    Difference: 2.002e-5
    Numeric difference exceeds tolerance: abs=2.002e-5, rel=0.002002%

  root.results[1].innovation:
    Type: value
    Python:     0.037669407129627075
    TypeScript: 0.020964338611804578
    Difference: 1.671e-2
    Numeric difference exceeds tolerance: abs=1.671e-2, rel=44.346513%

  root.results[1].normalized_innovation:
    Type: value
    Python:     0.00441850329645455
    TypeScript: 0.007717363251018658
    Difference: 3.299e-3
    Numeric difference exceeds tolerance: abs=3.299e-3, rel=42.745946%

  root.results[1].kalman_confidence_upper:
    Type: value
    Python:     73.56315626723355
    TypeScript: 72.66413382742729
    Difference: 8.990e-1
    Numeric difference exceeds tolerance: abs=8.990e-1, rel=1.222110%

  root.results[1].kalman_confidence_lower:
    Type: value
    Python:     65.56150491850718
    TypeScript: 66.49393749534909
    Difference: 9.324e-1
    Numeric difference exceeds tolerance: abs=9.324e-1, rel=1.402282%

  root.results[1].kalman_variance:
    Type: value
    Python:     4.0016515191608955
    TypeScript: 2.3794576735244517
    Difference: 1.622e+0
    Numeric difference exceeds tolerance: abs=1.622e+0, rel=40.538109%

  root.results[1].prediction_error:
    Type: value
    Python:     0.037669407129627075
    TypeScript: 0.020964338611804578
    Difference: 1.671e-2
    Numeric difference exceeds tolerance: abs=1.671e-2, rel=44.346513%

  root.results[1].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:25:59.110000
    TypeScript: 2025-11-10T10:25:59.110Z
    Value mismatch: 2025-11-10T11:25:59.110000 !== 2025-11-10T10:25:59.110Z

  root.results[2].timestamp:
    Type: type
    Python:     1762770445510
    TypeScript: 2025-11-10T10:27:25.510Z
    Type mismatch: Python number, TypeScript string

  root.results[2].filtered_weight:
    Type: value
    Python:     69.6053590959197
    TypeScript: 69.77350743758205
    Difference: 1.681e-1
    Numeric difference exceeds tolerance: abs=1.681e-1, rel=0.240992%

  root.results[2].trend:
    Type: value
    Python:     0.000005904206563257916
    TypeScript: 0.00004503612453218308
    Difference: 3.913e-5
    Numeric difference exceeds tolerance: abs=3.913e-5, rel=86.890065%

  root.results[2].trend_weekly:
    Type: value
    Python:     0.00004132944594280541
    TypeScript: 0.00031525287172528156
    Difference: 2.739e-4
    Numeric difference exceeds tolerance: abs=2.739e-4, rel=86.890065%

  root.results[2].confidence:
    Type: value
    Python:     0.9979153833407085
    TypeScript: 0.9937231174566352
    Difference: 4.192e-3
    Numeric difference exceeds tolerance: abs=4.192e-3, rel=0.420102%

  root.results[2].innovation:
    Type: value
    Python:     0.4646409040802979
    TypeScript: 0.29649256241793864
    Difference: 1.681e-1
    Numeric difference exceeds tolerance: abs=1.681e-1, rel=36.188881%

  root.results[2].normalized_innovation:
    Type: value
    Python:     0.06460328934325016
    TypeScript: 0.11222000708841132
    Difference: 4.762e-2
    Numeric difference exceeds tolerance: abs=4.762e-2, rel=42.431576%

  root.results[2].kalman_confidence_upper:
    Type: value
    Python:     73.62617496591466
    TypeScript: 72.58811782367049
    Difference: 1.038e+0
    Numeric difference exceeds tolerance: abs=1.038e+0, rel=1.409902%

  root.results[2].kalman_confidence_lower:
    Type: value
    Python:     65.58454322592473
    TypeScript: 66.95889705149362
    Difference: 1.374e+0
    Numeric difference exceeds tolerance: abs=1.374e+0, rel=2.052534%

  root.results[2].kalman_variance:
    Type: value
    Python:     4.041740065100835
    TypeScript: 1.9805079063692226
    Difference: 2.061e+0
    Numeric difference exceeds tolerance: abs=2.061e+0, rel=50.998632%

  root.results[2].prediction_error:
    Type: value
    Python:     0.4646409040802979
    TypeScript: 0.29649256241793864
    Difference: 1.681e-1
    Numeric difference exceeds tolerance: abs=1.681e-1, rel=36.188881%

  root.results[2].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:27:25.510000
    TypeScript: 2025-11-10T10:27:25.510Z
    Value mismatch: 2025-11-10T11:27:25.510000 !== 2025-11-10T10:27:25.510Z

  root.results[2].quality_score:
    Type: value
    Python:     0.8901620889765289
    TypeScript: 0.9199841300071664
    Difference: 2.982e-2
    Numeric difference exceeds tolerance: abs=2.982e-2, rel=3.241582%

  root.results[2].quality_components.kalman_fit:
    Type: value
    Python:     0.9901220103742958
    TypeScript: 0.990384400900456
    Difference: 2.624e-4
    Numeric difference exceeds tolerance: abs=2.624e-4, rel=0.026494%

  root.results[2].quality_components.temporal_consistency:
    Type: value
    Python:     0.7914133025655209
    TypeScript: 0.8752477858127202
    Difference: 8.383e-2
    Numeric difference exceeds tolerance: abs=8.383e-2, rel=9.578371%

  root.results[3].timestamp:
    Type: type
    Python:     1762770531910
    TypeScript: 2025-11-10T10:28:51.910Z
    Type mismatch: Python number, TypeScript string

  root.results[3].filtered_weight:
    Type: value
    Python:     69.63781267255982
    TypeScript: 69.81609309238165
    Difference: 1.783e-1
    Numeric difference exceeds tolerance: abs=1.783e-1, rel=0.255357%

  root.results[3].trend:
    Type: value
    Python:     0.00001589608071506883
    TypeScript: 0.00007096817696325693
    Difference: 5.507e-5
    Numeric difference exceeds tolerance: abs=5.507e-5, rel=77.601114%

  root.results[3].trend_weekly:
    Type: value
    Python:     0.00011127256500548181
    TypeScript: 0.0004967772387427985
    Difference: 3.855e-4
    Numeric difference exceeds tolerance: abs=3.855e-4, rel=77.601114%

  root.results[3].confidence:
    Type: value
    Python:     0.9991511550180204
    TypeScript: 0.9996000750593443
    Difference: 4.489e-4
    Numeric difference exceeds tolerance: abs=4.489e-4, rel=0.044910%

  root.results[3].innovation:
    Type: value
    Python:     0.2521873274401827
    TypeScript: 0.07390690761835117
    Difference: 1.783e-1
    Numeric difference exceeds tolerance: abs=1.783e-1, rel=70.693647%

  root.results[3].normalized_innovation:
    Type: value
    Python:     0.0412117812011586
    TypeScript: 0.028284445618127697
    Difference: 1.293e-2
    Numeric difference exceeds tolerance: abs=1.293e-2, rel=31.368058%

  root.results[3].kalman_confidence_upper:
    Type: value
    Python:     73.55311830689307
    TypeScript: 72.51994557147793
    Difference: 1.033e+0
    Numeric difference exceeds tolerance: abs=1.033e+0, rel=1.404662%

  root.results[3].kalman_confidence_lower:
    Type: value
    Python:     65.72250703822657
    TypeScript: 67.11224061328537
    Difference: 1.390e+0
    Numeric difference exceeds tolerance: abs=1.390e+0, rel=2.070760%

  root.results[3].kalman_variance:
    Type: value
    Python:     3.8324045525604276
    TypeScript: 1.827704557178785
    Difference: 2.005e+0
    Numeric difference exceeds tolerance: abs=2.005e+0, rel=52.309196%

  root.results[3].prediction_error:
    Type: value
    Python:     0.2521873274401827
    TypeScript: 0.07390690761835117
    Difference: 1.783e-1
    Numeric difference exceeds tolerance: abs=1.783e-1, rel=70.693647%

  root.results[3].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:28:51.910000
    TypeScript: 2025-11-10T10:28:51.910Z
    Value mismatch: 2025-11-10T11:28:51.910000 !== 2025-11-10T10:28:51.910Z

  root.results[3].quality_score:
    Type: value
    Python:     0.8935392243805096
    TypeScript: 0.9071352730793679
    Difference: 1.360e-2
    Numeric difference exceeds tolerance: abs=1.360e-2, rel=1.498790%

  root.results[3].quality_components.kalman_fit:
    Type: value
    Python:     0.9944454323790577
    TypeScript: 0.9977057237199334
    Difference: 3.260e-3
    Numeric difference exceeds tolerance: abs=3.260e-3, rel=0.326779%

  root.results[3].quality_components.temporal_consistency:
    Type: value
    Python:     0.9134753026673629
    TypeScript: 0.9585984788807085
    Difference: 4.512e-2
    Numeric difference exceeds tolerance: abs=4.512e-2, rel=4.707203%

  root.results[4].timestamp:
    Type: type
    Python:     1762770618310
    TypeScript: 2025-11-10T10:30:18.310Z
    Type mismatch: Python number, TypeScript string

  root.results[4].filtered_weight:
    Type: value
    Python:     69.62527261696026
    TypeScript: 69.72216366121945
    Difference: 9.689e-2
    Numeric difference exceeds tolerance: abs=9.689e-2, rel=0.138967%

  root.results[4].trend:
    Type: value
    Python:     0.000009198793110317915
    TypeScript: -0.00003277873735952792
    Difference: 4.198e-5
    Numeric difference exceeds tolerance: abs=4.198e-5, rel=128.063293%

  root.results[4].trend_weekly:
    Type: value
    Python:     0.0000643915517722254
    TypeScript: -0.00022945116151669543
    Difference: 2.938e-4
    Numeric difference exceeds tolerance: abs=2.938e-4, rel=128.063293%

  root.results[4].confidence:
    Type: value
    Python:     0.9998974856760499
    TypeScript: 0.9978117055757455
    Difference: 2.086e-3
    Numeric difference exceeds tolerance: abs=2.086e-3, rel=0.208599%

  root.results[4].innovation:
    Type: value
    Python:     -0.07527261696026244
    TypeScript: -0.17216366121945725
    Difference: 9.689e-2
    Numeric difference exceeds tolerance: abs=9.689e-2, rel=56.278452%

  root.results[4].normalized_innovation:
    Type: value
    Python:     0.01431918844785784
    TypeScript: 0.06619202730269165
    Difference: 5.187e-2
    Numeric difference exceeds tolerance: abs=5.187e-2, rel=78.367201%

  root.results[4].kalman_confidence_upper:
    Type: value
    Python:     73.34194594088689
    TypeScript: 72.37927442183116
    Difference: 9.627e-1
    Numeric difference exceeds tolerance: abs=9.627e-1, rel=1.312580%

  root.results[4].kalman_confidence_lower:
    Type: value
    Python:     65.90859929303363
    TypeScript: 67.06505290060775
    Difference: 1.156e+0
    Numeric difference exceeds tolerance: abs=1.156e+0, rel=1.724376%

  root.results[4].kalman_variance:
    Type: value
    Python:     3.453415149196944
    TypeScript: 1.7650593985396297
    Difference: 1.688e+0
    Numeric difference exceeds tolerance: abs=1.688e+0, rel=48.889452%

  root.results[4].prediction_error:
    Type: value
    Python:     -0.07527261696026244
    TypeScript: -0.17216366121945725
    Difference: 9.689e-2
    Numeric difference exceeds tolerance: abs=9.689e-2, rel=56.278452%

  root.results[4].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.results[4].quality_score:
    Type: value
    Python:     0.909561513787953
    TypeScript: 0.8946202982577565
    Difference: 1.494e-2
    Numeric difference exceeds tolerance: abs=1.494e-2, rel=1.642683%

  root.results[4].quality_components.kalman_fit:
    Type: value
    Python:     0.9982802639482513
    TypeScript: 0.9947629007575469
    Difference: 3.517e-3
    Numeric difference exceeds tolerance: abs=3.517e-3, rel=0.352342%

  root.results[4].quality_components.temporal_consistency:
    Type: value
    Python:     0.9679184759159902
    TypeScript: 0.9177442262666116
    Difference: 5.017e-2
    Numeric difference exceeds tolerance: abs=5.017e-2, rel=5.183727%

  root.results[5].timestamp:
    Type: type
    Python:     1762770704710
    TypeScript: 2025-11-10T10:31:44.710Z
    Type mismatch: Python number, TypeScript string

  root.results[5].filtered_weight:
    Type: value
    Python:     69.76001648816205
    TypeScript: 69.96830720947777
    Difference: 2.083e-1
    Numeric difference exceeds tolerance: abs=2.083e-1, rel=0.297693%

  root.results[5].trend:
    Type: value
    Python:     0.00011974253872188602
    TypeScript: 0.0003780569539299731
    Difference: 2.583e-4
    Numeric difference exceeds tolerance: abs=2.583e-4, rel=68.326852%

  root.results[5].trend_weekly:
    Type: value
    Python:     0.0008381977710532021
    TypeScript: 0.0026463986775098116
    Difference: 1.808e-3
    Numeric difference exceeds tolerance: abs=1.808e-3, rel=68.326852%

  root.results[5].confidence:
    Type: value
    Python:     0.9892916331664391
    TypeScript: 0.98430833755207
    Difference: 4.983e-3
    Numeric difference exceeds tolerance: abs=4.983e-3, rel=0.503724%

  root.results[5].innovation:
    Type: value
    Python:     0.6699835118379553
    TypeScript: 0.46169279052223544
    Difference: 2.083e-1
    Numeric difference exceeds tolerance: abs=2.083e-1, rel=31.088932%

  root.results[5].normalized_innovation:
    Type: value
    Python:     0.146738638508628
    TypeScript: 0.1778543215291966
    Difference: 3.112e-2
    Numeric difference exceeds tolerance: abs=3.112e-2, rel=17.495039%

  root.results[5].kalman_confidence_upper:
    Type: value
    Python:     73.21830586276712
    TypeScript: 72.60551703469908
    Difference: 6.128e-1
    Numeric difference exceeds tolerance: abs=6.128e-1, rel=0.836934%

  root.results[5].kalman_confidence_lower:
    Type: value
    Python:     66.30172711355698
    TypeScript: 67.33109738425647
    Difference: 1.029e+0
    Numeric difference exceeds tolerance: abs=1.029e+0, rel=1.528818%

  root.results[5].kalman_variance:
    Type: value
    Python:     2.989941349626596
    TypeScript: 1.7387189155609415
    Difference: 1.251e+0
    Numeric difference exceeds tolerance: abs=1.251e+0, rel=41.847725%

  root.results[5].prediction_error:
    Type: value
    Python:     0.6699835118379553
    TypeScript: 0.46169279052223544
    Difference: 2.083e-1
    Numeric difference exceeds tolerance: abs=2.083e-1, rel=31.088932%

  root.results[5].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:31:44.710000
    TypeScript: 2025-11-10T10:31:44.710Z
    Value mismatch: 2025-11-10T11:31:44.710000 !== 2025-11-10T10:31:44.710Z

  root.results[5].quality_score:
    Type: value
    Python:     0.4661584533390423
    TypeScript: 0.49399895241721303
    Difference: 2.784e-2
    Numeric difference exceeds tolerance: abs=2.784e-2, rel=5.635741%

  root.results[5].quality_components.kalman_fit:
    Type: value
    Python:     0.9843156898703053
    TypeScript: 0.9861253584309051
    Difference: 1.810e-3
    Numeric difference exceeds tolerance: abs=1.810e-3, rel=0.183513%

  root.results[5].quality_components.temporal_consistency:
    Type: value
    Python:     0.4380641095179947
    TypeScript: 0.5312748311191138
    Difference: 9.321e-2
    Numeric difference exceeds tolerance: abs=9.321e-2, rel=17.544728%

  root.results[6].timestamp:
    Type: type
    Python:     1762770791110
    TypeScript: 2025-11-10T10:33:11.110Z
    Type mismatch: Python number, TypeScript string

  root.results[6].filtered_weight:
    Type: value
    Python:     69.86354986279649
    TypeScript: 70.0898475328516
    Difference: 2.263e-1
    Numeric difference exceeds tolerance: abs=2.263e-1, rel=0.322868%

  root.results[6].trend:
    Type: value
    Python:     0.00024196956971566568
    TypeScript: 0.0006543143777515837
    Difference: 4.123e-4
    Numeric difference exceeds tolerance: abs=4.123e-4, rel=63.019371%

  root.results[6].trend_weekly:
    Type: value
    Python:     0.0016937869880096598
    TypeScript: 0.004580200644261086
    Difference: 2.886e-3
    Numeric difference exceeds tolerance: abs=2.886e-3, rel=63.019371%

  root.results[6].confidence:
    Type: value
    Python:     0.993564751924342
    TypeScript: 0.9960709483857568
    Difference: 2.506e-3
    Numeric difference exceeds tolerance: abs=2.506e-3, rel=0.251608%

  root.results[6].innovation:
    Type: value
    Python:     0.4564501372035039
    TypeScript: 0.2301524671483861
    Difference: 2.263e-1
    Numeric difference exceeds tolerance: abs=2.263e-1, rel=49.577742%

  root.results[6].normalized_innovation:
    Type: value
    Python:     0.11363136493576545
    TypeScript: 0.08873320252845887
    Difference: 2.490e-2
    Numeric difference exceeds tolerance: abs=2.490e-2, rel=21.911347%

  root.results[6].kalman_confidence_upper:
    Type: value
    Python:     73.03692950223387
    TypeScript: 72.71859730085806
    Difference: 3.183e-1
    Numeric difference exceeds tolerance: abs=3.183e-1, rel=0.435851%

  root.results[6].kalman_confidence_lower:
    Type: value
    Python:     66.6901702233591
    TypeScript: 67.46109776484515
    Difference: 7.709e-1
    Numeric difference exceeds tolerance: abs=7.709e-1, rel=1.142773%

  root.results[6].kalman_variance:
    Type: value
    Python:     2.517584583998928
    TypeScript: 1.7275813356984977
    Difference: 7.900e-1
    Numeric difference exceeds tolerance: abs=7.900e-1, rel=31.379412%

  root.results[6].prediction_error:
    Type: value
    Python:     0.4564501372035039
    TypeScript: 0.2301524671483861
    Difference: 2.263e-1
    Numeric difference exceeds tolerance: abs=2.263e-1, rel=49.577742%

  root.results[6].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:33:11.110000
    TypeScript: 2025-11-10T10:33:11.110Z
    Value mismatch: 2025-11-10T11:33:11.110000 !== 2025-11-10T10:33:11.110Z

  root.results[6].quality_score:
    Type: value
    Python:     0.7612391154514953
    TypeScript: 0.816385406237894
    Difference: 5.515e-2
    Numeric difference exceeds tolerance: abs=5.515e-2, rel=6.754933%

  root.results[6].quality_components.kalman_fit:
    Type: value
    Python:     0.9890323074969029
    TypeScript: 0.9930819659001825
    Difference: 4.050e-3
    Numeric difference exceeds tolerance: abs=4.050e-3, rel=0.407787%

  root.results[6].quality_components.temporal_consistency:
    Type: value
    Python:     0.7131278587506958
    TypeScript: 0.8992936932315927
    Difference: 1.862e-1
    Numeric difference exceeds tolerance: abs=1.862e-1, rel=20.701339%

  root.results[7].timestamp:
    Type: type
    Python:     1762770877510
    TypeScript: 2025-11-10T10:34:37.510Z
    Type mismatch: Python number, TypeScript string

  root.results[7].filtered_weight:
    Type: value
    Python:     69.90938341577476
    TypeScript: 70.09338879035677
    Difference: 1.840e-1
    Numeric difference exceeds tolerance: abs=1.840e-1, rel=0.262515%

  root.results[7].trend:
    Type: value
    Python:     0.000316812908999397
    TypeScript: 0.0006643992068045659
    Difference: 3.476e-4
    Numeric difference exceeds tolerance: abs=3.476e-4, rel=52.315881%

  root.results[7].trend_weekly:
    Type: value
    Python:     0.002217690362995779
    TypeScript: 0.004650794447631961
    Difference: 2.433e-3
    Numeric difference exceeds tolerance: abs=2.433e-3, rel=52.315881%

  root.results[7].confidence:
    Type: value
    Python:     0.9985888523573752
    TypeScript: 0.999996749326322
    Difference: 1.408e-3
    Numeric difference exceeds tolerance: abs=1.408e-3, rel=0.140790%

  root.results[7].innovation:
    Type: value
    Python:     0.1906165842252392
    TypeScript: 0.0066112096432249245
    Difference: 1.840e-1
    Numeric difference exceeds tolerance: abs=1.840e-1, rel=96.531671%

  root.results[7].normalized_innovation:
    Type: value
    Python:     0.053144035397069814
    TypeScript: 0.002549776053502904
    Difference: 5.059e-2
    Numeric difference exceeds tolerance: abs=5.059e-2, rel=95.202141%

  root.results[7].kalman_confidence_upper:
    Type: value
    Python:     72.79944821519184
    TypeScript: 72.71858782952235
    Difference: 8.086e-2
    Numeric difference exceeds tolerance: abs=8.086e-2, rel=0.111073%

  root.results[7].kalman_confidence_lower:
    Type: value
    Python:     67.01931861635767
    TypeScript: 67.46818975119119
    Difference: 4.489e-1
    Numeric difference exceeds tolerance: abs=4.489e-1, rel=0.665308%

  root.results[7].kalman_variance:
    Type: value
    Python:     2.0881186362074304
    TypeScript: 1.7229174988089617
    Difference: 3.652e-1
    Numeric difference exceeds tolerance: abs=3.652e-1, rel=17.489482%

  root.results[7].prediction_error:
    Type: value
    Python:     0.1906165842252392
    TypeScript: 0.0066112096432249245
    Difference: 1.840e-1
    Numeric difference exceeds tolerance: abs=1.840e-1, rel=96.531671%

  root.results[7].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:34:37.510000
    TypeScript: 2025-11-10T10:34:37.510Z
    Value mismatch: 2025-11-10T11:34:37.510000 !== 2025-11-10T10:34:37.510Z

  root.results[7].quality_score:
    Type: value
    Python:     0.7773749829236973
    TypeScript: 0.7927475133667357
    Difference: 1.537e-2
    Numeric difference exceeds tolerance: abs=1.537e-2, rel=1.939146%

  root.results[7].quality_components.kalman_fit:
    Type: value
    Python:     0.9953430273240147
    TypeScript: 0.9998008795705958
    Difference: 4.458e-3
    Numeric difference exceeds tolerance: abs=4.458e-3, rel=0.445874%

  root.results[7].quality_components.temporal_consistency:
    Type: value
    Python:     0.9249027060807666
    TypeScript: 0.9959977878108687
    Difference: 7.110e-2
    Numeric difference exceeds tolerance: abs=7.110e-2, rel=7.138076%

  root.results[8].timestamp:
    Type: type
    Python:     1762770963910
    TypeScript: 2025-11-10T10:36:03.910Z
    Type mismatch: Python number, TypeScript string

  root.results[8].filtered_weight:
    Type: value
    Python:     69.96010854415746
    TypeScript: 70.11980241024361
    Difference: 1.597e-1
    Numeric difference exceeds tolerance: abs=1.597e-1, rel=0.227744%

  root.results[8].trend:
    Type: value
    Python:     0.00042842827197489227
    TypeScript: 0.0007577391197213973
    Difference: 3.293e-4
    Numeric difference exceeds tolerance: abs=3.293e-4, rel=43.459660%

  root.results[8].trend_weekly:
    Type: value
    Python:     0.002998997903824246
    TypeScript: 0.005304173838049781
    Difference: 2.305e-3
    Numeric difference exceeds tolerance: abs=2.305e-3, rel=43.459660%

  root.results[8].confidence:
    Type: value
    Python:     0.9979238101769797
    TypeScript: 0.9998125613493881
    Difference: 1.889e-3
    Numeric difference exceeds tolerance: abs=1.889e-3, rel=0.188911%

  root.results[8].innovation:
    Type: value
    Python:     0.20989145584253777
    TypeScript: 0.05019758975639377
    Difference: 1.597e-1
    Numeric difference exceeds tolerance: abs=1.597e-1, rel=76.084024%

  root.results[8].normalized_innovation:
    Type: value
    Python:     0.06447244516763724
    TypeScript: 0.01936265578019663
    Difference: 4.511e-2
    Numeric difference exceeds tolerance: abs=4.511e-2, rel=69.967549%

  root.results[8].kalman_confidence_upper:
    Type: value
    Python:     72.58766507247513
    TypeScript: 72.74356361573297
    Difference: 1.559e-1
    Numeric difference exceeds tolerance: abs=1.559e-1, rel=0.214312%

  root.results[8].kalman_confidence_lower:
    Type: value
    Python:     67.3325520158398
    TypeScript: 67.49604120475425
    Difference: 1.635e-1
    Numeric difference exceeds tolerance: abs=1.635e-1, rel=0.242220%

  root.results[8].kalman_variance:
    Type: value
    Python:     1.7260133273761955
    TypeScript: 1.7210307158577463
    Difference: 4.983e-3
    Numeric difference exceeds tolerance: abs=4.983e-3, rel=0.288677%

  root.results[8].prediction_error:
    Type: value
    Python:     0.20989145584253777
    TypeScript: 0.05019758975639377
    Difference: 1.597e-1
    Numeric difference exceeds tolerance: abs=1.597e-1, rel=76.084024%

  root.results[8].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:36:03.910000
    TypeScript: 2025-11-10T10:36:03.910Z
    Value mismatch: 2025-11-10T11:36:03.910000 !== 2025-11-10T10:36:03.910Z

  root.results[8].quality_score:
    Type: value
    Python:     0.7743640699347982
    TypeScript: 0.7860241689480075
    Difference: 1.166e-2
    Numeric difference exceeds tolerance: abs=1.166e-2, rel=1.483428%

  root.results[8].quality_components.kalman_fit:
    Type: value
    Python:     0.9948569776966091
    TypeScript: 0.9984899469579456
    Difference: 3.633e-3
    Numeric difference exceeds tolerance: abs=3.633e-3, rel=0.363846%

  root.results[8].quality_components.temporal_consistency:
    Type: value
    Python:     0.9190351298797528
    TypeScript: 0.971705566704649
    Difference: 5.267e-2
    Numeric difference exceeds tolerance: abs=5.267e-2, rel=5.420411%

  root.results[9].timestamp:
    Type: type
    Python:     1762771050310
    TypeScript: 2025-11-10T10:37:30.310Z
    Type mismatch: Python number, TypeScript string

  root.results[9].filtered_weight:
    Type: value
    Python:     69.97334581855951
    TypeScript: 70.08895398891374
    Difference: 1.156e-1
    Numeric difference exceeds tolerance: abs=1.156e-1, rel=0.164945%

  root.results[9].trend:
    Type: value
    Python:     0.0004667853739103754
    TypeScript: 0.0006281076661637135
    Difference: 1.613e-4
    Numeric difference exceeds tolerance: abs=1.613e-4, rel=25.683860%

  root.results[9].trend_weekly:
    Type: value
    Python:     0.003267497617372628
    TypeScript: 0.0043967536631459945
    Difference: 1.129e-3
    Numeric difference exceeds tolerance: abs=1.129e-3, rel=25.683860%

  root.results[9].confidence:
    Type: value
    Python:     0.9998223034455657
    TypeScript: 0.9997414473210958
    Difference: 8.086e-5
    Numeric difference exceeds tolerance: abs=8.086e-5, rel=0.008087%

  root.results[9].innovation:
    Type: value
    Python:     0.056654181440492835
    TypeScript: -0.058953988913742705
    Difference: 1.156e-1
    Numeric difference exceeds tolerance: abs=1.156e-1, rel=196.098979%

  root.results[9].normalized_innovation:
    Type: value
    Python:     0.01885271038008296
    TypeScript: 0.022741420774023528
    Difference: 3.889e-3
    Numeric difference exceeds tolerance: abs=3.889e-3, rel=17.099681%

  root.results[9].kalman_confidence_upper:
    Type: value
    Python:     72.36904276828402
    TypeScript: 72.71218813188251
    Difference: 3.431e-1
    Numeric difference exceeds tolerance: abs=3.431e-1, rel=0.471923%

  root.results[9].kalman_confidence_lower:
    Type: value
    Python:     67.577648868835
    TypeScript: 67.46571984594497
    Difference: 1.119e-1
    Numeric difference exceeds tolerance: abs=1.119e-1, rel=0.165630%

  root.results[9].kalman_variance:
    Type: value
    Python:     1.4348409687298243
    TypeScript: 1.7203393422092734
    Difference: 2.855e-1
    Numeric difference exceeds tolerance: abs=2.855e-1, rel=16.595468%

  root.results[9].prediction_error:
    Type: value
    Python:     0.056654181440492835
    TypeScript: -0.058953988913742705
    Difference: 1.156e-1
    Numeric difference exceeds tolerance: abs=1.156e-1, rel=196.098979%

  root.results[9].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:37:30.310000
    TypeScript: 2025-11-10T10:37:30.310Z
    Value mismatch: 2025-11-10T11:37:30.310000 !== 2025-11-10T10:37:30.310Z

  root.results[9].quality_score:
    Type: value
    Python:     0.782962178012945
    TypeScript: 0.7815486827473337
    Difference: 1.413e-3
    Numeric difference exceeds tolerance: abs=1.413e-3, rel=0.180532%

  root.results[9].quality_components.kalman_fit:
    Type: value
    Python:     0.9986162613355426
    TypeScript: 0.9982271286418527
    Difference: 3.891e-4
    Numeric difference exceeds tolerance: abs=3.891e-4, rel=0.038967%

  root.results[9].quality_components.temporal_consistency:
    Type: value
    Python:     0.9740183066335732
    TypeScript: 0.9672545572606939
    Difference: 6.764e-3
    Numeric difference exceeds tolerance: abs=6.764e-3, rel=0.694417%

  root.finalState.kalman_params.transition_covariance[0][0]:
    Type: value
    Python:     0.04209952319851201
    TypeScript: 0.8999999999999999
    Difference: 8.579e-1
    Numeric difference exceeds tolerance: abs=8.579e-1, rel=95.322275%

  root.finalState.kalman_params.transition_covariance[1][1]:
    Type: value
    Python:     0.0002806634879900801
    TypeScript: 0.006
    Difference: 5.719e-3
    Numeric difference exceeds tolerance: abs=5.719e-3, rel=95.322275%

  root.finalState.last_state[0][0]:
    Type: type
    Python:     69.96010854415746
    TypeScript: [
  70.11980241024361
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[0][1]:
    Type: type
    Python:     0.00042842827197489227
    TypeScript: [
  0.0007577391197213973
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][0]:
    Type: type
    Python:     69.97334581855951
    TypeScript: [
  70.08895398891374
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][1]:
    Type: type
    Python:     0.0004667853739103754
    TypeScript: [
  0.0006281076661637135
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_covariance[0][0][0]:
    Type: value
    Python:     1.7260133273761955
    TypeScript: 1.7210307158577463
    Difference: 4.983e-3
    Numeric difference exceeds tolerance: abs=4.983e-3, rel=0.288677%

  root.finalState.last_covariance[0][0][1]:
    Type: value
    Python:     0.003800286133368291
    TypeScript: 0.006097079738770134
    Difference: 2.297e-3
    Numeric difference exceeds tolerance: abs=2.297e-3, rel=37.670388%

  root.finalState.last_covariance[0][1][0]:
    Type: value
    Python:     0.0038002861333682903
    TypeScript: 0.006097079738770133
    Difference: 2.297e-3
    Numeric difference exceeds tolerance: abs=2.297e-3, rel=37.670388%

  root.finalState.last_covariance[0][1][1]:
    Type: value
    Python:     0.01342336863742711
    TypeScript: 0.04897210549826418
    Difference: 3.555e-2
    Numeric difference exceeds tolerance: abs=3.555e-2, rel=72.589766%

  root.finalState.last_covariance[1][0][0]:
    Type: value
    Python:     1.4348409687298243
    TypeScript: 1.7203393422092734
    Difference: 2.855e-1
    Numeric difference exceeds tolerance: abs=2.855e-1, rel=16.595468%

  root.finalState.last_covariance[1][0][1]:
    Type: value
    Python:     0.004171179409022243
    TypeScript: 0.007211508263968155
    Difference: 3.040e-3
    Numeric difference exceeds tolerance: abs=3.040e-3, rel=42.159403%

  root.finalState.last_covariance[1][1][0]:
    Type: value
    Python:     0.004171179409022243
    TypeScript: 0.007211508263968154
    Difference: 3.040e-3
    Numeric difference exceeds tolerance: abs=3.040e-3, rel=42.159403%

  root.finalState.last_covariance[1][1][1]:
    Type: value
    Python:     0.013701208073428174
    TypeScript: 0.054956248415209644
    Difference: 4.126e-2
    Numeric difference exceeds tolerance: abs=4.126e-2, rel=75.068880%

  root.finalState.reset_parameters.quality_acceptance_threshold:
    Type: missing
    Python:     0.25
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_safety_weight:
    Type: missing
    Python:     0.5
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_plausibility_weight:
    Type: missing
    Python:     0.05
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_consistency_weight:
    Type: missing
    Python:     0.05
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_reliability_weight:
    Type: missing
    Python:     0.4
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.enabled:
    Type: extra
    Python:     undefined
    TypeScript: true
    Extra key in TypeScript output

  root.finalState.reset_timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.finalState.reset_events:
    Type: missing
    Python:     [
  {
    "timestamp": 1762770272710,
    "type": "initial",
    "source": "withings",
    "weight":
    TypeScript: []
    Array length mismatch: Python 1, TypeScript 0

  root.finalState.last_timestamp:
    Type: type
    Python:     1762771050310
    TypeScript: 2025-11-10T10:37:30.310Z
    Type mismatch: Python number, TypeScript string

  root.finalState.last_accepted_timestamp:
    Type: type
    Python:     1762771050310
    TypeScript: 2025-11-10T10:37:30.310Z
    Type mismatch: Python number, TypeScript string

  root.finalState.measurement_history[0].timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

  root.finalState.measurement_history[1].timestamp:
    Type: value
    Python:     2025-11-10T11:25:59.110000
    TypeScript: 2025-11-10T10:25:59.110Z
    Value mismatch: 2025-11-10T11:25:59.110000 !== 2025-11-10T10:25:59.110Z

  root.finalState.measurement_history[2].timestamp:
    Type: value
    Python:     2025-11-10T11:27:25.510000
    TypeScript: 2025-11-10T10:27:25.510Z
    Value mismatch: 2025-11-10T11:27:25.510000 !== 2025-11-10T10:27:25.510Z

  root.finalState.measurement_history[2].quality_score:
    Type: value
    Python:     0.8901620889765289
    TypeScript: 0.9199841300071664
    Difference: 2.982e-2
    Numeric difference exceeds tolerance: abs=2.982e-2, rel=3.241582%

  root.finalState.measurement_history[3].timestamp:
    Type: value
    Python:     2025-11-10T11:28:51.910000
    TypeScript: 2025-11-10T10:28:51.910Z
    Value mismatch: 2025-11-10T11:28:51.910000 !== 2025-11-10T10:28:51.910Z

  root.finalState.measurement_history[3].quality_score:
    Type: value
    Python:     0.8935392243805096
    TypeScript: 0.9071352730793679
    Difference: 1.360e-2
    Numeric difference exceeds tolerance: abs=1.360e-2, rel=1.498790%

  root.finalState.measurement_history[4].timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.finalState.measurement_history[4].quality_score:
    Type: value
    Python:     0.909561513787953
    TypeScript: 0.8946202982577565
    Difference: 1.494e-2
    Numeric difference exceeds tolerance: abs=1.494e-2, rel=1.642683%

  root.finalState.measurement_history[5].timestamp:
    Type: value
    Python:     2025-11-10T11:31:44.710000
    TypeScript: 2025-11-10T10:31:44.710Z
    Value mismatch: 2025-11-10T11:31:44.710000 !== 2025-11-10T10:31:44.710Z

  root.finalState.measurement_history[5].quality_score:
    Type: value
    Python:     0.4661584533390423
    TypeScript: 0.49399895241721303
    Difference: 2.784e-2
    Numeric difference exceeds tolerance: abs=2.784e-2, rel=5.635741%

  root.finalState.measurement_history[6].timestamp:
    Type: value
    Python:     2025-11-10T11:33:11.110000
    TypeScript: 2025-11-10T10:33:11.110Z
    Value mismatch: 2025-11-10T11:33:11.110000 !== 2025-11-10T10:33:11.110Z

  root.finalState.measurement_history[6].quality_score:
    Type: value
    Python:     0.7612391154514953
    TypeScript: 0.816385406237894
    Difference: 5.515e-2
    Numeric difference exceeds tolerance: abs=5.515e-2, rel=6.754933%

  root.finalState.measurement_history[7].timestamp:
    Type: value
    Python:     2025-11-10T11:34:37.510000
    TypeScript: 2025-11-10T10:34:37.510Z
    Value mismatch: 2025-11-10T11:34:37.510000 !== 2025-11-10T10:34:37.510Z

  root.finalState.measurement_history[7].quality_score:
    Type: value
    Python:     0.7773749829236973
    TypeScript: 0.7927475133667357
    Difference: 1.537e-2
    Numeric difference exceeds tolerance: abs=1.537e-2, rel=1.939146%

  root.finalState.measurement_history[8].timestamp:
    Type: value
    Python:     2025-11-10T11:36:03.910000
    TypeScript: 2025-11-10T10:36:03.910Z
    Value mismatch: 2025-11-10T11:36:03.910000 !== 2025-11-10T10:36:03.910Z

  root.finalState.measurement_history[8].quality_score:
    Type: value
    Python:     0.7743640699347982
    TypeScript: 0.7860241689480075
    Difference: 1.166e-2
    Numeric difference exceeds tolerance: abs=1.166e-2, rel=1.483428%

  root.finalState.measurement_history[9].timestamp:
    Type: value
    Python:     2025-11-10T11:37:30.310000
    TypeScript: 2025-11-10T10:37:30.310Z
    Value mismatch: 2025-11-10T11:37:30.310000 !== 2025-11-10T10:37:30.310Z

  root.finalState.measurement_history[9].quality_score:
    Type: value
    Python:     0.782962178012945
    TypeScript: 0.7815486827473337
    Difference: 1.413e-3
    Numeric difference exceeds tolerance: abs=1.413e-3, rel=0.180532%

  root.finalState.temporal_baseline.last_timestamp:
    Type: value
    Python:     2025-11-10T11:37:30.310000
    TypeScript: 2025-11-10T10:37:30.310Z
    Value mismatch: 2025-11-10T11:37:30.310000 !== 2025-11-10T10:37:30.310Z

```

### Test 3: Reset Scenario
**Description**: Process measurements with a large change that triggers reset

**Comparison**: ✗ Found 174 difference(s): 136 numeric, 38 structural

**Differences**:
```
Found 174 difference(s):

  root.results[0].timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.results[0].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

  root.results[0].reset_reason:
    Type: value
    Python:     initial_measurement
    TypeScript: initial reset triggered
    Value mismatch: initial_measurement !== initial reset triggered

  root.results[0].gap_days:
    Type: value
    Python:     null
    TypeScript: 0
    Python value is null, TypeScript value is 0

  root.results[0].reset_event.gap_days:
    Type: missing
    Python:     null
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_event.reason:
    Type: value
    Python:     initial_measurement
    TypeScript: initial reset triggered
    Value mismatch: initial_measurement !== initial reset triggered

  root.results[1].timestamp:
    Type: type
    Python:     1762770359110
    TypeScript: 2025-11-10T10:25:59.110Z
    Type mismatch: Python number, TypeScript string

  root.results[1].filtered_weight:
    Type: value
    Python:     69.88855472219821
    TypeScript: 69.93824497440752
    Difference: 4.969e-2
    Numeric difference exceeds tolerance: abs=4.969e-2, rel=0.071049%

  root.results[1].trend:
    Type: value
    Python:     1.6314692073911941e-7
    TypeScript: 0.0000012471942535407356
    Difference: 1.084e-6
    Numeric difference exceeds tolerance: abs=1.084e-6, rel=86.918885%

  root.results[1].trend_weekly:
    Type: value
    Python:     0.000001142028445173836
    TypeScript: 0.000008730359774785148
    Difference: 7.588e-6
    Numeric difference exceeds tolerance: abs=7.588e-6, rel=86.918885%

  root.results[1].confidence:
    Type: value
    Python:     0.9999136330673903
    TypeScript: 0.9997365509991943
    Difference: 1.771e-4
    Numeric difference exceeds tolerance: abs=1.771e-4, rel=0.017710%

  root.results[1].innovation:
    Type: value
    Python:     0.11204996488635288
    TypeScript: 0.06235971267703633
    Difference: 4.969e-2
    Numeric difference exceeds tolerance: abs=4.969e-2, rel=44.346513%

  root.results[1].normalized_innovation:
    Type: value
    Python:     0.013143109407436756
    TypeScript: 0.022955770934116628
    Difference: 9.813e-3
    Numeric difference exceeds tolerance: abs=9.813e-3, rel=42.745946%

  root.results[1].kalman_confidence_upper:
    Type: value
    Python:     73.88938039656139
    TypeScript: 73.02334314044663
    Difference: 8.660e-1
    Numeric difference exceeds tolerance: abs=8.660e-1, rel=1.172073%

  root.results[1].kalman_confidence_lower:
    Type: value
    Python:     65.88772904783502
    TypeScript: 66.85314680836842
    Difference: 9.654e-1
    Numeric difference exceeds tolerance: abs=9.654e-1, rel=1.444087%

  root.results[1].kalman_variance:
    Type: value
    Python:     4.0016515191608955
    TypeScript: 2.3794576735244517
    Difference: 1.622e+0
    Numeric difference exceeds tolerance: abs=1.622e+0, rel=40.538109%

  root.results[1].prediction_error:
    Type: value
    Python:     0.11204996488635288
    TypeScript: 0.06235971267703633
    Difference: 4.969e-2
    Numeric difference exceeds tolerance: abs=4.969e-2, rel=44.346513%

  root.results[1].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:25:59.110000
    TypeScript: 2025-11-10T10:25:59.110Z
    Value mismatch: 2025-11-10T11:25:59.110000 !== 2025-11-10T10:25:59.110Z

  root.results[2].timestamp:
    Type: type
    Python:     1762770445510
    TypeScript: 2025-11-10T10:27:25.510Z
    Type mismatch: Python number, TypeScript string

  root.results[2].filtered_weight:
    Type: value
    Python:     69.89917473309745
    TypeScript: 69.96819397178386
    Difference: 6.902e-2
    Numeric difference exceeds tolerance: abs=6.902e-2, rel=0.098644%

  root.results[2].trend:
    Type: value
    Python:     0.0000016068450810693616
    TypeScript: 0.000008118239022466214
    Difference: 6.511e-6
    Numeric difference exceeds tolerance: abs=6.511e-6, rel=80.206975%

  root.results[2].trend_weekly:
    Type: value
    Python:     0.000011247915567485531
    TypeScript: 0.000056827673157263496
    Difference: 4.558e-5
    Numeric difference exceeds tolerance: abs=4.558e-5, rel=80.206975%

  root.results[2].confidence:
    Type: value
    Python:     0.9998728876085181
    TypeScript: 0.9998506771931183
    Difference: 2.221e-5
    Numeric difference exceeds tolerance: abs=2.221e-5, rel=0.002221%

  root.results[2].innovation:
    Type: value
    Python:     0.11467943721925167
    TypeScript: 0.04566019853284331
    Difference: 6.902e-2
    Numeric difference exceeds tolerance: abs=6.902e-2, rel=60.184494%

  root.results[2].normalized_innovation:
    Type: value
    Python:     0.015944934678235004
    TypeScript: 0.017282011262701166
    Difference: 1.337e-3
    Numeric difference exceeds tolerance: abs=1.337e-3, rel=7.736811%

  root.results[2].kalman_confidence_upper:
    Type: value
    Python:     73.91999060309242
    TypeScript: 72.7828043578723
    Difference: 1.137e+0
    Numeric difference exceeds tolerance: abs=1.137e+0, rel=1.538402%

  root.results[2].kalman_confidence_lower:
    Type: value
    Python:     65.87835886310249
    TypeScript: 67.15358358569543
    Difference: 1.275e+0
    Numeric difference exceeds tolerance: abs=1.275e+0, rel=1.898967%

  root.results[2].kalman_variance:
    Type: value
    Python:     4.041740065100835
    TypeScript: 1.9805079063692226
    Difference: 2.061e+0
    Numeric difference exceeds tolerance: abs=2.061e+0, rel=50.998632%

  root.results[2].prediction_error:
    Type: value
    Python:     0.11467943721925167
    TypeScript: 0.04566019853284331
    Difference: 6.902e-2
    Numeric difference exceeds tolerance: abs=6.902e-2, rel=60.184494%

  root.results[2].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:27:25.510000
    TypeScript: 2025-11-10T10:27:25.510Z
    Value mismatch: 2025-11-10T11:27:25.510000 !== 2025-11-10T10:27:25.510Z

  root.results[2].quality_score:
    Type: value
    Python:     0.9477124619928798
    TypeScript: 0.9527858133476765
    Difference: 5.073e-3
    Numeric difference exceeds tolerance: abs=5.073e-3, rel=0.532476%

  root.results[2].quality_components.kalman_fit:
    Type: value
    Python:     0.997552861181671
    TypeScript: 0.9985131276522633
    Difference: 9.603e-4
    Numeric difference exceeds tolerance: abs=9.603e-4, rel=0.096170%

  root.results[2].quality_components.temporal_consistency:
    Type: value
    Python:     0.9558417133453186
    TypeScript: 0.9720484701765703
    Difference: 1.621e-2
    Numeric difference exceeds tolerance: abs=1.621e-2, rel=1.667279%

  root.results[3].timestamp:
    Type: type
    Python:     1762770531910
    TypeScript: 2025-11-10T10:28:51.910Z
    Type mismatch: Python number, TypeScript string

  root.results[3].filtered_weight:
    Type: value
    Python:     69.92231645736025
    TypeScript: 70.01715947165995
    Difference: 9.484e-2
    Numeric difference exceeds tolerance: abs=9.484e-2, rel=0.135457%

  root.results[3].trend:
    Type: value
    Python:     0.0000087318470573641
    TypeScript: 0.00003793788509685788
    Difference: 2.921e-5
    Numeric difference exceeds tolerance: abs=2.921e-5, rel=76.983833%

  root.results[3].trend_weekly:
    Type: value
    Python:     0.0000611229294015487
    TypeScript: 0.00026556519567800515
    Difference: 2.044e-4
    Numeric difference exceeds tolerance: abs=2.044e-4, rel=76.983833%

  root.results[3].confidence:
    Type: value
    Python:     0.9995682870458482
    TypeScript: 0.9994712118879342
    Difference: 9.708e-5
    Numeric difference exceeds tolerance: abs=9.708e-5, rel=0.009712%

  root.results[3].innovation:
    Type: value
    Python:     0.17982964748230756
    TypeScript: 0.08498663318260924
    Difference: 9.484e-2
    Numeric difference exceeds tolerance: abs=9.484e-2, rel=52.740477%

  root.results[3].normalized_innovation:
    Type: value
    Python:     0.029387281909636038
    TypeScript: 0.03252469737922591
    Difference: 3.137e-3
    Numeric difference exceeds tolerance: abs=3.137e-3, rel=9.646256%

  root.results[3].kalman_confidence_upper:
    Type: value
    Python:     73.8376220916935
    TypeScript: 72.72101195075624
    Difference: 1.117e+0
    Numeric difference exceeds tolerance: abs=1.117e+0, rel=1.512251%

  root.results[3].kalman_confidence_lower:
    Type: value
    Python:     66.007010823027
    TypeScript: 67.31330699256367
    Difference: 1.306e+0
    Numeric difference exceeds tolerance: abs=1.306e+0, rel=1.940621%

  root.results[3].kalman_variance:
    Type: value
    Python:     3.8324045525604276
    TypeScript: 1.827704557178785
    Difference: 2.005e+0
    Numeric difference exceeds tolerance: abs=2.005e+0, rel=52.309196%

  root.results[3].prediction_error:
    Type: value
    Python:     0.17982964748230756
    TypeScript: 0.08498663318260924
    Difference: 9.484e-2
    Numeric difference exceeds tolerance: abs=9.484e-2, rel=52.740477%

  root.results[3].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:28:51.910000
    TypeScript: 2025-11-10T10:28:51.910Z
    Value mismatch: 2025-11-10T11:28:51.910000 !== 2025-11-10T10:28:51.910Z

  root.results[3].quality_score:
    Type: value
    Python:     0.9181140831089427
    TypeScript: 0.9241937239986877
    Difference: 6.080e-3
    Numeric difference exceeds tolerance: abs=6.080e-3, rel=0.657832%

  root.results[3].quality_components.kalman_fit:
    Type: value
    Python:     0.9960359871626436
    TypeScript: 0.9973622321734481
    Difference: 1.326e-3
    Numeric difference exceeds tolerance: abs=1.326e-3, rel=0.132975%

  root.results[3].quality_components.temporal_consistency:
    Type: value
    Python:     0.933512182633808
    TypeScript: 0.953179888138847
    Difference: 1.967e-2
    Numeric difference exceeds tolerance: abs=1.967e-2, rel=2.063378%

  root.results[4].timestamp:
    Type: type
    Python:     1762770618310
    TypeScript: 2025-11-10T10:30:18.310Z
    Type mismatch: Python number, TypeScript string

  root.results[4].filtered_weight:
    Type: value
    Python:     69.95256501595752
    TypeScript: 70.05844543507864
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=0.151132%

  root.results[4].trend:
    Type: value
    Python:     0.000024884229398918055
    TypeScript: 0.00008353141306498834
    Difference: 5.865e-5
    Numeric difference exceeds tolerance: abs=5.865e-5, rel=70.209735%

  root.results[4].trend_weekly:
    Type: value
    Python:     0.0001741896057924264
    TypeScript: 0.0005847198914549184
    Difference: 4.105e-4
    Numeric difference exceeds tolerance: abs=4.105e-4, rel=70.209735%

  root.results[4].confidence:
    Type: value
    Python:     0.999403854042189
    TypeScript: 0.9995769949821603
    Difference: 1.731e-4
    Numeric difference exceeds tolerance: abs=1.731e-4, rel=0.017321%

  root.results[4].innovation:
    Type: value
    Python:     0.18154097012779857
    TypeScript: 0.07566055100667768
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=58.323154%

  root.results[4].normalized_innovation:
    Type: value
    Python:     0.034534728128812156
    TypeScript: 0.02908932827344348
    Difference: 5.445e-3
    Numeric difference exceeds tolerance: abs=5.445e-3, rel=15.767896%

  root.results[4].kalman_confidence_upper:
    Type: value
    Python:     73.66923833988415
    TypeScript: 72.71555619569034
    Difference: 9.537e-1
    Numeric difference exceeds tolerance: abs=9.537e-1, rel=1.294546%

  root.results[4].kalman_confidence_lower:
    Type: value
    Python:     66.23589169203089
    TypeScript: 67.40133467446694
    Difference: 1.165e+0
    Numeric difference exceeds tolerance: abs=1.165e+0, rel=1.729110%

  root.results[4].kalman_variance:
    Type: value
    Python:     3.453415149196944
    TypeScript: 1.7650593985396297
    Difference: 1.688e+0
    Numeric difference exceeds tolerance: abs=1.688e+0, rel=48.889452%

  root.results[4].prediction_error:
    Type: value
    Python:     0.18154097012779857
    TypeScript: 0.07566055100667768
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=58.323154%

  root.results[4].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.results[4].quality_score:
    Type: value
    Python:     0.915025196010972
    TypeScript: 0.9233942774026027
    Difference: 8.369e-3
    Numeric difference exceeds tolerance: abs=8.369e-3, rel=0.906339%

  root.results[4].quality_components.kalman_fit:
    Type: value
    Python:     0.9958574092838801
    TypeScript: 0.9976950710932042
    Difference: 1.838e-3
    Numeric difference exceeds tolerance: abs=1.838e-3, rel=0.184191%

  root.results[4].quality_components.temporal_consistency:
    Type: value
    Python:     0.9311885240584878
    TypeScript: 0.9584551963484061
    Difference: 2.727e-2
    Numeric difference exceeds tolerance: abs=2.727e-2, rel=2.844856%

  root.results[5].timestamp:
    Type: type
    Python:     1762770704710
    TypeScript: 2025-11-10T10:31:44.710Z
    Type mismatch: Python number, TypeScript string

  root.results[5].quality_components.kalman_fit:
    Type: value
    Python:     0.8224108364823537
    TypeScript: 0.8199267787143306
    Difference: 2.484e-3
    Numeric difference exceeds tolerance: abs=2.484e-3, rel=0.302046%

  root.results[5].quality_details.rejection_reason:
    Type: missing
    Python:     Quality score 0.00 below threshold 0.46 (weakest: anomaly_detection=0.00)
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].quality_details.rejectionReason:
    Type: extra
    Python:     undefined
    TypeScript: Quality score 0.00 below threshold 0.46 (weakest: anomaly_detection=0.00)
    Extra key in TypeScript output

  root.results[5].quality_details.components.kalman_fit:
    Type: value
    Python:     0.8224108364823537
    TypeScript: 0.8199267787143306
    Difference: 2.484e-3
    Numeric difference exceeds tolerance: abs=2.484e-3, rel=0.302046%

  root.results[5].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     -9.952567504380454
    TypeScript: -10.05845378821995
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=1.052709%

  root.results[5].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     69.95256750438045
    TypeScript: 70.05845378821995
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=0.151140%

  root.results[5].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     0.9775760385574465
    TypeScript: 0.9927011848081173
    Difference: 1.513e-2
    Numeric difference exceeds tolerance: abs=1.513e-2, rel=1.523635%

  root.results[5].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     0.95565491116167
    TypeScript: 0.9854556423194398
    Difference: 2.980e-2
    Numeric difference exceeds tolerance: abs=2.980e-2, rel=3.024056%

  root.results[5].quality_details.metadata.kalman_fit.p_value:
    Type: value
    Python:     0.32828405313327047
    TypeScript: 0.3208555973831103
    Difference: 7.428e-3
    Numeric difference exceeds tolerance: abs=7.428e-3, rel=2.262813%

  root.results[5].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.8224108364823537
    TypeScript: 0.8199267787143306
    Difference: 2.484e-3
    Numeric difference exceeds tolerance: abs=2.484e-3, rel=0.302046%

  root.results[5].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     9.952565015957518
    TypeScript: 10.058445435078639
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=1.052652%

  root.results[6].timestamp:
    Type: type
    Python:     1762770791110
    TypeScript: 2025-11-10T10:33:11.110Z
    Type mismatch: Python number, TypeScript string

  root.results[6].quality_components.kalman_fit:
    Type: value
    Python:     0.8248052640992843
    TypeScript: 0.8223254066037136
    Difference: 2.480e-3
    Numeric difference exceeds tolerance: abs=2.480e-3, rel=0.300660%

  root.results[6].quality_details.rejection_reason:
    Type: missing
    Python:     Quality score 0.00 below threshold 0.46 (weakest: anomaly_detection=0.00)
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[6].quality_details.rejectionReason:
    Type: extra
    Python:     undefined
    TypeScript: Quality score 0.00 below threshold 0.46 (weakest: anomaly_detection=0.00)
    Extra key in TypeScript output

  root.results[6].quality_details.components.kalman_fit:
    Type: value
    Python:     0.8248052640992843
    TypeScript: 0.8223254066037136
    Difference: 2.480e-3
    Numeric difference exceeds tolerance: abs=2.480e-3, rel=0.300660%

  root.results[6].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     -9.80457625925483
    TypeScript: -9.910462543094326
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=1.068429%

  root.results[6].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     69.95256750438045
    TypeScript: 70.05845378821995
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=0.151140%

  root.results[6].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     0.9630398201306519
    TypeScript: 0.978095452409218
    Difference: 1.506e-2
    Numeric difference exceeds tolerance: abs=1.506e-2, rel=1.539280%

  root.results[6].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     0.9274456951572784
    TypeScript: 0.9566707140235928
    Difference: 2.923e-2
    Numeric difference exceeds tolerance: abs=2.923e-2, rel=3.054867%

  root.results[6].quality_details.metadata.kalman_fit.p_value:
    Type: value
    Python:     0.3355275403554654
    TypeScript: 0.32802711682101293
    Difference: 7.500e-3
    Numeric difference exceeds tolerance: abs=7.500e-3, rel=2.235412%

  root.results[6].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.8248052640992843
    TypeScript: 0.8223254066037136
    Difference: 2.480e-3
    Numeric difference exceeds tolerance: abs=2.480e-3, rel=0.300660%

  root.results[6].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     9.804573770831894
    TypeScript: 9.910454189953015
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=1.068371%

  root.results[7].timestamp:
    Type: type
    Python:     1762770877510
    TypeScript: 2025-11-10T10:34:37.510Z
    Type mismatch: Python number, TypeScript string

  root.results[7].quality_components.kalman_fit:
    Type: value
    Python:     0.8184182995940397
    TypeScript: 0.8159273124883419
    Difference: 2.491e-3
    Numeric difference exceeds tolerance: abs=2.491e-3, rel=0.304366%

  root.results[7].quality_details.rejection_reason:
    Type: missing
    Python:     Quality score 0.00 below threshold 0.46 (weakest: anomaly_detection=0.00)
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[7].quality_details.rejectionReason:
    Type: extra
    Python:     undefined
    TypeScript: Quality score 0.00 below threshold 0.46 (weakest: anomaly_detection=0.00)
    Extra key in TypeScript output

  root.results[7].quality_details.components.kalman_fit:
    Type: value
    Python:     0.8184182995940397
    TypeScript: 0.8159273124883419
    Difference: 2.491e-3
    Numeric difference exceeds tolerance: abs=2.491e-3, rel=0.304366%

  root.results[7].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     -10.200293178711064
    TypeScript: -10.30617946255056
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=1.027406%

  root.results[7].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     69.95256750438045
    TypeScript: 70.05845378821995
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=0.151140%

  root.results[7].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     1.0019085219346757
    TypeScript: 1.0171500290930506
    Difference: 1.524e-2
    Numeric difference exceeds tolerance: abs=1.524e-2, rel=1.498452%

  root.results[7].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     1.0038206863253265
    TypeScript: 1.0345941816839936
    Difference: 3.077e-2
    Numeric difference exceeds tolerance: abs=3.077e-2, rel=2.974451%

  root.results[7].quality_details.metadata.kalman_fit.p_value:
    Type: value
    Python:     0.3163877763595022
    TypeScript: 0.3090820637317666
    Difference: 7.306e-3
    Numeric difference exceeds tolerance: abs=7.306e-3, rel=2.309101%

  root.results[7].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.8184182995940397
    TypeScript: 0.8159273124883419
    Difference: 2.491e-3
    Numeric difference exceeds tolerance: abs=2.491e-3, rel=0.304366%

  root.results[7].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     10.200290690288128
    TypeScript: 10.306171109409249
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=1.027350%

  root.results[8].timestamp:
    Type: type
    Python:     1762770963910
    TypeScript: 2025-11-10T10:36:03.910Z
    Type mismatch: Python number, TypeScript string

  root.results[8].quality_score:
    Type: value
    Python:     0.3108894486276372
    TypeScript: 0.3107463108713578
    Difference: 1.431e-4
    Numeric difference exceeds tolerance: abs=1.431e-4, rel=0.046041%

  root.results[8].quality_components.kalman_fit:
    Type: value
    Python:     0.8206629541503064
    TypeScript: 0.8181758513523245
    Difference: 2.487e-3
    Numeric difference exceeds tolerance: abs=2.487e-3, rel=0.303060%

  root.results[8].quality_details.rejection_reason:
    Type: missing
    Python:     Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20)
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[8].quality_details.rejectionReason:
    Type: extra
    Python:     undefined
    TypeScript: Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20)
    Extra key in TypeScript output

  root.results[8].quality_details.overall:
    Type: value
    Python:     0.3108894486276372
    TypeScript: 0.3107463108713578
    Difference: 1.431e-4
    Numeric difference exceeds tolerance: abs=1.431e-4, rel=0.046041%

  root.results[8].quality_details.components.kalman_fit:
    Type: value
    Python:     0.8206629541503064
    TypeScript: 0.8181758513523245
    Difference: 2.487e-3
    Numeric difference exceeds tolerance: abs=2.487e-3, rel=0.303060%

  root.results[8].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     -10.060870351093804
    TypeScript: -10.1667566349333
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=1.041495%

  root.results[8].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     69.95256750438045
    TypeScript: 70.05845378821995
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=0.151140%

  root.results[8].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     0.9882139234859267
    TypeScript: 1.003389941401735
    Difference: 1.518e-2
    Numeric difference exceeds tolerance: abs=1.518e-2, rel=1.512475%

  root.results[8].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     0.976566758571449
    TypeScript: 1.0067913745061774
    Difference: 3.022e-2
    Numeric difference exceeds tolerance: abs=3.022e-2, rel=3.002073%

  root.results[8].quality_details.metadata.kalman_fit.p_value:
    Type: value
    Python:     0.3230478905699903
    TypeScript: 0.31567275535911143
    Difference: 7.375e-3
    Numeric difference exceeds tolerance: abs=7.375e-3, rel=2.282985%

  root.results[8].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.8206629541503064
    TypeScript: 0.8181758513523245
    Difference: 2.487e-3
    Numeric difference exceeds tolerance: abs=2.487e-3, rel=0.303060%

  root.results[8].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     10.060867862670868
    TypeScript: 10.166748281791989
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=1.041438%

  root.results[9].timestamp:
    Type: type
    Python:     1762771050310
    TypeScript: 2025-11-10T10:37:30.310Z
    Type mismatch: Python number, TypeScript string

  root.results[9].quality_score:
    Type: value
    Python:     0.3111106221270487
    TypeScript: 0.31096901104378943
    Difference: 1.416e-4
    Numeric difference exceeds tolerance: abs=1.416e-4, rel=0.045518%

  root.results[9].quality_components.kalman_fit:
    Type: value
    Python:     0.8245312906625418
    TypeScript: 0.8220509508775709
    Difference: 2.480e-3
    Numeric difference exceeds tolerance: abs=2.480e-3, rel=0.300818%

  root.results[9].quality_details.rejection_reason:
    Type: missing
    Python:     Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20)
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[9].quality_details.rejectionReason:
    Type: extra
    Python:     undefined
    TypeScript: Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20)
    Extra key in TypeScript output

  root.results[9].quality_details.overall:
    Type: value
    Python:     0.3111106221270487
    TypeScript: 0.31096901104378943
    Difference: 1.416e-4
    Numeric difference exceeds tolerance: abs=1.416e-4, rel=0.045518%

  root.results[9].quality_details.components.kalman_fit:
    Type: value
    Python:     0.8245312906625418
    TypeScript: 0.8220509508775709
    Difference: 2.480e-3
    Numeric difference exceeds tolerance: abs=2.480e-3, rel=0.300818%

  root.results[9].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     -9.82148782276871
    TypeScript: -9.927374106608205
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=1.066609%

  root.results[9].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     69.95256750438045
    TypeScript: 70.05845378821995
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=0.151140%

  root.results[9].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     0.964700933130733
    TypeScript: 0.9797645090545691
    Difference: 1.506e-2
    Numeric difference exceeds tolerance: abs=1.506e-2, rel=1.537469%

  root.results[9].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     0.930647890383307
    TypeScript: 0.9599384932029409
    Difference: 2.929e-2
    Numeric difference exceeds tolerance: abs=2.929e-2, rel=3.051300%

  root.results[9].quality_details.metadata.kalman_fit.p_value:
    Type: value
    Python:     0.33469462774573233
    TypeScript: 0.3272023748692141
    Difference: 7.492e-3
    Numeric difference exceeds tolerance: abs=7.492e-3, rel=2.238534%

  root.results[9].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.8245312906625418
    TypeScript: 0.8220509508775709
    Difference: 2.480e-3
    Numeric difference exceeds tolerance: abs=2.480e-3, rel=0.300818%

  root.results[9].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     9.821485334345773
    TypeScript: 9.927365753466894
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=1.066551%

  root.results[10].timestamp:
    Type: type
    Python:     1762771136710
    TypeScript: 2025-11-10T10:38:56.710Z
    Type mismatch: Python number, TypeScript string

  root.results[10].quality_score:
    Type: value
    Python:     0.31120596722967214
    TypeScript: 0.3110650134482761
    Difference: 1.410e-4
    Numeric difference exceeds tolerance: abs=1.410e-4, rel=0.045293%

  root.results[10].quality_components.kalman_fit:
    Type: value
    Python:     0.8262084426231454
    TypeScript: 0.8237310620453411
    Difference: 2.477e-3
    Numeric difference exceeds tolerance: abs=2.477e-3, rel=0.299849%

  root.results[10].quality_details.rejection_reason:
    Type: missing
    Python:     Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20)
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[10].quality_details.rejectionReason:
    Type: extra
    Python:     undefined
    TypeScript: Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20)
    Extra key in TypeScript output

  root.results[10].quality_details.overall:
    Type: value
    Python:     0.31120596722967214
    TypeScript: 0.3110650134482761
    Difference: 1.410e-4
    Numeric difference exceeds tolerance: abs=1.410e-4, rel=0.045293%

  root.results[10].quality_details.components.kalman_fit:
    Type: value
    Python:     0.8262084426231454
    TypeScript: 0.8237310620453411
    Difference: 2.477e-3
    Numeric difference exceeds tolerance: abs=2.477e-3, rel=0.299849%

  root.results[10].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     -9.718050197194025
    TypeScript: -9.82393648103352
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=1.077840%

  root.results[10].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     69.95256750438045
    TypeScript: 70.05845378821995
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=0.151140%

  root.results[10].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     0.9545409272626408
    TypeScript: 0.969555916797379
    Difference: 1.501e-2
    Numeric difference exceeds tolerance: abs=1.501e-2, rel=1.548646%

  root.results[10].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     0.9111483818194221
    TypeScript: 0.940038675796806
    Difference: 2.889e-2
    Numeric difference exceeds tolerance: abs=2.889e-2, rel=3.073309%

  root.results[10].quality_details.metadata.kalman_fit.p_value:
    Type: value
    Python:     0.3398099027482664
    TypeScript: 0.33226789543047386
    Difference: 7.542e-3
    Numeric difference exceeds tolerance: abs=7.542e-3, rel=2.219478%

  root.results[10].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.8262084426231454
    TypeScript: 0.8237310620453411
    Difference: 2.477e-3
    Numeric difference exceeds tolerance: abs=2.477e-3, rel=0.299849%

  root.results[10].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     9.718047708771088
    TypeScript: 9.82392812789221
    Difference: 1.059e-1
    Numeric difference exceeds tolerance: abs=1.059e-1, rel=1.077781%

  root.finalState.kalman_params.transition_covariance[0][0]:
    Type: value
    Python:     0.19607272887128602
    TypeScript: 0.8999999999999999
    Difference: 7.039e-1
    Numeric difference exceeds tolerance: abs=7.039e-1, rel=78.214141%

  root.finalState.kalman_params.transition_covariance[1][1]:
    Type: value
    Python:     0.0013071515258085737
    TypeScript: 0.006
    Difference: 4.693e-3
    Numeric difference exceeds tolerance: abs=4.693e-3, rel=78.214141%

  root.finalState.last_state[0][0]:
    Type: type
    Python:     69.92231645736025
    TypeScript: [
  70.01715947165995
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[0][1]:
    Type: type
    Python:     0.0000087318470573641
    TypeScript: [
  0.00003793788509685788
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][0]:
    Type: type
    Python:     69.95256501595752
    TypeScript: [
  70.05844543507864
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][1]:
    Type: type
    Python:     0.000024884229398918055
    TypeScript: [
  0.00008353141306498834
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_covariance[0][0][0]:
    Type: value
    Python:     3.8324045525604276
    TypeScript: 1.827704557178785
    Difference: 2.005e+0
    Numeric difference exceeds tolerance: abs=2.005e+0, rel=52.309196%

  root.finalState.last_covariance[0][0][1]:
    Type: value
    Python:     0.0011799500893092253
    TypeScript: 0.0011130777135867397
    Difference: 6.687e-5
    Numeric difference exceeds tolerance: abs=6.687e-5, rel=5.667390%

  root.finalState.last_covariance[0][1][0]:
    Type: value
    Python:     0.0011799500893092255
    TypeScript: 0.0011130777135867397
    Difference: 6.687e-5
    Numeric difference exceeds tolerance: abs=6.687e-5, rel=5.667390%

  root.finalState.last_covariance[0][1][1]:
    Type: value
    Python:     0.009714504341111076
    TypeScript: 0.018999540025122084
    Difference: 9.285e-3
    Numeric difference exceeds tolerance: abs=9.285e-3, rel=48.869792%

  root.finalState.last_covariance[1][0][0]:
    Type: value
    Python:     3.453415149196944
    TypeScript: 1.7650593985396297
    Difference: 1.688e+0
    Numeric difference exceeds tolerance: abs=1.688e+0, rel=48.889452%

  root.finalState.last_covariance[1][0][1]:
    Type: value
    Python:     0.001844137199694154
    TypeScript: 0.0019493957263792606
    Difference: 1.053e-4
    Numeric difference exceeds tolerance: abs=1.053e-4, rel=5.399546%

  root.finalState.last_covariance[1][1][0]:
    Type: value
    Python:     0.001844137199694154
    TypeScript: 0.0019493957263792606
    Difference: 1.053e-4
    Numeric difference exceeds tolerance: abs=1.053e-4, rel=5.399546%

  root.finalState.last_covariance[1][1][1]:
    Type: value
    Python:     0.011021491787102742
    TypeScript: 0.024998365306891923
    Difference: 1.398e-2
    Numeric difference exceeds tolerance: abs=1.398e-2, rel=55.911150%

  root.finalState.reset_parameters.quality_acceptance_threshold:
    Type: missing
    Python:     0.25
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_safety_weight:
    Type: missing
    Python:     0.5
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_plausibility_weight:
    Type: missing
    Python:     0.05
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_consistency_weight:
    Type: missing
    Python:     0.05
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_reliability_weight:
    Type: missing
    Python:     0.4
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.enabled:
    Type: extra
    Python:     undefined
    TypeScript: true
    Extra key in TypeScript output

  root.finalState.reset_timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.finalState.reset_events:
    Type: missing
    Python:     [
  {
    "timestamp": 1762770272710,
    "type": "initial",
    "source": "withings",
    "weight":
    TypeScript: []
    Array length mismatch: Python 1, TypeScript 0

  root.finalState.last_timestamp:
    Type: type
    Python:     1762770618310
    TypeScript: 2025-11-10T10:30:18.310Z
    Type mismatch: Python number, TypeScript string

  root.finalState.last_accepted_timestamp:
    Type: type
    Python:     1762770618310
    TypeScript: 2025-11-10T10:30:18.310Z
    Type mismatch: Python number, TypeScript string

  root.finalState.measurement_history[0].timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

  root.finalState.measurement_history[1].timestamp:
    Type: value
    Python:     2025-11-10T11:25:59.110000
    TypeScript: 2025-11-10T10:25:59.110Z
    Value mismatch: 2025-11-10T11:25:59.110000 !== 2025-11-10T10:25:59.110Z

  root.finalState.measurement_history[2].timestamp:
    Type: value
    Python:     2025-11-10T11:27:25.510000
    TypeScript: 2025-11-10T10:27:25.510Z
    Value mismatch: 2025-11-10T11:27:25.510000 !== 2025-11-10T10:27:25.510Z

  root.finalState.measurement_history[2].quality_score:
    Type: value
    Python:     0.9477124619928798
    TypeScript: 0.9527858133476765
    Difference: 5.073e-3
    Numeric difference exceeds tolerance: abs=5.073e-3, rel=0.532476%

  root.finalState.measurement_history[3].timestamp:
    Type: value
    Python:     2025-11-10T11:28:51.910000
    TypeScript: 2025-11-10T10:28:51.910Z
    Value mismatch: 2025-11-10T11:28:51.910000 !== 2025-11-10T10:28:51.910Z

  root.finalState.measurement_history[3].quality_score:
    Type: value
    Python:     0.9181140831089427
    TypeScript: 0.9241937239986877
    Difference: 6.080e-3
    Numeric difference exceeds tolerance: abs=6.080e-3, rel=0.657832%

  root.finalState.measurement_history[4].timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.finalState.measurement_history[4].quality_score:
    Type: value
    Python:     0.915025196010972
    TypeScript: 0.9233942774026027
    Difference: 8.369e-3
    Numeric difference exceeds tolerance: abs=8.369e-3, rel=0.906339%

  root.finalState.temporal_baseline.last_timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

```

### Test 4: Quality Rejection
**Description**: Mix of good and bad measurements - verify rejection logic

**Comparison**: ✗ Found 76 difference(s): 53 numeric, 23 structural

**Differences**:
```
Found 76 difference(s):

  root.results[0].timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.results[0].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

  root.results[0].reset_reason:
    Type: value
    Python:     initial_measurement
    TypeScript: initial reset triggered
    Value mismatch: initial_measurement !== initial reset triggered

  root.results[0].gap_days:
    Type: value
    Python:     null
    TypeScript: 0
    Python value is null, TypeScript value is 0

  root.results[0].reset_event.gap_days:
    Type: missing
    Python:     null
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_event.reason:
    Type: value
    Python:     initial_measurement
    TypeScript: initial reset triggered
    Value mismatch: initial_measurement !== initial reset triggered

  root.results[1].timestamp:
    Type: type
    Python:     1762770359110
    TypeScript: 2025-11-10T10:25:59.110Z
    Type mismatch: Python number, TypeScript string

  root.results[1].filtered_weight:
    Type: value
    Python:     70.01165296435182
    TypeScript: 70.09517830694098
    Difference: 8.353e-2
    Numeric difference exceeds tolerance: abs=8.353e-2, rel=0.119160%

  root.results[1].trend:
    Type: value
    Python:     2.742369346345332e-7
    TypeScript: 0.000002096433861180469
    Difference: 1.822e-6
    Numeric difference exceeds tolerance: abs=1.822e-6, rel=86.918885%

  root.results[1].trend_weekly:
    Type: value
    Python:     0.000001919658542441732
    TypeScript: 0.000014675037028263284
    Difference: 1.276e-5
    Numeric difference exceeds tolerance: abs=1.276e-5, rel=86.918885%

  root.results[1].confidence:
    Type: value
    Python:     0.9997559901329914
    TypeScript: 0.9992558058555713
    Difference: 5.002e-4
    Numeric difference exceeds tolerance: abs=5.002e-4, rel=0.050031%

  root.results[1].innovation:
    Type: value
    Python:     0.188347035648178
    TypeScript: 0.10482169305902289
    Difference: 8.353e-2
    Numeric difference exceeds tolerance: abs=8.353e-2, rel=44.346513%

  root.results[1].normalized_innovation:
    Type: value
    Python:     0.022092516482277752
    TypeScript: 0.03858681625509329
    Difference: 1.649e-2
    Numeric difference exceeds tolerance: abs=1.649e-2, rel=42.745946%

  root.results[1].kalman_confidence_upper:
    Type: value
    Python:     74.01247863871501
    TypeScript: 73.18027647298008
    Difference: 8.322e-1
    Numeric difference exceeds tolerance: abs=8.322e-1, rel=1.124408%

  root.results[1].kalman_confidence_lower:
    Type: value
    Python:     66.01082728998864
    TypeScript: 67.01008014090188
    Difference: 9.993e-1
    Numeric difference exceeds tolerance: abs=9.993e-1, rel=1.491198%

  root.results[1].kalman_variance:
    Type: value
    Python:     4.0016515191608955
    TypeScript: 2.3794576735244517
    Difference: 1.622e+0
    Numeric difference exceeds tolerance: abs=1.622e+0, rel=40.538109%

  root.results[1].prediction_error:
    Type: value
    Python:     0.188347035648178
    TypeScript: 0.10482169305902289
    Difference: 8.353e-2
    Numeric difference exceeds tolerance: abs=8.353e-2, rel=44.346513%

  root.results[1].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:25:59.110000
    TypeScript: 2025-11-10T10:25:59.110Z
    Value mismatch: 2025-11-10T11:25:59.110000 !== 2025-11-10T10:25:59.110Z

  root.results[2].quality_score:
    Type: extra
    Python:     undefined
    TypeScript: 0
    Extra key in TypeScript output

  root.results[2].timestamp:
    Type: type
    Python:     1762770445510
    TypeScript: 2025-11-10T10:27:25.510Z
    Type mismatch: Python number, TypeScript string

  root.results[2].metadata.timestamp:
    Type: value
    Python:     2025-11-10T11:27:25.510000
    TypeScript: 2025-11-10T10:27:25.510Z
    Value mismatch: 2025-11-10T11:27:25.510000 !== 2025-11-10T10:27:25.510Z

  root.results[3].timestamp:
    Type: type
    Python:     1762770531910
    TypeScript: 2025-11-10T10:28:51.910Z
    Type mismatch: Python number, TypeScript string

  root.results[3].quality_components.kalman_fit:
    Type: value
    Python:     0.0296130733009591
    TypeScript: 0.028998574394115302
    Difference: 6.145e-4
    Numeric difference exceeds tolerance: abs=6.145e-4, rel=2.075093%

  root.results[3].quality_details.rejection_reason:
    Type: missing
    Python:     Quality score 0.00 below threshold 0.46 (weakest: anomaly_detection=0.00)
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[3].quality_details.rejectionReason:
    Type: extra
    Python:     undefined
    TypeScript: Quality score 0.00 below threshold 0.46 (weakest: anomaly_detection=0.00)
    Extra key in TypeScript output

  root.results[3].quality_details.components.kalman_fit:
    Type: value
    Python:     0.0296130733009591
    TypeScript: 0.028998574394115302
    Difference: 6.145e-4
    Numeric difference exceeds tolerance: abs=6.145e-4, rel=2.075093%

  root.results[3].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     179.98834700822448
    TypeScript: 179.90482148341562
    Difference: 8.353e-2
    Numeric difference exceeds tolerance: abs=8.353e-2, rel=0.046406%

  root.results[3].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     70.01165299177552
    TypeScript: 70.09517851658437
    Difference: 8.353e-2
    Numeric difference exceeds tolerance: abs=8.353e-2, rel=0.119160%

  root.results[3].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     17.59769674792355
    TypeScript: 17.70254304513804
    Difference: 1.048e-1
    Numeric difference exceeds tolerance: abs=1.048e-1, rel=0.592267%

  root.results[3].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     309.6789308318791
    TypeScript: 313.38003026496517
    Difference: 3.701e+0
    Numeric difference exceeds tolerance: abs=3.701e+0, rel=1.181026%

  root.results[3].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.0296130733009591
    TypeScript: 0.028998574394115302
    Difference: 6.145e-4
    Numeric difference exceeds tolerance: abs=6.145e-4, rel=2.075093%

  root.results[3].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     179.9883470356482
    TypeScript: 179.90482169305903
    Difference: 8.353e-2
    Numeric difference exceeds tolerance: abs=8.353e-2, rel=0.046406%

  root.results[4].timestamp:
    Type: type
    Python:     1762770618310
    TypeScript: 2025-11-10T10:30:18.310Z
    Type mismatch: Python number, TypeScript string

  root.results[4].filtered_weight:
    Type: value
    Python:     70.01914101255421
    TypeScript: 70.09708831378938
    Difference: 7.795e-2
    Numeric difference exceeds tolerance: abs=7.795e-2, rel=0.111199%

  root.results[4].trend:
    Type: value
    Python:     0.0000012921698287282735
    TypeScript: 0.0000025345906574496616
    Difference: 1.242e-6
    Numeric difference exceeds tolerance: abs=1.242e-6, rel=49.018599%

  root.results[4].trend_weekly:
    Type: value
    Python:     0.000009045188801097915
    TypeScript: 0.000017742134602147632
    Difference: 8.697e-6
    Numeric difference exceeds tolerance: abs=8.697e-6, rel=49.018599%

  root.results[4].confidence:
    Type: value
    Python:     0.9999368043389365
    TypeScript: 0.9999993927437567
    Difference: 6.259e-5
    Numeric difference exceeds tolerance: abs=6.259e-5, rel=0.006259%

  root.results[4].innovation:
    Type: value
    Python:     0.08085898744577946
    TypeScript: 0.002911686210609332
    Difference: 7.795e-2
    Numeric difference exceeds tolerance: abs=7.795e-2, rel=96.399057%

  root.results[4].normalized_innovation:
    Type: value
    Python:     0.011242567143979145
    TypeScript: 0.0011020493887911418
    Difference: 1.014e-2
    Numeric difference exceeds tolerance: abs=1.014e-2, rel=90.197529%

  root.results[4].kalman_confidence_upper:
    Type: value
    Python:     74.03995688254918
    TypeScript: 72.91169869987782
    Difference: 1.128e+0
    Numeric difference exceeds tolerance: abs=1.128e+0, rel=1.523850%

  root.results[4].kalman_confidence_lower:
    Type: value
    Python:     65.99832514255925
    TypeScript: 67.28247792770095
    Difference: 1.284e+0
    Numeric difference exceeds tolerance: abs=1.284e+0, rel=1.908599%

  root.results[4].kalman_variance:
    Type: value
    Python:     4.041740065100835
    TypeScript: 1.9805079063692226
    Difference: 2.061e+0
    Numeric difference exceeds tolerance: abs=2.061e+0, rel=50.998632%

  root.results[4].prediction_error:
    Type: value
    Python:     0.08085898744577946
    TypeScript: 0.002911686210609332
    Difference: 7.795e-2
    Numeric difference exceeds tolerance: abs=7.795e-2, rel=96.399057%

  root.results[4].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.results[4].quality_score:
    Type: value
    Python:     0.9630516768743426
    TypeScript: 0.9724176355122152
    Difference: 9.366e-3
    Numeric difference exceeds tolerance: abs=9.366e-3, rel=0.963162%

  root.results[4].quality_components.kalman_fit:
    Type: value
    Python:     0.9982739308009785
    TypeScript: 0.999905118197791
    Difference: 1.631e-3
    Numeric difference exceeds tolerance: abs=1.631e-3, rel=0.163134%

  root.results[4].quality_components.temporal_consistency:
    Type: value
    Python:     0.9680018768103804
    TypeScript: 0.9981060190295834
    Difference: 3.010e-2
    Numeric difference exceeds tolerance: abs=3.010e-2, rel=3.016127%

  root.finalState.kalman_params.transition_covariance[0][0]:
    Type: value
    Python:     0.41430814635138935
    TypeScript: 0.8999999999999999
    Difference: 4.857e-1
    Numeric difference exceeds tolerance: abs=4.857e-1, rel=53.965762%

  root.finalState.kalman_params.transition_covariance[1][1]:
    Type: value
    Python:     0.002762054309009263
    TypeScript: 0.006
    Difference: 3.238e-3
    Numeric difference exceeds tolerance: abs=3.238e-3, rel=53.965762%

  root.finalState.last_state[0][0]:
    Type: type
    Python:     70.01165296435182
    TypeScript: [
  70.09517830694098
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[0][1]:
    Type: type
    Python:     2.742369346345332e-7
    TypeScript: [
  0.000002096433861180469
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][0]:
    Type: type
    Python:     70.01914101255421
    TypeScript: [
  70.09708831378938
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][1]:
    Type: type
    Python:     0.0000012921698287282735
    TypeScript: [
  0.0000025345906574496616
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_covariance[0][0][0]:
    Type: value
    Python:     4.0016515191608955
    TypeScript: 2.3794576735244517
    Difference: 1.622e+0
    Numeric difference exceeds tolerance: abs=1.622e+0, rel=40.538109%

  root.finalState.last_covariance[0][0][1]:
    Type: value
    Python:     0.00009417351782408612
    TypeScript: 0.00005241084652951098
    Difference: 4.176e-5
    Numeric difference exceeds tolerance: abs=4.176e-5, rel=44.346513%

  root.finalState.last_covariance[0][1][0]:
    Type: value
    Python:     0.00009417351782408611
    TypeScript: 0.00005241084652951098
    Difference: 4.176e-5
    Numeric difference exceeds tolerance: abs=4.176e-5, rel=44.346513%

  root.finalState.last_covariance[0][1][1]:
    Type: value
    Python:     0.0050614817335710915
    TypeScript: 0.006999998951783069
    Difference: 1.939e-3
    Numeric difference exceeds tolerance: abs=1.939e-3, rel=27.693107%

  root.finalState.last_covariance[1][0][0]:
    Type: value
    Python:     4.041740065100835
    TypeScript: 1.9805079063692226
    Difference: 2.061e+0
    Numeric difference exceeds tolerance: abs=2.061e+0, rel=50.998632%

  root.finalState.last_covariance[1][0][1]:
    Type: value
    Python:     0.0005494402704312545
    TypeScript: 0.00045437965714992517
    Difference: 9.506e-5
    Numeric difference exceeds tolerance: abs=9.506e-5, rel=17.301355%

  root.finalState.last_covariance[1][1][0]:
    Type: value
    Python:     0.0005494402704312545
    TypeScript: 0.00045437965714992517
    Difference: 9.506e-5
    Numeric difference exceeds tolerance: abs=9.506e-5, rel=17.301355%

  root.finalState.last_covariance[1][1][1]:
    Type: value
    Python:     0.007823529125682899
    TypeScript: 0.012999930575756099
    Difference: 5.176e-3
    Numeric difference exceeds tolerance: abs=5.176e-3, rel=39.818685%

  root.finalState.reset_parameters.quality_acceptance_threshold:
    Type: missing
    Python:     0.25
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_safety_weight:
    Type: missing
    Python:     0.5
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_plausibility_weight:
    Type: missing
    Python:     0.05
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_consistency_weight:
    Type: missing
    Python:     0.05
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_reliability_weight:
    Type: missing
    Python:     0.4
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.enabled:
    Type: extra
    Python:     undefined
    TypeScript: true
    Extra key in TypeScript output

  root.finalState.reset_timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.finalState.reset_events:
    Type: missing
    Python:     [
  {
    "timestamp": 1762770272710,
    "type": "initial",
    "source": "withings",
    "weight":
    TypeScript: []
    Array length mismatch: Python 1, TypeScript 0

  root.finalState.last_timestamp:
    Type: type
    Python:     1762770618310
    TypeScript: 2025-11-10T10:30:18.310Z
    Type mismatch: Python number, TypeScript string

  root.finalState.last_accepted_timestamp:
    Type: type
    Python:     1762770618310
    TypeScript: 2025-11-10T10:30:18.310Z
    Type mismatch: Python number, TypeScript string

  root.finalState.measurement_history[0].timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

  root.finalState.measurement_history[1].timestamp:
    Type: value
    Python:     2025-11-10T11:25:59.110000
    TypeScript: 2025-11-10T10:25:59.110Z
    Value mismatch: 2025-11-10T11:25:59.110000 !== 2025-11-10T10:25:59.110Z

  root.finalState.measurement_history[2].timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.finalState.measurement_history[2].quality_score:
    Type: value
    Python:     0.9630516768743426
    TypeScript: 0.9724176355122152
    Difference: 9.366e-3
    Numeric difference exceeds tolerance: abs=9.366e-3, rel=0.963162%

  root.finalState.temporal_baseline.last_timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

```

### Test 5: State Persistence
**Description**: Process in batches - verify state persistence works correctly

**Comparison**: ✗ Found 159 difference(s): 133 numeric, 26 structural

**Differences**:
```
Found 159 difference(s):

  root.results[0].timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.results[0].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

  root.results[0].reset_reason:
    Type: value
    Python:     initial_measurement
    TypeScript: initial reset triggered
    Value mismatch: initial_measurement !== initial reset triggered

  root.results[0].gap_days:
    Type: value
    Python:     null
    TypeScript: 0
    Python value is null, TypeScript value is 0

  root.results[0].reset_event.gap_days:
    Type: missing
    Python:     null
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_event.reason:
    Type: value
    Python:     initial_measurement
    TypeScript: initial reset triggered
    Value mismatch: initial_measurement !== initial reset triggered

  root.results[1].timestamp:
    Type: type
    Python:     1762770359110
    TypeScript: 2025-11-10T10:25:59.110Z
    Type mismatch: Python number, TypeScript string

  root.results[1].filtered_weight:
    Type: value
    Python:     70.00582648217592
    TypeScript: 70.04758915347048
    Difference: 4.176e-2
    Numeric difference exceeds tolerance: abs=4.176e-2, rel=0.059620%

  root.results[1].trend:
    Type: value
    Python:     1.3711846731725685e-7
    TypeScript: 0.00000104821693059016
    Difference: 9.111e-7
    Numeric difference exceeds tolerance: abs=9.111e-7, rel=86.918885%

  root.results[1].trend_weekly:
    Type: value
    Python:     9.598292712207979e-7
    TypeScript: 0.00000733751851413112
    Difference: 6.378e-6
    Numeric difference exceeds tolerance: abs=6.378e-6, rel=86.918885%

  root.results[1].confidence:
    Type: value
    Python:     0.9999389919505018
    TypeScript: 0.999813899520255
    Difference: 1.251e-4
    Numeric difference exceeds tolerance: abs=1.251e-4, rel=0.012510%

  root.results[1].innovation:
    Type: value
    Python:     0.09417351782407479
    TypeScript: 0.052410846529511446
    Difference: 4.176e-2
    Numeric difference exceeds tolerance: abs=4.176e-2, rel=44.346513%

  root.results[1].normalized_innovation:
    Type: value
    Python:     0.011046258241137209
    TypeScript: 0.019293408127546645
    Difference: 8.247e-3
    Numeric difference exceeds tolerance: abs=8.247e-3, rel=42.745946%

  root.results[1].kalman_confidence_upper:
    Type: value
    Python:     74.0066521565391
    TypeScript: 73.13268731950959
    Difference: 8.740e-1
    Numeric difference exceeds tolerance: abs=8.740e-1, rel=1.180927%

  root.results[1].kalman_confidence_lower:
    Type: value
    Python:     66.00500080781273
    TypeScript: 66.96249098743138
    Difference: 9.575e-1
    Numeric difference exceeds tolerance: abs=9.575e-1, rel=1.429890%

  root.results[1].kalman_variance:
    Type: value
    Python:     4.0016515191608955
    TypeScript: 2.3794576735244517
    Difference: 1.622e+0
    Numeric difference exceeds tolerance: abs=1.622e+0, rel=40.538109%

  root.results[1].prediction_error:
    Type: value
    Python:     0.09417351782407479
    TypeScript: 0.052410846529511446
    Difference: 4.176e-2
    Numeric difference exceeds tolerance: abs=4.176e-2, rel=44.346513%

  root.results[1].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:25:59.110000
    TypeScript: 2025-11-10T10:25:59.110Z
    Value mismatch: 2025-11-10T11:25:59.110000 !== 2025-11-10T10:25:59.110Z

  root.results[2].timestamp:
    Type: type
    Python:     1762770445510
    TypeScript: 2025-11-10T10:27:25.510Z
    Type mismatch: Python number, TypeScript string

  root.results[2].filtered_weight:
    Type: value
    Python:     70.02228404507574
    TypeScript: 70.10795939408577
    Difference: 8.568e-2
    Numeric difference exceeds tolerance: abs=8.568e-2, rel=0.122205%

  root.results[2].trend:
    Type: value
    Python:     0.000002374382648296539
    TypeScript: 0.000014898685043223674
    Difference: 1.252e-5
    Numeric difference exceeds tolerance: abs=1.252e-5, rel=84.063140%

  root.results[2].trend_weekly:
    Type: value
    Python:     0.00001662067853807577
    TypeScript: 0.00010429079530256571
    Difference: 8.767e-5
    Numeric difference exceeds tolerance: abs=8.767e-5, rel=84.063140%

  root.results[2].confidence:
    Type: value
    Python:     0.9996947673850899
    TypeScript: 0.9993933891668693
    Difference: 3.014e-4
    Numeric difference exceeds tolerance: abs=3.014e-4, rel=0.030147%

  root.results[2].innovation:
    Type: value
    Python:     0.17771595492426684
    TypeScript: 0.09204060591423513
    Difference: 8.568e-2
    Numeric difference exceeds tolerance: abs=8.568e-2, rel=48.209149%

  root.results[2].normalized_innovation:
    Type: value
    Python:     0.024709480280496986
    TypeScript: 0.03483661567725119
    Difference: 1.013e-2
    Numeric difference exceeds tolerance: abs=1.013e-2, rel=29.070377%

  root.results[2].kalman_confidence_upper:
    Type: value
    Python:     74.0430999150707
    TypeScript: 72.9225697801742
    Difference: 1.121e+0
    Numeric difference exceeds tolerance: abs=1.121e+0, rel=1.513348%

  root.results[2].kalman_confidence_lower:
    Type: value
    Python:     66.00146817508077
    TypeScript: 67.29334900799734
    Difference: 1.292e+0
    Numeric difference exceeds tolerance: abs=1.292e+0, rel=1.919775%

  root.results[2].kalman_variance:
    Type: value
    Python:     4.041740065100835
    TypeScript: 1.9805079063692226
    Difference: 2.061e+0
    Numeric difference exceeds tolerance: abs=2.061e+0, rel=50.998632%

  root.results[2].prediction_error:
    Type: value
    Python:     0.17771595492426684
    TypeScript: 0.09204060591423513
    Difference: 8.568e-2
    Numeric difference exceeds tolerance: abs=8.568e-2, rel=48.209149%

  root.results[2].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:27:25.510000
    TypeScript: 2025-11-10T10:27:25.510Z
    Value mismatch: 2025-11-10T11:27:25.510000 !== 2025-11-10T10:27:25.510Z

  root.results[2].quality_score:
    Type: value
    Python:     0.9412561190105361
    TypeScript: 0.9450858309686435
    Difference: 3.830e-3
    Numeric difference exceeds tolerance: abs=3.830e-3, rel=0.405224%

  root.results[2].quality_components.kalman_fit:
    Type: value
    Python:     0.9962102795524286
    TypeScript: 0.9970050658774887
    Difference: 7.948e-4
    Numeric difference exceeds tolerance: abs=7.948e-4, rel=0.079717%

  root.results[2].quality_components.temporal_consistency:
    Type: value
    Python:     0.9358715060747236
    TypeScript: 0.9476524034852276
    Difference: 1.178e-2
    Numeric difference exceeds tolerance: abs=1.178e-2, rel=1.243167%

  root.results[3].timestamp:
    Type: type
    Python:     1762770531910
    TypeScript: 2025-11-10T10:28:51.910Z
    Type mismatch: Python number, TypeScript string

  root.results[3].filtered_weight:
    Type: value
    Python:     70.05394776769158
    TypeScript: 70.17815903746495
    Difference: 1.242e-1
    Numeric difference exceeds tolerance: abs=1.242e-1, rel=0.176994%

  root.results[3].trend:
    Type: value
    Python:     0.000012123179167744857
    TypeScript: 0.0000576495770847014
    Difference: 4.553e-5
    Numeric difference exceeds tolerance: abs=4.553e-5, rel=78.970914%

  root.results[3].trend_weekly:
    Type: value
    Python:     0.000084862254174214
    TypeScript: 0.00040354703959290977
    Difference: 3.187e-4
    Numeric difference exceeds tolerance: abs=3.187e-4, rel=78.970914%

  root.results[3].confidence:
    Type: value
    Python:     0.9991919367647333
    TypeScript: 0.998913459596839
    Difference: 2.785e-4
    Numeric difference exceeds tolerance: abs=2.785e-4, rel=0.027870%

  root.results[3].innovation:
    Type: value
    Python:     0.2460522323084149
    TypeScript: 0.12184096253504606
    Difference: 1.242e-1
    Numeric difference exceeds tolerance: abs=1.242e-1, rel=50.481667%

  root.results[3].normalized_innovation:
    Type: value
    Python:     0.04020920029915558
    TypeScript: 0.04662898489379019
    Difference: 6.420e-3
    Numeric difference exceeds tolerance: abs=6.420e-3, rel=13.767798%

  root.results[3].kalman_confidence_upper:
    Type: value
    Python:     73.96925340202483
    TypeScript: 72.88201151656123
    Difference: 1.087e+0
    Numeric difference exceeds tolerance: abs=1.087e+0, rel=1.469857%

  root.results[3].kalman_confidence_lower:
    Type: value
    Python:     66.13864213335833
    TypeScript: 67.47430655836867
    Difference: 1.336e+0
    Numeric difference exceeds tolerance: abs=1.336e+0, rel=1.979516%

  root.results[3].kalman_variance:
    Type: value
    Python:     3.8324045525604276
    TypeScript: 1.827704557178785
    Difference: 2.005e+0
    Numeric difference exceeds tolerance: abs=2.005e+0, rel=52.309196%

  root.results[3].prediction_error:
    Type: value
    Python:     0.2460522323084149
    TypeScript: 0.12184096253504606
    Difference: 1.242e-1
    Numeric difference exceeds tolerance: abs=1.242e-1, rel=50.481667%

  root.results[3].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:28:51.910000
    TypeScript: 2025-11-10T10:28:51.910Z
    Value mismatch: 2025-11-10T11:28:51.910000 !== 2025-11-10T10:28:51.910Z

  root.results[3].quality_score:
    Type: value
    Python:     0.9342633391947578
    TypeScript: 0.941435178044982
    Difference: 7.172e-3
    Numeric difference exceeds tolerance: abs=7.172e-3, rel=0.761798%

  root.results[3].quality_components.kalman_fit:
    Type: value
    Python:     0.9945801944395222
    TypeScript: 0.9962205319228531
    Difference: 1.640e-3
    Numeric difference exceeds tolerance: abs=1.640e-3, rel=0.164656%

  root.results[3].quality_components.temporal_consistency:
    Type: value
    Python:     0.9150507210816894
    TypeScript: 0.9364497410261658
    Difference: 2.140e-2
    Numeric difference exceeds tolerance: abs=2.140e-2, rel=2.285122%

  root.results[4].timestamp:
    Type: type
    Python:     1762770618310
    TypeScript: 2025-11-10T10:30:18.310Z
    Type mismatch: Python number, TypeScript string

  root.results[4].filtered_weight:
    Type: value
    Python:     70.10337203509553
    TypeScript: 70.25647526250482
    Difference: 1.531e-1
    Numeric difference exceeds tolerance: abs=1.531e-1, rel=0.217920%

  root.results[4].trend:
    Type: value
    Python:     0.00003851528393043969
    TypeScript: 0.0001441384943082579
    Difference: 1.056e-4
    Numeric difference exceeds tolerance: abs=1.056e-4, rel=73.278974%

  root.results[4].trend_weekly:
    Type: value
    Python:     0.00026960698751307785
    TypeScript: 0.0010089694601578053
    Difference: 7.394e-4
    Numeric difference exceeds tolerance: abs=7.394e-4, rel=73.278974%

  root.results[4].confidence:
    Type: value
    Python:     0.998409215670912
    TypeScript: 0.9984786772772689
    Difference: 6.946e-5
    Numeric difference exceeds tolerance: abs=6.946e-5, rel=0.006957%

  root.results[4].innovation:
    Type: value
    Python:     0.29662796490447363
    TypeScript: 0.14352473749518424
    Difference: 1.531e-1
    Numeric difference exceeds tolerance: abs=1.531e-1, rel=51.614563%

  root.results[4].normalized_innovation:
    Type: value
    Python:     0.056427847202575994
    TypeScript: 0.05518117630399935
    Difference: 1.247e-3
    Numeric difference exceeds tolerance: abs=1.247e-3, rel=2.209319%

  root.results[4].kalman_confidence_upper:
    Type: value
    Python:     73.82004535902216
    TypeScript: 72.91358602311652
    Difference: 9.065e-1
    Numeric difference exceeds tolerance: abs=9.065e-1, rel=1.227931%

  root.results[4].kalman_confidence_lower:
    Type: value
    Python:     66.3866987111689
    TypeScript: 67.59936450189312
    Difference: 1.213e+0
    Numeric difference exceeds tolerance: abs=1.213e+0, rel=1.793901%

  root.results[4].kalman_variance:
    Type: value
    Python:     3.453415149196944
    TypeScript: 1.7650593985396297
    Difference: 1.688e+0
    Numeric difference exceeds tolerance: abs=1.688e+0, rel=48.889452%

  root.results[4].prediction_error:
    Type: value
    Python:     0.29662796490447363
    TypeScript: 0.14352473749518424
    Difference: 1.531e-1
    Numeric difference exceeds tolerance: abs=1.531e-1, rel=51.614563%

  root.results[4].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.results[4].quality_score:
    Type: value
    Python:     0.9291662865760396
    TypeScript: 0.938828596578805
    Difference: 9.662e-3
    Numeric difference exceeds tolerance: abs=9.662e-3, rel=1.029188%

  root.results[4].quality_components.kalman_fit:
    Type: value
    Python:     0.993240126989953
    TypeScript: 0.9956321713394507
    Difference: 2.392e-3
    Numeric difference exceeds tolerance: abs=2.392e-3, rel=0.240254%

  root.results[4].quality_components.temporal_consistency:
    Type: value
    Python:     0.9004151054030848
    TypeScript: 0.9285891671189228
    Difference: 2.817e-2
    Numeric difference exceeds tolerance: abs=2.817e-2, rel=3.034072%

  root.results[5].timestamp:
    Type: type
    Python:     1762770704710
    TypeScript: 2025-11-10T10:31:44.710Z
    Type mismatch: Python number, TypeScript string

  root.results[5].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:31:44.710000
    TypeScript: 2025-11-10T10:31:44.710Z
    Value mismatch: 2025-11-10T11:31:44.710000 !== 2025-11-10T10:31:44.710Z

  root.results[5].reset_reason:
    Type: value
    Python:     initial_measurement
    TypeScript: initial reset triggered
    Value mismatch: initial_measurement !== initial reset triggered

  root.results[5].gap_days:
    Type: value
    Python:     null
    TypeScript: 0
    Python value is null, TypeScript value is 0

  root.results[5].reset_event.gap_days:
    Type: missing
    Python:     null
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].reset_event.reason:
    Type: value
    Python:     initial_measurement
    TypeScript: initial reset triggered
    Value mismatch: initial_measurement !== initial reset triggered

  root.results[6].timestamp:
    Type: type
    Python:     1762770791110
    TypeScript: 2025-11-10T10:33:11.110Z
    Type mismatch: Python number, TypeScript string

  root.results[6].filtered_weight:
    Type: value
    Python:     70.50582648217592
    TypeScript: 70.54758915347048
    Difference: 4.176e-2
    Numeric difference exceeds tolerance: abs=4.176e-2, rel=0.059198%

  root.results[6].trend:
    Type: value
    Python:     1.3711846731725685e-7
    TypeScript: 0.00000104821693059016
    Difference: 9.111e-7
    Numeric difference exceeds tolerance: abs=9.111e-7, rel=86.918885%

  root.results[6].trend_weekly:
    Type: value
    Python:     9.598292712207979e-7
    TypeScript: 0.00000733751851413112
    Difference: 6.378e-6
    Numeric difference exceeds tolerance: abs=6.378e-6, rel=86.918885%

  root.results[6].confidence:
    Type: value
    Python:     0.9999389919505018
    TypeScript: 0.999813899520255
    Difference: 1.251e-4
    Numeric difference exceeds tolerance: abs=1.251e-4, rel=0.012510%

  root.results[6].innovation:
    Type: value
    Python:     0.09417351782407479
    TypeScript: 0.052410846529511446
    Difference: 4.176e-2
    Numeric difference exceeds tolerance: abs=4.176e-2, rel=44.346513%

  root.results[6].normalized_innovation:
    Type: value
    Python:     0.011046258241137209
    TypeScript: 0.019293408127546645
    Difference: 8.247e-3
    Numeric difference exceeds tolerance: abs=8.247e-3, rel=42.745946%

  root.results[6].kalman_confidence_upper:
    Type: value
    Python:     74.5066521565391
    TypeScript: 73.63268731950959
    Difference: 8.740e-1
    Numeric difference exceeds tolerance: abs=8.740e-1, rel=1.173002%

  root.results[6].kalman_confidence_lower:
    Type: value
    Python:     66.50500080781273
    TypeScript: 67.46249098743138
    Difference: 9.575e-1
    Numeric difference exceeds tolerance: abs=9.575e-1, rel=1.419293%

  root.results[6].kalman_variance:
    Type: value
    Python:     4.0016515191608955
    TypeScript: 2.3794576735244517
    Difference: 1.622e+0
    Numeric difference exceeds tolerance: abs=1.622e+0, rel=40.538109%

  root.results[6].prediction_error:
    Type: value
    Python:     0.09417351782407479
    TypeScript: 0.052410846529511446
    Difference: 4.176e-2
    Numeric difference exceeds tolerance: abs=4.176e-2, rel=44.346513%

  root.results[6].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:33:11.110000
    TypeScript: 2025-11-10T10:33:11.110Z
    Value mismatch: 2025-11-10T11:33:11.110000 !== 2025-11-10T10:33:11.110Z

  root.results[7].timestamp:
    Type: type
    Python:     1762770877510
    TypeScript: 2025-11-10T10:34:37.510Z
    Type mismatch: Python number, TypeScript string

  root.results[7].filtered_weight:
    Type: value
    Python:     70.52228404507574
    TypeScript: 70.60795939408577
    Difference: 8.568e-2
    Numeric difference exceeds tolerance: abs=8.568e-2, rel=0.121340%

  root.results[7].trend:
    Type: value
    Python:     0.000002374382648296539
    TypeScript: 0.000014898685043223674
    Difference: 1.252e-5
    Numeric difference exceeds tolerance: abs=1.252e-5, rel=84.063140%

  root.results[7].trend_weekly:
    Type: value
    Python:     0.00001662067853807577
    TypeScript: 0.00010429079530256571
    Difference: 8.767e-5
    Numeric difference exceeds tolerance: abs=8.767e-5, rel=84.063140%

  root.results[7].confidence:
    Type: value
    Python:     0.9996947673850899
    TypeScript: 0.9993933891668693
    Difference: 3.014e-4
    Numeric difference exceeds tolerance: abs=3.014e-4, rel=0.030147%

  root.results[7].innovation:
    Type: value
    Python:     0.17771595492426684
    TypeScript: 0.09204060591423513
    Difference: 8.568e-2
    Numeric difference exceeds tolerance: abs=8.568e-2, rel=48.209149%

  root.results[7].normalized_innovation:
    Type: value
    Python:     0.024709480280496986
    TypeScript: 0.03483661567725119
    Difference: 1.013e-2
    Numeric difference exceeds tolerance: abs=1.013e-2, rel=29.070377%

  root.results[7].kalman_confidence_upper:
    Type: value
    Python:     74.5430999150707
    TypeScript: 73.4225697801742
    Difference: 1.121e+0
    Numeric difference exceeds tolerance: abs=1.121e+0, rel=1.503198%

  root.results[7].kalman_confidence_lower:
    Type: value
    Python:     66.50146817508077
    TypeScript: 67.79334900799734
    Difference: 1.292e+0
    Numeric difference exceeds tolerance: abs=1.292e+0, rel=1.905616%

  root.results[7].kalman_variance:
    Type: value
    Python:     4.041740065100835
    TypeScript: 1.9805079063692226
    Difference: 2.061e+0
    Numeric difference exceeds tolerance: abs=2.061e+0, rel=50.998632%

  root.results[7].prediction_error:
    Type: value
    Python:     0.17771595492426684
    TypeScript: 0.09204060591423513
    Difference: 8.568e-2
    Numeric difference exceeds tolerance: abs=8.568e-2, rel=48.209149%

  root.results[7].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:34:37.510000
    TypeScript: 2025-11-10T10:34:37.510Z
    Value mismatch: 2025-11-10T11:34:37.510000 !== 2025-11-10T10:34:37.510Z

  root.results[7].quality_score:
    Type: value
    Python:     0.9412561190105361
    TypeScript: 0.9450858309686435
    Difference: 3.830e-3
    Numeric difference exceeds tolerance: abs=3.830e-3, rel=0.405224%

  root.results[7].quality_components.kalman_fit:
    Type: value
    Python:     0.9962102795524286
    TypeScript: 0.9970050658774887
    Difference: 7.948e-4
    Numeric difference exceeds tolerance: abs=7.948e-4, rel=0.079717%

  root.results[7].quality_components.temporal_consistency:
    Type: value
    Python:     0.9358715060747236
    TypeScript: 0.9476524034852276
    Difference: 1.178e-2
    Numeric difference exceeds tolerance: abs=1.178e-2, rel=1.243167%

  root.results[8].timestamp:
    Type: type
    Python:     1762770963910
    TypeScript: 2025-11-10T10:36:03.910Z
    Type mismatch: Python number, TypeScript string

  root.results[8].filtered_weight:
    Type: value
    Python:     70.55394776769158
    TypeScript: 70.67815903746495
    Difference: 1.242e-1
    Numeric difference exceeds tolerance: abs=1.242e-1, rel=0.175742%

  root.results[8].trend:
    Type: value
    Python:     0.000012123179167744857
    TypeScript: 0.0000576495770847014
    Difference: 4.553e-5
    Numeric difference exceeds tolerance: abs=4.553e-5, rel=78.970914%

  root.results[8].trend_weekly:
    Type: value
    Python:     0.000084862254174214
    TypeScript: 0.00040354703959290977
    Difference: 3.187e-4
    Numeric difference exceeds tolerance: abs=3.187e-4, rel=78.970914%

  root.results[8].confidence:
    Type: value
    Python:     0.9991919367647333
    TypeScript: 0.998913459596839
    Difference: 2.785e-4
    Numeric difference exceeds tolerance: abs=2.785e-4, rel=0.027870%

  root.results[8].innovation:
    Type: value
    Python:     0.2460522323084149
    TypeScript: 0.12184096253504606
    Difference: 1.242e-1
    Numeric difference exceeds tolerance: abs=1.242e-1, rel=50.481667%

  root.results[8].normalized_innovation:
    Type: value
    Python:     0.04020920029915558
    TypeScript: 0.04662898489379019
    Difference: 6.420e-3
    Numeric difference exceeds tolerance: abs=6.420e-3, rel=13.767798%

  root.results[8].kalman_confidence_upper:
    Type: value
    Python:     74.46925340202483
    TypeScript: 73.38201151656123
    Difference: 1.087e+0
    Numeric difference exceeds tolerance: abs=1.087e+0, rel=1.459988%

  root.results[8].kalman_confidence_lower:
    Type: value
    Python:     66.63864213335833
    TypeScript: 67.97430655836867
    Difference: 1.336e+0
    Numeric difference exceeds tolerance: abs=1.336e+0, rel=1.964955%

  root.results[8].kalman_variance:
    Type: value
    Python:     3.8324045525604276
    TypeScript: 1.827704557178785
    Difference: 2.005e+0
    Numeric difference exceeds tolerance: abs=2.005e+0, rel=52.309196%

  root.results[8].prediction_error:
    Type: value
    Python:     0.2460522323084149
    TypeScript: 0.12184096253504606
    Difference: 1.242e-1
    Numeric difference exceeds tolerance: abs=1.242e-1, rel=50.481667%

  root.results[8].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:36:03.910000
    TypeScript: 2025-11-10T10:36:03.910Z
    Value mismatch: 2025-11-10T11:36:03.910000 !== 2025-11-10T10:36:03.910Z

  root.results[8].quality_score:
    Type: value
    Python:     0.9342633391947578
    TypeScript: 0.941435178044982
    Difference: 7.172e-3
    Numeric difference exceeds tolerance: abs=7.172e-3, rel=0.761798%

  root.results[8].quality_components.kalman_fit:
    Type: value
    Python:     0.9945801944395222
    TypeScript: 0.9962205319228531
    Difference: 1.640e-3
    Numeric difference exceeds tolerance: abs=1.640e-3, rel=0.164656%

  root.results[8].quality_components.temporal_consistency:
    Type: value
    Python:     0.9150507210816894
    TypeScript: 0.9364497410261658
    Difference: 2.140e-2
    Numeric difference exceeds tolerance: abs=2.140e-2, rel=2.285122%

  root.results[9].timestamp:
    Type: type
    Python:     1762771050310
    TypeScript: 2025-11-10T10:37:30.310Z
    Type mismatch: Python number, TypeScript string

  root.results[9].filtered_weight:
    Type: value
    Python:     70.60337203509553
    TypeScript: 70.75647526250482
    Difference: 1.531e-1
    Numeric difference exceeds tolerance: abs=1.531e-1, rel=0.216381%

  root.results[9].trend:
    Type: value
    Python:     0.00003851528393043969
    TypeScript: 0.0001441384943082579
    Difference: 1.056e-4
    Numeric difference exceeds tolerance: abs=1.056e-4, rel=73.278974%

  root.results[9].trend_weekly:
    Type: value
    Python:     0.00026960698751307785
    TypeScript: 0.0010089694601578053
    Difference: 7.394e-4
    Numeric difference exceeds tolerance: abs=7.394e-4, rel=73.278974%

  root.results[9].confidence:
    Type: value
    Python:     0.998409215670912
    TypeScript: 0.9984786772772689
    Difference: 6.946e-5
    Numeric difference exceeds tolerance: abs=6.946e-5, rel=0.006957%

  root.results[9].innovation:
    Type: value
    Python:     0.29662796490447363
    TypeScript: 0.14352473749518424
    Difference: 1.531e-1
    Numeric difference exceeds tolerance: abs=1.531e-1, rel=51.614563%

  root.results[9].normalized_innovation:
    Type: value
    Python:     0.056427847202575994
    TypeScript: 0.05518117630399935
    Difference: 1.247e-3
    Numeric difference exceeds tolerance: abs=1.247e-3, rel=2.209319%

  root.results[9].kalman_confidence_upper:
    Type: value
    Python:     74.32004535902216
    TypeScript: 73.41358602311652
    Difference: 9.065e-1
    Numeric difference exceeds tolerance: abs=9.065e-1, rel=1.219670%

  root.results[9].kalman_confidence_lower:
    Type: value
    Python:     66.8866987111689
    TypeScript: 68.09936450189312
    Difference: 1.213e+0
    Numeric difference exceeds tolerance: abs=1.213e+0, rel=1.780730%

  root.results[9].kalman_variance:
    Type: value
    Python:     3.453415149196944
    TypeScript: 1.7650593985396297
    Difference: 1.688e+0
    Numeric difference exceeds tolerance: abs=1.688e+0, rel=48.889452%

  root.results[9].prediction_error:
    Type: value
    Python:     0.29662796490447363
    TypeScript: 0.14352473749518424
    Difference: 1.531e-1
    Numeric difference exceeds tolerance: abs=1.531e-1, rel=51.614563%

  root.results[9].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:37:30.310000
    TypeScript: 2025-11-10T10:37:30.310Z
    Value mismatch: 2025-11-10T11:37:30.310000 !== 2025-11-10T10:37:30.310Z

  root.results[9].quality_score:
    Type: value
    Python:     0.9291662865760396
    TypeScript: 0.938828596578805
    Difference: 9.662e-3
    Numeric difference exceeds tolerance: abs=9.662e-3, rel=1.029188%

  root.results[9].quality_components.kalman_fit:
    Type: value
    Python:     0.993240126989953
    TypeScript: 0.9956321713394507
    Difference: 2.392e-3
    Numeric difference exceeds tolerance: abs=2.392e-3, rel=0.240254%

  root.results[9].quality_components.temporal_consistency:
    Type: value
    Python:     0.9004151054030848
    TypeScript: 0.9285891671189228
    Difference: 2.817e-2
    Numeric difference exceeds tolerance: abs=2.817e-2, rel=3.034072%

  root.finalState.kalman_params.transition_covariance[0][0]:
    Type: value
    Python:     0.19607272887128602
    TypeScript: 0.8999999999999999
    Difference: 7.039e-1
    Numeric difference exceeds tolerance: abs=7.039e-1, rel=78.214141%

  root.finalState.kalman_params.transition_covariance[1][1]:
    Type: value
    Python:     0.0013071515258085737
    TypeScript: 0.006
    Difference: 4.693e-3
    Numeric difference exceeds tolerance: abs=4.693e-3, rel=78.214141%

  root.finalState.last_state[0][0]:
    Type: type
    Python:     70.55394776769158
    TypeScript: [
  70.67815903746495
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[0][1]:
    Type: type
    Python:     0.000012123179167744857
    TypeScript: [
  0.0000576495770847014
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][0]:
    Type: type
    Python:     70.60337203509553
    TypeScript: [
  70.75647526250482
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][1]:
    Type: type
    Python:     0.00003851528393043969
    TypeScript: [
  0.0001441384943082579
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_covariance[0][0][0]:
    Type: value
    Python:     3.8324045525604276
    TypeScript: 1.827704557178785
    Difference: 2.005e+0
    Numeric difference exceeds tolerance: abs=2.005e+0, rel=52.309196%

  root.finalState.last_covariance[0][0][1]:
    Type: value
    Python:     0.0011799500893092253
    TypeScript: 0.0011130777135867397
    Difference: 6.687e-5
    Numeric difference exceeds tolerance: abs=6.687e-5, rel=5.667390%

  root.finalState.last_covariance[0][1][0]:
    Type: value
    Python:     0.0011799500893092255
    TypeScript: 0.0011130777135867397
    Difference: 6.687e-5
    Numeric difference exceeds tolerance: abs=6.687e-5, rel=5.667390%

  root.finalState.last_covariance[0][1][1]:
    Type: value
    Python:     0.009714504341111076
    TypeScript: 0.018999540025122084
    Difference: 9.285e-3
    Numeric difference exceeds tolerance: abs=9.285e-3, rel=48.869792%

  root.finalState.last_covariance[1][0][0]:
    Type: value
    Python:     3.453415149196944
    TypeScript: 1.7650593985396297
    Difference: 1.688e+0
    Numeric difference exceeds tolerance: abs=1.688e+0, rel=48.889452%

  root.finalState.last_covariance[1][0][1]:
    Type: value
    Python:     0.001844137199694154
    TypeScript: 0.0019493957263792606
    Difference: 1.053e-4
    Numeric difference exceeds tolerance: abs=1.053e-4, rel=5.399546%

  root.finalState.last_covariance[1][1][0]:
    Type: value
    Python:     0.001844137199694154
    TypeScript: 0.0019493957263792606
    Difference: 1.053e-4
    Numeric difference exceeds tolerance: abs=1.053e-4, rel=5.399546%

  root.finalState.last_covariance[1][1][1]:
    Type: value
    Python:     0.011021491787102742
    TypeScript: 0.024998365306891923
    Difference: 1.398e-2
    Numeric difference exceeds tolerance: abs=1.398e-2, rel=55.911150%

  root.finalState.reset_parameters.quality_acceptance_threshold:
    Type: missing
    Python:     0.25
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_safety_weight:
    Type: missing
    Python:     0.5
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_plausibility_weight:
    Type: missing
    Python:     0.05
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_consistency_weight:
    Type: missing
    Python:     0.05
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.quality_reliability_weight:
    Type: missing
    Python:     0.4
    TypeScript: undefined
    Key missing in TypeScript output

  root.finalState.reset_parameters.enabled:
    Type: extra
    Python:     undefined
    TypeScript: true
    Extra key in TypeScript output

  root.finalState.reset_timestamp:
    Type: type
    Python:     1762770704710
    TypeScript: 2025-11-10T10:31:44.710Z
    Type mismatch: Python number, TypeScript string

  root.finalState.reset_events:
    Type: missing
    Python:     [
  {
    "timestamp": 1762770704710,
    "type": "initial",
    "source": "withings",
    "weight":
    TypeScript: []
    Array length mismatch: Python 1, TypeScript 0

  root.finalState.last_timestamp:
    Type: type
    Python:     1762771050310
    TypeScript: 2025-11-10T10:37:30.310Z
    Type mismatch: Python number, TypeScript string

  root.finalState.last_accepted_timestamp:
    Type: type
    Python:     1762771050310
    TypeScript: 2025-11-10T10:37:30.310Z
    Type mismatch: Python number, TypeScript string

  root.finalState.measurement_history[0].timestamp:
    Type: value
    Python:     2025-11-10T11:31:44.710000
    TypeScript: 2025-11-10T10:31:44.710Z
    Value mismatch: 2025-11-10T11:31:44.710000 !== 2025-11-10T10:31:44.710Z

  root.finalState.measurement_history[1].timestamp:
    Type: value
    Python:     2025-11-10T11:33:11.110000
    TypeScript: 2025-11-10T10:33:11.110Z
    Value mismatch: 2025-11-10T11:33:11.110000 !== 2025-11-10T10:33:11.110Z

  root.finalState.measurement_history[2].timestamp:
    Type: value
    Python:     2025-11-10T11:34:37.510000
    TypeScript: 2025-11-10T10:34:37.510Z
    Value mismatch: 2025-11-10T11:34:37.510000 !== 2025-11-10T10:34:37.510Z

  root.finalState.measurement_history[2].quality_score:
    Type: value
    Python:     0.9412561190105361
    TypeScript: 0.9450858309686435
    Difference: 3.830e-3
    Numeric difference exceeds tolerance: abs=3.830e-3, rel=0.405224%

  root.finalState.measurement_history[3].timestamp:
    Type: value
    Python:     2025-11-10T11:36:03.910000
    TypeScript: 2025-11-10T10:36:03.910Z
    Value mismatch: 2025-11-10T11:36:03.910000 !== 2025-11-10T10:36:03.910Z

  root.finalState.measurement_history[3].quality_score:
    Type: value
    Python:     0.9342633391947578
    TypeScript: 0.941435178044982
    Difference: 7.172e-3
    Numeric difference exceeds tolerance: abs=7.172e-3, rel=0.761798%

  root.finalState.measurement_history[4].timestamp:
    Type: value
    Python:     2025-11-10T11:37:30.310000
    TypeScript: 2025-11-10T10:37:30.310Z
    Value mismatch: 2025-11-10T11:37:30.310000 !== 2025-11-10T10:37:30.310Z

  root.finalState.measurement_history[4].quality_score:
    Type: value
    Python:     0.9291662865760396
    TypeScript: 0.938828596578805
    Difference: 9.662e-3
    Numeric difference exceeds tolerance: abs=9.662e-3, rel=1.029188%

  root.finalState.temporal_baseline.last_timestamp:
    Type: value
    Python:     2025-11-10T11:37:30.310000
    TypeScript: 2025-11-10T10:37:30.310Z
    Value mismatch: 2025-11-10T11:37:30.310000 !== 2025-11-10T10:37:30.310Z

```

## All Test Results

| Test Name | Status | Py Time | TS Time | Differences |
|-----------|--------|---------|---------|-------------|
<<<<<<< Updated upstream
| Test 1: Single Measurement Processing | ❌ | 204.74ms | 2798.10ms | 24 |
| Test 2: Multi-Measurement Sequence | ❌ | 148.19ms | 4.00ms | 183 |
| Test 3: Reset Scenario | ❌ | 119.34ms | 1.70ms | 176 |
| Test 4: Quality Rejection | ❌ | 122.31ms | 0.92ms | 78 |
| Test 5: State Persistence | ❌ | 289.10ms | 1.91ms | 161 |
=======
| Test 1: Single Measurement Processing | ❌ | 234.22ms | 2792.09ms | 22 |
| Test 2: Multi-Measurement Sequence | ❌ | 146.56ms | 4.58ms | 181 |
| Test 3: Reset Scenario | ❌ | 122.18ms | 1.64ms | 174 |
| Test 4: Quality Rejection | ❌ | 112.16ms | 0.88ms | 76 |
| Test 5: State Persistence | ❌ | 224.93ms | 1.99ms | 159 |
>>>>>>> Stashed changes
