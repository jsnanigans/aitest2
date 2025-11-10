# Cross-Language Test Report
**Test Suite**: Phase 1: Integration Tests
**Date**: 2025-11-10T13:11:27.114Z
**Total Tests**: 5
**Passed**: 0
**Failed**: 5
**Success Rate**: 0.0%
**Duration**: 3.88s

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Passed | 0 | 0.0% |
| ❌ Failed | 5 | 100.0% |

## Performance Comparison

- **Python avg**: 188.00ms
- **TypeScript avg**: 586.11ms
- **Speed ratio**: Python is 0.68x faster

## Failed Tests

### Test 1: Single Measurement Processing
**Description**: Process a single measurement and verify initialization

**Comparison**: ✗ Found 35 difference(s): 13 numeric, 22 structural

**Differences**:
```
Found 35 difference(s):

  root.results[0].was_reset:
    Type: missing
    Python:     true
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_reason:
    Type: missing
    Python:     initial_measurement
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_type:
    Type: missing
    Python:     initial
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].gap_days:
    Type: missing
    Python:     null
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.results[0].kalman_confidence_upper:
    Type: value
    Python:     73.81575680566779
    TypeScript: 71.20664825031987
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.534623%

  root.results[0].kalman_confidence_lower:
    Type: value
    Python:     66.18424319433221
    TypeScript: 68.79335174968013
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.792675%

  root.results[0].kalman_variance:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.results[0].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

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

  root.finalState.adaptation_state:
    Type: extra
    Python:     undefined
    TypeScript: {}
    Extra key in TypeScript output

  root.finalState.version:
    Type: extra
    Python:     undefined
    TypeScript: 1
    Extra key in TypeScript output

  root.finalState.kalman_params.initial_state_covariance[0][0]:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.finalState.kalman_params.transition_covariance[0][0]:
    Type: value
    Python:     0.8999999999999999
    TypeScript: 0.018
    Difference: 8.820e-1
    Numeric difference exceeds tolerance: abs=8.820e-1, rel=98.000000%

  root.finalState.kalman_params.transition_covariance[1][1]:
    Type: value
    Python:     0.006
    TypeScript: 0.00012
    Difference: 5.880e-3
    Numeric difference exceeds tolerance: abs=5.880e-3, rel=98.000000%

  root.finalState.kalman_params.observation_covariance[0][0]:
    Type: value
    Python:     100
    TypeScript: 5
    Difference: 9.500e+1
    Numeric difference exceeds tolerance: abs=9.500e+1, rel=95.000000%

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

  root.finalState.last_covariance[0][0][0]:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.finalState.last_covariance[1][0][0]:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

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

**Comparison**: ✗ Found 192 difference(s): 144 numeric, 48 structural

**Differences**:
```
Found 192 difference(s):

  root.results[0].was_reset:
    Type: missing
    Python:     true
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_reason:
    Type: missing
    Python:     initial_measurement
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_type:
    Type: missing
    Python:     initial
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].gap_days:
    Type: missing
    Python:     null
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.results[0].kalman_confidence_upper:
    Type: value
    Python:     73.37575680566779
    TypeScript: 70.76664825031988
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.555818%

  root.results[0].kalman_confidence_lower:
    Type: value
    Python:     65.74424319433221
    TypeScript: 68.35335174968013
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.817089%

  root.results[0].kalman_variance:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.results[0].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

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
    TypeScript: 69.5628391623204
    Difference: 5.086e-4
    Numeric difference exceeds tolerance: abs=5.086e-4, rel=0.000731%

  root.results[1].trend:
    Type: value
    Python:     5.4847386926894944e-8
    TypeScript: 7.432167535919116e-7
    Difference: 6.884e-7
    Numeric difference exceeds tolerance: abs=6.884e-7, rel=92.620270%

  root.results[1].trend_weekly:
    Type: value
    Python:     3.839317084882646e-7
    TypeScript: 0.000005202517275143381
    Difference: 4.819e-6
    Numeric difference exceeds tolerance: abs=4.819e-6, rel=92.620270%

  root.results[1].confidence:
    Type: value
    Python:     0.9999902384619538
    TypeScript: 0.999871067615968
    Difference: 1.192e-4
    Numeric difference exceeds tolerance: abs=1.192e-4, rel=0.011917%

  root.results[1].innovation:
    Type: value
    Python:     0.037669407129627075
    TypeScript: 0.03716083767959333
    Difference: 5.086e-4
    Numeric difference exceeds tolerance: abs=5.086e-4, rel=1.350086%

  root.results[1].normalized_innovation:
    Type: value
    Python:     0.00441850329645455
    TypeScript: 0.016058685906781164
    Difference: 1.164e-2
    Numeric difference exceeds tolerance: abs=1.164e-2, rel=72.485275%

  root.results[1].kalman_confidence_upper:
    Type: value
    Python:     73.56315626723355
    TypeScript: 70.75430093686397
    Difference: 2.809e+0
    Numeric difference exceeds tolerance: abs=2.809e+0, rel=3.818291%

  root.results[1].kalman_confidence_lower:
    Type: value
    Python:     65.56150491850718
    TypeScript: 68.37137738777683
    Difference: 2.810e+0
    Numeric difference exceeds tolerance: abs=2.810e+0, rel=4.109720%

  root.results[1].kalman_variance:
    Type: value
    Python:     4.0016515191608955
    TypeScript: 0.35489529004962833
    Difference: 3.647e+0
    Numeric difference exceeds tolerance: abs=3.647e+0, rel=91.131279%

  root.results[1].prediction_error:
    Type: value
    Python:     0.037669407129627075
    TypeScript: 0.03716083767959333
    Difference: 5.086e-4
    Numeric difference exceeds tolerance: abs=5.086e-4, rel=1.350086%

  root.results[1].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:25:59.110000
    TypeScript: 2025-11-10T10:25:59.110Z
    Value mismatch: 2025-11-10T11:25:59.110000 !== 2025-11-10T10:25:59.110Z

  root.results[1].quality_score:
    Type: value
    Python:     0.9566546516157678
    TypeScript: 0.9556776396374415
    Difference: 9.770e-4
    Numeric difference exceeds tolerance: abs=9.770e-4, rel=0.102128%

  root.results[1].quality_components.kalman_fit:
    Type: value
    Python:     0.99921787021379
    TypeScript: 0.9965575381083879
    Difference: 2.660e-3
    Numeric difference exceeds tolerance: abs=2.660e-3, rel=0.266241%

  root.results[2].timestamp:
    Type: type
    Python:     1762770445510
    TypeScript: 2025-11-10T10:27:25.510Z
    Type mismatch: Python number, TypeScript string

  root.results[2].filtered_weight:
    Type: value
    Python:     69.6053590959197
    TypeScript: 69.59804035352437
    Difference: 7.319e-3
    Numeric difference exceeds tolerance: abs=7.319e-3, rel=0.010515%

  root.results[2].trend:
    Type: value
    Python:     0.000005904206563257916
    TypeScript: 0.00002008430320323208
    Difference: 1.418e-5
    Numeric difference exceeds tolerance: abs=1.418e-5, rel=70.602881%

  root.results[2].trend_weekly:
    Type: value
    Python:     0.00004132944594280541
    TypeScript: 0.00014059012242262456
    Difference: 9.926e-5
    Numeric difference exceeds tolerance: abs=9.926e-5, rel=70.602881%

  root.results[2].confidence:
    Type: value
    Python:     0.9979153833407085
    TypeScript: 0.9793865285029923
    Difference: 1.853e-2
    Numeric difference exceeds tolerance: abs=1.853e-2, rel=1.856756%

  root.results[2].innovation:
    Type: value
    Python:     0.4646409040802979
    TypeScript: 0.47195964647562505
    Difference: 7.319e-3
    Numeric difference exceeds tolerance: abs=7.319e-3, rel=1.550714%

  root.results[2].normalized_innovation:
    Type: value
    Python:     0.06460328934325016
    TypeScript: 0.20410239908130254
    Difference: 1.395e-1
    Numeric difference exceeds tolerance: abs=1.395e-1, rel=68.347609%

  root.results[2].kalman_confidence_upper:
    Type: value
    Python:     73.62617496591466
    TypeScript: 70.77624409091833
    Difference: 2.850e+0
    Numeric difference exceeds tolerance: abs=2.850e+0, rel=3.870812%

  root.results[2].kalman_confidence_lower:
    Type: value
    Python:     65.58454322592473
    TypeScript: 68.41983661613041
    Difference: 2.835e+0
    Numeric difference exceeds tolerance: abs=2.835e+0, rel=4.143964%

  root.results[2].kalman_variance:
    Type: value
    Python:     4.041740065100835
    TypeScript: 0.34704101170227364
    Difference: 3.695e+0
    Numeric difference exceeds tolerance: abs=3.695e+0, rel=91.413574%

  root.results[2].prediction_error:
    Type: value
    Python:     0.4646409040802979
    TypeScript: 0.47195964647562505
    Difference: 7.319e-3
    Numeric difference exceeds tolerance: abs=7.319e-3, rel=1.550714%

  root.results[2].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:27:25.510000
    TypeScript: 2025-11-10T10:27:25.510Z
    Value mismatch: 2025-11-10T11:27:25.510000 !== 2025-11-10T10:27:25.510Z

  root.results[2].quality_score:
    Type: value
    Python:     0.8901620889765289
    TypeScript: 0.8795778809475164
    Difference: 1.058e-2
    Numeric difference exceeds tolerance: abs=1.058e-2, rel=1.189020%

  root.results[2].quality_components.kalman_fit:
    Type: value
    Python:     0.9901220103742958
    TypeScript: 0.9571843351680731
    Difference: 3.294e-2
    Numeric difference exceeds tolerance: abs=3.294e-2, rel=3.326628%

  root.results[2].quality_components.temporal_consistency:
    Type: value
    Python:     0.7914133025655209
    TypeScript: 0.792215080414027
    Difference: 8.018e-4
    Numeric difference exceeds tolerance: abs=8.018e-4, rel=0.101207%

  root.results[3].timestamp:
    Type: type
    Python:     1762770531910
    TypeScript: 2025-11-10T10:28:51.910Z
    Type mismatch: Python number, TypeScript string

  root.results[3].filtered_weight:
    Type: value
    Python:     69.63781267255982
    TypeScript: 69.61790991965903
    Difference: 1.990e-2
    Numeric difference exceeds tolerance: abs=1.990e-2, rel=0.028580%

  root.results[3].trend:
    Type: value
    Python:     0.00001589608071506883
    TypeScript: 0.00003720851378517793
    Difference: 2.131e-5
    Numeric difference exceeds tolerance: abs=2.131e-5, rel=57.278378%

  root.results[3].trend_weekly:
    Type: value
    Python:     0.00011127256500548181
    TypeScript: 0.0002604595964962455
    Difference: 1.492e-4
    Numeric difference exceeds tolerance: abs=1.492e-4, rel=57.278378%

  root.results[3].confidence:
    Type: value
    Python:     0.9991511550180204
    TypeScript: 0.9930923592108624
    Difference: 6.059e-3
    Numeric difference exceeds tolerance: abs=6.059e-3, rel=0.606394%

  root.results[3].innovation:
    Type: value
    Python:     0.2521873274401827
    TypeScript: 0.2720900803409734
    Difference: 1.990e-2
    Numeric difference exceeds tolerance: abs=1.990e-2, rel=7.314766%

  root.results[3].normalized_innovation:
    Type: value
    Python:     0.0412117812011586
    TypeScript: 0.11774216729201478
    Difference: 7.653e-2
    Numeric difference exceeds tolerance: abs=7.653e-2, rel=64.998282%

  root.results[3].kalman_confidence_upper:
    Type: value
    Python:     73.55311830689307
    TypeScript: 70.78452428153247
    Difference: 2.769e+0
    Numeric difference exceeds tolerance: abs=2.769e+0, rel=3.764074%

  root.results[3].kalman_confidence_lower:
    Type: value
    Python:     65.72250703822657
    TypeScript: 68.45129555778558
    Difference: 2.729e+0
    Numeric difference exceeds tolerance: abs=2.729e+0, rel=3.986467%

  root.results[3].kalman_variance:
    Type: value
    Python:     3.8324045525604276
    TypeScript: 0.3402472673323422
    Difference: 3.492e+0
    Numeric difference exceeds tolerance: abs=3.492e+0, rel=91.121833%

  root.results[3].prediction_error:
    Type: value
    Python:     0.2521873274401827
    TypeScript: 0.2720900803409734
    Difference: 1.990e-2
    Numeric difference exceeds tolerance: abs=1.990e-2, rel=7.314766%

  root.results[3].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:28:51.910000
    TypeScript: 2025-11-10T10:28:51.910Z
    Value mismatch: 2025-11-10T11:28:51.910000 !== 2025-11-10T10:28:51.910Z

  root.results[3].quality_score:
    Type: value
    Python:     0.8935392243805096
    TypeScript: 0.8867496822737152
    Difference: 6.790e-3
    Numeric difference exceeds tolerance: abs=6.790e-3, rel=0.759848%

  root.results[3].quality_components.kalman_fit:
    Type: value
    Python:     0.9944454323790577
    TypeScript: 0.9751057737348243
    Difference: 1.934e-2
    Numeric difference exceeds tolerance: abs=1.934e-2, rel=1.944768%

  root.results[3].quality_components.temporal_consistency:
    Type: value
    Python:     0.9134753026673629
    TypeScript: 0.9118337382096611
    Difference: 1.642e-3
    Numeric difference exceeds tolerance: abs=1.642e-3, rel=0.179705%

  root.results[4].timestamp:
    Type: type
    Python:     1762770618310
    TypeScript: 2025-11-10T10:30:18.310Z
    Type mismatch: Python number, TypeScript string

  root.results[4].filtered_weight:
    Type: value
    Python:     69.62527261696026
    TypeScript: 69.61337214518043
    Difference: 1.190e-2
    Numeric difference exceeds tolerance: abs=1.190e-2, rel=0.017092%

  root.results[4].trend:
    Type: value
    Python:     0.000009198793110317915
    TypeScript: 0.00003176785648127964
    Difference: 2.257e-5
    Numeric difference exceeds tolerance: abs=2.257e-5, rel=71.043709%

  root.results[4].trend_weekly:
    Type: value
    Python:     0.0000643915517722254
    TypeScript: 0.00022237499536895746
    Difference: 1.580e-4
    Numeric difference exceeds tolerance: abs=1.580e-4, rel=71.043709%

  root.results[4].confidence:
    Type: value
    Python:     0.9998974856760499
    TypeScript: 0.9996236404720705
    Difference: 2.738e-4
    Numeric difference exceeds tolerance: abs=2.738e-4, rel=0.027387%

  root.results[4].innovation:
    Type: value
    Python:     -0.07527261696026244
    TypeScript: -0.06337214518043766
    Difference: 1.190e-2
    Numeric difference exceeds tolerance: abs=1.190e-2, rel=15.809829%

  root.results[4].normalized_innovation:
    Type: value
    Python:     0.01431918844785784
    TypeScript: 0.027438307854226768
    Difference: 1.312e-2
    Numeric difference exceeds tolerance: abs=1.312e-2, rel=47.813150%

  root.results[4].kalman_confidence_upper:
    Type: value
    Python:     73.34194594088689
    TypeScript: 70.76984629091437
    Difference: 2.572e+0
    Numeric difference exceeds tolerance: abs=2.572e+0, rel=3.506997%

  root.results[4].kalman_confidence_lower:
    Type: value
    Python:     65.90859929303363
    TypeScript: 68.4568979994465
    Difference: 2.548e+0
    Numeric difference exceeds tolerance: abs=2.548e+0, rel=3.722486%

  root.results[4].kalman_variance:
    Type: value
    Python:     3.453415149196944
    TypeScript: 0.33435811243776226
    Difference: 3.119e+0
    Numeric difference exceeds tolerance: abs=3.119e+0, rel=90.318045%

  root.results[4].prediction_error:
    Type: value
    Python:     -0.07527261696026244
    TypeScript: -0.06337214518043766
    Difference: 1.190e-2
    Numeric difference exceeds tolerance: abs=1.190e-2, rel=15.809829%

  root.results[4].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.results[4].quality_score:
    Type: value
    Python:     0.909561513787953
    TypeScript: 0.909969958739884
    Difference: 4.084e-4
    Numeric difference exceeds tolerance: abs=4.084e-4, rel=0.044886%

  root.results[4].quality_components.kalman_fit:
    Type: value
    Python:     0.9982802639482513
    TypeScript: 0.9941494168296441
    Difference: 4.131e-3
    Numeric difference exceeds tolerance: abs=4.131e-3, rel=0.413796%

  root.results[4].quality_components.temporal_consistency:
    Type: value
    Python:     0.9679184759159902
    TypeScript: 0.9747062211655353
    Difference: 6.788e-3
    Numeric difference exceeds tolerance: abs=6.788e-3, rel=0.696389%

  root.results[5].filtered_weight:
    Type: missing
    Python:     69.76001648816205
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].trend:
    Type: missing
    Python:     0.00011974253872188602
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].trend_weekly:
    Type: missing
    Python:     0.0008381977710532021
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].confidence:
    Type: missing
    Python:     0.9892916331664391
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].innovation:
    Type: missing
    Python:     0.6699835118379553
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].normalized_innovation:
    Type: missing
    Python:     0.146738638508628
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].kalman_confidence_upper:
    Type: missing
    Python:     73.21830586276712
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].kalman_confidence_lower:
    Type: missing
    Python:     66.30172711355698
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].kalman_variance:
    Type: missing
    Python:     2.989941349626596
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].prediction_error:
    Type: missing
    Python:     0.6699835118379553
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].preprocessing:
    Type: missing
    Python:     {
  "original_weight": 70.43,
  "original_unit": "kg",
  "source": "withings",
  "timestamp": "2025-
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].noise_multiplier:
    Type: missing
    Python:     1
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].bmi_details:
    Type: missing
    Python:     {
  "user_height_m": 1.67,
  "implied_bmi": 25.3,
  "original_weight": 70.43,
  "original_unit": "kg
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].cleaned_weight:
    Type: extra
    Python:     undefined
    TypeScript: 70.43
    Extra key in TypeScript output

  root.results[5].reason:
    Type: extra
    Python:     undefined
    TypeScript: Quality score 0.46 below threshold 0.46 (weakest: anomaly_detection=0.21)
    Extra key in TypeScript output

  root.results[5].quality_details:
    Type: extra
    Python:     undefined
    TypeScript: {
  "overall": 0.4577709129994638,
  "components": {
    "kalman_fit": 0.9318388893374059,
    "temp
    Extra key in TypeScript output

  root.results[5].timestamp:
    Type: type
    Python:     1762770704710
    TypeScript: 2025-11-10T10:31:44.710Z
    Type mismatch: Python number, TypeScript string

  root.results[5].accepted:
    Type: value
    Python:     true
    TypeScript: false
    Value mismatch: true !== false

  root.results[5].stage:
    Type: value
    Python:     accepted
    TypeScript: unified_quality_scoring
    Value mismatch: accepted !== unified_quality_scoring

  root.results[5].quality_score:
    Type: value
    Python:     0.4661584533390423
    TypeScript: 0.4577709129994638
    Difference: 8.388e-3
    Numeric difference exceeds tolerance: abs=8.388e-3, rel=1.799290%

  root.results[5].quality_components.kalman_fit:
    Type: value
    Python:     0.9843156898703053
    TypeScript: 0.9318388893374059
    Difference: 5.248e-2
    Numeric difference exceeds tolerance: abs=5.248e-2, rel=5.331298%

  root.results[5].quality_components.temporal_consistency:
    Type: value
    Python:     0.4380641095179947
    TypeScript: 0.42780647100475716
    Difference: 1.026e-2
    Numeric difference exceeds tolerance: abs=1.026e-2, rel=2.341584%

  root.results[6].timestamp:
    Type: type
    Python:     1762770791110
    TypeScript: 2025-11-10T10:33:11.110Z
    Type mismatch: Python number, TypeScript string

  root.results[6].filtered_weight:
    Type: value
    Python:     69.86354986279649
    TypeScript: 69.65990576379795
    Difference: 2.036e-1
    Numeric difference exceeds tolerance: abs=2.036e-1, rel=0.291488%

  root.results[6].trend:
    Type: value
    Python:     0.00024196956971566568
    TypeScript: 0.00010418689861239646
    Difference: 1.378e-4
    Numeric difference exceeds tolerance: abs=1.378e-4, rel=56.942148%

  root.results[6].trend_weekly:
    Type: value
    Python:     0.0016937869880096598
    TypeScript: 0.0007293082902867752
    Difference: 9.645e-4
    Numeric difference exceeds tolerance: abs=9.645e-4, rel=56.942148%

  root.results[6].confidence:
    Type: value
    Python:     0.993564751924342
    TypeScript: 0.9599438318505354
    Difference: 3.362e-2
    Numeric difference exceeds tolerance: abs=3.362e-2, rel=3.383868%

  root.results[6].innovation:
    Type: value
    Python:     0.4564501372035039
    TypeScript: 0.6600942362020419
    Difference: 2.036e-1
    Numeric difference exceeds tolerance: abs=2.036e-1, rel=30.850762%

  root.results[6].normalized_innovation:
    Type: value
    Python:     0.11363136493576545
    TypeScript: 0.2859388211522499
    Difference: 1.723e-1
    Numeric difference exceeds tolerance: abs=1.723e-1, rel=60.260253%

  root.results[6].kalman_confidence_upper:
    Type: value
    Python:     73.03692950223387
    TypeScript: 70.80750218638632
    Difference: 2.229e+0
    Numeric difference exceeds tolerance: abs=2.229e+0, rel=3.052466%

  root.results[6].kalman_confidence_lower:
    Type: value
    Python:     66.6901702233591
    TypeScript: 68.51230934120959
    Difference: 1.822e+0
    Numeric difference exceeds tolerance: abs=1.822e+0, rel=2.659579%

  root.results[6].kalman_variance:
    Type: value
    Python:     2.517584583998928
    TypeScript: 0.3292443872844016
    Difference: 2.188e+0
    Numeric difference exceeds tolerance: abs=2.188e+0, rel=86.922211%

  root.results[6].prediction_error:
    Type: value
    Python:     0.4564501372035039
    TypeScript: 0.6600942362020419
    Difference: 2.036e-1
    Numeric difference exceeds tolerance: abs=2.036e-1, rel=30.850762%

  root.results[6].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:33:11.110000
    TypeScript: 2025-11-10T10:33:11.110Z
    Value mismatch: 2025-11-10T11:33:11.110000 !== 2025-11-10T10:33:11.110Z

  root.results[6].quality_score:
    Type: value
    Python:     0.7612391154514953
    TypeScript: 0.6600491717336802
    Difference: 1.012e-1
    Numeric difference exceeds tolerance: abs=1.012e-1, rel=13.292794%

  root.results[6].quality_components.kalman_fit:
    Type: value
    Python:     0.9890323074969029
    TypeScript: 0.9407422429729183
    Difference: 4.829e-2
    Numeric difference exceeds tolerance: abs=4.829e-2, rel=4.882557%

  root.results[6].quality_components.temporal_consistency:
    Type: value
    Python:     0.7131278587506958
    TypeScript: 0.5359043116682439
    Difference: 1.772e-1
    Numeric difference exceeds tolerance: abs=1.772e-1, rel=24.851581%

  root.results[6].quality_components.anomaly_detection:
    Type: value
    Python:     0.5324767979862408
    TypeScript: 0.4827617362875701
    Difference: 4.972e-2
    Numeric difference exceeds tolerance: abs=4.972e-2, rel=9.336569%

  root.results[6].quality_components.trend_alignment:
    Type: value
    Python:     0.9915360228629635
    TypeScript: 0.9388261130901062
    Difference: 5.271e-2
    Numeric difference exceeds tolerance: abs=5.271e-2, rel=5.315985%

  root.results[7].timestamp:
    Type: type
    Python:     1762770877510
    TypeScript: 2025-11-10T10:34:37.510Z
    Type mismatch: Python number, TypeScript string

  root.results[7].filtered_weight:
    Type: value
    Python:     69.90938341577476
    TypeScript: 69.6885038774724
    Difference: 2.209e-1
    Numeric difference exceeds tolerance: abs=2.209e-1, rel=0.315951%

  root.results[7].trend:
    Type: value
    Python:     0.000316812908999397
    TypeScript: 0.00015952633385148617
    Difference: 1.573e-4
    Numeric difference exceeds tolerance: abs=1.573e-4, rel=49.646517%

  root.results[7].trend_weekly:
    Type: value
    Python:     0.002217690362995779
    TypeScript: 0.0011166843369604033
    Difference: 1.101e-3
    Numeric difference exceeds tolerance: abs=1.101e-3, rel=49.646517%

  root.results[7].confidence:
    Type: value
    Python:     0.9985888523573752
    TypeScript: 0.984225693995948
    Difference: 1.436e-2
    Numeric difference exceeds tolerance: abs=1.436e-2, rel=1.438346%

  root.results[7].innovation:
    Type: value
    Python:     0.1906165842252392
    TypeScript: 0.4114961225275948
    Difference: 2.209e-1
    Numeric difference exceeds tolerance: abs=2.209e-1, rel=53.677186%

  root.results[7].normalized_innovation:
    Type: value
    Python:     0.053144035397069814
    TypeScript: 0.17832579405527502
    Difference: 1.252e-1
    Numeric difference exceeds tolerance: abs=1.252e-1, rel=70.198346%

  root.results[7].kalman_confidence_upper:
    Type: value
    Python:     72.79944821519184
    TypeScript: 70.82832543901604
    Difference: 1.971e+0
    Numeric difference exceeds tolerance: abs=1.971e+0, rel=2.707607%

  root.results[7].kalman_confidence_lower:
    Type: value
    Python:     67.01931861635767
    TypeScript: 68.54868231592876
    Difference: 1.529e+0
    Numeric difference exceeds tolerance: abs=1.529e+0, rel=2.231062%

  root.results[7].kalman_variance:
    Type: value
    Python:     2.0881186362074304
    TypeScript: 0.3247982980399501
    Difference: 1.763e+0
    Numeric difference exceeds tolerance: abs=1.763e+0, rel=84.445410%

  root.results[7].prediction_error:
    Type: value
    Python:     0.1906165842252392
    TypeScript: 0.4114961225275948
    Difference: 2.209e-1
    Numeric difference exceeds tolerance: abs=2.209e-1, rel=53.677186%

  root.results[7].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:34:37.510000
    TypeScript: 2025-11-10T10:34:37.510Z
    Value mismatch: 2025-11-10T11:34:37.510000 !== 2025-11-10T10:34:37.510Z

  root.results[7].quality_score:
    Type: value
    Python:     0.7773749829236973
    TypeScript: 0.8057882214618054
    Difference: 2.841e-2
    Numeric difference exceeds tolerance: abs=2.841e-2, rel=3.526142%

  root.results[7].quality_components.kalman_fit:
    Type: value
    Python:     0.9953430273240147
    TypeScript: 0.9626529202861055
    Difference: 3.269e-2
    Numeric difference exceeds tolerance: abs=3.269e-2, rel=3.284306%

  root.results[7].quality_components.temporal_consistency:
    Type: value
    Python:     0.9249027060807666
    TypeScript: 0.8832685575142813
    Difference: 4.163e-2
    Numeric difference exceeds tolerance: abs=4.163e-2, rel=4.501463%

  root.results[7].quality_components.anomaly_detection:
    Type: value
    Python:     0.44872947451330913
    TypeScript: 0.5363931609733014
    Difference: 8.766e-2
    Numeric difference exceeds tolerance: abs=8.766e-2, rel=16.343178%

  root.results[7].quality_components.trend_alignment:
    Type: value
    Python:     0.9625066673625294
    TypeScript: 0.9902148558062565
    Difference: 2.771e-2
    Numeric difference exceeds tolerance: abs=2.771e-2, rel=2.798200%

  root.results[8].timestamp:
    Type: type
    Python:     1762770963910
    TypeScript: 2025-11-10T10:36:03.910Z
    Type mismatch: Python number, TypeScript string

  root.results[8].filtered_weight:
    Type: value
    Python:     69.96010854415746
    TypeScript: 69.71942404571614
    Difference: 2.407e-1
    Numeric difference exceeds tolerance: abs=2.407e-1, rel=0.344031%

  root.results[8].trend:
    Type: value
    Python:     0.00042842827197489227
    TypeScript: 0.00023168310981248007
    Difference: 1.967e-4
    Numeric difference exceeds tolerance: abs=1.967e-4, rel=45.922544%

  root.results[8].trend_weekly:
    Type: value
    Python:     0.002998997903824246
    TypeScript: 0.0016217817686873604
    Difference: 1.377e-3
    Numeric difference exceeds tolerance: abs=1.377e-3, rel=45.922544%

  root.results[8].confidence:
    Type: value
    Python:     0.9979238101769797
    TypeScript: 0.981103449688045
    Difference: 1.682e-2
    Numeric difference exceeds tolerance: abs=1.682e-2, rel=1.685536%

  root.results[8].innovation:
    Type: value
    Python:     0.20989145584253777
    TypeScript: 0.4505759542838632
    Difference: 2.407e-1
    Numeric difference exceeds tolerance: abs=2.407e-1, rel=53.417076%

  root.results[8].normalized_innovation:
    Type: value
    Python:     0.06447244516763724
    TypeScript: 0.19533239196724672
    Difference: 1.309e-1
    Numeric difference exceeds tolerance: abs=1.309e-1, rel=66.993470%

  root.results[8].kalman_confidence_upper:
    Type: value
    Python:     72.58766507247513
    TypeScript: 70.85243643242548
    Difference: 1.735e+0
    Numeric difference exceeds tolerance: abs=1.735e+0, rel=2.390528%

  root.results[8].kalman_confidence_lower:
    Type: value
    Python:     67.3325520158398
    TypeScript: 68.5864116590068
    Difference: 1.254e+0
    Numeric difference exceeds tolerance: abs=1.254e+0, rel=1.828146%

  root.results[8].kalman_variance:
    Type: value
    Python:     1.7260133273761955
    TypeScript: 0.32092926710920067
    Difference: 1.405e+0
    Numeric difference exceeds tolerance: abs=1.405e+0, rel=81.406327%

  root.results[8].prediction_error:
    Type: value
    Python:     0.20989145584253777
    TypeScript: 0.4505759542838632
    Difference: 2.407e-1
    Numeric difference exceeds tolerance: abs=2.407e-1, rel=53.417076%

  root.results[8].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:36:03.910000
    TypeScript: 2025-11-10T10:36:03.910Z
    Value mismatch: 2025-11-10T11:36:03.910000 !== 2025-11-10T10:36:03.910Z

  root.results[8].quality_score:
    Type: value
    Python:     0.7743640699347982
    TypeScript: 0.759584794650416
    Difference: 1.478e-2
    Numeric difference exceeds tolerance: abs=1.478e-2, rel=1.908569%

  root.results[8].quality_components.kalman_fit:
    Type: value
    Python:     0.9948569776966091
    TypeScript: 0.9591959496249064
    Difference: 3.566e-2
    Numeric difference exceeds tolerance: abs=3.566e-2, rel=3.584538%

  root.results[8].quality_components.temporal_consistency:
    Type: value
    Python:     0.9190351298797528
    TypeScript: 0.8766797838916072
    Difference: 4.236e-2
    Numeric difference exceeds tolerance: abs=4.236e-2, rel=4.608675%

  root.results[8].quality_components.anomaly_detection:
    Type: value
    Python:     0.44534773507219133
    TypeScript: 0.450046862306353
    Difference: 4.699e-3
    Numeric difference exceeds tolerance: abs=4.699e-3, rel=1.044142%

  root.results[8].quality_components.trend_alignment:
    Type: value
    Python:     0.9747005337838514
    TypeScript: 0.9921736438053014
    Difference: 1.747e-2
    Numeric difference exceeds tolerance: abs=1.747e-2, rel=1.761094%

  root.results[9].timestamp:
    Type: type
    Python:     1762771050310
    TypeScript: 2025-11-10T10:37:30.310Z
    Type mismatch: Python number, TypeScript string

  root.results[9].filtered_weight:
    Type: value
    Python:     69.97334581855951
    TypeScript: 69.73917108713276
    Difference: 2.342e-1
    Numeric difference exceeds tolerance: abs=2.342e-1, rel=0.334663%

  root.results[9].trend:
    Type: value
    Python:     0.0004667853739103754
    TypeScript: 0.0002859686575619165
    Difference: 1.808e-4
    Numeric difference exceeds tolerance: abs=1.808e-4, rel=38.736586%

  root.results[9].trend_weekly:
    Type: value
    Python:     0.003267497617372628
    TypeScript: 0.0020017806029334153
    Difference: 1.266e-3
    Numeric difference exceeds tolerance: abs=1.266e-3, rel=38.736586%

  root.results[9].confidence:
    Type: value
    Python:     0.9998223034455657
    TypeScript: 0.9920785101094451
    Difference: 7.744e-3
    Numeric difference exceeds tolerance: abs=7.744e-3, rel=0.774517%

  root.results[9].innovation:
    Type: value
    Python:     0.056654181440492835
    TypeScript: 0.2908289128672408
    Difference: 2.342e-1
    Numeric difference exceeds tolerance: abs=2.342e-1, rel=80.519756%

  root.results[9].normalized_innovation:
    Type: value
    Python:     0.01885271038008296
    TypeScript: 0.12611924177886155
    Difference: 1.073e-1
    Numeric difference exceeds tolerance: abs=1.073e-1, rel=85.051678%

  root.results[9].kalman_confidence_upper:
    Type: value
    Python:     72.36904276828402
    TypeScript: 70.86622160503806
    Difference: 1.503e+0
    Numeric difference exceeds tolerance: abs=1.503e+0, rel=2.076608%

  root.results[9].kalman_confidence_lower:
    Type: value
    Python:     67.577648868835
    TypeScript: 68.61212056922746
    Difference: 1.034e+0
    Numeric difference exceeds tolerance: abs=1.034e+0, rel=1.507710%

  root.results[9].kalman_variance:
    Type: value
    Python:     1.4348409687298243
    TypeScript: 0.3175607174776533
    Difference: 1.117e+0
    Numeric difference exceeds tolerance: abs=1.117e+0, rel=77.867881%

  root.results[9].prediction_error:
    Type: value
    Python:     0.056654181440492835
    TypeScript: 0.2908289128672408
    Difference: 2.342e-1
    Numeric difference exceeds tolerance: abs=2.342e-1, rel=80.519756%

  root.results[9].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:37:30.310000
    TypeScript: 2025-11-10T10:37:30.310Z
    Value mismatch: 2025-11-10T11:37:30.310000 !== 2025-11-10T10:37:30.310Z

  root.results[9].quality_score:
    Type: value
    Python:     0.782962178012945
    TypeScript: 0.7685288239225214
    Difference: 1.443e-2
    Numeric difference exceeds tolerance: abs=1.443e-2, rel=1.843429%

  root.results[9].quality_components.kalman_fit:
    Type: value
    Python:     0.9986162613355426
    TypeScript: 0.9734779172854322
    Difference: 2.514e-2
    Numeric difference exceeds tolerance: abs=2.514e-2, rel=2.517318%

  root.results[9].quality_components.temporal_consistency:
    Type: value
    Python:     0.9740183066335732
    TypeScript: 0.9077644069412598
    Difference: 6.625e-2
    Numeric difference exceeds tolerance: abs=6.625e-2, rel=6.802121%

  root.results[9].quality_components.anomaly_detection:
    Type: value
    Python:     0.4406212906963603
    TypeScript: 0.4487909869929939
    Difference: 8.170e-3
    Numeric difference exceeds tolerance: abs=8.170e-3, rel=1.820379%

  root.results[9].quality_components.trend_alignment:
    Type: value
    Python:     0.9571267564259217
    TypeScript: 0.9677631928271441
    Difference: 1.064e-2
    Numeric difference exceeds tolerance: abs=1.064e-2, rel=1.099074%

  root.finalState.adaptation_state:
    Type: extra
    Python:     undefined
    TypeScript: {}
    Extra key in TypeScript output

  root.finalState.version:
    Type: extra
    Python:     undefined
    TypeScript: 1
    Extra key in TypeScript output

  root.finalState.kalman_params.initial_state_covariance[0][0]:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.finalState.kalman_params.transition_covariance[0][0]:
    Type: value
    Python:     0.04209952319851201
    TypeScript: 0.018
    Difference: 2.410e-2
    Numeric difference exceeds tolerance: abs=2.410e-2, rel=57.244171%

  root.finalState.kalman_params.transition_covariance[1][1]:
    Type: value
    Python:     0.0002806634879900801
    TypeScript: 0.00012
    Difference: 1.607e-4
    Numeric difference exceeds tolerance: abs=1.607e-4, rel=57.244171%

  root.finalState.kalman_params.observation_covariance[0][0]:
    Type: value
    Python:     100
    TypeScript: 5
    Difference: 9.500e+1
    Numeric difference exceeds tolerance: abs=9.500e+1, rel=95.000000%

  root.finalState.last_state[0][0]:
    Type: type
    Python:     69.96010854415746
    TypeScript: [
  69.71942404571614
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[0][1]:
    Type: type
    Python:     0.00042842827197489227
    TypeScript: [
  0.00023168310981248007
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][0]:
    Type: type
    Python:     69.97334581855951
    TypeScript: [
  69.73917108713276
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][1]:
    Type: type
    Python:     0.0004667853739103754
    TypeScript: [
  0.0002859686575619165
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_covariance[0][0][0]:
    Type: value
    Python:     1.7260133273761955
    TypeScript: 0.32092926710920067
    Difference: 1.405e+0
    Numeric difference exceeds tolerance: abs=1.405e+0, rel=81.406327%

  root.finalState.last_covariance[0][0][1]:
    Type: value
    Python:     0.003800286133368291
    TypeScript: 0.0007493224069523752
    Difference: 3.051e-3
    Numeric difference exceeds tolerance: abs=3.051e-3, rel=80.282474%

  root.finalState.last_covariance[0][1][0]:
    Type: value
    Python:     0.0038002861333682903
    TypeScript: 0.0007493224069523752
    Difference: 3.051e-3
    Numeric difference exceeds tolerance: abs=3.051e-3, rel=80.282474%

  root.finalState.last_covariance[0][1][1]:
    Type: value
    Python:     0.01342336863742711
    TypeScript: 0.0018396767095361507
    Difference: 1.158e-2
    Numeric difference exceeds tolerance: abs=1.158e-2, rel=86.294970%

  root.finalState.last_covariance[1][0][0]:
    Type: value
    Python:     1.4348409687298243
    TypeScript: 0.3175607174776533
    Difference: 1.117e+0
    Numeric difference exceeds tolerance: abs=1.117e+0, rel=77.867881%

  root.finalState.last_covariance[1][0][1]:
    Type: value
    Python:     0.004171179409022243
    TypeScript: 0.00087401482455507
    Difference: 3.297e-3
    Numeric difference exceeds tolerance: abs=3.297e-3, rel=79.046338%

  root.finalState.last_covariance[1][1][0]:
    Type: value
    Python:     0.004171179409022243
    TypeScript: 0.00087401482455507
    Difference: 3.297e-3
    Numeric difference exceeds tolerance: abs=3.297e-3, rel=79.046338%

  root.finalState.last_covariance[1][1][1]:
    Type: value
    Python:     0.013701208073428174
    TypeScript: 0.0019595135676634106
    Difference: 1.174e-2
    Numeric difference exceeds tolerance: abs=1.174e-2, rel=85.698242%

  root.finalState.measurements_since_reset:
    Type: value
    Python:     10
    TypeScript: 9
    Difference: 1.000e+0
    Numeric difference exceeds tolerance: abs=1.000e+0, rel=10.000000%

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

  root.finalState.measurement_history:
    Type: missing
    Python:     [
  {
    "weight": 69.56,
    "timestamp": "2025-11-10T11:24:32.710000",
    "quality_score": 0.975
    TypeScript: [
  {
    "weight": 69.56,
    "timestamp": "2025-11-10T10:24:32.710Z",
    "quality_score": 0.97560
    Array length mismatch: Python 10, TypeScript 9

  root.finalState.temporal_baseline.last_timestamp:
    Type: value
    Python:     2025-11-10T11:37:30.310000
    TypeScript: 2025-11-10T10:37:30.310Z
    Value mismatch: 2025-11-10T11:37:30.310000 !== 2025-11-10T10:37:30.310Z

  root.finalState.temporal_baseline.rolling_avg_change_rate:
    Type: value
    Python:     2.0115946270000307
    TypeScript: 2.217216610000003
    Difference: 2.056e-1
    Numeric difference exceeds tolerance: abs=2.056e-1, rel=9.273879%

```

### Test 3: Reset Scenario
**Description**: Process measurements with a large change that triggers reset

**Comparison**: ✗ Found 189 difference(s): 145 numeric, 44 structural

**Differences**:
```
Found 189 difference(s):

  root.results[0].was_reset:
    Type: missing
    Python:     true
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_reason:
    Type: missing
    Python:     initial_measurement
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_type:
    Type: missing
    Python:     initial
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].gap_days:
    Type: missing
    Python:     null
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.results[0].kalman_confidence_upper:
    Type: value
    Python:     73.69737903624514
    TypeScript: 71.08827048089722
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.540300%

  root.results[0].kalman_confidence_lower:
    Type: value
    Python:     66.06586542490956
    TypeScript: 68.67497398025748
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.799213%

  root.results[0].kalman_variance:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.results[0].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

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
    TypeScript: 69.89006749325993
    Difference: 1.513e-3
    Numeric difference exceeds tolerance: abs=1.513e-3, rel=0.002165%

  root.results[1].trend:
    Type: value
    Python:     1.6314692073911941e-7
    TypeScript: 0.0000022107438764924537
    Difference: 2.048e-6
    Numeric difference exceeds tolerance: abs=2.048e-6, rel=92.620270%

  root.results[1].trend_weekly:
    Type: value
    Python:     0.000001142028445173836
    TypeScript: 0.000015475207135447175
    Difference: 1.433e-5
    Numeric difference exceeds tolerance: abs=1.433e-5, rel=92.620270%

  root.results[1].confidence:
    Type: value
    Python:     0.9999136330673903
    TypeScript: 0.9988597812498967
    Difference: 1.054e-3
    Numeric difference exceeds tolerance: abs=1.054e-3, rel=0.105394%

  root.results[1].innovation:
    Type: value
    Python:     0.11204996488635288
    TypeScript: 0.11053719382462646
    Difference: 1.513e-3
    Numeric difference exceeds tolerance: abs=1.513e-3, rel=1.350086%

  root.results[1].normalized_innovation:
    Type: value
    Python:     0.013143109407436756
    TypeScript: 0.047767547436678034
    Difference: 3.462e-2
    Numeric difference exceeds tolerance: abs=3.462e-2, rel=72.485275%

  root.results[1].kalman_confidence_upper:
    Type: value
    Python:     73.88938039656139
    TypeScript: 71.0815292678035
    Difference: 2.808e+0
    Numeric difference exceeds tolerance: abs=2.808e+0, rel=3.800074%

  root.results[1].kalman_confidence_lower:
    Type: value
    Python:     65.88772904783502
    TypeScript: 68.69860571871637
    Difference: 2.811e+0
    Numeric difference exceeds tolerance: abs=2.811e+0, rel=4.091607%

  root.results[1].kalman_variance:
    Type: value
    Python:     4.0016515191608955
    TypeScript: 0.35489529004962833
    Difference: 3.647e+0
    Numeric difference exceeds tolerance: abs=3.647e+0, rel=91.131279%

  root.results[1].prediction_error:
    Type: value
    Python:     0.11204996488635288
    TypeScript: 0.11053719382462646
    Difference: 1.513e-3
    Numeric difference exceeds tolerance: abs=1.513e-3, rel=1.350086%

  root.results[1].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:25:59.110000
    TypeScript: 2025-11-10T10:25:59.110Z
    Value mismatch: 2025-11-10T11:25:59.110000 !== 2025-11-10T10:25:59.110Z

  root.results[1].quality_score:
    Type: value
    Python:     0.9483375916895864
    TypeScript: 0.9454754935198453
    Difference: 2.862e-3
    Numeric difference exceeds tolerance: abs=2.862e-3, rel=0.301802%

  root.results[1].quality_components.kalman_fit:
    Type: value
    Python:     0.9976753029370461
    TypeScript: 0.9897949485862356
    Difference: 7.880e-3
    Numeric difference exceeds tolerance: abs=7.880e-3, rel=0.789872%

  root.results[2].timestamp:
    Type: type
    Python:     1762770445510
    TypeScript: 2025-11-10T10:27:25.510Z
    Type mismatch: Python number, TypeScript string

  root.results[2].filtered_weight:
    Type: value
    Python:     69.89917473309745
    TypeScript: 69.89865950971816
    Difference: 5.152e-4
    Numeric difference exceeds tolerance: abs=5.152e-4, rel=0.000737%

  root.results[2].trend:
    Type: value
    Python:     0.0000016068450810693616
    TypeScript: 0.000006931465035199739
    Difference: 5.325e-6
    Numeric difference exceeds tolerance: abs=5.325e-6, rel=76.818103%

  root.results[2].trend_weekly:
    Type: value
    Python:     0.000011247915567485531
    TypeScript: 0.00004852025524639817
    Difference: 3.727e-5
    Numeric difference exceeds tolerance: abs=3.727e-5, rel=76.818103%

  root.results[2].confidence:
    Type: value
    Python:     0.9998728876085181
    TypeScript: 0.9987599141050354
    Difference: 1.113e-3
    Numeric difference exceeds tolerance: abs=1.113e-3, rel=0.111311%

  root.results[2].innovation:
    Type: value
    Python:     0.11467943721925167
    TypeScript: 0.11519466059854722
    Difference: 5.152e-4
    Numeric difference exceeds tolerance: abs=5.152e-4, rel=0.447263%

  root.results[2].normalized_innovation:
    Type: value
    Python:     0.015944934678235004
    TypeScript: 0.049816773033654185
    Difference: 3.387e-2
    Numeric difference exceeds tolerance: abs=3.387e-2, rel=67.992839%

  root.results[2].kalman_confidence_upper:
    Type: value
    Python:     73.91999060309242
    TypeScript: 71.07686324711212
    Difference: 2.843e+0
    Numeric difference exceeds tolerance: abs=2.843e+0, rel=3.846223%

  root.results[2].kalman_confidence_lower:
    Type: value
    Python:     65.87835886310249
    TypeScript: 68.7204557723242
    Difference: 2.842e+0
    Numeric difference exceeds tolerance: abs=2.842e+0, rel=4.135736%

  root.results[2].kalman_variance:
    Type: value
    Python:     4.041740065100835
    TypeScript: 0.34704101170227364
    Difference: 3.695e+0
    Numeric difference exceeds tolerance: abs=3.695e+0, rel=91.413574%

  root.results[2].prediction_error:
    Type: value
    Python:     0.11467943721925167
    TypeScript: 0.11519466059854722
    Difference: 5.152e-4
    Numeric difference exceeds tolerance: abs=5.152e-4, rel=0.447263%

  root.results[2].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:27:25.510000
    TypeScript: 2025-11-10T10:27:25.510Z
    Value mismatch: 2025-11-10T11:27:25.510000 !== 2025-11-10T10:27:25.510Z

  root.results[2].quality_score:
    Type: value
    Python:     0.9477124619928798
    TypeScript: 0.9448830987236454
    Difference: 2.829e-3
    Numeric difference exceeds tolerance: abs=2.829e-3, rel=0.298547%

  root.results[2].quality_components.kalman_fit:
    Type: value
    Python:     0.997552861181671
    TypeScript: 0.9893761844180352
    Difference: 8.177e-3
    Numeric difference exceeds tolerance: abs=8.177e-3, rel=0.819674%

  root.results[2].quality_components.temporal_consistency:
    Type: value
    Python:     0.9558417133453186
    TypeScript: 0.9563118148844911
    Difference: 4.701e-4
    Numeric difference exceeds tolerance: abs=4.701e-4, rel=0.049158%

  root.results[3].timestamp:
    Type: type
    Python:     1762770531910
    TypeScript: 2025-11-10T10:28:51.910Z
    Type mismatch: Python number, TypeScript string

  root.results[3].filtered_weight:
    Type: value
    Python:     69.92231645736025
    TypeScript: 69.91250730728238
    Difference: 9.809e-3
    Numeric difference exceeds tolerance: abs=9.809e-3, rel=0.014029%

  root.results[3].trend:
    Type: value
    Python:     0.0000087318470573641
    TypeScript: 0.0000188665370520033
    Difference: 1.013e-5
    Numeric difference exceeds tolerance: abs=1.013e-5, rel=53.717807%

  root.results[3].trend_weekly:
    Type: value
    Python:     0.0000611229294015487
    TypeScript: 0.0001320657593640231
    Difference: 7.094e-5
    Numeric difference exceeds tolerance: abs=7.094e-5, rel=53.717807%

  root.results[3].confidence:
    Type: value
    Python:     0.9995682870458482
    TypeScript: 0.9966385081655432
    Difference: 2.930e-3
    Numeric difference exceeds tolerance: abs=2.930e-3, rel=0.293104%

  root.results[3].innovation:
    Type: value
    Python:     0.17982964748230756
    TypeScript: 0.18963879756017832
    Difference: 9.809e-3
    Numeric difference exceeds tolerance: abs=9.809e-3, rel=5.172544%

  root.results[3].normalized_innovation:
    Type: value
    Python:     0.029387281909636038
    TypeScript: 0.08206283374758021
    Difference: 5.268e-2
    Numeric difference exceeds tolerance: abs=5.268e-2, rel=64.189292%

  root.results[3].kalman_confidence_upper:
    Type: value
    Python:     73.8376220916935
    TypeScript: 71.07912166915582
    Difference: 2.759e+0
    Numeric difference exceeds tolerance: abs=2.759e+0, rel=3.735901%

  root.results[3].kalman_confidence_lower:
    Type: value
    Python:     66.007010823027
    TypeScript: 68.74589294540894
    Difference: 2.739e+0
    Numeric difference exceeds tolerance: abs=2.739e+0, rel=3.984067%

  root.results[3].kalman_variance:
    Type: value
    Python:     3.8324045525604276
    TypeScript: 0.3402472673323422
    Difference: 3.492e+0
    Numeric difference exceeds tolerance: abs=3.492e+0, rel=91.121833%

  root.results[3].prediction_error:
    Type: value
    Python:     0.17982964748230756
    TypeScript: 0.18963879756017832
    Difference: 9.809e-3
    Numeric difference exceeds tolerance: abs=9.809e-3, rel=5.172544%

  root.results[3].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:28:51.910000
    TypeScript: 2025-11-10T10:28:51.910Z
    Value mismatch: 2025-11-10T11:28:51.910000 !== 2025-11-10T10:28:51.910Z

  root.results[3].quality_score:
    Type: value
    Python:     0.9181140831089427
    TypeScript: 0.913463369176142
    Difference: 4.651e-3
    Numeric difference exceeds tolerance: abs=4.651e-3, rel=0.506551%

  root.results[3].quality_components.kalman_fit:
    Type: value
    Python:     0.9960359871626436
    TypeScript: 0.982583293755753
    Difference: 1.345e-2
    Numeric difference exceeds tolerance: abs=1.345e-2, rel=1.350623%

  root.results[3].quality_components.temporal_consistency:
    Type: value
    Python:     0.933512182633808
    TypeScript: 0.9333752918340021
    Difference: 1.369e-4
    Numeric difference exceeds tolerance: abs=1.369e-4, rel=0.014664%

  root.results[4].timestamp:
    Type: type
    Python:     1762770618310
    TypeScript: 2025-11-10T10:30:18.310Z
    Type mismatch: Python number, TypeScript string

  root.results[4].filtered_weight:
    Type: value
    Python:     69.95256501595752
    TypeScript: 69.92732773096515
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.036078%

  root.results[4].trend:
    Type: value
    Python:     0.000024884229398918055
    TypeScript: 0.000036618967884249657
    Difference: 1.173e-5
    Numeric difference exceeds tolerance: abs=1.173e-5, rel=32.045519%

  root.results[4].trend_weekly:
    Type: value
    Python:     0.0001741896057924264
    TypeScript: 0.0002563327751897476
    Difference: 8.214e-5
    Numeric difference exceeds tolerance: abs=8.214e-5, rel=32.045519%

  root.results[4].confidence:
    Type: value
    Python:     0.999403854042189
    TypeScript: 0.9960002983806998
    Difference: 3.404e-3
    Numeric difference exceeds tolerance: abs=3.404e-3, rel=0.340559%

  root.results[4].innovation:
    Type: value
    Python:     0.18154097012779857
    TypeScript: 0.20677825512017023
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=12.204999%

  root.results[4].normalized_innovation:
    Type: value
    Python:     0.034534728128812156
    TypeScript: 0.08952901003102655
    Difference: 5.499e-2
    Numeric difference exceeds tolerance: abs=5.499e-2, rel=61.426215%

  root.results[4].kalman_confidence_upper:
    Type: value
    Python:     73.66923833988415
    TypeScript: 71.08380187669908
    Difference: 2.585e+0
    Numeric difference exceeds tolerance: abs=2.585e+0, rel=3.509520%

  root.results[4].kalman_confidence_lower:
    Type: value
    Python:     66.23589169203089
    TypeScript: 68.77085358523121
    Difference: 2.535e+0
    Numeric difference exceeds tolerance: abs=2.535e+0, rel=3.686099%

  root.results[4].kalman_variance:
    Type: value
    Python:     3.453415149196944
    TypeScript: 0.33435811243776226
    Difference: 3.119e+0
    Numeric difference exceeds tolerance: abs=3.119e+0, rel=90.318045%

  root.results[4].prediction_error:
    Type: value
    Python:     0.18154097012779857
    TypeScript: 0.20677825512017023
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=12.204999%

  root.results[4].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.results[4].quality_score:
    Type: value
    Python:     0.915025196010972
    TypeScript: 0.909244155629429
    Difference: 5.781e-3
    Numeric difference exceeds tolerance: abs=5.781e-3, rel=0.631790%

  root.results[4].quality_components.kalman_fit:
    Type: value
    Python:     0.9958574092838801
    TypeScript: 0.9810360726506892
    Difference: 1.482e-2
    Numeric difference exceeds tolerance: abs=1.482e-2, rel=1.488299%

  root.results[4].quality_components.temporal_consistency:
    Type: value
    Python:     0.9311885240584878
    TypeScript: 0.9286512131397915
    Difference: 2.537e-3
    Numeric difference exceeds tolerance: abs=2.537e-3, rel=0.272481%

  root.results[5].timestamp:
    Type: type
    Python:     1762770704710
    TypeScript: 2025-11-10T10:31:44.710Z
    Type mismatch: Python number, TypeScript string

  root.results[5].quality_components.kalman_fit:
    Type: value
    Python:     0.8224108364823537
    TypeScript: 0.4239264931245335
    Difference: 3.985e-1
    Numeric difference exceeds tolerance: abs=3.985e-1, rel=48.453197%

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
    TypeScript: 0.4239264931245335
    Difference: 3.985e-1
    Numeric difference exceeds tolerance: abs=3.985e-1, rel=48.453197%

  root.results[5].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     -9.952567504380454
    TypeScript: -9.92733139286193
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.253564%

  root.results[5].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     69.95256750438045
    TypeScript: 69.92733139286193
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.036076%

  root.results[5].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     0.9775760385574465
    TypeScript: 4.290976020260009
    Difference: 3.313e+0
    Numeric difference exceeds tolerance: abs=3.313e+0, rel=77.217863%

  root.results[5].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     0.95565491116167
    TypeScript: 18.412475206446427
    Difference: 1.746e+1
    Numeric difference exceeds tolerance: abs=1.746e+1, rel=94.809742%

  root.results[5].quality_details.metadata.kalman_fit.p_value:
    Type: value
    Python:     0.32828405313327047
    TypeScript: 0.000017788952555441995
    Difference: 3.283e-1
    Numeric difference exceeds tolerance: abs=3.283e-1, rel=99.994581%

  root.results[5].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.8224108364823537
    TypeScript: 0.4239264931245335
    Difference: 3.985e-1
    Numeric difference exceeds tolerance: abs=3.985e-1, rel=48.453197%

  root.results[5].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     9.952565015957518
    TypeScript: 9.927327730965146
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.253576%

  root.results[6].timestamp:
    Type: type
    Python:     1762770791110
    TypeScript: 2025-11-10T10:33:11.110Z
    Type mismatch: Python number, TypeScript string

  root.results[6].quality_components.kalman_fit:
    Type: value
    Python:     0.8248052640992843
    TypeScript: 0.4293848406418379
    Difference: 3.954e-1
    Numeric difference exceeds tolerance: abs=3.954e-1, rel=47.941064%

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
    TypeScript: 0.4293848406418379
    Difference: 3.954e-1
    Numeric difference exceeds tolerance: abs=3.954e-1, rel=47.941064%

  root.results[6].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     -9.80457625925483
    TypeScript: -9.779340147736306
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.257391%

  root.results[6].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     69.95256750438045
    TypeScript: 69.92733139286193
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.036076%

  root.results[6].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     0.9630398201306519
    TypeScript: 4.2270084887138095
    Difference: 3.264e+0
    Numeric difference exceeds tolerance: abs=3.264e+0, rel=77.216989%

  root.results[6].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     0.9274456951572784
    TypeScript: 17.867600763658604
    Difference: 1.694e+1
    Numeric difference exceeds tolerance: abs=1.694e+1, rel=94.809344%

  root.results[6].quality_details.metadata.kalman_fit.p_value:
    Type: value
    Python:     0.3355275403554654
    TypeScript: 0.000023681873678738086
    Difference: 3.355e-1
    Numeric difference exceeds tolerance: abs=3.355e-1, rel=99.992942%

  root.results[6].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.8248052640992843
    TypeScript: 0.4293848406418379
    Difference: 3.954e-1
    Numeric difference exceeds tolerance: abs=3.954e-1, rel=47.941064%

  root.results[6].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     9.804573770831894
    TypeScript: 9.779336485839522
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.257403%

  root.results[7].timestamp:
    Type: type
    Python:     1762770877510
    TypeScript: 2025-11-10T10:34:37.510Z
    Type mismatch: Python number, TypeScript string

  root.results[7].quality_components.kalman_fit:
    Type: value
    Python:     0.8184182995940397
    TypeScript: 0.41494449081587526
    Difference: 4.035e-1
    Numeric difference exceeds tolerance: abs=4.035e-1, rel=49.299216%

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
    TypeScript: 0.41494449081587526
    Difference: 4.035e-1
    Numeric difference exceeds tolerance: abs=4.035e-1, rel=49.299216%

  root.results[7].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     -10.200293178711064
    TypeScript: -10.17505706719254
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.247406%

  root.results[7].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     69.95256750438045
    TypeScript: 69.92733139286193
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.036076%

  root.results[7].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     1.0019085219346757
    TypeScript: 4.398052623839467
    Difference: 3.396e+0
    Numeric difference exceeds tolerance: abs=3.396e+0, rel=77.219269%

  root.results[7].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     1.0038206863253265
    TypeScript: 19.34286688206122
    Difference: 1.834e+1
    Numeric difference exceeds tolerance: abs=1.834e+1, rel=94.810383%

  root.results[7].quality_details.metadata.kalman_fit.p_value:
    Type: value
    Python:     0.3163877763595022
    TypeScript: 0.000010922649883005064
    Difference: 3.164e-1
    Numeric difference exceeds tolerance: abs=3.164e-1, rel=99.996548%

  root.results[7].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.8184182995940397
    TypeScript: 0.41494449081587526
    Difference: 4.035e-1
    Numeric difference exceeds tolerance: abs=4.035e-1, rel=49.299216%

  root.results[7].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     10.200290690288128
    TypeScript: 10.175053405295756
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.247417%

  root.results[8].timestamp:
    Type: type
    Python:     1762770963910
    TypeScript: 2025-11-10T10:36:03.910Z
    Type mismatch: Python number, TypeScript string

  root.results[8].reason:
    Type: value
    Python:     Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20)
    TypeScript: Quality score 0.27 below threshold 0.46 (weakest: temporal_consistency=0.20)
    Value mismatch: Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20) !== Quality score 0.27 below threshold 0.46 (weakest: temporal_consistency=0.20)

  root.results[8].quality_score:
    Type: value
    Python:     0.3108894486276372
    TypeScript: 0.27162088167720494
    Difference: 3.927e-2
    Numeric difference exceeds tolerance: abs=3.927e-2, rel=12.631039%

  root.results[8].quality_components.kalman_fit:
    Type: value
    Python:     0.8206629541503064
    TypeScript: 0.4199759888892275
    Difference: 4.007e-1
    Numeric difference exceeds tolerance: abs=4.007e-1, rel=48.824790%

  root.results[8].quality_details.rejection_reason:
    Type: missing
    Python:     Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20)
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[8].quality_details.rejectionReason:
    Type: extra
    Python:     undefined
    TypeScript: Quality score 0.27 below threshold 0.46 (weakest: temporal_consistency=0.20)
    Extra key in TypeScript output

  root.results[8].quality_details.overall:
    Type: value
    Python:     0.3108894486276372
    TypeScript: 0.27162088167720494
    Difference: 3.927e-2
    Numeric difference exceeds tolerance: abs=3.927e-2, rel=12.631039%

  root.results[8].quality_details.components.kalman_fit:
    Type: value
    Python:     0.8206629541503064
    TypeScript: 0.4199759888892275
    Difference: 4.007e-1
    Numeric difference exceeds tolerance: abs=4.007e-1, rel=48.824790%

  root.results[8].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     -10.060870351093804
    TypeScript: -10.03563423957528
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.250834%

  root.results[8].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     69.95256750438045
    TypeScript: 69.92733139286193
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.036076%

  root.results[8].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     0.9882139234859267
    TypeScript: 4.337788693251568
    Difference: 3.350e+0
    Numeric difference exceeds tolerance: abs=3.350e+0, rel=77.218486%

  root.results[8].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     0.976566758571449
    TypeScript: 18.816410747301145
    Difference: 1.784e+1
    Numeric difference exceeds tolerance: abs=1.784e+1, rel=94.810026%

  root.results[8].quality_details.metadata.kalman_fit.p_value:
    Type: value
    Python:     0.3230478905699903
    TypeScript: 0.000014392342337576913
    Difference: 3.230e-1
    Numeric difference exceeds tolerance: abs=3.230e-1, rel=99.995545%

  root.results[8].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.8206629541503064
    TypeScript: 0.4199759888892275
    Difference: 4.007e-1
    Numeric difference exceeds tolerance: abs=4.007e-1, rel=48.824790%

  root.results[8].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     10.060867862670868
    TypeScript: 10.035630577678496
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.250846%

  root.results[9].timestamp:
    Type: type
    Python:     1762771050310
    TypeScript: 2025-11-10T10:37:30.310Z
    Type mismatch: Python number, TypeScript string

  root.results[9].reason:
    Type: value
    Python:     Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20)
    TypeScript: Quality score 0.27 below threshold 0.46 (weakest: temporal_consistency=0.20)
    Value mismatch: Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20) !== Quality score 0.27 below threshold 0.46 (weakest: temporal_consistency=0.20)

  root.results[9].quality_score:
    Type: value
    Python:     0.3111106221270487
    TypeScript: 0.27306775063363475
    Difference: 3.804e-2
    Numeric difference exceeds tolerance: abs=3.804e-2, rel=12.228085%

  root.results[9].quality_components.kalman_fit:
    Type: value
    Python:     0.8245312906625418
    TypeScript: 0.42875755317717884
    Difference: 3.958e-1
    Numeric difference exceeds tolerance: abs=3.958e-1, rel=47.999845%

  root.results[9].quality_details.rejection_reason:
    Type: missing
    Python:     Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20)
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[9].quality_details.rejectionReason:
    Type: extra
    Python:     undefined
    TypeScript: Quality score 0.27 below threshold 0.46 (weakest: temporal_consistency=0.20)
    Extra key in TypeScript output

  root.results[9].quality_details.overall:
    Type: value
    Python:     0.3111106221270487
    TypeScript: 0.27306775063363475
    Difference: 3.804e-2
    Numeric difference exceeds tolerance: abs=3.804e-2, rel=12.228085%

  root.results[9].quality_details.components.kalman_fit:
    Type: value
    Python:     0.8245312906625418
    TypeScript: 0.42875755317717884
    Difference: 3.958e-1
    Numeric difference exceeds tolerance: abs=3.958e-1, rel=47.999845%

  root.results[9].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     -9.82148782276871
    TypeScript: -9.796251711250186
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.256948%

  root.results[9].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     69.95256750438045
    TypeScript: 69.92733139286193
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.036076%

  root.results[9].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     0.964700933130733
    TypeScript: 4.234318319586922
    Difference: 3.270e+0
    Numeric difference exceeds tolerance: abs=3.270e+0, rel=77.217090%

  root.results[9].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     0.930647890383307
    TypeScript: 17.929451631589416
    Difference: 1.700e+1
    Numeric difference exceeds tolerance: abs=1.700e+1, rel=94.809390%

  root.results[9].quality_details.metadata.kalman_fit.p_value:
    Type: value
    Python:     0.33469462774573233
    TypeScript: 0.000022924603160445756
    Difference: 3.347e-1
    Numeric difference exceeds tolerance: abs=3.347e-1, rel=99.993151%

  root.results[9].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.8245312906625418
    TypeScript: 0.42875755317717884
    Difference: 3.958e-1
    Numeric difference exceeds tolerance: abs=3.958e-1, rel=47.999845%

  root.results[9].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     9.821485334345773
    TypeScript: 9.796248049353402
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.256960%

  root.results[10].timestamp:
    Type: type
    Python:     1762771136710
    TypeScript: 2025-11-10T10:38:56.710Z
    Type mismatch: Python number, TypeScript string

  root.results[10].reason:
    Type: value
    Python:     Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20)
    TypeScript: Quality score 0.27 below threshold 0.46 (weakest: temporal_consistency=0.20)
    Value mismatch: Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20) !== Quality score 0.27 below threshold 0.46 (weakest: temporal_consistency=0.20)

  root.results[10].quality_score:
    Type: value
    Python:     0.31120596722967214
    TypeScript: 0.2736884294047871
    Difference: 3.752e-2
    Numeric difference exceeds tolerance: abs=3.752e-2, rel=12.055533%

  root.results[10].quality_components.kalman_fit:
    Type: value
    Python:     0.8262084426231454
    TypeScript: 0.43260867321946955
    Difference: 3.936e-1
    Numeric difference exceeds tolerance: abs=3.936e-1, rel=47.639282%

  root.results[10].quality_details.rejection_reason:
    Type: missing
    Python:     Quality score 0.31 below threshold 0.46 (weakest: temporal_consistency=0.20)
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[10].quality_details.rejectionReason:
    Type: extra
    Python:     undefined
    TypeScript: Quality score 0.27 below threshold 0.46 (weakest: temporal_consistency=0.20)
    Extra key in TypeScript output

  root.results[10].quality_details.overall:
    Type: value
    Python:     0.31120596722967214
    TypeScript: 0.2736884294047871
    Difference: 3.752e-2
    Numeric difference exceeds tolerance: abs=3.752e-2, rel=12.055533%

  root.results[10].quality_details.components.kalman_fit:
    Type: value
    Python:     0.8262084426231454
    TypeScript: 0.43260867321946955
    Difference: 3.936e-1
    Numeric difference exceeds tolerance: abs=3.936e-1, rel=47.639282%

  root.results[10].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     -9.718050197194025
    TypeScript: -9.6928140856755
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.259683%

  root.results[10].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     69.95256750438045
    TypeScript: 69.92733139286193
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.036076%

  root.results[10].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     0.9545409272626408
    TypeScript: 4.18960858306571
    Difference: 3.235e+0
    Numeric difference exceeds tolerance: abs=3.235e+0, rel=77.216465%

  root.results[10].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     0.9111483818194221
    TypeScript: 17.552820079297867
    Difference: 1.664e+1
    Numeric difference exceeds tolerance: abs=1.664e+1, rel=94.809105%

  root.results[10].quality_details.metadata.kalman_fit.p_value:
    Type: value
    Python:     0.3398099027482664
    TypeScript: 0.000027943604420710422
    Difference: 3.398e-1
    Numeric difference exceeds tolerance: abs=3.398e-1, rel=99.991777%

  root.results[10].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.8262084426231454
    TypeScript: 0.43260867321946955
    Difference: 3.936e-1
    Numeric difference exceeds tolerance: abs=3.936e-1, rel=47.639282%

  root.results[10].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     9.718047708771088
    TypeScript: 9.692810423778717
    Difference: 2.524e-2
    Numeric difference exceeds tolerance: abs=2.524e-2, rel=0.259695%

  root.finalState.adaptation_state:
    Type: extra
    Python:     undefined
    TypeScript: {}
    Extra key in TypeScript output

  root.finalState.version:
    Type: extra
    Python:     undefined
    TypeScript: 1
    Extra key in TypeScript output

  root.finalState.kalman_params.initial_state_covariance[0][0]:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.finalState.kalman_params.transition_covariance[0][0]:
    Type: value
    Python:     0.19607272887128602
    TypeScript: 0.018
    Difference: 1.781e-1
    Numeric difference exceeds tolerance: abs=1.781e-1, rel=90.819733%

  root.finalState.kalman_params.transition_covariance[1][1]:
    Type: value
    Python:     0.0013071515258085737
    TypeScript: 0.00012
    Difference: 1.187e-3
    Numeric difference exceeds tolerance: abs=1.187e-3, rel=90.819733%

  root.finalState.kalman_params.observation_covariance[0][0]:
    Type: value
    Python:     100
    TypeScript: 5
    Difference: 9.500e+1
    Numeric difference exceeds tolerance: abs=9.500e+1, rel=95.000000%

  root.finalState.last_state[0][0]:
    Type: type
    Python:     69.92231645736025
    TypeScript: [
  69.91250730728238
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[0][1]:
    Type: type
    Python:     0.0000087318470573641
    TypeScript: [
  0.0000188665370520033
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][0]:
    Type: type
    Python:     69.95256501595752
    TypeScript: [
  69.92732773096515
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][1]:
    Type: type
    Python:     0.000024884229398918055
    TypeScript: [
  0.000036618967884249657
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_covariance[0][0][0]:
    Type: value
    Python:     3.8324045525604276
    TypeScript: 0.3402472673323422
    Difference: 3.492e+0
    Numeric difference exceeds tolerance: abs=3.492e+0, rel=91.121833%

  root.finalState.last_covariance[0][0][1]:
    Type: value
    Python:     0.0011799500893092253
    TypeScript: 0.00029326532946001945
    Difference: 8.867e-4
    Numeric difference exceeds tolerance: abs=8.867e-4, rel=75.145955%

  root.finalState.last_covariance[0][1][0]:
    Type: value
    Python:     0.0011799500893092255
    TypeScript: 0.00029326532946001945
    Difference: 8.867e-4
    Numeric difference exceeds tolerance: abs=8.867e-4, rel=75.145955%

  root.finalState.last_covariance[0][1][1]:
    Type: value
    Python:     0.009714504341111076
    TypeScript: 0.0013599718709253784
    Difference: 8.355e-3
    Numeric difference exceeds tolerance: abs=8.355e-3, rel=86.000605%

  root.finalState.last_covariance[1][0][0]:
    Type: value
    Python:     3.453415149196944
    TypeScript: 0.33435811243776226
    Difference: 3.119e+0
    Numeric difference exceeds tolerance: abs=3.119e+0, rel=90.318045%

  root.finalState.last_covariance[1][0][1]:
    Type: value
    Python:     0.001844137199694154
    TypeScript: 0.00040055703559759796
    Difference: 1.444e-3
    Numeric difference exceeds tolerance: abs=1.444e-3, rel=78.279434%

  root.finalState.last_covariance[1][1][0]:
    Type: value
    Python:     0.001844137199694154
    TypeScript: 0.00040055703559759796
    Difference: 1.444e-3
    Numeric difference exceeds tolerance: abs=1.444e-3, rel=78.279434%

  root.finalState.last_covariance[1][1][1]:
    Type: value
    Python:     0.011021491787102742
    TypeScript: 0.0014799374821011539
    Difference: 9.542e-3
    Numeric difference exceeds tolerance: abs=9.542e-3, rel=86.572258%

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

  root.finalState.measurement_history[1].quality_score:
    Type: value
    Python:     0.9483375916895864
    TypeScript: 0.9454754935198453
    Difference: 2.862e-3
    Numeric difference exceeds tolerance: abs=2.862e-3, rel=0.301802%

  root.finalState.measurement_history[2].timestamp:
    Type: value
    Python:     2025-11-10T11:27:25.510000
    TypeScript: 2025-11-10T10:27:25.510Z
    Value mismatch: 2025-11-10T11:27:25.510000 !== 2025-11-10T10:27:25.510Z

  root.finalState.measurement_history[2].quality_score:
    Type: value
    Python:     0.9477124619928798
    TypeScript: 0.9448830987236454
    Difference: 2.829e-3
    Numeric difference exceeds tolerance: abs=2.829e-3, rel=0.298547%

  root.finalState.measurement_history[3].timestamp:
    Type: value
    Python:     2025-11-10T11:28:51.910000
    TypeScript: 2025-11-10T10:28:51.910Z
    Value mismatch: 2025-11-10T11:28:51.910000 !== 2025-11-10T10:28:51.910Z

  root.finalState.measurement_history[3].quality_score:
    Type: value
    Python:     0.9181140831089427
    TypeScript: 0.913463369176142
    Difference: 4.651e-3
    Numeric difference exceeds tolerance: abs=4.651e-3, rel=0.506551%

  root.finalState.measurement_history[4].timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.finalState.measurement_history[4].quality_score:
    Type: value
    Python:     0.915025196010972
    TypeScript: 0.909244155629429
    Difference: 5.781e-3
    Numeric difference exceeds tolerance: abs=5.781e-3, rel=0.631790%

  root.finalState.temporal_baseline.last_timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

```

### Test 4: Quality Rejection
**Description**: Mix of good and bad measurements - verify rejection logic

**Comparison**: ✗ Found 88 difference(s): 59 numeric, 29 structural

**Differences**:
```
Found 88 difference(s):

  root.results[0].was_reset:
    Type: missing
    Python:     true
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_reason:
    Type: missing
    Python:     initial_measurement
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_type:
    Type: missing
    Python:     initial
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].gap_days:
    Type: missing
    Python:     null
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.results[0].kalman_confidence_upper:
    Type: value
    Python:     73.81575680566779
    TypeScript: 71.20664825031987
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.534623%

  root.results[0].kalman_confidence_lower:
    Type: value
    Python:     66.18424319433221
    TypeScript: 68.79335174968013
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.792675%

  root.results[0].kalman_variance:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.results[0].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

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
    TypeScript: 70.01419581160198
    Difference: 2.543e-3
    Numeric difference exceeds tolerance: abs=2.543e-3, rel=0.003632%

  root.results[1].trend:
    Type: value
    Python:     2.742369346345332e-7
    TypeScript: 0.0000037160837679603503
    Difference: 3.442e-6
    Numeric difference exceeds tolerance: abs=3.442e-6, rel=92.620270%

  root.results[1].trend_weekly:
    Type: value
    Python:     0.000001919658542441732
    TypeScript: 0.00002601258637572245
    Difference: 2.409e-5
    Numeric difference exceeds tolerance: abs=2.409e-5, rel=92.620270%

  root.results[1].confidence:
    Type: value
    Python:     0.9997559901329914
    TypeScript: 0.9967816725409628
    Difference: 2.974e-3
    Numeric difference exceeds tolerance: abs=2.974e-3, rel=0.297504%

  root.results[1].innovation:
    Type: value
    Python:     0.188347035648178
    TypeScript: 0.18580418839802348
    Difference: 2.543e-3
    Numeric difference exceeds tolerance: abs=2.543e-3, rel=1.350086%

  root.results[1].normalized_innovation:
    Type: value
    Python:     0.022092516482277752
    TypeScript: 0.08029342953393039
    Difference: 5.820e-2
    Numeric difference exceeds tolerance: abs=5.820e-2, rel=72.485275%

  root.results[1].kalman_confidence_upper:
    Type: value
    Python:     74.01247863871501
    TypeScript: 71.20565758614555
    Difference: 2.807e+0
    Numeric difference exceeds tolerance: abs=2.807e+0, rel=3.792362%

  root.results[1].kalman_confidence_lower:
    Type: value
    Python:     66.01082728998864
    TypeScript: 68.82273403705841
    Difference: 2.812e+0
    Numeric difference exceeds tolerance: abs=2.812e+0, rel=4.085724%

  root.results[1].kalman_variance:
    Type: value
    Python:     4.0016515191608955
    TypeScript: 0.35489529004962833
    Difference: 3.647e+0
    Numeric difference exceeds tolerance: abs=3.647e+0, rel=91.131279%

  root.results[1].prediction_error:
    Type: value
    Python:     0.188347035648178
    TypeScript: 0.18580418839802348
    Difference: 2.543e-3
    Numeric difference exceeds tolerance: abs=2.543e-3, rel=1.350086%

  root.results[1].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:25:59.110000
    TypeScript: 2025-11-10T10:25:59.110Z
    Value mismatch: 2025-11-10T11:25:59.110000 !== 2025-11-10T10:25:59.110Z

  root.results[1].quality_score:
    Type: value
    Python:     0.9407390439689578
    TypeScript: 0.9359941824316373
    Difference: 4.745e-3
    Numeric difference exceeds tolerance: abs=4.745e-3, rel=0.504376%

  root.results[1].quality_components.kalman_fit:
    Type: value
    Python:     0.9960954635563457
    TypeScript: 0.9829057887319245
    Difference: 1.319e-2
    Numeric difference exceeds tolerance: abs=1.319e-2, rel=1.324138%

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
    TypeScript: 1.8011100717405372e-7
    Difference: 2.961e-2
    Numeric difference exceeds tolerance: abs=2.961e-2, rel=99.999392%

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
    TypeScript: 1.8011100717405372e-7
    Difference: 2.961e-2
    Numeric difference exceeds tolerance: abs=2.961e-2, rel=99.999392%

  root.results[3].quality_details.metadata.kalman_fit.innovation:
    Type: value
    Python:     179.98834700822448
    TypeScript: 179.98580381678966
    Difference: 2.543e-3
    Numeric difference exceeds tolerance: abs=2.543e-3, rel=0.001413%

  root.results[3].quality_details.metadata.kalman_fit.prediction:
    Type: value
    Python:     70.01165299177552
    TypeScript: 70.01419618321036
    Difference: 2.543e-3
    Numeric difference exceeds tolerance: abs=2.543e-3, rel=0.003632%

  root.results[3].quality_details.metadata.kalman_fit.normalized_innovation:
    Type: value
    Python:     17.59769674792355
    TypeScript: 77.648462348095
    Difference: 6.005e+1
    Numeric difference exceeds tolerance: abs=6.005e+1, rel=77.336709%

  root.results[3].quality_details.metadata.kalman_fit.chi_squared:
    Type: value
    Python:     309.6789308318791
    TypeScript: 6029.283705023526
    Difference: 5.720e+3
    Numeric difference exceeds tolerance: abs=5.720e+3, rel=94.863753%

  root.results[3].quality_details.metadata.kalman_fit.score:
    Type: value
    Python:     0.0296130733009591
    TypeScript: 1.8011100717405372e-7
    Difference: 2.961e-2
    Numeric difference exceeds tolerance: abs=2.961e-2, rel=99.999392%

  root.results[3].quality_details.metadata.temporal_consistency.actual_change:
    Type: value
    Python:     179.9883470356482
    TypeScript: 179.98580418839802
    Difference: 2.543e-3
    Numeric difference exceeds tolerance: abs=2.543e-3, rel=0.001413%

  root.results[4].timestamp:
    Type: type
    Python:     1762770618310
    TypeScript: 2025-11-10T10:30:18.310Z
    Type mismatch: Python number, TypeScript string

  root.results[4].filtered_weight:
    Type: value
    Python:     70.01914101255421
    TypeScript: 70.02015167188767
    Difference: 1.011e-3
    Numeric difference exceeds tolerance: abs=1.011e-3, rel=0.001443%

  root.results[4].trend:
    Type: value
    Python:     0.0000012921698287282735
    TypeScript: 0.000006988298730430238
    Difference: 5.696e-6
    Numeric difference exceeds tolerance: abs=5.696e-6, rel=81.509522%

  root.results[4].trend_weekly:
    Type: value
    Python:     0.000009045188801097915
    TypeScript: 0.00004891809111301166
    Difference: 3.987e-5
    Numeric difference exceeds tolerance: abs=3.987e-5, rel=81.509522%

  root.results[4].confidence:
    Type: value
    Python:     0.9999368043389365
    TypeScript: 0.9994039829440177
    Difference: 5.328e-4
    Numeric difference exceeds tolerance: abs=5.328e-4, rel=0.053286%

  root.results[4].innovation:
    Type: value
    Python:     0.08085898744577946
    TypeScript: 0.07984832811231968
    Difference: 1.011e-3
    Numeric difference exceeds tolerance: abs=1.011e-3, rel=1.249904%

  root.results[4].normalized_innovation:
    Type: value
    Python:     0.011242567143979145
    TypeScript: 0.034530993172945244
    Difference: 2.329e-2
    Numeric difference exceeds tolerance: abs=2.329e-2, rel=67.442097%

  root.results[4].kalman_confidence_upper:
    Type: value
    Python:     74.03995688254918
    TypeScript: 71.19835540928163
    Difference: 2.842e+0
    Numeric difference exceeds tolerance: abs=2.842e+0, rel=3.837930%

  root.results[4].kalman_confidence_lower:
    Type: value
    Python:     65.99832514255925
    TypeScript: 68.84194793449372
    Difference: 2.844e+0
    Numeric difference exceeds tolerance: abs=2.844e+0, rel=4.130654%

  root.results[4].kalman_variance:
    Type: value
    Python:     4.041740065100835
    TypeScript: 0.34704101170227364
    Difference: 3.695e+0
    Numeric difference exceeds tolerance: abs=3.695e+0, rel=91.413574%

  root.results[4].prediction_error:
    Type: value
    Python:     0.08085898744577946
    TypeScript: 0.07984832811231968
    Difference: 1.011e-3
    Numeric difference exceeds tolerance: abs=1.011e-3, rel=1.249904%

  root.results[4].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.results[4].quality_score:
    Type: value
    Python:     0.9630516768743426
    TypeScript: 0.961190717010834
    Difference: 1.861e-3
    Numeric difference exceeds tolerance: abs=1.861e-3, rel=0.193236%

  root.results[4].quality_components.kalman_fit:
    Type: value
    Python:     0.9982739308009785
    TypeScript: 0.9926239377293191
    Difference: 5.650e-3
    Numeric difference exceeds tolerance: abs=5.650e-3, rel=0.565976%

  root.results[4].quality_components.temporal_consistency:
    Type: value
    Python:     0.9680018768103804
    TypeScript: 0.9688470293216866
    Difference: 8.452e-4
    Numeric difference exceeds tolerance: abs=8.452e-4, rel=0.087233%

  root.finalState.adaptation_state:
    Type: extra
    Python:     undefined
    TypeScript: {}
    Extra key in TypeScript output

  root.finalState.version:
    Type: extra
    Python:     undefined
    TypeScript: 1
    Extra key in TypeScript output

  root.finalState.kalman_params.initial_state_covariance[0][0]:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.finalState.kalman_params.transition_covariance[0][0]:
    Type: value
    Python:     0.41430814635138935
    TypeScript: 0.018
    Difference: 3.963e-1
    Numeric difference exceeds tolerance: abs=3.963e-1, rel=95.655408%

  root.finalState.kalman_params.transition_covariance[1][1]:
    Type: value
    Python:     0.002762054309009263
    TypeScript: 0.00012
    Difference: 2.642e-3
    Numeric difference exceeds tolerance: abs=2.642e-3, rel=95.655408%

  root.finalState.kalman_params.observation_covariance[0][0]:
    Type: value
    Python:     100
    TypeScript: 5
    Difference: 9.500e+1
    Numeric difference exceeds tolerance: abs=9.500e+1, rel=95.000000%

  root.finalState.last_state[0][0]:
    Type: type
    Python:     70.01165296435182
    TypeScript: [
  70.01419581160198
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[0][1]:
    Type: type
    Python:     2.742369346345332e-7
    TypeScript: [
  0.0000037160837679603503
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][0]:
    Type: type
    Python:     70.01914101255421
    TypeScript: [
  70.02015167188767
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][1]:
    Type: type
    Python:     0.0000012921698287282735
    TypeScript: [
  0.000006988298730430238
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_covariance[0][0][0]:
    Type: value
    Python:     4.0016515191608955
    TypeScript: 0.35489529004962833
    Difference: 3.647e+0
    Numeric difference exceeds tolerance: abs=3.647e+0, rel=91.131279%

  root.finalState.last_covariance[0][0][1]:
    Type: value
    Python:     0.00009417351782408612
    TypeScript: 0.00009290209419900744
    Difference: 1.271e-6
    Numeric difference exceeds tolerance: abs=1.271e-6, rel=1.350086%

  root.finalState.last_covariance[0][1][0]:
    Type: value
    Python:     0.00009417351782408611
    TypeScript: 0.00009290209419900744
    Difference: 1.271e-6
    Numeric difference exceeds tolerance: abs=1.271e-6, rel=1.350086%

  root.finalState.last_covariance[0][1][1]:
    Type: value
    Python:     0.0050614817335710915
    TypeScript: 0.0011199981419581161
    Difference: 3.941e-3
    Numeric difference exceeds tolerance: abs=3.941e-3, rel=77.872129%

  root.finalState.last_covariance[1][0][0]:
    Type: value
    Python:     4.041740065100835
    TypeScript: 0.34704101170227364
    Difference: 3.695e+0
    Numeric difference exceeds tolerance: abs=3.695e+0, rel=91.413574%

  root.finalState.last_covariance[1][0][1]:
    Type: value
    Python:     0.0005494402704312545
    TypeScript: 0.00019068003527700613
    Difference: 3.588e-4
    Numeric difference exceeds tolerance: abs=3.588e-4, rel=65.295584%

  root.finalState.last_covariance[1][1][0]:
    Type: value
    Python:     0.0005494402704312545
    TypeScript: 0.00019068003527700613
    Difference: 3.588e-4
    Numeric difference exceeds tolerance: abs=3.588e-4, rel=65.295584%

  root.finalState.last_covariance[1][1][1]:
    Type: value
    Python:     0.007823529125682899
    TypeScript: 0.001239990327817492
    Difference: 6.584e-3
    Numeric difference exceeds tolerance: abs=6.584e-3, rel=84.150499%

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

  root.finalState.measurement_history[1].quality_score:
    Type: value
    Python:     0.9407390439689578
    TypeScript: 0.9359941824316373
    Difference: 4.745e-3
    Numeric difference exceeds tolerance: abs=4.745e-3, rel=0.504376%

  root.finalState.measurement_history[2].timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.finalState.measurement_history[2].quality_score:
    Type: value
    Python:     0.9630516768743426
    TypeScript: 0.961190717010834
    Difference: 1.861e-3
    Numeric difference exceeds tolerance: abs=1.861e-3, rel=0.193236%

  root.finalState.temporal_baseline.last_timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

```

### Test 5: State Persistence
**Description**: Process in batches - verify state persistence works correctly

**Comparison**: ✗ Found 178 difference(s): 142 numeric, 36 structural

**Differences**:
```
Found 178 difference(s):

  root.results[0].was_reset:
    Type: missing
    Python:     true
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_reason:
    Type: missing
    Python:     initial_measurement
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].reset_type:
    Type: missing
    Python:     initial
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].gap_days:
    Type: missing
    Python:     null
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[0].timestamp:
    Type: type
    Python:     1762770272710
    TypeScript: 2025-11-10T10:24:32.710Z
    Type mismatch: Python number, TypeScript string

  root.results[0].kalman_confidence_upper:
    Type: value
    Python:     73.81575680566779
    TypeScript: 71.20664825031987
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.534623%

  root.results[0].kalman_confidence_lower:
    Type: value
    Python:     66.18424319433221
    TypeScript: 68.79335174968013
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.792675%

  root.results[0].kalman_variance:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.results[0].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:24:32.710000
    TypeScript: 2025-11-10T10:24:32.710Z
    Value mismatch: 2025-11-10T11:24:32.710000 !== 2025-11-10T10:24:32.710Z

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
    TypeScript: 70.00709790580099
    Difference: 1.271e-3
    Numeric difference exceeds tolerance: abs=1.271e-3, rel=0.001816%

  root.results[1].trend:
    Type: value
    Python:     1.3711846731725685e-7
    TypeScript: 0.000001858041883980043
    Difference: 1.721e-6
    Numeric difference exceeds tolerance: abs=1.721e-6, rel=92.620270%

  root.results[1].trend_weekly:
    Type: value
    Python:     9.598292712207979e-7
    TypeScript: 0.0000130062931878603
    Difference: 1.205e-5
    Numeric difference exceeds tolerance: abs=1.205e-5, rel=92.620270%

  root.results[1].confidence:
    Type: value
    Python:     0.9999389919505018
    TypeScript: 0.9991944452802651
    Difference: 7.445e-4
    Numeric difference exceeds tolerance: abs=7.445e-4, rel=0.074459%

  root.results[1].innovation:
    Type: value
    Python:     0.09417351782407479
    TypeScript: 0.09290209419900464
    Difference: 1.271e-3
    Numeric difference exceeds tolerance: abs=1.271e-3, rel=1.350086%

  root.results[1].normalized_innovation:
    Type: value
    Python:     0.011046258241137209
    TypeScript: 0.04014671476696212
    Difference: 2.910e-2
    Numeric difference exceeds tolerance: abs=2.910e-2, rel=72.485275%

  root.results[1].kalman_confidence_upper:
    Type: value
    Python:     74.0066521565391
    TypeScript: 71.19855968034456
    Difference: 2.808e+0
    Numeric difference exceeds tolerance: abs=2.808e+0, rel=3.794378%

  root.results[1].kalman_confidence_lower:
    Type: value
    Python:     66.00500080781273
    TypeScript: 68.81563613125742
    Difference: 2.811e+0
    Numeric difference exceeds tolerance: abs=2.811e+0, rel=4.084298%

  root.results[1].kalman_variance:
    Type: value
    Python:     4.0016515191608955
    TypeScript: 0.35489529004962833
    Difference: 3.647e+0
    Numeric difference exceeds tolerance: abs=3.647e+0, rel=91.131279%

  root.results[1].prediction_error:
    Type: value
    Python:     0.09417351782407479
    TypeScript: 0.09290209419900464
    Difference: 1.271e-3
    Numeric difference exceeds tolerance: abs=1.271e-3, rel=1.350086%

  root.results[1].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:25:59.110000
    TypeScript: 2025-11-10T10:25:59.110Z
    Value mismatch: 2025-11-10T11:25:59.110000 !== 2025-11-10T10:25:59.110Z

  root.results[1].quality_score:
    Type: value
    Python:     0.9502517432046031
    TypeScript: 0.9478378128444289
    Difference: 2.414e-3
    Numeric difference exceeds tolerance: abs=2.414e-3, rel=0.254031%

  root.results[1].quality_components.kalman_fit:
    Type: value
    Python:     0.9980458223730743
    TypeScript: 0.9914160522867913
    Difference: 6.630e-3
    Numeric difference exceeds tolerance: abs=6.630e-3, rel=0.664275%

  root.results[2].timestamp:
    Type: type
    Python:     1762770445510
    TypeScript: 2025-11-10T10:27:25.510Z
    Type mismatch: Python number, TypeScript string

  root.results[2].filtered_weight:
    Type: value
    Python:     70.02228404507574
    TypeScript: 70.0204870662949
    Difference: 1.797e-3
    Numeric difference exceeds tolerance: abs=1.797e-3, rel=0.002566%

  root.results[2].trend:
    Type: value
    Python:     0.000002374382648296539
    TypeScript: 0.000009214550423525659
    Difference: 6.840e-6
    Numeric difference exceeds tolerance: abs=6.840e-6, rel=74.232246%

  root.results[2].trend_weekly:
    Type: value
    Python:     0.00001662067853807577
    TypeScript: 0.00006450185296467961
    Difference: 4.788e-5
    Numeric difference exceeds tolerance: abs=4.788e-5, rel=74.232246%

  root.results[2].confidence:
    Type: value
    Python:     0.9996947673850899
    TypeScript: 0.9969911966481986
    Difference: 2.704e-3
    Numeric difference exceeds tolerance: abs=2.704e-3, rel=0.270440%

  root.results[2].innovation:
    Type: value
    Python:     0.17771595492426684
    TypeScript: 0.17951293370509802
    Difference: 1.797e-3
    Numeric difference exceeds tolerance: abs=1.797e-3, rel=1.001030%

  root.results[2].normalized_innovation:
    Type: value
    Python:     0.024709480280496986
    TypeScript: 0.07763168039669593
    Difference: 5.292e-2
    Numeric difference exceeds tolerance: abs=5.292e-2, rel=68.170881%

  root.results[2].kalman_confidence_upper:
    Type: value
    Python:     74.0430999150707
    TypeScript: 71.19869080368886
    Difference: 2.844e+0
    Numeric difference exceeds tolerance: abs=2.844e+0, rel=3.841559%

  root.results[2].kalman_confidence_lower:
    Type: value
    Python:     66.00146817508077
    TypeScript: 68.84228332890095
    Difference: 2.841e+0
    Numeric difference exceeds tolerance: abs=2.841e+0, rel=4.126556%

  root.results[2].kalman_variance:
    Type: value
    Python:     4.041740065100835
    TypeScript: 0.34704101170227364
    Difference: 3.695e+0
    Numeric difference exceeds tolerance: abs=3.695e+0, rel=91.413574%

  root.results[2].prediction_error:
    Type: value
    Python:     0.17771595492426684
    TypeScript: 0.17951293370509802
    Difference: 1.797e-3
    Numeric difference exceeds tolerance: abs=1.797e-3, rel=1.001030%

  root.results[2].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:27:25.510000
    TypeScript: 2025-11-10T10:27:25.510Z
    Value mismatch: 2025-11-10T11:27:25.510000 !== 2025-11-10T10:27:25.510Z

  root.results[2].quality_score:
    Type: value
    Python:     0.9412561190105361
    TypeScript: 0.9367822904471772
    Difference: 4.474e-3
    Numeric difference exceeds tolerance: abs=4.474e-3, rel=0.475304%

  root.results[2].quality_components.kalman_fit:
    Type: value
    Python:     0.9962102795524286
    TypeScript: 0.9834936171616375
    Difference: 1.272e-2
    Numeric difference exceeds tolerance: abs=1.272e-2, rel=1.276504%

  root.results[2].quality_components.temporal_consistency:
    Type: value
    Python:     0.9358715060747236
    TypeScript: 0.9362158948773998
    Difference: 3.444e-4
    Numeric difference exceeds tolerance: abs=3.444e-4, rel=0.036785%

  root.results[3].timestamp:
    Type: type
    Python:     1762770531910
    TypeScript: 2025-11-10T10:28:51.910Z
    Type mismatch: Python number, TypeScript string

  root.results[3].filtered_weight:
    Type: value
    Python:     70.05394776769158
    TypeScript: 70.03950862742087
    Difference: 1.444e-2
    Numeric difference exceeds tolerance: abs=1.444e-2, rel=0.020611%

  root.results[3].trend:
    Type: value
    Python:     0.000012123179167744857
    TypeScript: 0.00002560878689563456
    Difference: 1.349e-5
    Numeric difference exceeds tolerance: abs=1.349e-5, rel=52.660080%

  root.results[3].trend_weekly:
    Type: value
    Python:     0.000084862254174214
    TypeScript: 0.00017926150826944192
    Difference: 9.440e-5
    Numeric difference exceeds tolerance: abs=9.440e-5, rel=52.660080%

  root.results[3].confidence:
    Type: value
    Python:     0.9991919367647333
    TypeScript: 0.9936668982182464
    Difference: 5.525e-3
    Numeric difference exceeds tolerance: abs=5.525e-3, rel=0.552951%

  root.results[3].innovation:
    Type: value
    Python:     0.2460522323084149
    TypeScript: 0.2604913725791249
    Difference: 1.444e-2
    Numeric difference exceeds tolerance: abs=1.444e-2, rel=5.543040%

  root.results[3].normalized_innovation:
    Type: value
    Python:     0.04020920029915558
    TypeScript: 0.1127230317617692
    Difference: 7.251e-2
    Numeric difference exceeds tolerance: abs=7.251e-2, rel=64.329206%

  root.results[3].kalman_confidence_upper:
    Type: value
    Python:     73.96925340202483
    TypeScript: 71.20612298929431
    Difference: 2.763e+0
    Numeric difference exceeds tolerance: abs=2.763e+0, rel=3.735512%

  root.results[3].kalman_confidence_lower:
    Type: value
    Python:     66.13864213335833
    TypeScript: 68.87289426554743
    Difference: 2.734e+0
    Numeric difference exceeds tolerance: abs=2.734e+0, rel=3.969997%

  root.results[3].kalman_variance:
    Type: value
    Python:     3.8324045525604276
    TypeScript: 0.3402472673323422
    Difference: 3.492e+0
    Numeric difference exceeds tolerance: abs=3.492e+0, rel=91.121833%

  root.results[3].prediction_error:
    Type: value
    Python:     0.2460522323084149
    TypeScript: 0.2604913725791249
    Difference: 1.444e-2
    Numeric difference exceeds tolerance: abs=1.444e-2, rel=5.543040%

  root.results[3].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:28:51.910000
    TypeScript: 2025-11-10T10:28:51.910Z
    Value mismatch: 2025-11-10T11:28:51.910000 !== 2025-11-10T10:28:51.910Z

  root.results[3].quality_score:
    Type: value
    Python:     0.9342633391947578
    TypeScript: 0.9275569702102858
    Difference: 6.706e-3
    Numeric difference exceeds tolerance: abs=6.706e-3, rel=0.717824%

  root.results[3].quality_components.kalman_fit:
    Type: value
    Python:     0.9945801944395222
    TypeScript: 0.9761542129758899
    Difference: 1.843e-2
    Numeric difference exceeds tolerance: abs=1.843e-2, rel=1.852639%

  root.results[3].quality_components.temporal_consistency:
    Type: value
    Python:     0.9150507210816894
    TypeScript: 0.9146398210793958
    Difference: 4.109e-4
    Numeric difference exceeds tolerance: abs=4.109e-4, rel=0.044905%

  root.results[4].timestamp:
    Type: type
    Python:     1762770618310
    TypeScript: 2025-11-10T10:30:18.310Z
    Type mismatch: Python number, TypeScript string

  root.results[4].filtered_weight:
    Type: value
    Python:     70.10337203509553
    TypeScript: 70.06361766002658
    Difference: 3.975e-2
    Numeric difference exceeds tolerance: abs=3.975e-2, rel=0.056708%

  root.results[4].trend:
    Type: value
    Python:     0.00003851528393043969
    TypeScript: 0.00005448805285180108
    Difference: 1.597e-5
    Numeric difference exceeds tolerance: abs=1.597e-5, rel=29.314259%

  root.results[4].trend_weekly:
    Type: value
    Python:     0.00026960698751307785
    TypeScript: 0.00038141636996260757
    Difference: 1.118e-4
    Numeric difference exceeds tolerance: abs=1.118e-4, rel=29.314259%

  root.results[4].confidence:
    Type: value
    Python:     0.998409215670912
    TypeScript: 0.98944998276146
    Difference: 8.959e-3
    Numeric difference exceeds tolerance: abs=8.959e-3, rel=0.897351%

  root.results[4].innovation:
    Type: value
    Python:     0.29662796490447363
    TypeScript: 0.3363823399734258
    Difference: 3.975e-2
    Numeric difference exceeds tolerance: abs=3.975e-2, rel=11.818211%

  root.results[4].normalized_innovation:
    Type: value
    Python:     0.056427847202575994
    TypeScript: 0.14564383412675075
    Difference: 8.922e-2
    Numeric difference exceeds tolerance: abs=8.922e-2, rel=61.256275%

  root.results[4].kalman_confidence_upper:
    Type: value
    Python:     73.82004535902216
    TypeScript: 71.22009180576052
    Difference: 2.600e+0
    Numeric difference exceeds tolerance: abs=2.600e+0, rel=3.522016%

  root.results[4].kalman_confidence_lower:
    Type: value
    Python:     66.3866987111689
    TypeScript: 68.90714351429264
    Difference: 2.520e+0
    Numeric difference exceeds tolerance: abs=2.520e+0, rel=3.657741%

  root.results[4].kalman_variance:
    Type: value
    Python:     3.453415149196944
    TypeScript: 0.33435811243776226
    Difference: 3.119e+0
    Numeric difference exceeds tolerance: abs=3.119e+0, rel=90.318045%

  root.results[4].prediction_error:
    Type: value
    Python:     0.29662796490447363
    TypeScript: 0.3363823399734258
    Difference: 3.975e-2
    Numeric difference exceeds tolerance: abs=3.975e-2, rel=11.818211%

  root.results[4].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:30:18.310000
    TypeScript: 2025-11-10T10:30:18.310Z
    Value mismatch: 2025-11-10T11:30:18.310000 !== 2025-11-10T10:30:18.310Z

  root.results[4].quality_score:
    Type: value
    Python:     0.9291662865760396
    TypeScript: 0.9197751759991581
    Difference: 9.391e-3
    Numeric difference exceeds tolerance: abs=9.391e-3, rel=1.010703%

  root.results[4].quality_components.kalman_fit:
    Type: value
    Python:     0.993240126989953
    TypeScript: 0.969333678272057
    Difference: 2.391e-2
    Numeric difference exceeds tolerance: abs=2.391e-2, rel=2.406915%

  root.results[4].quality_components.temporal_consistency:
    Type: value
    Python:     0.9004151054030848
    TypeScript: 0.8975693791061289
    Difference: 2.846e-3
    Numeric difference exceeds tolerance: abs=2.846e-3, rel=0.316046%

  root.results[5].was_reset:
    Type: missing
    Python:     true
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].reset_reason:
    Type: missing
    Python:     initial_measurement
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].reset_type:
    Type: missing
    Python:     initial
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].gap_days:
    Type: missing
    Python:     null
    TypeScript: undefined
    Key missing in TypeScript output

  root.results[5].timestamp:
    Type: type
    Python:     1762770704710
    TypeScript: 2025-11-10T10:31:44.710Z
    Type mismatch: Python number, TypeScript string

  root.results[5].kalman_confidence_upper:
    Type: value
    Python:     74.31575680566779
    TypeScript: 71.70664825031987
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.510842%

  root.results[5].kalman_confidence_lower:
    Type: value
    Python:     66.68424319433221
    TypeScript: 69.29335174968013
    Difference: 2.609e+0
    Numeric difference exceeds tolerance: abs=2.609e+0, rel=3.765309%

  root.results[5].kalman_variance:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.results[5].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:31:44.710000
    TypeScript: 2025-11-10T10:31:44.710Z
    Value mismatch: 2025-11-10T11:31:44.710000 !== 2025-11-10T10:31:44.710Z

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
    TypeScript: 70.50709790580099
    Difference: 1.271e-3
    Numeric difference exceeds tolerance: abs=1.271e-3, rel=0.001803%

  root.results[6].trend:
    Type: value
    Python:     1.3711846731725685e-7
    TypeScript: 0.000001858041883980043
    Difference: 1.721e-6
    Numeric difference exceeds tolerance: abs=1.721e-6, rel=92.620270%

  root.results[6].trend_weekly:
    Type: value
    Python:     9.598292712207979e-7
    TypeScript: 0.0000130062931878603
    Difference: 1.205e-5
    Numeric difference exceeds tolerance: abs=1.205e-5, rel=92.620270%

  root.results[6].confidence:
    Type: value
    Python:     0.9999389919505018
    TypeScript: 0.9991944452802651
    Difference: 7.445e-4
    Numeric difference exceeds tolerance: abs=7.445e-4, rel=0.074459%

  root.results[6].innovation:
    Type: value
    Python:     0.09417351782407479
    TypeScript: 0.09290209419900464
    Difference: 1.271e-3
    Numeric difference exceeds tolerance: abs=1.271e-3, rel=1.350086%

  root.results[6].normalized_innovation:
    Type: value
    Python:     0.011046258241137209
    TypeScript: 0.04014671476696212
    Difference: 2.910e-2
    Numeric difference exceeds tolerance: abs=2.910e-2, rel=72.485275%

  root.results[6].kalman_confidence_upper:
    Type: value
    Python:     74.5066521565391
    TypeScript: 71.69855968034456
    Difference: 2.808e+0
    Numeric difference exceeds tolerance: abs=2.808e+0, rel=3.768915%

  root.results[6].kalman_confidence_lower:
    Type: value
    Python:     66.50500080781273
    TypeScript: 69.31563613125742
    Difference: 2.811e+0
    Numeric difference exceeds tolerance: abs=2.811e+0, rel=4.054836%

  root.results[6].kalman_variance:
    Type: value
    Python:     4.0016515191608955
    TypeScript: 0.35489529004962833
    Difference: 3.647e+0
    Numeric difference exceeds tolerance: abs=3.647e+0, rel=91.131279%

  root.results[6].prediction_error:
    Type: value
    Python:     0.09417351782407479
    TypeScript: 0.09290209419900464
    Difference: 1.271e-3
    Numeric difference exceeds tolerance: abs=1.271e-3, rel=1.350086%

  root.results[6].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:33:11.110000
    TypeScript: 2025-11-10T10:33:11.110Z
    Value mismatch: 2025-11-10T11:33:11.110000 !== 2025-11-10T10:33:11.110Z

  root.results[6].quality_score:
    Type: value
    Python:     0.9502517432046031
    TypeScript: 0.9478378128444289
    Difference: 2.414e-3
    Numeric difference exceeds tolerance: abs=2.414e-3, rel=0.254031%

  root.results[6].quality_components.kalman_fit:
    Type: value
    Python:     0.9980458223730743
    TypeScript: 0.9914160522867913
    Difference: 6.630e-3
    Numeric difference exceeds tolerance: abs=6.630e-3, rel=0.664275%

  root.results[7].timestamp:
    Type: type
    Python:     1762770877510
    TypeScript: 2025-11-10T10:34:37.510Z
    Type mismatch: Python number, TypeScript string

  root.results[7].filtered_weight:
    Type: value
    Python:     70.52228404507574
    TypeScript: 70.5204870662949
    Difference: 1.797e-3
    Numeric difference exceeds tolerance: abs=1.797e-3, rel=0.002548%

  root.results[7].trend:
    Type: value
    Python:     0.000002374382648296539
    TypeScript: 0.000009214550423525659
    Difference: 6.840e-6
    Numeric difference exceeds tolerance: abs=6.840e-6, rel=74.232246%

  root.results[7].trend_weekly:
    Type: value
    Python:     0.00001662067853807577
    TypeScript: 0.00006450185296467961
    Difference: 4.788e-5
    Numeric difference exceeds tolerance: abs=4.788e-5, rel=74.232246%

  root.results[7].confidence:
    Type: value
    Python:     0.9996947673850899
    TypeScript: 0.9969911966481986
    Difference: 2.704e-3
    Numeric difference exceeds tolerance: abs=2.704e-3, rel=0.270440%

  root.results[7].innovation:
    Type: value
    Python:     0.17771595492426684
    TypeScript: 0.17951293370509802
    Difference: 1.797e-3
    Numeric difference exceeds tolerance: abs=1.797e-3, rel=1.001030%

  root.results[7].normalized_innovation:
    Type: value
    Python:     0.024709480280496986
    TypeScript: 0.07763168039669593
    Difference: 5.292e-2
    Numeric difference exceeds tolerance: abs=5.292e-2, rel=68.170881%

  root.results[7].kalman_confidence_upper:
    Type: value
    Python:     74.5430999150707
    TypeScript: 71.69869080368886
    Difference: 2.844e+0
    Numeric difference exceeds tolerance: abs=2.844e+0, rel=3.815791%

  root.results[7].kalman_confidence_lower:
    Type: value
    Python:     66.50146817508077
    TypeScript: 69.34228332890095
    Difference: 2.841e+0
    Numeric difference exceeds tolerance: abs=2.841e+0, rel=4.096801%

  root.results[7].kalman_variance:
    Type: value
    Python:     4.041740065100835
    TypeScript: 0.34704101170227364
    Difference: 3.695e+0
    Numeric difference exceeds tolerance: abs=3.695e+0, rel=91.413574%

  root.results[7].prediction_error:
    Type: value
    Python:     0.17771595492426684
    TypeScript: 0.17951293370509802
    Difference: 1.797e-3
    Numeric difference exceeds tolerance: abs=1.797e-3, rel=1.001030%

  root.results[7].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:34:37.510000
    TypeScript: 2025-11-10T10:34:37.510Z
    Value mismatch: 2025-11-10T11:34:37.510000 !== 2025-11-10T10:34:37.510Z

  root.results[7].quality_score:
    Type: value
    Python:     0.9412561190105361
    TypeScript: 0.9367822904471772
    Difference: 4.474e-3
    Numeric difference exceeds tolerance: abs=4.474e-3, rel=0.475304%

  root.results[7].quality_components.kalman_fit:
    Type: value
    Python:     0.9962102795524286
    TypeScript: 0.9834936171616375
    Difference: 1.272e-2
    Numeric difference exceeds tolerance: abs=1.272e-2, rel=1.276504%

  root.results[7].quality_components.temporal_consistency:
    Type: value
    Python:     0.9358715060747236
    TypeScript: 0.9362158948773998
    Difference: 3.444e-4
    Numeric difference exceeds tolerance: abs=3.444e-4, rel=0.036785%

  root.results[8].timestamp:
    Type: type
    Python:     1762770963910
    TypeScript: 2025-11-10T10:36:03.910Z
    Type mismatch: Python number, TypeScript string

  root.results[8].filtered_weight:
    Type: value
    Python:     70.55394776769158
    TypeScript: 70.53950862742087
    Difference: 1.444e-2
    Numeric difference exceeds tolerance: abs=1.444e-2, rel=0.020465%

  root.results[8].trend:
    Type: value
    Python:     0.000012123179167744857
    TypeScript: 0.00002560878689563456
    Difference: 1.349e-5
    Numeric difference exceeds tolerance: abs=1.349e-5, rel=52.660080%

  root.results[8].trend_weekly:
    Type: value
    Python:     0.000084862254174214
    TypeScript: 0.00017926150826944192
    Difference: 9.440e-5
    Numeric difference exceeds tolerance: abs=9.440e-5, rel=52.660080%

  root.results[8].confidence:
    Type: value
    Python:     0.9991919367647333
    TypeScript: 0.9936668982182464
    Difference: 5.525e-3
    Numeric difference exceeds tolerance: abs=5.525e-3, rel=0.552951%

  root.results[8].innovation:
    Type: value
    Python:     0.2460522323084149
    TypeScript: 0.2604913725791249
    Difference: 1.444e-2
    Numeric difference exceeds tolerance: abs=1.444e-2, rel=5.543040%

  root.results[8].normalized_innovation:
    Type: value
    Python:     0.04020920029915558
    TypeScript: 0.1127230317617692
    Difference: 7.251e-2
    Numeric difference exceeds tolerance: abs=7.251e-2, rel=64.329206%

  root.results[8].kalman_confidence_upper:
    Type: value
    Python:     74.46925340202483
    TypeScript: 71.70612298929431
    Difference: 2.763e+0
    Numeric difference exceeds tolerance: abs=2.763e+0, rel=3.710431%

  root.results[8].kalman_confidence_lower:
    Type: value
    Python:     66.63864213335833
    TypeScript: 69.37289426554743
    Difference: 2.734e+0
    Numeric difference exceeds tolerance: abs=2.734e+0, rel=3.941384%

  root.results[8].kalman_variance:
    Type: value
    Python:     3.8324045525604276
    TypeScript: 0.3402472673323422
    Difference: 3.492e+0
    Numeric difference exceeds tolerance: abs=3.492e+0, rel=91.121833%

  root.results[8].prediction_error:
    Type: value
    Python:     0.2460522323084149
    TypeScript: 0.2604913725791249
    Difference: 1.444e-2
    Numeric difference exceeds tolerance: abs=1.444e-2, rel=5.543040%

  root.results[8].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:36:03.910000
    TypeScript: 2025-11-10T10:36:03.910Z
    Value mismatch: 2025-11-10T11:36:03.910000 !== 2025-11-10T10:36:03.910Z

  root.results[8].quality_score:
    Type: value
    Python:     0.9342633391947578
    TypeScript: 0.9275569702102858
    Difference: 6.706e-3
    Numeric difference exceeds tolerance: abs=6.706e-3, rel=0.717824%

  root.results[8].quality_components.kalman_fit:
    Type: value
    Python:     0.9945801944395222
    TypeScript: 0.9761542129758899
    Difference: 1.843e-2
    Numeric difference exceeds tolerance: abs=1.843e-2, rel=1.852639%

  root.results[8].quality_components.temporal_consistency:
    Type: value
    Python:     0.9150507210816894
    TypeScript: 0.9146398210793958
    Difference: 4.109e-4
    Numeric difference exceeds tolerance: abs=4.109e-4, rel=0.044905%

  root.results[9].timestamp:
    Type: type
    Python:     1762771050310
    TypeScript: 2025-11-10T10:37:30.310Z
    Type mismatch: Python number, TypeScript string

  root.results[9].filtered_weight:
    Type: value
    Python:     70.60337203509553
    TypeScript: 70.56361766002658
    Difference: 3.975e-2
    Numeric difference exceeds tolerance: abs=3.975e-2, rel=0.056307%

  root.results[9].trend:
    Type: value
    Python:     0.00003851528393043969
    TypeScript: 0.00005448805285180108
    Difference: 1.597e-5
    Numeric difference exceeds tolerance: abs=1.597e-5, rel=29.314259%

  root.results[9].trend_weekly:
    Type: value
    Python:     0.00026960698751307785
    TypeScript: 0.00038141636996260757
    Difference: 1.118e-4
    Numeric difference exceeds tolerance: abs=1.118e-4, rel=29.314259%

  root.results[9].confidence:
    Type: value
    Python:     0.998409215670912
    TypeScript: 0.98944998276146
    Difference: 8.959e-3
    Numeric difference exceeds tolerance: abs=8.959e-3, rel=0.897351%

  root.results[9].innovation:
    Type: value
    Python:     0.29662796490447363
    TypeScript: 0.3363823399734258
    Difference: 3.975e-2
    Numeric difference exceeds tolerance: abs=3.975e-2, rel=11.818211%

  root.results[9].normalized_innovation:
    Type: value
    Python:     0.056427847202575994
    TypeScript: 0.14564383412675075
    Difference: 8.922e-2
    Numeric difference exceeds tolerance: abs=8.922e-2, rel=61.256275%

  root.results[9].kalman_confidence_upper:
    Type: value
    Python:     74.32004535902216
    TypeScript: 71.72009180576052
    Difference: 2.600e+0
    Numeric difference exceeds tolerance: abs=2.600e+0, rel=3.498321%

  root.results[9].kalman_confidence_lower:
    Type: value
    Python:     66.8866987111689
    TypeScript: 69.40714351429264
    Difference: 2.520e+0
    Numeric difference exceeds tolerance: abs=2.520e+0, rel=3.631391%

  root.results[9].kalman_variance:
    Type: value
    Python:     3.453415149196944
    TypeScript: 0.33435811243776226
    Difference: 3.119e+0
    Numeric difference exceeds tolerance: abs=3.119e+0, rel=90.318045%

  root.results[9].prediction_error:
    Type: value
    Python:     0.29662796490447363
    TypeScript: 0.3363823399734258
    Difference: 3.975e-2
    Numeric difference exceeds tolerance: abs=3.975e-2, rel=11.818211%

  root.results[9].preprocessing.timestamp:
    Type: value
    Python:     2025-11-10T11:37:30.310000
    TypeScript: 2025-11-10T10:37:30.310Z
    Value mismatch: 2025-11-10T11:37:30.310000 !== 2025-11-10T10:37:30.310Z

  root.results[9].quality_score:
    Type: value
    Python:     0.9291662865760396
    TypeScript: 0.9197751759991581
    Difference: 9.391e-3
    Numeric difference exceeds tolerance: abs=9.391e-3, rel=1.010703%

  root.results[9].quality_components.kalman_fit:
    Type: value
    Python:     0.993240126989953
    TypeScript: 0.969333678272057
    Difference: 2.391e-2
    Numeric difference exceeds tolerance: abs=2.391e-2, rel=2.406915%

  root.results[9].quality_components.temporal_consistency:
    Type: value
    Python:     0.9004151054030848
    TypeScript: 0.8975693791061289
    Difference: 2.846e-3
    Numeric difference exceeds tolerance: abs=2.846e-3, rel=0.316046%

  root.finalState.adaptation_state:
    Type: extra
    Python:     undefined
    TypeScript: {}
    Extra key in TypeScript output

  root.finalState.version:
    Type: extra
    Python:     undefined
    TypeScript: 1
    Extra key in TypeScript output

  root.finalState.kalman_params.initial_state_covariance[0][0]:
    Type: value
    Python:     3.6399999999999997
    TypeScript: 0.364
    Difference: 3.276e+0
    Numeric difference exceeds tolerance: abs=3.276e+0, rel=90.000000%

  root.finalState.kalman_params.transition_covariance[0][0]:
    Type: value
    Python:     0.19607272887128602
    TypeScript: 0.018
    Difference: 1.781e-1
    Numeric difference exceeds tolerance: abs=1.781e-1, rel=90.819733%

  root.finalState.kalman_params.transition_covariance[1][1]:
    Type: value
    Python:     0.0013071515258085737
    TypeScript: 0.00012
    Difference: 1.187e-3
    Numeric difference exceeds tolerance: abs=1.187e-3, rel=90.819733%

  root.finalState.kalman_params.observation_covariance[0][0]:
    Type: value
    Python:     100
    TypeScript: 5
    Difference: 9.500e+1
    Numeric difference exceeds tolerance: abs=9.500e+1, rel=95.000000%

  root.finalState.last_state[0][0]:
    Type: type
    Python:     70.55394776769158
    TypeScript: [
  70.53950862742087
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[0][1]:
    Type: type
    Python:     0.000012123179167744857
    TypeScript: [
  0.00002560878689563456
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][0]:
    Type: type
    Python:     70.60337203509553
    TypeScript: [
  70.56361766002658
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_state[1][1]:
    Type: type
    Python:     0.00003851528393043969
    TypeScript: [
  0.00005448805285180108
]
    Type mismatch: Python number, TypeScript object

  root.finalState.last_covariance[0][0][0]:
    Type: value
    Python:     3.8324045525604276
    TypeScript: 0.3402472673323422
    Difference: 3.492e+0
    Numeric difference exceeds tolerance: abs=3.492e+0, rel=91.121833%

  root.finalState.last_covariance[0][0][1]:
    Type: value
    Python:     0.0011799500893092253
    TypeScript: 0.00029326532946001945
    Difference: 8.867e-4
    Numeric difference exceeds tolerance: abs=8.867e-4, rel=75.145955%

  root.finalState.last_covariance[0][1][0]:
    Type: value
    Python:     0.0011799500893092255
    TypeScript: 0.00029326532946001945
    Difference: 8.867e-4
    Numeric difference exceeds tolerance: abs=8.867e-4, rel=75.145955%

  root.finalState.last_covariance[0][1][1]:
    Type: value
    Python:     0.009714504341111076
    TypeScript: 0.0013599718709253784
    Difference: 8.355e-3
    Numeric difference exceeds tolerance: abs=8.355e-3, rel=86.000605%

  root.finalState.last_covariance[1][0][0]:
    Type: value
    Python:     3.453415149196944
    TypeScript: 0.33435811243776226
    Difference: 3.119e+0
    Numeric difference exceeds tolerance: abs=3.119e+0, rel=90.318045%

  root.finalState.last_covariance[1][0][1]:
    Type: value
    Python:     0.001844137199694154
    TypeScript: 0.00040055703559759796
    Difference: 1.444e-3
    Numeric difference exceeds tolerance: abs=1.444e-3, rel=78.279434%

  root.finalState.last_covariance[1][1][0]:
    Type: value
    Python:     0.001844137199694154
    TypeScript: 0.00040055703559759796
    Difference: 1.444e-3
    Numeric difference exceeds tolerance: abs=1.444e-3, rel=78.279434%

  root.finalState.last_covariance[1][1][1]:
    Type: value
    Python:     0.011021491787102742
    TypeScript: 0.0014799374821011539
    Difference: 9.542e-3
    Numeric difference exceeds tolerance: abs=9.542e-3, rel=86.572258%

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

  root.finalState.measurement_history[1].quality_score:
    Type: value
    Python:     0.9502517432046031
    TypeScript: 0.9478378128444289
    Difference: 2.414e-3
    Numeric difference exceeds tolerance: abs=2.414e-3, rel=0.254031%

  root.finalState.measurement_history[2].timestamp:
    Type: value
    Python:     2025-11-10T11:34:37.510000
    TypeScript: 2025-11-10T10:34:37.510Z
    Value mismatch: 2025-11-10T11:34:37.510000 !== 2025-11-10T10:34:37.510Z

  root.finalState.measurement_history[2].quality_score:
    Type: value
    Python:     0.9412561190105361
    TypeScript: 0.9367822904471772
    Difference: 4.474e-3
    Numeric difference exceeds tolerance: abs=4.474e-3, rel=0.475304%

  root.finalState.measurement_history[3].timestamp:
    Type: value
    Python:     2025-11-10T11:36:03.910000
    TypeScript: 2025-11-10T10:36:03.910Z
    Value mismatch: 2025-11-10T11:36:03.910000 !== 2025-11-10T10:36:03.910Z

  root.finalState.measurement_history[3].quality_score:
    Type: value
    Python:     0.9342633391947578
    TypeScript: 0.9275569702102858
    Difference: 6.706e-3
    Numeric difference exceeds tolerance: abs=6.706e-3, rel=0.717824%

  root.finalState.measurement_history[4].timestamp:
    Type: value
    Python:     2025-11-10T11:37:30.310000
    TypeScript: 2025-11-10T10:37:30.310Z
    Value mismatch: 2025-11-10T11:37:30.310000 !== 2025-11-10T10:37:30.310Z

  root.finalState.measurement_history[4].quality_score:
    Type: value
    Python:     0.9291662865760396
    TypeScript: 0.9197751759991581
    Difference: 9.391e-3
    Numeric difference exceeds tolerance: abs=9.391e-3, rel=1.010703%

  root.finalState.temporal_baseline.last_timestamp:
    Type: value
    Python:     2025-11-10T11:37:30.310000
    TypeScript: 2025-11-10T10:37:30.310Z
    Value mismatch: 2025-11-10T11:37:30.310000 !== 2025-11-10T10:37:30.310Z

```

## All Test Results

| Test Name | Status | Py Time | TS Time | Differences |
|-----------|--------|---------|---------|-------------|
| Test 1: Single Measurement Processing | ❌ | 348.59ms | 2921.76ms | 35 |
| Test 2: Multi-Measurement Sequence | ❌ | 148.90ms | 3.92ms | 192 |
| Test 3: Reset Scenario | ❌ | 129.16ms | 2.18ms | 189 |
| Test 4: Quality Rejection | ❌ | 106.38ms | 0.94ms | 88 |
| Test 5: State Persistence | ❌ | 206.97ms | 1.77ms | 178 |
