---
theme: seriph
background: https://images.unsplash.com/photo-1559724087-a45f6a7a35d7?q=80&w=3852&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&w=1920
title: Weight Measurement Filtering Impact
info: |
  ## Data Quality Pipeline Implementation
  Technical analysis of filtering system performance
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Weight Measurement Filtering Impact

## Data Quality Pipeline Implementation

<div class="pt-12 text-gray-200 text-orange">
    Cohort: Employer Apple
</div>

<div class="pt-12 text-gray-400">
September 22, 2025
</div>

---
layout: default
---

# Analysis Methodology

### Two Processing Approaches

<div grid="~ cols-2 gap-8" class="mt-12">
<div class="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg">

**Raw Data**
- Original measurements
- No quality checks
- Direct from sources

</div>
<div class="bg-blue-500/10 p-6 rounded-lg">

**Filtered Data**
- Adaptive Kalman filtering
- Statistical outlier detection
- Quality score threshold: 0.46

</div>
</div>

---
layout: center
---

# Dataset

<div class="text-6xl font-bold mb-8">4,133 users</div>
<div class="text-3xl mb-4">346,497 measurements</div>
<div class="text-xl text-gray-400">3.0% removal rate | 97% retention</div>

---
layout: default
---

# Key Improvements

<div class="grid grid-cols-3 gap-8 mt-16">
<div class="text-center">
<div class="text-5xl font-bold text-blue-500">2.6%</div>
<div class="text-lg mt-4">CI Improvement</div>
        <hr />
<div class="text-xs mt-4 text-gray-200">&gt;5% would be considered a "meaningful statistical change"
        <br /> <br />
the data is still quite chaotic. we still have a few big jumps in the data which are hard/impossible to handle automatically.


</div>

</div>

<div class="text-center">
<div class="text-5xl font-bold text-green-500">42.9%</div>
<div class="text-lg mt-4">Variance Reduction</div>
        <hr />
<div class="text-xs mt-4 text-gray-200">Lower variance means more consistent measurements. <br /> <br /> Variance Reduction is HIGH! commonly a 15-40% is considered a substantial improvement.

</div>
</div>

<div class="text-center">
<div class="text-5xl font-bold text-purple-500">0.347</div>
<div class="text-lg mt-4">Effect Size Gain</div>
        <hr />
<div class="text-xs mt-4 text-gray-200">Effect Size Imp. is a bit low -- in statistics a <0.5 is considered to be a "small effect", but in our case I think thins is good because it means that it only has a small effect on the overall data.<br /> <br />
In medical fields a "Effect Size Imp." of 0.1 could already be highly significant for health <br />
(psychology: .07 (small), .16 (medium) and .32 (large))
</div>

<div class="text-md mt-4"></div>
</div>
</div>

---
layout: default
---

# Weight Change Statistics

<div class="mt-16">

| Metric | Raw | Filtered | Impact |
|:-------|:----|:---------|:-------|
| **Mean Change** | -3.73% | -4.88% | -1.15% improvement |
| **Std Deviation** | 8.04% | 7.15% | 11% reduction |

</div>

<br />

>filtering does slightly change the outcomes for the reports, but overall the filtered data is still very consistent with the raw data

---
layout: default
---

# Clinical Success Thresholds

<div class="mt-12">

| Weight Loss | Raw | Filtered | Δ |
|:------------|:----|:---------|:--|
| **≥5%** | 47.3% | 48.3% | +1.0% |
| **≥10%** | 27.7% | 26.7% | -1.0% |
| **Missing** | 25% | 25% | 0% |

</div>

<br/>

>outliers and unreasonable changes in data are filtered out, this leads to a reduction in outcomes for >5% weight loss, but an increase for outcomes that are <=5% weight loss.
>
>the 25% with Missing data either have no start or end weight within the 14 day limit used in this analysis.

---
layout: center
---

# 90+ Day Cohort

<div class="mt-8">
<div class="text-5xl font-bold mb-8">3,633 users</div>
<div class="text-2xl text-gray-400">Quarterly reporting baseline</div>
</div>


---
layout: default
---

# Quarterly Metrics

<div class="mt-16">

| Metric | Raw | Filtered | Improvement |
|:-------|:----|:---------|:------------|
| **Avg Loss** | 6.03% | 6.08% | +0.05% |
| **Median** | 5.12% | 5.13% | +0.01% |
| **Std Dev** | 8.04% | 7.15% | -11% |

</div>

---
layout: default
---

# Weight Loss Progression


| Days | Raw | Filtered | Δ | Gain
|:-----|:----|:---------|:--|:--|
| 90 | 2.51% | 2.61% | +0.10% | |
| 120 | 3.30% | 3.42% | +0.12% | +0.79 / +0.81 |
| 150 | 4.18% | 4.33% | **+0.15%** | +0.88 / +0.91 |
| 180 | 5.23% | 5.27% | +0.04% | +1.05 / +0.94 |
| 210 | 6.10% | 6.22% | +0.12% | +0.78 / +0.95 |

<div class="mt-8 text-gray-400">
Peak improvement at 135-150 days (retention period)
</div>

>**Average Improvement Across All Checkpoints:** +0.10%
>
>**Maximum Improvement:** +0.15% at 135 days

---
layout: center
---

# Quality Improvements

<div class="grid grid-cols-3 gap-8 mt-12">
<div>
<div class="text-4xl font-bold">9,904</div>
<div class="text-sm mt-2">Outliers Removed</div>
</div>

<div>
<div class="text-4xl font-bold">5,434</div>
<div class="text-sm mt-2">Impossible Changes</div>
</div>

<div>
<div class="text-4xl font-bold">32</div>
<div class="text-sm mt-2">Direction Errors</div>
</div>
</div>

---
layout: default
---

# Daily Volatility Reduction

<div class="mt-16 text-center">
<div class="text-6xl font-bold">0.59kg</div>
<div class="text-2xl mt-4 text-gray-400">Average reduction in daily weight fluctuation</div>
<div class="mt-8">
<span class="text-xl">Before: ±1.82kg → After: ±1.23kg</span>
</div>
</div>

---
layout: default
---

# Technical Configuration

```toml
[kalman]
quality_threshold = 0.46
initial_variance = 0.364

[outlier_detection]
iqr_multiplier = 1.5
mad_threshold = 3.0
temporal_threshold = 2.0

[quality_scoring]
kalman_weight = 0.40
temporal_weight = 0.30
statistical_weight = 0.30
```

---
layout: default
---

# Future Improvements

<div class="grid grid-cols-2 gap-8 mt-12">
<div class="p-6 bg-blue-50 dark:bg-blue-900/20 rounded-lg">

### Technical
- Source monitoring
- Threshold optimization
- Validation cycles

</div>
<div class="p-6 bg-green-50 dark:bg-green-900/20 rounded-lg">

### Operational
- Uncertainty alerts
- PMP weight graphs
- Reset notifications

</div>
</div>

---
layout: center
class: text-center
---

# Summary

<div class="grid grid-cols-3 gap-12 mt-16">
<div>
<div class="text-6xl font-bold text-blue-500">97%</div>
<div class="text-lg mt-4">Data Retained</div>
</div>

<div>
<div class="text-6xl font-bold text-green-500">43%</div>
<div class="text-lg mt-4">Less Variance</div>
</div>

<div>
<div class="text-6xl font-bold text-purple-500">32</div>
<div class="text-lg mt-4">Report Errors Prevented</div>
</div>
</div>

---
layout: end
---

# Questions / Graphs
