# Plan: Prove Filtering Effectiveness with Charts and Examples

## Decision
**Approach**: Create comprehensive visualization suite comparing raw vs filtered data impact on weight loss metrics
**Why**: Visual evidence needed to validate that filtering improves clinical reliability without distorting outcomes
**Risk Level**: Low

## Implementation Steps

### 1. **90-Day Weight Loss Comparison** - Primary proof point
- Modify `simple_report.py:182-404` to export detailed 90-day metrics
- Filter users with `start_date <= '2025-06-14'` (90+ days in program)
- Calculate weight loss from actual start date (not first measurement)
- Generate comparison table: Raw vs Filtered weight loss percentages
- Export to `data/90_day_weight_loss_comparison.csv`

### 2. **Visualization Dashboard** - Create `viz/filtering_effectiveness.py`
- **Chart 1: Weight Loss Distribution** (Histogram)
  - X-axis: Weight loss percentage bins (-5%, 0%, 5%, 10%, 15%+)
  - Y-axis: Number of users
  - Overlay: Raw (blue) vs Filtered (green) distributions
  - Highlight: Success rate difference (% who lost weight)

- **Chart 2: Individual Journey Comparison** (Line plots)
  - Select 6 representative users (2 high success, 2 moderate, 2 minimal)
  - Dual plots per user: Raw measurements vs Filtered trend
  - Show outliers as red dots, accepted as green
  - Annotate: Start weight, 90-day weight, % change

- **Chart 3: Filtering Impact by Time** (Area chart)
  - X-axis: Days from start (0-360)
  - Y-axis: Average weight loss %
  - Lines: Raw average, Filtered average, Difference band
  - Highlight: 90-day mark with vertical line

- **Chart 4: Data Quality Metrics** (Multi-panel)
  - Panel A: Variance reduction per user (scatter plot)
  - Panel B: Smoothness improvement (before/after)
  - Panel C: Outlier removal rate by weight range
  - Panel D: Temporal consistency (autocorrelation)

### 3. **Statistical Evidence Table** - Generate `reports/filtering_evidence.md`
- Shapiro-Wilk normality test results
- Variance reduction percentages
- Trend smoothness improvements
- Clinical plausibility scores
- 90-day success rate comparison

### 4. **Example Case Studies** - Document specific users
- User A: Heavy filtering (>30% removed) - show improvement
- User B: Light filtering (<10% removed) - minimal impact
- User C: Critical outlier caught - prevented false plateau
- Include actual weight trajectories and filtering decisions

## Files to Change
- `simple_report.py:899-910` - Add 90-day specific analysis export
- `simple_report.py:182-247` - Enhance interval calculation for start dates
- NEW: `viz/filtering_effectiveness.py` - Main visualization script
- NEW: `scripts/generate_evidence_report.py` - Statistical analysis
- `data/2025-09-17-user-employers.csv` - Read for start dates

## Acceptance Criteria
- [ ] 90-day weight loss comparison shows <2% average difference between raw/filtered
- [ ] Visualizations clearly demonstrate filtering improves trend clarity
- [ ] Statistical tests confirm improved data quality (p < 0.05)
- [ ] At least 100 users with 90+ days of data analyzed
- [ ] Case studies show both positive and edge cases

## Risks & Mitigations
**Main Risk**: Filtering might show worse outcomes for some users
**Mitigation**: Present honestly, explain why (e.g., removing false low readings that inflate loss)

**Secondary Risk**: Insufficient 90-day data
**Mitigation**: Adjust to 60-day if needed, clearly document sample size

## Out of Scope
- Real-time dashboard implementation
- Individual user reports
- Predictive modeling
- Cross-employer comparisons