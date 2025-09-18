# Plan: Strict Unit Validation - No BMI Detection, No Unit Assumptions

## Decision
**Approach**: Remove all BMI detection logic and enforce strict unit validation from source data
**Why**: Data integrity requires explicit units - guessing causes corruption. BMI values don't exist in source data.
**Risk Level**: Medium (will reject previously accepted data)

## Implementation Steps

### Phase 1: Remove BMI Detection
1. **Remove BMI detection logic** - Delete `validation.py:691-706` (BMI heuristic detection)
2. **Clean up BMI references** - Remove BMI-related warnings and metadata from preprocessing
3. **Update tests** - Remove/update `test_bmi_validator.py` to reflect no BMI conversion

### Phase 2: Implement Strict Unit Validation
1. **Define supported units whitelist** - Create explicit list in `constants.py`:
   ```python
   SUPPORTED_WEIGHT_UNITS = {
       'kg', 'kilogram', 'kilograms',
       'lb', 'lbs', 'pound', 'pounds', 
       'st', 'stone', 'stones',
       'g', 'gram', 'grams'  # Add if needed
   }
   ```

2. **Update preprocessing** - Modify `validation.py:653-690`:
   - Check if unit in SUPPORTED_WEIGHT_UNITS
   - If not supported: return (None, {'rejected': f'Unsupported unit: {unit}'})
   - If supported: perform appropriate conversion
   - Never default to 'kg' if unit is missing/unknown

3. **Add unit validation at entry** - Update `main.py:364-368`:
   - Extract unit from CSV
   - Validate against whitelist before processing
   - Skip row entirely if unit unsupported
   - Add counter for unit-rejected measurements

### Phase 3: Improve Unit Handling
1. **Standardize unit extraction** - `main.py:364`:
   - Use exact field name from CSV (no fallbacks)
   - Preserve original unit in metadata
   - Log all unit rejections with user_id and timestamp

2. **Update conversion functions** - `validation.py:680-688`:
   - Keep existing conversion factors (they're correct)
   - Add 'g' → 'kg' conversion (divide by 1000) if needed
   - Remove any unit guessing or defaulting logic

3. **Enhanced logging** - Add unit rejection tracking:
   - Track rejected units and their frequency
   - Report in final statistics
   - Save to separate CSV for analysis

## Files to Change

### Core Changes
- `src/processing/validation.py:653-706` - Remove BMI detection, enforce strict validation
- `src/constants.py` - Add SUPPORTED_WEIGHT_UNITS constant
- `main.py:364-368` - Add unit validation before processing

### Test Updates  
- `tests/test_bmi_validator.py` - Remove or update for new behavior
- `tests/test_processor.py` - Add tests for unit rejection
- Add new test file: `tests/test_unit_validation.py`

### Documentation
- `docs/IMPLEMENTATION_GUIDE.md` - Document supported units
- `README.md` - Add note about strict unit requirements

## Acceptance Criteria
- [ ] BMI detection code completely removed
- [ ] Only whitelisted units are processed
- [ ] Unknown/missing units cause measurement rejection
- [ ] No unit defaulting or assumptions
- [ ] Unit rejections are logged with details
- [ ] Tests pass with strict validation
- [ ] Documentation lists all supported units

## Risks & Mitigations

### Main Risk: Data Loss from Strict Rejection
**Risk**: Valid measurements rejected due to unit typos or variations
**Mitigation**: 
- Log all rejections to CSV for review
- Provide script to analyze rejected data patterns
- Consider adding common typo mappings in future iteration

### Secondary Risk: Breaking Existing Pipelines
**Risk**: Systems expecting BMI conversion will fail
**Mitigation**:
- Add feature flag for transition period
- Document breaking change prominently
- Provide migration script for historical data

## Out of Scope
- Fixing upstream data quality issues
- Adding new unit types beyond current conversions
- Automatic unit inference from value ranges
- Fuzzy matching for unit typos

## Rollout Strategy
1. Deploy with feature flag disabled (old behavior)
2. Run in shadow mode - log what would be rejected
3. Analyze rejection patterns for 1 week
4. Enable strict mode for new data only
5. Full rollout after validation

## Success Metrics
- Zero BMI misidentifications
- <1% increase in rejection rate
- All rejections have clear unit-related reasons
- No data corruption from unit confusion