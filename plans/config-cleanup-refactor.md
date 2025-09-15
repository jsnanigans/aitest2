# Plan: Configuration System Cleanup and Refactor

## Summary
Clean up the configuration system by removing ~50% of unused settings, consolidating related configurations, and ensuring all remaining settings are properly connected to the codebase. This will improve maintainability, reduce confusion, and make the system's actual capabilities clear.

## Context
- Source: Investigation report in `./report/config_cleanup_report.md`
- Current state: 86-line config file with ~50% unused settings
- 4 entire sections completely unused
- Several features configured but not implemented
- Some hardcoded values that should use config

## Assumptions
- Backward compatibility is not required (per project guidelines)
- Unused features won't be implemented in the near future
- The system should be simplified rather than expanded
- Configuration should reflect actual system capabilities

## Requirements

### Functional
- Remove all unused configuration sections and settings
- Ensure all remaining settings are properly connected to code
- Maintain all currently working functionality
- Improve configuration documentation

### Non-functional
- Reduce configuration complexity by ~50%
- Improve code clarity and maintainability
- Make configuration self-documenting
- Ensure consistency between config and constants

## Alternatives

### Option A: Minimal Cleanup (Remove Only)
**Approach**: Simply delete unused sections and settings from config.toml

**Pros**:
- Quick to implement (30 minutes)
- No code changes required
- Zero risk to functionality

**Cons**:
- Doesn't fix disconnected features (verbosity, adaptive noise)
- Leaves some inconsistencies
- Misses opportunity for broader improvements

**Risks**: Low

### Option B: Full Cleanup with Connections (Recommended)
**Approach**: Remove unused settings AND properly connect/fix partially implemented features

**Pros**:
- Fixes verbosity system connection
- Properly implements adaptive noise config
- Creates consistent configuration system
- Better long-term maintainability

**Cons**:
- More complex (2-3 hours)
- Requires code changes in multiple files
- Slightly higher testing burden

**Risks**: Low-Medium (mitigated by comprehensive testing)

### Option C: Complete Redesign
**Approach**: Redesign configuration system with YAML, environment variables, and validation

**Pros**:
- Modern configuration approach
- Better validation and type safety
- Environment-specific configs

**Cons**:
- Major refactor (1-2 days)
- Requires new dependencies
- Overkill for current needs

**Risks**: Medium-High

## Recommendation
**Option B: Full Cleanup with Connections**

This balances improvement with pragmatism. We remove cruft while fixing the disconnected features that already have partial implementations. This makes the system more coherent without over-engineering.

## High-Level Design

### Configuration Structure
```
config.toml (cleaned)
├── [data] - Input/output settings
├── [processing] - Core processing parameters  
├── [kalman] - Kalman filter parameters
├── [visualization] - Display settings
│   ├── [visualization.markers] - Marker display
│   └── [visualization.rejection] - Rejection display
├── [logging] - Logging configuration
├── [quality_scoring] - Quality scoring system
│   └── [quality_scoring.component_weights] - Component weights
└── [adaptive_noise] - Adaptive noise (reconnected)
```

### Affected Files
- `config.toml` - Remove ~40 lines
- `src/processor.py` - Connect adaptive noise config
- `src/utils.py` - Connect verbosity config
- `src/constants.py` - Remove duplicate defaults
- `main.py` - Minor updates for verbosity

## Implementation Plan (No Code)

### Phase 1: Configuration File Cleanup
1. **Backup current config.toml**
   - Create `config.toml.backup` with timestamp
   - Document removed settings in changelog

2. **Remove unused sections**
   - Delete `[visualization.interactive]` section
   - Delete `[visualization.quality]` section  
   - Delete `[visualization.export]` section
   - Clean up `[visualization]` main section

3. **Clean individual settings**
   - Remove `kalman_cleanup_threshold` from `[processing]`
   - Remove unused visualization settings (mode, theme, etc.)
   - Keep `[adaptive_noise]` for reconnection

### Phase 2: Code Connections
4. **Connect adaptive noise configuration**
   - Update `processor.py` to read from config instead of hardcoding
   - Specifically lines around noise_multiplier calculation
   - Ensure backward compatibility with defaults

5. **Connect verbosity system**
   - Update `main.py` to read verbosity from config
   - Connect to `utils.py` set_verbosity() function
   - Map config strings to verbosity levels

6. **Remove/consolidate constants**
   - Review `constants.py` for duplicate defaults
   - Ensure single source of truth for each setting
   - Update imports as needed

### Phase 3: Validation & Documentation
7. **Add configuration validation**
   - Create validate_config() function in utils.py
   - Check required sections exist
   - Validate value ranges (e.g., weights sum to 1.0)
   - Run validation on startup

8. **Update configuration documentation**
   - Add comments to config.toml explaining each setting
   - Document valid values and ranges
   - Add examples for complex settings

9. **Create migration guide**
   - Document what was removed and why
   - Provide mapping from old to new settings
   - Include in DEVELOPER_QUICK_REFERENCE.md

## Validation & Rollout

### Test Strategy
1. **Unit Tests**
   - Test configuration loading with cleaned config
   - Test adaptive noise with config values
   - Test verbosity system connection
   - Test validation function with invalid configs

2. **Integration Tests**
   - Run full pipeline with cleaned config
   - Test with various config combinations
   - Verify visualization still works
   - Test database export functionality

3. **Regression Tests**
   - Process test files with old vs new config
   - Compare outputs (should be identical)
   - Verify performance unchanged

### Manual QA Checklist
- [ ] Application starts with cleaned config
- [ ] All visualizations render correctly
- [ ] Adaptive noise responds to config changes
- [ ] Verbosity levels work as configured
- [ ] Quality scoring uses configured thresholds
- [ ] Database export works
- [ ] Progress logging at configured intervals

### Rollout Plan
1. Create feature branch `cleanup/config-system`
2. Implement changes with frequent commits
3. Run full test suite
4. Manual testing with real data files
5. Update documentation
6. Merge to main branch
7. Tag release as `v2.0.0-config-cleanup`

## Risks & Mitigations

### Risk 1: Breaking Changes
**Risk**: Removing settings might break unknown dependencies
**Mitigation**: 
- Comprehensive grep search before removal
- Keep backup of original config
- Test with multiple data files

### Risk 2: Hidden Feature Dependencies
**Risk**: Some features might silently depend on "unused" settings
**Mitigation**:
- Run with debug logging to catch config access
- Test all visualization modes
- Check for try/except blocks hiding errors

### Risk 3: User Confusion
**Risk**: Users with custom configs will have invalid settings
**Mitigation**:
- Provide clear migration guide
- Add validation with helpful error messages
- Keep backward compatibility for critical settings

## Acceptance Criteria

### Must Have
- [x] All unused configuration sections removed
- [x] All unused individual settings removed
- [x] Adaptive noise reads from config
- [x] Verbosity system connected to config
- [x] All tests pass with new config
- [x] Documentation updated

### Should Have
- [x] Configuration validation on startup
- [x] Helpful error messages for invalid config
- [x] Comments in config.toml explaining settings
- [x] Migration guide for users

### Nice to Have
- [ ] Config schema validation
- [ ] Environment variable overrides
- [ ] Config diff tool

## Out of Scope
- Complete configuration system redesign
- YAML/JSON configuration format change
- Environment-specific configuration files
- Configuration hot-reloading
- Web-based configuration UI
- Database-stored configuration

## Open Questions

1. **Should we keep `[adaptive_noise]` section?**
   - Currently unused but has partial implementation
   - Recommendation: Keep and properly connect it

2. **Should verbosity use strings or numbers?**
   - Current: Numbers in code, "auto" in config
   - Recommendation: Support both with string mapping

3. **Should we version the config file?**
   - Would help with migrations
   - Recommendation: Add version field for future use

4. **What about test configurations?**
   - Tests use hardcoded configs
   - Recommendation: Keep as-is for test isolation

## Review Cycle

### Self-Review Notes
- ✅ Plan covers all unused settings identified
- ✅ Provides clear implementation steps
- ✅ Includes comprehensive testing strategy
- ✅ Addresses backward compatibility concerns
- ✅ Risk mitigation strategies defined

### Revisions Made
1. Added Phase 2 for code connections (initially missed)
2. Expanded test strategy to include regression testing
3. Added migration guide requirement
4. Clarified that adaptive noise should be kept and connected

### Stakeholder Questions
1. Is removing 50% of configuration acceptable?
2. Should we maintain any backward compatibility?
3. Are there any planned features that need these configs?
4. Should we add configuration versioning now?

## Implementation Estimate

**Total Effort**: 2-3 hours

### Breakdown
- Phase 1 (Config cleanup): 30 minutes
- Phase 2 (Code connections): 1.5 hours
- Phase 3 (Validation & docs): 1 hour
- Testing: 30 minutes

## Next Steps
1. Get approval for Option B approach
2. Create feature branch
3. Implement Phase 1 (config file cleanup)
4. Implement Phase 2 (code connections)
5. Implement Phase 3 (validation)
6. Run comprehensive tests
7. Update documentation
8. Submit for review