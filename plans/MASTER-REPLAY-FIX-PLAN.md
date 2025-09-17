# Master Implementation Plan: Replay System Critical Fixes
**Generated: September 17, 2025**
**Priority: CRITICAL - Blocking Production**

## Executive Summary

The council review identified 5 HIGH severity issues that must be fixed before the replay feature can go to production. This master plan coordinates the implementation of all fixes with proper sequencing to avoid conflicts and maximize efficiency.

**Total Estimated Effort**: 8-10 developer weeks
**Recommended Team Size**: 2-3 senior developers
**Implementation Timeline**: 4-6 weeks with parallel work streams

## Critical Dependencies Graph

```
[Singleton Removal] ──┬──> [Memory Limits]
                      ├──> [WAL Implementation]
                      └──> [Lost Update Fix]

[State Race Fix] ─────────> Independent

[WAL Implementation] ─────> [Lost Update Fix] (optional optimization)
```

## Implementation Phases

### Phase 0: Foundation (Week 1)
**Goal**: Remove architectural blockers and enable parallel development

#### 0.1 Remove Singleton Anti-Pattern ⚡ BLOCKER
- **Plan**: `plans/remove-singleton-replay-buffer.md`
- **Owner**: Senior Architect
- **Duration**: 3 days
- **Why First**: Blocks all other testing and development
- **Deliverables**:
  - BufferFactory implementation
  - Migration helpers
  - Backward compatibility layer
  - Updated tests with isolation

#### 0.2 Fix State Restoration Race Condition
- **Plan**: `plans/fix-replay-state-restoration-race.md`
- **Owner**: Backend Developer
- **Duration**: 2 days
- **Can Run**: In parallel with 0.1
- **Deliverables**:
  - Atomic check_and_restore_snapshot method
  - Transaction support in database layer
  - Validation framework
  - Race condition tests

### Phase 1: Core Safety (Week 2)
**Goal**: Implement critical safety mechanisms

#### 1.1 Implement Global Memory Limits
- **Plan**: `plans/global-memory-limits-lru.md`
- **Owner**: Systems Developer
- **Duration**: 4 days
- **Dependencies**: Singleton removal (0.1)
- **Deliverables**:
  - MemoryManager class
  - LRU eviction with priority scoring
  - Emergency memory release
  - Memory pressure metrics
  - Performance benchmarks

#### 1.2 Add Write-Ahead Log - Part 1
- **Plan**: `plans/wal-crash-recovery.md`
- **Owner**: Database Developer
- **Duration**: 3 days
- **Dependencies**: Singleton removal (0.1)
- **Deliverables**:
  - WAL format specification
  - Basic write/read operations
  - Checkpointing mechanism
  - Corruption detection

### Phase 2: Consistency (Week 3)
**Goal**: Fix data consistency issues

#### 2.1 Fix Lost Update Problem
- **Plan**: `plans/replay-buffer-consistency-fix.md`
- **Owner**: Senior Developer
- **Duration**: 4 days
- **Dependencies**: Singleton removal (0.1), WAL Part 1 (1.2)
- **Deliverables**:
  - Buffer versioning system
  - Snapshot coordination
  - Pending queue implementation
  - Causal ordering guarantees
  - Integration tests

#### 2.2 Complete WAL Integration
- **Plan**: `plans/wal-crash-recovery.md` (continued)
- **Owner**: Database Developer
- **Duration**: 3 days
- **Dependencies**: Lost Update Fix (2.1)
- **Deliverables**:
  - Recovery process
  - Async write queue
  - Storage management
  - WAL tools and utilities

### Phase 3: Optimization & Hardening (Week 4)
**Goal**: Performance optimization and production readiness

#### 3.1 Optimize Outlier Detection
- **Plan**: Create new plan
- **Owner**: Algorithm Developer
- **Duration**: 3 days
- **Deliverables**:
  - Single-pass detection algorithm
  - Performance benchmarks
  - Accuracy validation

#### 3.2 Add Audit Logging
- **Plan**: Create new plan
- **Owner**: Backend Developer
- **Duration**: 2 days
- **Deliverables**:
  - Structured audit events
  - Audit trail persistence
  - Query interface
  - Compliance reports

#### 3.3 Simplify Configuration
- **Plan**: Create new plan
- **Owner**: Senior Developer
- **Duration**: 2 days
- **Deliverables**:
  - Consolidated config structure
  - Migration script
  - Documentation
  - Validation framework

### Phase 4: Integration Testing (Week 5)
**Goal**: Comprehensive system validation

#### 4.1 Integration Test Suite
- **Duration**: 3 days
- **Team**: All developers
- **Deliverables**:
  - End-to-end replay scenarios
  - Failure injection tests
  - Performance regression tests
  - Memory leak detection
  - Concurrency stress tests

#### 4.2 Production Simulation
- **Duration**: 2 days
- **Team**: All developers + SRE
- **Deliverables**:
  - Load testing with production data
  - Failover scenarios
  - Recovery procedures
  - Monitoring validation

## Risk Mitigation Strategy

### Rollback Plan
Each phase includes:
1. Feature flags for gradual rollout
2. Backward compatibility maintenance
3. Automated rollback triggers
4. Data migration reversibility

### Testing Requirements
- Unit tests: 90% coverage minimum
- Integration tests: All critical paths
- Performance tests: No >10% degradation
- Chaos tests: Failure injection scenarios
- Soak tests: 72-hour stability runs

### Monitoring & Alerts
Required metrics before production:
- Memory usage (global and per-user)
- WAL write latency (p50, p95, p99)
- Replay success/failure rate
- Buffer overflow frequency
- State restoration time
- Outlier detection accuracy

## Success Criteria

### Phase 0 Complete When:
- [ ] All tests run in isolation
- [ ] No global state dependencies
- [ ] Race condition impossible
- [ ] CI/CD pipeline green

### Phase 1 Complete When:
- [ ] Memory never exceeds limits
- [ ] WAL survives crash tests
- [ ] <10ms performance overhead
- [ ] Zero data loss scenarios

### Phase 2 Complete When:
- [ ] No lost updates in any test
- [ ] Causal consistency maintained
- [ ] Recovery completes <30s
- [ ] All edges cases handled

### Phase 3 Complete When:
- [ ] Outlier detection <100ms
- [ ] Full audit trail available
- [ ] Config reduced to 5 params
- [ ] Documentation complete

### Phase 4 Complete When:
- [ ] 72-hour soak test passes
- [ ] Load test at 2x capacity
- [ ] All alerts configured
- [ ] Runbook completed

## Resource Allocation

### Team Structure
```
Tech Lead (20%)
├── Architecture Track: Senior Architect (100%)
│   ├── Singleton Removal
│   └── Configuration Simplification
│
├── Safety Track: Systems Developer (100%)
│   ├── Memory Limits
│   ├── Performance Optimization
│   └── Monitoring
│
└── Data Track: Database Developer (100%)
    ├── State Race Fix
    ├── WAL Implementation
    ├── Lost Update Fix
    └── Audit Logging
```

### External Dependencies
- DBA review for WAL design
- Security review for audit logging
- SRE partnership for monitoring
- Product sign-off on configuration changes

## Communication Plan

### Daily Standups
- 15 min daily sync
- Blocker identification
- Dependency coordination

### Weekly Council Review
- Progress against plan
- Risk assessment
- Go/no-go decisions

### Stakeholder Updates
- Weekly email to leadership
- Bi-weekly demo to product
- Phase completion announcements

## Contingency Plans

### If Behind Schedule:
1. **Week 2**: Reduce Phase 3 scope (defer optimization)
2. **Week 3**: Add 1 developer from another team
3. **Week 4**: Descope config simplification
4. **Week 5**: Extend timeline by 1 week

### If Critical Bug Found:
1. Stop all work
2. Emergency council session
3. Hotfix or rollback decision
4. Root cause analysis
5. Plan adjustment

## Appendix: Individual Plan References

1. **State Restoration Race**: `plans/fix-replay-state-restoration-race.md`
2. **Lost Update Problem**: `plans/replay-buffer-consistency-fix.md`
3. **Global Memory Limits**: `plans/global-memory-limits-lru.md`
4. **Write-Ahead Log**: `plans/wal-crash-recovery.md`
5. **Singleton Removal**: `plans/remove-singleton-replay-buffer.md`

## Council Approval

**Butler Lampson**: "Good sequencing. Ship after Phase 2 if needed."
**Nancy Leveson**: "Phase 1 safety measures are non-negotiable."
**Barbara Liskov**: "Singleton removal first is correct. Don't skip it."
**Leslie Lamport**: "Phase 2 consistency fixes must be atomic."
**Martin Kleppmann**: "WAL design needs DBA review before coding."

## Next Steps

1. [ ] Get team assignments confirmed
2. [ ] Set up project tracking board
3. [ ] Schedule kick-off meeting
4. [ ] Create feature flag infrastructure
5. [ ] Establish monitoring baseline
6. [ ] Begin Phase 0 implementation

---

**Document Status**: APPROVED FOR IMPLEMENTATION
**Review Date**: September 17, 2025
**Next Review**: End of Phase 1