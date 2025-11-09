import { describe, it, expect } from 'bun:test';
import { Matrix } from 'ml-matrix';
import { InMemoryStore } from '../src/weight-processor-lib/core/database/memory_store';
import type { KalmanState } from '../src/weight-processor-lib/core/types';

describe('Deep Copy Snapshot Tests', () => {
  it('should create independent snapshots that do not share state references', async () => {
    const store = new InMemoryStore();
    const userId = 'test-user';

    // Create initial state with Matrix objects
    const initialState: KalmanState = {
      kalman_params: { test: 'param' },
      last_state: [new Matrix([[100], [0]])],  // weight=100, velocity=0
      last_covariance: [new Matrix([[1, 0], [0, 1]])],
      last_timestamp: new Date('2025-01-01T00:00:00Z'),
      last_accepted_timestamp: new Date('2025-01-01T00:00:00Z'),
      last_source: 'test',
      last_raw_weight: 100,
      measurement_history: [],
      reset_events: [],
      measurements_since_reset: 1,
      adaptation_state: {},
      version: 1,
    };

    // Save initial state and snapshot
    await store.saveState(userId, initialState);
    const snapshot1Timestamp = new Date('2025-01-01T01:00:00Z');
    await store.saveStateSnapshot(userId, snapshot1Timestamp);

    // Modify the state (Matrix and primitives)
    const state = await store.getState(userId);
    if (state && state.last_state && state.last_state[0]) {
      state.last_state[0].set(0, 0, 200);  // Change weight to 200
    }
    if (state) {
      state.last_raw_weight = 200;
      state.measurements_since_reset = 2;
    }
    await store.saveState(userId, state!);

    // Save second snapshot
    const snapshot2Timestamp = new Date('2025-01-01T02:00:00Z');
    await store.saveStateSnapshot(userId, snapshot2Timestamp);

    // Modify state again
    const state2 = await store.getState(userId);
    if (state2 && state2.last_state && state2.last_state[0]) {
      state2.last_state[0].set(0, 0, 300);  // Change weight to 300
    }
    if (state2) {
      state2.last_raw_weight = 300;
      state2.measurements_since_reset = 3;
    }
    await store.saveState(userId, state2!);

    // Verify current state is 300
    const currentState = await store.getState(userId);
    expect(currentState?.last_raw_weight).toBe(300);
    expect(currentState?.measurements_since_reset).toBe(3);
    if (currentState?.last_state?.[0]) {
      expect(currentState.last_state[0].get(0, 0)).toBe(300);
    }

    // Restore first snapshot and verify it has weight=100
    const snapshot1 = await store.getSnapshot(userId, snapshot1Timestamp);
    expect(snapshot1).not.toBeNull();
    expect(snapshot1?.last_raw_weight).toBe(100);
    expect(snapshot1?.measurements_since_reset).toBe(1);
    if (snapshot1?.last_state?.[0]) {
      expect(snapshot1.last_state[0].get(0, 0)).toBe(100);
    }

    // Restore second snapshot and verify it has weight=200
    const snapshot2 = await store.getSnapshot(userId, snapshot2Timestamp);
    expect(snapshot2).not.toBeNull();
    expect(snapshot2?.last_raw_weight).toBe(200);
    expect(snapshot2?.measurements_since_reset).toBe(2);
    if (snapshot2?.last_state?.[0]) {
      expect(snapshot2.last_state[0].get(0, 0)).toBe(200);
    }

    // Verify current state is still 300
    const finalState = await store.getState(userId);
    expect(finalState?.last_raw_weight).toBe(300);
    expect(finalState?.measurements_since_reset).toBe(3);
    if (finalState?.last_state?.[0]) {
      expect(finalState.last_state[0].get(0, 0)).toBe(300);
    }
  });

  it('should handle Matrix arrays correctly in deep copy', async () => {
    const store = new InMemoryStore();
    const userId = 'test-matrix-user';

    const state: KalmanState = {
      kalman_params: {},
      last_state: [
        new Matrix([[50], [0.1]]),
        new Matrix([[51], [0.2]]),
      ],
      last_covariance: [
        new Matrix([[1, 0], [0, 1]]),
        new Matrix([[2, 0], [0, 2]]),
      ],
      last_timestamp: new Date(),
      last_accepted_timestamp: new Date(),
      last_source: 'test',
      last_raw_weight: 50,
      measurement_history: [],
      reset_events: [],
      measurements_since_reset: 1,
      adaptation_state: {},
      version: 1,
    };

    await store.saveState(userId, state);
    const snapshotTime = new Date();
    await store.saveStateSnapshot(userId, snapshotTime);

    // Modify all matrices
    const currentState = await store.getState(userId);
    if (currentState?.last_state) {
      currentState.last_state[0].set(0, 0, 999);
      currentState.last_state[1].set(0, 0, 888);
    }
    if (currentState?.last_covariance) {
      currentState.last_covariance[0].set(0, 0, 777);
      currentState.last_covariance[1].set(0, 0, 666);
    }

    // Verify snapshot still has original values
    const snapshot = await store.getSnapshot(userId, snapshotTime);
    expect(snapshot?.last_state?.[0].get(0, 0)).toBe(50);
    expect(snapshot?.last_state?.[1].get(0, 0)).toBe(51);
    expect(snapshot?.last_covariance?.[0].get(0, 0)).toBe(1);
    expect(snapshot?.last_covariance?.[1].get(0, 0)).toBe(2);
  });

  it('should deep copy nested objects and arrays', async () => {
    const store = new InMemoryStore();
    const userId = 'test-nested';

    const state: KalmanState = {
      kalman_params: { nested: { value: 123 } },
      last_state: undefined,
      last_covariance: undefined,
      last_timestamp: null,
      last_accepted_timestamp: null,
      last_source: null,
      last_raw_weight: null,
      measurement_history: [
        { id: '1', weight: 100 },
        { id: '2', weight: 101 },
      ],
      reset_events: [
        { type: 'initial', timestamp: new Date('2025-01-01') },
      ],
      measurements_since_reset: 0,
      adaptation_state: { phase: 'active', count: 5 },
      version: 1,
    };

    await store.saveState(userId, state);
    const snapshotTime = new Date();
    await store.saveStateSnapshot(userId, snapshotTime);

    // Modify nested objects and arrays
    const currentState = await store.getState(userId);
    if (currentState) {
      currentState.kalman_params.nested.value = 999;
      currentState.measurement_history[0].weight = 999;
      currentState.adaptation_state.count = 999;
    }

    // Verify snapshot has original values
    const snapshot = await store.getSnapshot(userId, snapshotTime);
    expect(snapshot?.kalman_params.nested.value).toBe(123);
    expect(snapshot?.measurement_history[0].weight).toBe(100);
    expect(snapshot?.adaptation_state.count).toBe(5);
  });
});
