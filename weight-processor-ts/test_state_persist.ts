#!/usr/bin/env bun
import { ProcessorStateDB } from './src/core/database/database';

const db = new ProcessorStateDB();
const userId = 'TEST_USER';

// Simulate first measurement
console.log('=== Test 1: Save and reload state ===');
let state: any = db.get_state(userId) || {};
console.log('1. Initial state lastAcceptedTimestamp:', state.lastAcceptedTimestamp);

state.lastAcceptedTimestamp = new Date('2025-01-14T23:33:34.522Z');
state.lastTimestamp = new Date('2025-01-14T23:33:34.522Z');
db.save_state(userId, state);
console.log('2. After save lastAcceptedTimestamp:', state.lastAcceptedTimestamp);

// Get state again (simulate second measurement)
const state2 = db.get_state(userId);
console.log('3. After reload lastAcceptedTimestamp:', state2?.lastAcceptedTimestamp);
console.log('   Type:', typeof state2?.lastAcceptedTimestamp);
console.log('   Is Date?:', state2?.lastAcceptedTimestamp instanceof Date);

// Check if it's a string that needs parsing
if (typeof state2?.lastAcceptedTimestamp === 'string') {
  console.log('   WARNING: lastAcceptedTimestamp is stored as string!');
  const parsed = new Date(state2.lastAcceptedTimestamp);
  console.log('   Parsed:', parsed);
}

console.log('\n=== Test 2: Check gap calculation ===');
const timestamp1 = new Date('2025-01-14T23:33:34.522Z');
const timestamp2 = new Date('2025-03-08T14:57:22.966Z'); // ~52 days later

db.save_state(userId, { lastAcceptedTimestamp: timestamp1 });
const reloaded = db.get_state(userId);

const lastTs = reloaded?.lastAcceptedTimestamp;
console.log('Stored timestamp:', lastTs);
console.log('Type:', typeof lastTs);

if (lastTs) {
  const last = typeof lastTs === 'string' ? new Date(lastTs) : lastTs;
  const gapDays = (timestamp2.getTime() - last.getTime()) / (86400.0 * 1000);
  console.log('Gap in days:', gapDays);
  console.log('Should trigger hard reset?', gapDays >= 30);
}
