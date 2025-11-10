/**
 * Run all test phases
 */

import { spawn } from 'bun';

const phases = [
  { name: 'Phase 1: Integration Tests', path: 'tests/phase1_integration/integration_tests.ts' },
  { name: 'Phase 2: Component Tests', path: 'tests/phase2_components/component_tests.ts' },
  // Phase 3, 4 will be added as they're implemented
  // { name: 'Phase 3: Sub-Component Tests', path: 'tests/phase3_subcomponents/subcomponent_tests.ts' },
  // { name: 'Phase 4: Edge Cases', path: 'tests/phase4_edge_cases/edge_case_tests.ts' },
];

console.log('🧪 Running All Test Phases\n');

let totalPassed = 0;
let totalFailed = 0;
let overallSuccess = true;

for (const phase of phases) {
  console.log(`${'='.repeat(80)}`);
  console.log(`Running: ${phase.name}`);
  console.log('='.repeat(80));

  const proc = spawn({
    cmd: ['bun', 'run', phase.path],
    stdout: 'inherit',
    stderr: 'inherit',
  });

  const exitCode = await proc.exited;

  if (exitCode !== 0) {
    overallSuccess = false;
    totalFailed++;
  } else {
    totalPassed++;
  }

  console.log('');
}

console.log(`${'='.repeat(80)}`);
console.log('📊 Overall Summary');
console.log('='.repeat(80));
console.log(`Total Phases Run: ${phases.length}`);
console.log(`✅ Passed: ${totalPassed}`);
console.log(`❌ Failed: ${totalFailed}`);
console.log('='.repeat(80));

process.exit(overallSuccess ? 0 : 1);
