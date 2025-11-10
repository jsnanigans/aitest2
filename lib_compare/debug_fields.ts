/**
 * Debug script to see which fields differ between Python and TypeScript
 */

import { pythonWrapper } from './utils/python_wrapper';
import { typescriptWrapper } from './utils/typescript_wrapper';

const input = {
  deviceId: 'test-device-struct',
  userId: 'test-user-struct',
  measurements: [
    {
      weight_kg: 70.0,
      timestamp: Date.now(),
      source: 'withings',
    },
  ],
};

async function debug() {
  console.log('Running Python...');
  const pyResult = await pythonWrapper.processMeasurements(input);

  console.log('\nRunning TypeScript...');
  const tsResult = await typescriptWrapper.processMeasurements(input);

  const pyResultKeys = Object.keys(pyResult.results[0] || {}).sort();
  const tsResultKeys = Object.keys(tsResult.results[0] || {}).sort();

  const pyStateKeys = Object.keys(pyResult.finalState || {}).sort();
  const tsStateKeys = Object.keys(tsResult.finalState || {}).sort();

  console.log('\n=== RESULT KEYS ===');
  console.log('\nPython keys (' + pyResultKeys.length + '):');
  console.log(pyResultKeys);

  console.log('\nTypeScript keys (' + tsResultKeys.length + '):');
  console.log(tsResultKeys);

  console.log('\n--- Missing in TypeScript ---');
  const missingInTS = pyResultKeys.filter(k => !tsResultKeys.includes(k));
  console.log(missingInTS);

  console.log('\n--- Extra in TypeScript ---');
  const extraInTS = tsResultKeys.filter(k => !pyResultKeys.includes(k));
  console.log(extraInTS);

  console.log('\n\n=== STATE KEYS ===');
  console.log('\nPython keys (' + pyStateKeys.length + '):');
  console.log(pyStateKeys);

  console.log('\nTypeScript keys (' + tsStateKeys.length + '):');
  console.log(tsStateKeys);

  console.log('\n--- Missing in TypeScript ---');
  const missingStateInTS = pyStateKeys.filter(k => !tsStateKeys.includes(k));
  console.log(missingStateInTS);

  console.log('\n--- Extra in TypeScript ---');
  const extraStateInTS = tsStateKeys.filter(k => !pyStateKeys.includes(k));
  console.log(extraStateInTS);
}

debug().catch(console.error);
