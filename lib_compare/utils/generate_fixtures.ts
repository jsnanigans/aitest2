/**
 * Script to generate all test fixtures
 */

import { generateAllFixtures } from './data_generator';
import { resolve } from 'path';

const fixturesDir = resolve(import.meta.dir, '../fixtures');

console.log('🔧 Generating test fixtures...\n');

await generateAllFixtures(fixturesDir);

console.log('\n✅ Fixture generation complete!');
