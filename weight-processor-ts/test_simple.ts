/**
 * Simple test script to process test_user.csv
 */

import { ConfigManager } from './src/config/config_manager';
import { ProcessorStateDB } from './src/core/database/database';
import { WeightProcessorService } from './src/services/weight_processor_service';

async function main() {
  console.log('🚀 Starting minimal weight processor test...\n');

  try {
    // 1. Load configuration
    console.log('📋 Loading configuration...');
    const config = ConfigManager.loadConfig('./config.toml');
    console.log('✅ Config loaded\n');

    // 2. Create in-memory database
    console.log('💾 Creating in-memory database...');
    const stateStore = new ProcessorStateDB();
    console.log('✅ Database created\n');

    // 3. Create service
    console.log('⚙️  Initializing processor service...');
    const service = new WeightProcessorService(stateStore, config);
    console.log('✅ Service initialized\n');

    // 4. Create test measurements from test_user.csv
    console.log('📊 Processing test measurements...\n');

    const userId = 'ADC64C0B-CB46-41F9-BDA0-CC11A35942D7';

    const testMeasurements = [
      {
        measurement_id: '03dac217-7020-44ca-9884-10084d0e2c39',
        weight: 57.2,
        unit: 'kg',
        timestamp: new Date('2025-07-27T00:02:53.375Z'),
        source: 'https://api.iglucose.com'
      },
      {
        measurement_id: '05088123-08fb-4093-8396-bad142004e3e',
        weight: 56.7,
        unit: 'kg',
        timestamp: new Date('2025-08-10T18:20:04.208Z'),
        source: 'https://api.iglucose.com'
      },
      {
        measurement_id: '06ff5b04-0ae0-4555-943a-f11b6aab1803',
        weight: 110.2,
        unit: 'kg',
        timestamp: new Date('2025-06-18T03:45:02.605Z'),
        source: 'https://api.iglucose.com'
      }
    ];

    // Process each measurement
    for (const measurement of testMeasurements) {
      console.log(`\n📍 Processing measurement ${measurement.measurement_id?.substring(0, 8)}...`);
      console.log(`   Weight: ${measurement.weight} kg`);
      console.log(`   Time: ${measurement.timestamp.toISOString()}`);

      const result = await service.process_single(userId, measurement);

      console.log(`   ✅ Accepted: ${result.accepted}`);
      if (result.qualityScore !== undefined) {
        console.log(`   📊 Quality Score: ${result.qualityScore.toFixed(2)}`);
      }
      if (result.kalmanEstimate !== undefined) {
        console.log(`   🎯 Kalman Estimate: ${result.kalmanEstimate.toFixed(2)} kg`);
      }
      if (result.rejectionReason) {
        console.log(`   ❌ Rejection Reason: ${result.rejectionReason}`);
      }
    }

    console.log('\n\n✨ Test completed successfully!\n');

  } catch (error) {
    console.error('❌ Error:', error);
    if (error instanceof Error) {
      console.error('Stack:', error.stack);
    }
    process.exit(1);
  }
}

main();
