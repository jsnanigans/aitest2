/**
 * Test a single measurement to debug the innovation_covariance calculation
 */
import { ConfigManager } from './src/config/config_manager';
import { processMeasurement } from './src/core/processing/processor';
import { InMemoryStateStorage } from './src/storage/in_memory_storage';

async function main() {
  // Initialize
  const configManager = ConfigManager.getInstance();
  await configManager.loadConfig('./config.toml');
  const config = configManager.getConfig();
  const db = new InMemoryStateStorage();

  // Process first measurement
  const result = await processMeasurement(
    'ADC64C0B-CB46-41F9-BDA0-CC11A35942D7',
    104.32616,
    new Date('2025-01-14T00:00:00.000Z'),
    'https://api.iglucose.com',
    config,
    'kg',
    db
  );

  console.log('Result:', result);
}

main().catch(console.error);
