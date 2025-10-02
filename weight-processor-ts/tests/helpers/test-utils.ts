/**
 * Test utilities and helper functions
 */

/**
 * Check if two numbers are approximately equal within a tolerance
 */
export function expectClose(actual: number, expected: number, tolerance = 0.001): boolean {
  const diff = Math.abs(actual - expected);
  return diff <= tolerance;
}

/**
 * Create a mock measurement object for testing
 */
export function createMockMeasurement(overrides: Partial<any> = {}): any {
  return {
    device_id: 'test-device',
    user_id: 'test-user',
    weight_kg: 75.0,
    timestamp: new Date().toISOString(),
    source: 'QUESTIONNAIRE_ONBOARDING',
    ...overrides,
  };
}

/**
 * Sleep for testing async operations
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
