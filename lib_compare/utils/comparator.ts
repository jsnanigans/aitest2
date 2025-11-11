/**
 * Comparator utility for cross-language test outputs
 * Handles numeric tolerance, structural comparison, and difference reporting
 */

export interface ComparisonConfig {
  absoluteTolerance: number;
  relativeTolerance: number;
  ignoreKeys?: string[];
  strictTypes?: boolean;
  fieldAliases?: Record<string, string[]>;  // Map field names to their aliases
}

export interface Difference {
  path: string;
  pythonValue: any;
  typescriptValue: any;
  difference?: number;
  type: 'value' | 'type' | 'missing' | 'extra';
  message: string;
}

export interface ComparisonResult {
  passed: boolean;
  differences: Difference[];
  summary: string;
  numericDifferences: number;
  structuralDifferences: number;
}

export class Comparator {
  private config: ComparisonConfig;

  constructor(config?: Partial<ComparisonConfig>) {
    this.config = {
      absoluteTolerance: config?.absoluteTolerance ?? 1e-10,
      relativeTolerance: config?.relativeTolerance ?? 1e-8,
      ignoreKeys: config?.ignoreKeys ?? ['enabled', 'adaptation_state', 'version'],  // TS-only fields
      strictTypes: config?.strictTypes ?? true,
      fieldAliases: config?.fieldAliases ?? {
        'rejection_reason': ['rejectionReason'],  // Python uses snake_case, TS uses camelCase
      },
    };
  }

  /**
   * Check if a field name or value appears to be a timestamp
   */
  private isTimestampField(path: string, value: number): boolean {
    const fieldName = path.split('.').pop()?.toLowerCase() || '';

    // Check field name
    if (fieldName.includes('timestamp') || fieldName === 'time') {
      return true;
    }

    // Check if value is in milliseconds range (1970-2100)
    // Timestamps in ms are typically > 1e12 and < 5e12
    if (value > 1e12 && value < 5e12) {
      return true;
    }

    return false;
  }

  /**
   * Compare two values with tolerance handling
   */
  compare(
    pythonValue: any,
    typescriptValue: any,
    path: string = 'root'
  ): ComparisonResult {
    const differences: Difference[] = [];
    this._compareRecursive(pythonValue, typescriptValue, path, differences);

    const numericDifferences = differences.filter(d => d.type === 'value').length;
    const structuralDifferences = differences.filter(d => d.type !== 'value').length;

    const passed = differences.length === 0;
    const summary = passed
      ? '✓ Values match within tolerance'
      : `✗ Found ${differences.length} difference(s): ${numericDifferences} numeric, ${structuralDifferences} structural`;

    return {
      passed,
      differences,
      summary,
      numericDifferences,
      structuralDifferences,
    };
  }

  /**
   * Unwrap single-element arrays for matrix comparison
   * Python outputs flat numbers, TypeScript may wrap them in arrays
   */
  private unwrapSingleElementArray(value: any): any {
    if (Array.isArray(value) && value.length === 1) {
      return value[0];
    }
    return value;
  }

  /**
   * Recursive comparison logic
   */
  private _compareRecursive(
    pyValue: any,
    tsValue: any,
    path: string,
    differences: Difference[]
  ): void {
    // Handle null/undefined
    if (pyValue === null || pyValue === undefined) {
      if (tsValue !== null && tsValue !== undefined) {
        differences.push({
          path,
          pythonValue: pyValue,
          typescriptValue: tsValue,
          type: 'value',
          message: `Python value is ${pyValue}, TypeScript value is ${tsValue}`,
        });
      }
      return;
    }

    if (tsValue === null || tsValue === undefined) {
      differences.push({
        path,
        pythonValue: pyValue,
        typescriptValue: tsValue,
        type: 'value',
        message: `TypeScript value is ${tsValue}, Python value is ${pyValue}`,
      });
      return;
    }

    // Check if key should be ignored
    const keyName = path.split('.').pop() || '';
    if (this.config.ignoreKeys?.includes(keyName)) {
      return;
    }

    // Handle matrix values: unwrap single-element arrays for comparison with flat numbers
    const unwrappedPyValue = this.unwrapSingleElementArray(pyValue);
    const unwrappedTsValue = this.unwrapSingleElementArray(tsValue);

    // Handle numbers (including unwrapped matrix values)
    if (typeof unwrappedPyValue === 'number' && typeof unwrappedTsValue === 'number') {
      this._compareNumbers(unwrappedPyValue, unwrappedTsValue, path, differences);
      return;
    }

    // Handle arrays
    if (Array.isArray(pyValue) && Array.isArray(tsValue)) {
      this._compareArrays(pyValue, tsValue, path, differences);
      return;
    }

    // Handle objects
    if (typeof pyValue === 'object' && typeof tsValue === 'object') {
      this._compareObjects(pyValue, tsValue, path, differences);
      return;
    }

    // Handle primitive types (string, boolean, etc.)
    if (this.config.strictTypes && typeof pyValue !== typeof tsValue) {
      differences.push({
        path,
        pythonValue: pyValue,
        typescriptValue: tsValue,
        type: 'type',
        message: `Type mismatch: Python ${typeof pyValue}, TypeScript ${typeof tsValue}`,
      });
      return;
    }

    if (pyValue !== tsValue) {
      differences.push({
        path,
        pythonValue: pyValue,
        typescriptValue: tsValue,
        type: 'value',
        message: `Value mismatch: ${pyValue} !== ${tsValue}`,
      });
    }
  }

  /**
   * Compare two numbers with tolerance
   */
  private _compareNumbers(
    pyNum: number,
    tsNum: number,
    path: string,
    differences: Difference[]
  ): void {
    // Handle special values
    if (Number.isNaN(pyNum) && Number.isNaN(tsNum)) {
      return; // Both NaN is okay
    }

    if (Number.isNaN(pyNum) || Number.isNaN(tsNum)) {
      differences.push({
        path,
        pythonValue: pyNum,
        typescriptValue: tsNum,
        type: 'value',
        message: `NaN mismatch: Python ${pyNum}, TypeScript ${tsNum}`,
      });
      return;
    }

    if (!Number.isFinite(pyNum) || !Number.isFinite(tsNum)) {
      if (pyNum !== tsNum) {
        differences.push({
          path,
          pythonValue: pyNum,
          typescriptValue: tsNum,
          type: 'value',
          message: `Infinity mismatch: Python ${pyNum}, TypeScript ${tsNum}`,
        });
      }
      return;
    }

    // Check absolute difference
    const absDiff = Math.abs(pyNum - tsNum);

    // Use larger tolerance for timestamps (3700000ms = ~1 hour + buffer for test execution time)
    const isTimestamp = this.isTimestampField(path, pyNum) || this.isTimestampField(path, tsNum);
    const absoluteTolerance = isTimestamp ? 3700000 : this.config.absoluteTolerance;

    if (absDiff <= absoluteTolerance) {
      return;
    }

    // Check relative difference
    const maxAbs = Math.max(Math.abs(pyNum), Math.abs(tsNum));
    const relDiff = maxAbs > 0 ? absDiff / maxAbs : 0;

    if (relDiff > this.config.relativeTolerance) {
      differences.push({
        path,
        pythonValue: pyNum,
        typescriptValue: tsNum,
        difference: absDiff,
        type: 'value',
        message: `Numeric difference exceeds tolerance: abs=${absDiff.toExponential(3)}, rel=${(relDiff * 100).toFixed(6)}%`,
      });
    }
  }

  /**
   * Compare two arrays
   */
  private _compareArrays(
    pyArr: any[],
    tsArr: any[],
    path: string,
    differences: Difference[]
  ): void {
    if (pyArr.length !== tsArr.length) {
      differences.push({
        path,
        pythonValue: pyArr,
        typescriptValue: tsArr,
        type: 'missing',
        message: `Array length mismatch: Python ${pyArr.length}, TypeScript ${tsArr.length}`,
      });
      return;
    }

    for (let i = 0; i < pyArr.length; i++) {
      this._compareRecursive(pyArr[i], tsArr[i], `${path}[${i}]`, differences);
    }
  }

  /**
   * Find alias for a key in TypeScript object
   */
  private findAliasKey(pyKey: string, tsKeys: Set<string>): string | null {
    const aliases = this.config.fieldAliases?.[pyKey] || [];
    for (const alias of aliases) {
      if (tsKeys.has(alias)) {
        return alias;
      }
    }
    return null;
  }

  /**
   * Check if a TS key is an alias for any Python key
   */
  private isAliasKey(tsKey: string, pyKeys: Set<string>): boolean {
    for (const [pyKey, aliases] of Object.entries(this.config.fieldAliases || {})) {
      if (aliases.includes(tsKey) && pyKeys.has(pyKey)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Compare two objects
   */
  private _compareObjects(
    pyObj: Record<string, any>,
    tsObj: Record<string, any>,
    path: string,
    differences: Difference[]
  ): void {
    const pyKeys = new Set(Object.keys(pyObj));
    const tsKeys = new Set(Object.keys(tsObj));

    // Find missing keys (check aliases too)
    for (const key of pyKeys) {
      if (this.config.ignoreKeys?.includes(key)) {
        continue;
      }

      const aliasKey = this.findAliasKey(key, tsKeys);
      if (!tsKeys.has(key) && !aliasKey) {
        differences.push({
          path: `${path}.${key}`,
          pythonValue: pyObj[key],
          typescriptValue: undefined,
          type: 'missing',
          message: `Key missing in TypeScript output`,
        });
      }
    }

    // Find extra keys (ignore if they're aliases)
    for (const key of tsKeys) {
      if (this.config.ignoreKeys?.includes(key)) {
        continue;
      }

      if (!pyKeys.has(key) && !this.isAliasKey(key, pyKeys)) {
        differences.push({
          path: `${path}.${key}`,
          pythonValue: undefined,
          typescriptValue: tsObj[key],
          type: 'extra',
          message: `Extra key in TypeScript output`,
        });
      }
    }

    // Compare common keys (including aliases)
    for (const key of pyKeys) {
      if (this.config.ignoreKeys?.includes(key)) {
        continue;
      }

      let tsKey = key;
      if (!tsKeys.has(key)) {
        const aliasKey = this.findAliasKey(key, tsKeys);
        if (aliasKey) {
          tsKey = aliasKey;
        } else {
          continue; // Already reported as missing above
        }
      }

      this._compareRecursive(pyObj[key], tsObj[tsKey], `${path}.${key}`, differences);
    }
  }

  /**
   * Format differences for display
   */
  formatDifferences(differences: Difference[]): string {
    if (differences.length === 0) {
      return '✓ No differences found';
    }

    const lines: string[] = [
      `Found ${differences.length} difference(s):\n`,
    ];

    for (const diff of differences) {
      lines.push(`  ${diff.path}:`);
      lines.push(`    Type: ${diff.type}`);
      lines.push(`    Python:     ${this._formatValue(diff.pythonValue)}`);
      lines.push(`    TypeScript: ${this._formatValue(diff.typescriptValue)}`);
      if (diff.difference !== undefined) {
        lines.push(`    Difference: ${diff.difference.toExponential(3)}`);
      }
      lines.push(`    ${diff.message}`);
      lines.push('');
    }

    return lines.join('\n');
  }

  private _formatValue(value: any): string {
    if (value === null) return 'null';
    if (value === undefined) return 'undefined';
    if (typeof value === 'number') {
      if (Number.isNaN(value)) return 'NaN';
      if (!Number.isFinite(value)) return value > 0 ? 'Infinity' : '-Infinity';
      return value.toString();
    }
    if (typeof value === 'object') {
      return JSON.stringify(value, null, 2).substring(0, 100);
    }
    return String(value);
  }
}

/**
 * Helper function for quick comparisons
 */
export function compare(
  pythonValue: any,
  typescriptValue: any,
  config?: Partial<ComparisonConfig>
): ComparisonResult {
  const comparator = new Comparator(config);
  return comparator.compare(pythonValue, typescriptValue);
}
