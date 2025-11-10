/**
 * Test runner for cross-language comparison tests
 * Orchestrates running Python and TypeScript implementations
 */

import { Comparator, ComparisonResult, ComparisonConfig } from './comparator';

export interface TestCase {
  name: string;
  description: string;
  input: any;
  runPython: () => Promise<any>;
  runTypeScript: () => Promise<any>;
  comparisonConfig?: Partial<ComparisonConfig>;
}

export interface TestResult {
  testName: string;
  description: string;
  passed: boolean;
  pythonResult: any;
  typescriptResult: any;
  comparison: ComparisonResult;
  pythonDuration: number;
  typescriptDuration: number;
  error?: string;
}

export interface TestSuiteResult {
  suiteName: string;
  totalTests: number;
  passed: number;
  failed: number;
  duration: number;
  results: TestResult[];
}

export class TestRunner {
  private suiteName: string;
  private results: TestResult[] = [];

  constructor(suiteName: string) {
    this.suiteName = suiteName;
  }

  /**
   * Run a single test case
   */
  async runTest(test: TestCase): Promise<TestResult> {
    console.log(`\n🧪 Running: ${test.name}`);
    console.log(`   ${test.description}`);

    try {
      // Run Python implementation
      console.log('   → Running Python...');
      const pyStart = performance.now();
      const pythonResult = await test.runPython();
      const pythonDuration = performance.now() - pyStart;
      console.log(`   ✓ Python completed in ${pythonDuration.toFixed(2)}ms`);

      // Run TypeScript implementation
      console.log('   → Running TypeScript...');
      const tsStart = performance.now();
      const typescriptResult = await test.runTypeScript();
      const typescriptDuration = performance.now() - tsStart;
      console.log(`   ✓ TypeScript completed in ${typescriptDuration.toFixed(2)}ms`);

      // Compare results
      console.log('   → Comparing results...');
      const comparator = new Comparator(test.comparisonConfig);
      const comparison = comparator.compare(pythonResult, typescriptResult);

      const result: TestResult = {
        testName: test.name,
        description: test.description,
        passed: comparison.passed,
        pythonResult,
        typescriptResult,
        comparison,
        pythonDuration,
        typescriptDuration,
      };

      if (comparison.passed) {
        console.log(`   ✅ PASSED - ${comparison.summary}`);
      } else {
        console.log(`   ❌ FAILED - ${comparison.summary}`);
        console.log(comparator.formatDifferences(comparison.differences));
      }

      this.results.push(result);
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.log(`   ❌ ERROR: ${errorMessage}`);

      const result: TestResult = {
        testName: test.name,
        description: test.description,
        passed: false,
        pythonResult: null,
        typescriptResult: null,
        comparison: {
          passed: false,
          differences: [],
          summary: 'Test execution failed',
          numericDifferences: 0,
          structuralDifferences: 0,
        },
        pythonDuration: 0,
        typescriptDuration: 0,
        error: errorMessage,
      };

      this.results.push(result);
      return result;
    }
  }

  /**
   * Run multiple test cases
   */
  async runTests(tests: TestCase[]): Promise<TestSuiteResult> {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`📦 Test Suite: ${this.suiteName}`);
    console.log(`   Running ${tests.length} tests`);
    console.log('='.repeat(80));

    const suiteStart = performance.now();
    this.results = [];

    for (const test of tests) {
      await this.runTest(test);
    }

    const duration = performance.now() - suiteStart;
    const passed = this.results.filter(r => r.passed).length;
    const failed = this.results.length - passed;

    const suiteResult: TestSuiteResult = {
      suiteName: this.suiteName,
      totalTests: this.results.length,
      passed,
      failed,
      duration,
      results: this.results,
    };

    this.printSummary(suiteResult);
    return suiteResult;
  }

  /**
   * Print test suite summary
   */
  private printSummary(suiteResult: TestSuiteResult): void {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`📊 Test Suite Summary: ${suiteResult.suiteName}`);
    console.log('='.repeat(80));
    console.log(`Total Tests:    ${suiteResult.totalTests}`);
    console.log(`✅ Passed:      ${suiteResult.passed}`);
    console.log(`❌ Failed:      ${suiteResult.failed}`);
    console.log(`Success Rate:   ${((suiteResult.passed / suiteResult.totalTests) * 100).toFixed(1)}%`);
    console.log(`Duration:       ${(suiteResult.duration / 1000).toFixed(2)}s`);

    if (suiteResult.failed > 0) {
      console.log(`\n⚠️  Failed Tests:`);
      for (const result of suiteResult.results.filter(r => !r.passed)) {
        console.log(`   - ${result.testName}: ${result.error || result.comparison.summary}`);
      }
    }

    console.log('='.repeat(80));
  }

  /**
   * Generate detailed report
   */
  async generateReport(outputPath: string, suiteResult: TestSuiteResult): Promise<void> {
    const report = this.buildMarkdownReport(suiteResult);
    await Bun.write(outputPath, report);
    console.log(`\n📄 Report generated: ${outputPath}`);
  }

  /**
   * Build markdown report
   */
  private buildMarkdownReport(suiteResult: TestSuiteResult): string {
    const lines: string[] = [];

    lines.push(`# Cross-Language Test Report`);
    lines.push(`**Test Suite**: ${suiteResult.suiteName}`);
    lines.push(`**Date**: ${new Date().toISOString()}`);
    lines.push(`**Total Tests**: ${suiteResult.totalTests}`);
    lines.push(`**Passed**: ${suiteResult.passed}`);
    lines.push(`**Failed**: ${suiteResult.failed}`);
    lines.push(`**Success Rate**: ${((suiteResult.passed / suiteResult.totalTests) * 100).toFixed(1)}%`);
    lines.push(`**Duration**: ${(suiteResult.duration / 1000).toFixed(2)}s`);
    lines.push(``);

    lines.push(`## Summary`);
    lines.push(``);
    lines.push(`| Status | Count | Percentage |`);
    lines.push(`|--------|-------|------------|`);
    lines.push(`| ✅ Passed | ${suiteResult.passed} | ${((suiteResult.passed / suiteResult.totalTests) * 100).toFixed(1)}% |`);
    lines.push(`| ❌ Failed | ${suiteResult.failed} | ${((suiteResult.failed / suiteResult.totalTests) * 100).toFixed(1)}% |`);
    lines.push(``);

    lines.push(`## Performance Comparison`);
    lines.push(``);
    const avgPyTime = suiteResult.results.reduce((sum, r) => sum + r.pythonDuration, 0) / suiteResult.totalTests;
    const avgTsTime = suiteResult.results.reduce((sum, r) => sum + r.typescriptDuration, 0) / suiteResult.totalTests;
    const speedRatio = avgPyTime / avgTsTime;

    lines.push(`- **Python avg**: ${avgPyTime.toFixed(2)}ms`);
    lines.push(`- **TypeScript avg**: ${avgTsTime.toFixed(2)}ms`);
    lines.push(`- **Speed ratio**: ${speedRatio > 1 ? 'TypeScript' : 'Python'} is ${Math.abs(speedRatio - 1).toFixed(2)}x faster`);
    lines.push(``);

    if (suiteResult.failed > 0) {
      lines.push(`## Failed Tests`);
      lines.push(``);

      for (const result of suiteResult.results.filter(r => !r.passed)) {
        lines.push(`### ${result.testName}`);
        lines.push(`**Description**: ${result.description}`);
        lines.push(``);

        if (result.error) {
          lines.push(`**Error**: ${result.error}`);
          lines.push(``);
        } else {
          lines.push(`**Comparison**: ${result.comparison.summary}`);
          lines.push(``);

          if (result.comparison.differences.length > 0) {
            lines.push(`**Differences**:`);
            lines.push(`\`\`\``);
            const comparator = new Comparator();
            lines.push(comparator.formatDifferences(result.comparison.differences));
            lines.push(`\`\`\``);
            lines.push(``);
          }
        }
      }
    }

    lines.push(`## All Test Results`);
    lines.push(``);
    lines.push(`| Test Name | Status | Py Time | TS Time | Differences |`);
    lines.push(`|-----------|--------|---------|---------|-------------|`);

    for (const result of suiteResult.results) {
      const status = result.passed ? '✅' : '❌';
      const diffCount = result.comparison.differences.length;
      lines.push(`| ${result.testName} | ${status} | ${result.pythonDuration.toFixed(2)}ms | ${result.typescriptDuration.toFixed(2)}ms | ${diffCount} |`);
    }

    lines.push(``);

    return lines.join('\n');
  }

  /**
   * Save detailed results as JSON
   */
  async saveResults(outputPath: string, suiteResult: TestSuiteResult): Promise<void> {
    await Bun.write(outputPath, JSON.stringify(suiteResult, null, 2));
    console.log(`💾 Results saved: ${outputPath}`);
  }
}

/**
 * Helper function to run a single test
 */
export async function runSingleTest(test: TestCase): Promise<TestResult> {
  const runner = new TestRunner('Single Test');
  return runner.runTest(test);
}
