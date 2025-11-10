/**
 * Wrapper for calling Python weight processor implementation
 */

import { spawn } from 'bun';
import type { Measurement } from './data_generator';

export interface PythonProcessorInput {
  deviceId: string;
  userId: string;
  measurements: Measurement[];
}

export interface PythonProcessorResult {
  results: any[];
  finalState: any;
}

export class PythonWrapper {
  private pythonPath: string;

  constructor(pythonPath: string = 'python3') {
    this.pythonPath = pythonPath;
  }

  /**
   * Process measurements using Python implementation
   */
  async processMeasurements(input: PythonProcessorInput): Promise<PythonProcessorResult> {
    const scriptPath = `${import.meta.dir}/../scripts/python_runner.py`;

    // Create input JSON for Python script
    const inputJson = JSON.stringify(input);

    // Run Python script using uv (for proper environment management)
    const proc = spawn({
      cmd: ['uv', 'run', this.pythonPath, scriptPath],
      stdin: 'pipe',
      stdout: 'pipe',
      stderr: 'pipe',
      cwd: '/Users/brendanmullins/Projects/aitest/strem_process_anchor/python_lib',
    });

    // Write input to stdin
    proc.stdin.write(inputJson);
    proc.stdin.end();

    // Wait for completion
    const output = await new Response(proc.stdout).text();
    const errorOutput = await new Response(proc.stderr).text();

    const exitCode = await proc.exited;

    if (exitCode !== 0) {
      throw new Error(`Python script failed with exit code ${exitCode}\nError: ${errorOutput}`);
    }

    if (errorOutput && !errorOutput.includes('UserWarning')) {
      console.warn('Python stderr:', errorOutput);
    }

    try {
      const result = JSON.parse(output);
      return result;
    } catch (error) {
      throw new Error(`Failed to parse Python output: ${error}\nOutput: ${output}`);
    }
  }

  /**
   * Test single measurement processing
   */
  async processSingle(
    deviceId: string,
    userId: string,
    measurement: Measurement
  ): Promise<any> {
    const result = await this.processMeasurements({
      deviceId,
      userId,
      measurements: [measurement],
    });

    return result.results[0];
  }

  /**
   * Process measurements one by one and return all results
   */
  async processSequence(
    deviceId: string,
    userId: string,
    measurements: Measurement[]
  ): Promise<any[]> {
    const result = await this.processMeasurements({
      deviceId,
      userId,
      measurements,
    });

    return result.results;
  }
}

/**
 * Default Python wrapper instance
 */
export const pythonWrapper = new PythonWrapper();
