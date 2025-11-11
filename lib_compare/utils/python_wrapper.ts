/**
 * Wrapper for calling Python weight processor implementation
 * Uses a persistent Python process to eliminate subprocess spawning overhead
 */

import { spawn } from 'bun';
import type { Subprocess } from 'bun';
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
  private serverProcess: Subprocess | null = null;
  private initialized: boolean = false;

  constructor(pythonPath: string = 'python3') {
    this.pythonPath = pythonPath;
  }

  /**
   * Initialize persistent Python server
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    const scriptPath = `${import.meta.dir}/../scripts/python_server.py`;

    // Start persistent Python server
    this.serverProcess = spawn({
      cmd: ['uv', 'run', this.pythonPath, scriptPath],
      stdin: 'pipe',
      stdout: 'pipe',
      stderr: 'pipe',
      cwd: '/Users/brendanmullins/Projects/aitest/strem_process_anchor/python_lib',
    });

    // Wait for "Python server ready" message on stderr
    const stderrReader = this.serverProcess.stderr.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await stderrReader.read();
      if (done) break;

      const text = decoder.decode(value);
      if (text.includes('Python server ready')) {
        stderrReader.releaseLock();
        break;
      }
    }

    this.initialized = true;
  }

  /**
   * Cleanup - terminate the persistent server
   */
  async cleanup(): Promise<void> {
    if (this.serverProcess) {
      this.serverProcess.kill();
      await this.serverProcess.exited;
      this.serverProcess = null;
      this.initialized = false;
    }
  }

  /**
   * Process measurements using Python implementation (persistent server)
   */
  async processMeasurements(input: PythonProcessorInput): Promise<PythonProcessorResult> {
    // Ensure server is initialized
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.serverProcess) {
      throw new Error('Python server process not available');
    }

    // Send request as newline-delimited JSON
    const requestJson = JSON.stringify(input) + '\n';
    this.serverProcess.stdin.write(requestJson);

    // Read response line from stdout
    const stdoutReader = this.serverProcess.stdout.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await stdoutReader.read();
      if (done) {
        throw new Error('Python server closed unexpectedly');
      }

      buffer += decoder.decode(value, { stream: true });

      // Check if we have a complete line
      const newlineIndex = buffer.indexOf('\n');
      if (newlineIndex !== -1) {
        const line = buffer.slice(0, newlineIndex);
        buffer = buffer.slice(newlineIndex + 1);

        // Release the reader so we can read again later
        stdoutReader.releaseLock();

        try {
          const result = JSON.parse(line);

          // Check for error response
          if (result.error) {
            throw new Error(`Python processing error: ${result.error}`);
          }

          return result;
        } catch (error) {
          if (error instanceof SyntaxError) {
            throw new Error(`Failed to parse Python response: ${error}\nResponse: ${line}`);
          }
          throw error;
        }
      }
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
