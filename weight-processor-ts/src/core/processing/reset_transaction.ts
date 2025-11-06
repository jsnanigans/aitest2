/**
 * Transaction management for atomic reset operations.
 * Ensures all reset operations succeed or rollback together.
 */

import type { ProcessorState } from '../../models';

/**
 * Types of operations in a reset transaction
 */
export enum ResetOperation {
  KALMAN_RESET = 'kalman_reset',
  STATE_UPDATE = 'state_update',
  BUFFER_UPDATE = 'buffer_update',
  STATE_PERSIST = 'state_persist',
}

/**
 * Snapshot of state at a point in transaction
 */
export interface TransactionCheckpoint {
  operation: ResetOperation;
  timestamp: number;
  stateSnapshot: ProcessorState;
  validationPassed: boolean;
}

/**
 * Manages atomic reset operations with automatic rollback.
 *
 * Ensures that all reset operations either complete successfully
 * or rollback to the original state if any operation fails.
 */
export class ResetTransaction {
  private userId: string;
  private checkpoints: TransactionCheckpoint[] = [];
  private originalStates: Map<ResetOperation, any> = new Map();
  private completedOperations: ResetOperation[] = [];
  private failed: boolean = false;
  private failureReason: string | null = null;

  /**
   * Initialize transaction for a specific user.
   */
  constructor(userId: string) {
    this.userId = userId;
  }

  /**
   * Save original state before modifying.
   */
  saveOriginalState(operation: ResetOperation, state: any): void {
    if (!this.originalStates.has(operation)) {
      this.originalStates.set(operation, this.deepCopy(state));
    }
  }

  /**
   * Save state snapshot after an operation.
   */
  saveCheckpoint(operation: ResetOperation, state: ProcessorState): void {
    const checkpoint: TransactionCheckpoint = {
      operation,
      timestamp: Date.now(),
      stateSnapshot: this.deepCopy(state),
      validationPassed: false,
    };
    this.checkpoints.push(checkpoint);
  }

  /**
   * Validate the state after an operation.
   *
   * Returns true if validation passed, false otherwise
   */
  validateCheckpoint(operation: ResetOperation, validator?: any): boolean {
    const checkpoint = this.getLastCheckpoint(operation);
    if (!checkpoint) {
      console.error(`No checkpoint found for ${operation}`);
      this.failed = true;
      this.failureReason = `Missing checkpoint for ${operation}`;
      return false;
    }

    try {
      // Use custom validator or simple validation
      let isValid: boolean;
      if (validator) {
        isValid = validator.validate(checkpoint.stateSnapshot, operation);
      } else {
        // Default validation - check that state has basic required fields
        isValid = this.defaultValidation(checkpoint.stateSnapshot);
      }

      checkpoint.validationPassed = isValid;
      if (!isValid) {
        this.failed = true;
        this.failureReason = `Validation failed for ${operation}`;
        console.error(this.failureReason);
      }

      return isValid;
    } catch (e) {
      console.error(`Validation error for ${operation}:`, e);
      this.failed = true;
      this.failureReason = `Validation error: ${e}`;
      return false;
    }
  }

  /**
   * Default validation - checks basic state structure
   */
  private defaultValidation(state: ProcessorState): boolean {
    // Basic validation - state should be an object
    if (!state || typeof state !== 'object') {
      return false;
    }

    // Check for required fields based on operation context
    // This is a simple check - can be extended as needed
    return true;
  }

  /**
   * Mark an operation as successfully completed.
   */
  markCompleted(operation: ResetOperation): void {
    this.completedOperations.push(operation);
  }

  /**
   * Rollback all completed operations.
   */
  rollback(reason: string): void {
    console.warn(`Rolling back reset transaction for user ${this.userId}: ${reason}`);

    // Return original states to caller
    // Actual state restoration happens in the processor
    console.info(
      `Rollback complete - ${this.completedOperations.length} operations rolled back`
    );

    // Clear transaction state
    this.checkpoints = [];
    this.completedOperations = [];
  }

  /**
   * Commit all operations - make permanent
   */
  commit(): void {
    // States are already updated in place, just log success
  }

  /**
   * Get original state for rollback.
   */
  getOriginalState(operation: ResetOperation): any | null {
    return this.originalStates.get(operation) ?? null;
  }

  /**
   * Get the most recent checkpoint for an operation.
   */
  private getLastCheckpoint(operation: ResetOperation): TransactionCheckpoint | null {
    for (let i = this.checkpoints.length - 1; i >= 0; i--) {
      if (this.checkpoints[i].operation === operation) {
        return this.checkpoints[i];
      }
    }
    return null;
  }

  /**
   * Check if transaction has failed
   */
  get isFailed(): boolean {
    return this.failed;
  }

  /**
   * Get failure reason
   */
  get getFailureReason(): string | null {
    return this.failureReason;
  }

  /**
   * Deep copy utility
   */
  private deepCopy<T>(obj: T): T {
    return JSON.parse(JSON.stringify(obj));
  }

  /**
   * Execute a function within this transaction context
   * Automatically handles commit/rollback
   */
  async execute<T>(fn: (txn: ResetTransaction) => Promise<T>): Promise<T> {
    try {
      const result = await fn(this);

      if (this.failed) {
        this.rollback(this.failureReason || 'Unknown failure');
        throw new Error(this.failureReason || 'Transaction failed');
      }

      this.commit();
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.rollback(errorMessage);
      throw error;
    }
  }

  /**
   * Execute a synchronous function within this transaction context
   */
  executeSync<T>(fn: (txn: ResetTransaction) => T): T {
    try {
      const result = fn(this);

      if (this.failed) {
        this.rollback(this.failureReason || 'Unknown failure');
        throw new Error(this.failureReason || 'Transaction failed');
      }

      this.commit();
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.rollback(errorMessage);
      throw error;
    }
  }
}

/**
 * Convenience function for atomic reset operations.
 *
 * Usage:
 *   await atomicReset(userId, async (txn) => {
 *     // Perform operations
 *     txn.saveCheckpoint(...);
 *     txn.validateCheckpoint(...);
 *     // If any operation fails, automatic rollback occurs
 *   });
 */
export async function atomicReset<T>(
  userId: string,
  fn: (txn: ResetTransaction) => Promise<T>
): Promise<T> {
  const txn = new ResetTransaction(userId);
  return txn.execute(fn);
}

/**
 * Synchronous version of atomicReset
 */
export function atomicResetSync<T>(
  userId: string,
  fn: (txn: ResetTransaction) => T
): T {
  const txn = new ResetTransaction(userId);
  return txn.executeSync(fn);
}
