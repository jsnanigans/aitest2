/**
 * Transaction management for atomic reset operations.
 * Ensures all reset operations succeed or rollback together.
 */

/**
 * Types of operations in a reset transaction
 */
export enum ResetOperation {
  KALMAN_RESET = "kalman_reset",
  STATE_UPDATE = "state_update",
  BUFFER_UPDATE = "buffer_update",
  STATE_PERSIST = "state_persist",
}

/**
 * Snapshot of state at a point in transaction
 */
export interface TransactionCheckpoint {
  operation: ResetOperation;
  timestamp: number;
  stateSnapshot: Record<string, any>;
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
  private checkpoints: TransactionCheckpoint[];
  private originalStates: Map<ResetOperation, any>;
  private completedOperations: ResetOperation[];
  private failed: boolean;
  private failureReason: string | null;

  /**
   * Initialize transaction for a specific user.
   *
   * @param userId - User identifier for logging
   */
  constructor(userId: string) {
    this.userId = userId;
    this.checkpoints = [];
    this.originalStates = new Map();
    this.completedOperations = [];
    this.failed = false;
    this.failureReason = null;
  }

  /**
   * Start transaction - capture initial state
   */
  start(): void {
    console.info(`Starting reset transaction for user ${this.userId}`);
  }

  /**
   * End transaction - commit or rollback
   *
   * @param error - Optional error that caused transaction failure
   * @returns true if successful, false if rollback occurred
   */
  end(error?: Error): boolean {
    if (error) {
      console.error(`Reset transaction failed with exception: ${error.message}`);
      this.rollback(error.message);
      return false;
    }

    if (this.failed) {
      this.rollback(this.failureReason || "Unknown failure");
      return false;
    }

    // All operations succeeded
    this.commit();
    return true;
  }

  /**
   * Save original state before modifying.
   *
   * @param operation - Type of operation
   * @param state - Original state to preserve
   */
  saveOriginalState(operation: ResetOperation, state: any): void {
    if (!this.originalStates.has(operation)) {
      // Deep clone the state
      this.originalStates.set(operation, this.deepClone(state));
      console.debug(`Saved original state for ${operation}`);
    }
  }

  /**
   * Save state snapshot after an operation.
   *
   * @param operation - Type of operation completed
   * @param state - New state after operation
   */
  saveCheckpoint(operation: ResetOperation, state: Record<string, any>): void {
    const checkpoint: TransactionCheckpoint = {
      operation,
      timestamp: Date.now() / 1000, // Convert to seconds to match Python's time.time()
      stateSnapshot: this.deepClone(state),
      validationPassed: false,
    };
    this.checkpoints.push(checkpoint);
    console.debug(`Checkpoint saved for ${operation}`);
  }

  /**
   * Validate the state after an operation.
   *
   * @param operation - Operation to validate
   * @param validator - Optional custom validator, otherwise uses StateValidator
   * @returns true if validation passed, false otherwise
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
      // Import here to avoid circular dependencies
      if (!validator) {
        // Lazy import to avoid circular dependency
        const { StateValidator } = require("./state_validator");
        validator = new StateValidator();
      }

      const isValid = validator.validate(checkpoint.stateSnapshot, operation);

      checkpoint.validationPassed = isValid;
      if (!isValid) {
        this.failed = true;
        this.failureReason = `Validation failed for ${operation}`;
        console.error(this.failureReason);
      }

      return isValid;
    } catch (e) {
      const error = e as Error;
      console.error(`Validation error for ${operation}: ${error.message}`);
      this.failed = true;
      this.failureReason = `Validation error: ${error.message}`;
      return false;
    }
  }

  /**
   * Mark an operation as successfully completed.
   *
   * @param operation - Operation that completed successfully
   */
  markCompleted(operation: ResetOperation): void {
    this.completedOperations.push(operation);
    console.info(`Operation completed: ${operation}`);
  }

  /**
   * Rollback all completed operations.
   *
   * @param reason - Reason for rollback (for logging)
   */
  rollback(reason: string): void {
    console.warn(
      `Rolling back reset transaction for user ${this.userId}: ${reason}`
    );

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
    console.info(
      `Committing reset transaction for user ${this.userId} - ${this.completedOperations.length} operations`
    );
    // States are already updated in place, just log success
  }

  /**
   * Get original state for rollback.
   *
   * @param operation - Operation to get original state for
   * @returns Original state if saved, undefined otherwise
   */
  getOriginalState(operation: ResetOperation): any | undefined {
    return this.originalStates.get(operation);
  }

  /**
   * Get the most recent checkpoint for an operation.
   *
   * @param operation - Operation to find checkpoint for
   * @returns Most recent checkpoint or null
   */
  private getLastCheckpoint(
    operation: ResetOperation
  ): TransactionCheckpoint | null {
    // Iterate in reverse to find most recent checkpoint
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
   * Deep clone an object (simple implementation for state objects)
   */
  private deepClone(obj: any): any {
    if (obj === null || typeof obj !== "object") {
      return obj;
    }

    if (obj instanceof Date) {
      return new Date(obj.getTime());
    }

    if (obj instanceof Array) {
      return obj.map((item) => this.deepClone(item));
    }

    if (obj instanceof Map) {
      const clonedMap = new Map();
      obj.forEach((value, key) => {
        clonedMap.set(key, this.deepClone(value));
      });
      return clonedMap;
    }

    if (obj instanceof Set) {
      const clonedSet = new Set();
      obj.forEach((value) => {
        clonedSet.add(this.deepClone(value));
      });
      return clonedSet;
    }

    // Plain object
    const clonedObj: any = {};
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        clonedObj[key] = this.deepClone(obj[key]);
      }
    }
    return clonedObj;
  }
}

/**
 * Convenience function for atomic reset operations.
 * Mimics Python's context manager pattern.
 *
 * Usage:
 *   const result = await atomicReset(userId, async (txn) => {
 *     // Perform operations
 *     txn.saveCheckpoint(...);
 *     txn.validateCheckpoint(...);
 *     // If any operation fails, automatic rollback occurs
 *   });
 *
 * @param userId - User identifier
 * @param operation - Async function to execute within transaction
 * @returns Result of the operation function
 */
export async function atomicReset<T>(
  userId: string,
  operation: (txn: ResetTransaction) => Promise<T>
): Promise<T> {
  const txn = new ResetTransaction(userId);
  txn.start();

  try {
    const result = await operation(txn);

    if (txn.isFailed) {
      throw new Error(txn["failureReason"] || "Transaction failed");
    }

    txn.commit();
    return result;
  } catch (error) {
    const err = error as Error;
    txn.rollback(err.message);
    throw error;
  }
}

/**
 * Synchronous version of atomicReset for non-async operations.
 *
 * @param userId - User identifier
 * @param operation - Sync function to execute within transaction
 * @returns Result of the operation function
 */
export function atomicResetSync<T>(
  userId: string,
  operation: (txn: ResetTransaction) => T
): T {
  const txn = new ResetTransaction(userId);
  txn.start();

  try {
    const result = operation(txn);

    if (txn.isFailed) {
      throw new Error(txn["failureReason"] || "Transaction failed");
    }

    txn.commit();
    return result;
  } catch (error) {
    const err = error as Error;
    txn.rollback(err.message);
    throw error;
  }
}
