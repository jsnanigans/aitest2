/**
 * Circuit breaker pattern for preventing cascading failures.
 */

/**
 * Circuit breaker states
 */
export enum CircuitState {
  CLOSED = "closed", // Normal operation
  OPEN = "open", // Failing, reject calls
  HALF_OPEN = "half_open", // Testing recovery
}

/**
 * Raised when circuit is open and rejecting calls
 */
export class CircuitOpenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CircuitOpenError";
    Object.setPrototypeOf(this, CircuitOpenError.prototype);
  }
}

/**
 * Circuit breaker status information
 */
export interface CircuitBreakerStatus {
  name: string;
  state: string;
  failure_count: number;
  success_count: number;
  last_failure?: string;
  recovery_in?: number;
  last_error?: string;
}

/**
 * Prevents cascading failures by breaking the circuit after repeated failures.
 *
 * The circuit breaker has three states:
 * - CLOSED: Normal operation, requests pass through
 * - OPEN: Too many failures, requests are rejected immediately
 * - HALF_OPEN: Testing if the system has recovered
 */
export class CircuitBreaker {
  private failure_threshold: number;
  private timeout: number; // Seconds before attempting recovery
  private success_threshold: number;
  private name: string;

  private state: CircuitState;
  private failure_count: number;
  private success_count: number;
  private last_failure_time: Date | null;
  private last_attempt_time: Date | null;
  private last_error: Error | null;

  /**
   * Initialize circuit breaker.
   *
   * @param failure_threshold - Number of failures before opening circuit
   * @param timeout - Seconds to wait before attempting recovery
   * @param success_threshold - Successes needed in HALF_OPEN to close circuit
   * @param name - Name for logging
   */
  constructor(
    failure_threshold: number = 3,
    timeout: number = 60,
    success_threshold: number = 2,
    name: string = "circuit"
  ) {
    this.failure_threshold = failure_threshold;
    this.timeout = timeout;
    this.success_threshold = success_threshold;
    this.name = name;

    this.state = CircuitState.CLOSED;
    this.failure_count = 0;
    this.success_count = 0;
    this.last_failure_time = null;
    this.last_attempt_time = null;
    this.last_error = null;
  }

  /**
   * Execute function with circuit breaker protection.
   *
   * @param func - Function to execute
   * @param args - Arguments for func
   * @returns Result from func
   * @throws {CircuitOpenError} If circuit is open
   * @throws {Error} Any exception from func
   */
  call<T>(func: (...args: any[]) => T, ...args: any[]): T {
    if (this.state === CircuitState.OPEN) {
      if (this._should_attempt_recovery()) {
        this.state = CircuitState.HALF_OPEN;
        this.success_count = 0;
        console.info(`Circuit breaker '${this.name}' entering HALF_OPEN state`);
      } else {
        const time_remaining = this._time_until_recovery();
        throw new CircuitOpenError(
          `Circuit '${this.name}' open due to ${this.failure_count} failures. ` +
            `Retry in ${Math.round(time_remaining)} seconds`
        );
      }
    }

    try {
      // Attempt the call
      this.last_attempt_time = new Date();
      const result = func(...args);
      this._on_success();
      return result;
    } catch (e) {
      this._on_failure(e instanceof Error ? e : new Error(String(e)));
      throw e;
    }
  }

  /**
   * Check if enough time has passed to attempt recovery
   */
  private _should_attempt_recovery(): boolean {
    if (this.last_failure_time === null) {
      return true;
    }

    const elapsed = Date.now() - this.last_failure_time.getTime();
    return elapsed / 1000 >= this.timeout;
  }

  /**
   * Calculate seconds until recovery attempt
   */
  private _time_until_recovery(): number {
    if (this.last_failure_time === null) {
      return 0;
    }

    const elapsed = (Date.now() - this.last_failure_time.getTime()) / 1000;
    const remaining = this.timeout - elapsed;
    return Math.max(0, remaining);
  }

  /**
   * Handle successful call
   */
  private _on_success(): void {
    if (this.state === CircuitState.HALF_OPEN) {
      this.success_count += 1;
      console.debug(
        `Circuit '${this.name}' success in HALF_OPEN: ${this.success_count}/${this.success_threshold}`
      );

      if (this.success_count >= this.success_threshold) {
        console.info(
          `Circuit breaker '${this.name}' recovered, entering CLOSED state`
        );
        this.state = CircuitState.CLOSED;
        this.failure_count = 0;
        this.success_count = 0;
        this.last_error = null;
      }
    } else {
      // Reset failure count on success in CLOSED state
      if (this.failure_count > 0) {
        console.debug(
          `Circuit '${this.name}' success in CLOSED, resetting failure count`
        );
      }
      this.failure_count = 0;
    }
  }

  /**
   * Handle failed call
   */
  private _on_failure(error: Error): void {
    this.failure_count += 1;
    this.last_failure_time = new Date();
    this.last_error = error;

    if (this.state === CircuitState.HALF_OPEN) {
      console.warn(
        `Circuit breaker '${this.name}' recovery failed, reopening circuit`
      );
      this.state = CircuitState.OPEN;
      this.success_count = 0;
    } else if (this.failure_count >= this.failure_threshold) {
      console.error(
        `Circuit breaker '${this.name}' opening after ${this.failure_count} failures: ${error}`
      );
      this.state = CircuitState.OPEN;
    } else {
      console.warn(
        `Circuit '${this.name}' failure ${this.failure_count}/${this.failure_threshold}: ${error}`
      );
    }
  }

  /**
   * Manually reset the circuit breaker to closed state
   */
  reset(): void {
    console.info(`Circuit breaker '${this.name}' manually reset`);
    this.state = CircuitState.CLOSED;
    this.failure_count = 0;
    this.success_count = 0;
    this.last_failure_time = null;
    this.last_error = null;
  }

  /**
   * Check if circuit is currently open
   */
  get is_open(): boolean {
    return this.state === CircuitState.OPEN;
  }

  /**
   * Check if circuit is currently closed
   */
  get is_closed(): boolean {
    return this.state === CircuitState.CLOSED;
  }

  /**
   * Get current circuit breaker status.
   *
   * @returns Dictionary with status information
   */
  get_status(): CircuitBreakerStatus {
    const status: CircuitBreakerStatus = {
      name: this.name,
      state: this.state,
      failure_count: this.failure_count,
      success_count: this.success_count,
    };

    if (this.last_failure_time) {
      status.last_failure = this.last_failure_time.toISOString();
      if (this.state === CircuitState.OPEN) {
        status.recovery_in = this._time_until_recovery();
      }
    }

    if (this.last_error) {
      status.last_error = String(this.last_error);
    }

    return status;
  }
}

/**
 * Manages multiple circuit breakers for different operations.
 */
export class MultiCircuitBreaker {
  private breakers: Map<string, CircuitBreaker>;

  /**
   * Initialize multi-circuit breaker manager
   */
  constructor() {
    this.breakers = new Map<string, CircuitBreaker>();
  }

  /**
   * Add a new circuit breaker.
   *
   * @param name - Name of the breaker
   * @param failure_threshold - Failures before opening
   * @param timeout - Recovery timeout in seconds
   * @param success_threshold - Successes needed to close
   * @returns The created circuit breaker
   */
  add_breaker(
    name: string,
    failure_threshold: number = 3,
    timeout: number = 60,
    success_threshold: number = 2
  ): CircuitBreaker {
    const breaker = new CircuitBreaker(
      failure_threshold,
      timeout,
      success_threshold,
      name
    );
    this.breakers.set(name, breaker);
    return breaker;
  }

  /**
   * Get a circuit breaker by name
   */
  get_breaker(name: string): CircuitBreaker | undefined {
    return this.breakers.get(name);
  }

  /**
   * Call function through named circuit breaker.
   *
   * @param breaker_name - Name of breaker to use
   * @param func - Function to call
   * @param args - Function arguments
   * @returns Function result
   * @throws {Error} If breaker doesn't exist
   * @throws {CircuitOpenError} If circuit is open
   */
  call<T>(breaker_name: string, func: (...args: any[]) => T, ...args: any[]): T {
    const breaker = this.breakers.get(breaker_name);
    if (!breaker) {
      throw new Error(`No circuit breaker named '${breaker_name}'`);
    }

    return breaker.call(func, ...args);
  }

  /**
   * Get status of all circuit breakers
   */
  get_status(): Record<string, CircuitBreakerStatus> {
    const status: Record<string, CircuitBreakerStatus> = {};
    for (const [name, breaker] of this.breakers.entries()) {
      status[name] = breaker.get_status();
    }
    return status;
  }

  /**
   * Reset all circuit breakers
   */
  reset_all(): void {
    for (const breaker of this.breakers.values()) {
      breaker.reset();
    }
  }
}
