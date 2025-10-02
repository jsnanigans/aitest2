/**
 * Circuit breaker pattern for fault tolerance.
 */

export class CircuitOpenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CircuitOpenError';
  }
}

enum CircuitState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open',
}

export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failureCount: number = 0;
  private successCount: number = 0;
  private lastFailureTime: number = 0;
  private name: string;
  private failureThreshold: number;
  private timeout: number;
  private successThreshold: number;

  constructor(params: {
    failureThreshold: number;
    timeout: number;
    successThreshold: number;
    name: string;
  }) {
    this.failureThreshold = params.failureThreshold;
    this.timeout = params.timeout * 1000; // Convert to milliseconds
    this.successThreshold = params.successThreshold;
    this.name = params.name;
  }

  async call<T>(fn: (...args: any[]) => T | Promise<T>, ...args: any[]): Promise<T> {
    if (this.state === CircuitState.OPEN) {
      // Check if timeout has elapsed
      if (Date.now() - this.lastFailureTime >= this.timeout) {
        this.state = CircuitState.HALF_OPEN;
        this.successCount = 0;
      } else {
        throw new CircuitOpenError(
          `Circuit breaker "${this.name}" is open`
        );
      }
    }

    try {
      const result = await Promise.resolve(fn(...args));
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failureCount = 0;

    if (this.state === CircuitState.HALF_OPEN) {
      this.successCount++;
      if (this.successCount >= this.successThreshold) {
        this.state = CircuitState.CLOSED;
        this.successCount = 0;
      }
    }
  }

  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this.failureCount >= this.failureThreshold) {
      this.state = CircuitState.OPEN;
    }
  }

  getState(): string {
    return this.state;
  }

  reset(): void {
    this.state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.lastFailureTime = 0;
  }
}
