/**
 * ReplayBuffer - 24-hour measurement buffering
 *
 * Manages in-memory storage of measurements for replay analysis.
 * Automatically manages buffer windows and provides safe operations.
 *
 * Note: Thread-safety not needed in JavaScript (single-threaded).
 * Use BufferFactory for instance management.
 *
 * Key features:
 * - Automatic 24-hour window rotation
 * - Memory-efficient storage with limits
 * - Proper cleanup() method for resource management
 * - Buffer state tracking and cleanup
 *
 * Ported from Python: weight_values/src/core/replay/replay_buffer.py
 */

import type { Config } from '../../models';

interface BufferData {
  measurements: Array<Record<string, any>>;
  first_timestamp: Date | null;
  last_timestamp: Date | null;
  created_at: Date;
}

interface BufferStats {
  total_measurements_buffered: number;
  buffers_created: number;
  buffers_triggered: number;
  buffers_cleaned: number;
  last_cleanup_time: Date;
}

interface TriggerResult {
  should_trigger: boolean;
  reason: string;
  buffer_age_hours?: number;
  trigger_threshold_hours?: number;
  measurement_count?: number;
  trigger_threshold?: number;
  min_required?: number;
}

export class ReplayBuffer {
  private config: Config;
  private buffer_hours: number;
  private max_buffer_measurements: number;
  private trigger_mode: string;
  private buffers: Map<string, BufferData>;
  private _stats: BufferStats;

  /**
   * Initialize replay buffer.
   */
  constructor(config?: Config) {
    this.config = config || ({} as Config);

    // Configuration
    const replay_config = this.config.replay || {};
    this.buffer_hours = replay_config.buffer_hours || 24;
    this.max_buffer_measurements = replay_config.max_buffer_measurements || 100;
    this.trigger_mode = replay_config.trigger_mode || 'time_based';

    // Buffer storage: user_id -> buffer_data
    this.buffers = new Map();

    // Statistics
    this._stats = {
      total_measurements_buffered: 0,
      buffers_created: 0,
      buffers_triggered: 0,
      buffers_cleaned: 0,
      last_cleanup_time: new Date()
    };
  }

  /**
   * Add a measurement to the user's buffer.
   */
  add_measurement(
    user_id: string,
    measurement: Record<string, any>
  ): {
    success: boolean;
    user_id?: string;
    buffer_size?: number;
    buffer_ready?: boolean;
    trigger_reason?: string;
    buffer_window_start?: Date | null;
    buffer_window_end?: Date | null;
    error?: string;
  } {
    try {
      // Ensure buffer exists for user
      if (!this.buffers.has(user_id)) {
        this._create_user_buffer(user_id);
      }

      const buffer_data = this.buffers.get(user_id)!;
      const timestamp = measurement.timestamp as Date;

      // Add measurement to buffer
      const measurement_copy = { ...measurement };
      buffer_data.measurements.push(measurement_copy);

      // Update buffer timestamps
      if (!buffer_data.first_timestamp || timestamp < buffer_data.first_timestamp) {
        buffer_data.first_timestamp = timestamp;
      }
      if (!buffer_data.last_timestamp || timestamp > buffer_data.last_timestamp) {
        buffer_data.last_timestamp = timestamp;
      }

      // Update statistics
      this._stats.total_measurements_buffered += 1;

      // Check buffer limits
      this._enforce_buffer_limits(user_id);

      // Check if buffer should be triggered for processing
      const trigger_result = this._check_buffer_trigger(user_id);

      console.debug(
        `Added measurement for user ${user_id}, buffer size: ${buffer_data.measurements.length}`
      );

      return {
        success: true,
        user_id,
        buffer_size: buffer_data.measurements.length,
        buffer_ready: trigger_result.should_trigger,
        trigger_reason: trigger_result.reason || 'not_ready',
        buffer_window_start: buffer_data.first_timestamp,
        buffer_window_end: buffer_data.last_timestamp
      };
    } catch (e) {
      const error = e as Error;
      console.error(`Failed to add measurement for user ${user_id}: ${error.message}`);
      return { success: false, error: error.message, user_id };
    }
  }

  /**
   * Get all buffered measurements for a user.
   */
  get_buffer_measurements(user_id: string): Array<Record<string, any>> | null {
    if (!this.buffers.has(user_id)) {
      return null;
    }

    // Return copy to prevent external modification
    const measurements = this.buffers.get(user_id)!.measurements;
    return measurements.map((m) => ({ ...m }));
  }

  /**
   * Clear buffer for a user (typically after processing).
   */
  clear_buffer(user_id: string): boolean {
    if (this.buffers.has(user_id)) {
      const old_size = this.buffers.get(user_id)!.measurements.length;
      this._create_user_buffer(user_id); // Reset to empty buffer
      this._stats.buffers_cleaned += 1;
      console.info(`Cleared buffer for user ${user_id} (was ${old_size} measurements)`);
      return true;
    }
    return false;
  }

  /**
   * Get information about a user's buffer.
   */
  get_buffer_info(user_id: string): Record<string, any> | null {
    if (!this.buffers.has(user_id)) {
      return null;
    }

    const buffer_data = this.buffers.get(user_id)!;
    const measurements = buffer_data.measurements;

    const info: Record<string, any> = {
      user_id,
      measurement_count: measurements.length,
      first_timestamp: buffer_data.first_timestamp,
      last_timestamp: buffer_data.last_timestamp,
      buffer_age_hours: 0,
      is_ready_for_processing: false,
      trigger_reason: null
    };

    // Calculate buffer age
    if (buffer_data.first_timestamp) {
      const age_delta = new Date().getTime() - buffer_data.first_timestamp.getTime();
      info.buffer_age_hours = age_delta / (3600 * 1000);
    }

    // Check if ready for processing
    const trigger_result = this._check_buffer_trigger(user_id);
    info.is_ready_for_processing = trigger_result.should_trigger;
    info.trigger_reason = trigger_result.reason;

    return info;
  }

  /**
   * Get list of user IDs with buffers ready for processing.
   */
  get_ready_buffers(): string[] {
    const ready_users: string[] = [];

    for (const user_id of this.buffers.keys()) {
      const trigger_result = this._check_buffer_trigger(user_id);
      if (trigger_result.should_trigger) {
        ready_users.push(user_id);
      }
    }

    return ready_users;
  }

  /**
   * Clean up old buffers that haven't been active.
   */
  cleanup_old_buffers(max_age_hours?: number): number {
    if (max_age_hours === undefined) {
      max_age_hours = this.buffer_hours * 2;
    }

    const current_time = new Date();
    const users_to_remove: string[] = [];

    for (const [user_id, buffer_data] of this.buffers.entries()) {
      const last_timestamp = buffer_data.last_timestamp;
      if (last_timestamp) {
        const age_hours = (current_time.getTime() - last_timestamp.getTime()) / (3600 * 1000);
        if (age_hours > max_age_hours) {
          users_to_remove.push(user_id);
        }
      }
    }

    // Remove old buffers
    for (const user_id of users_to_remove) {
      this.buffers.delete(user_id);
    }

    this._stats.buffers_cleaned += users_to_remove.length;
    this._stats.last_cleanup_time = current_time;

    if (users_to_remove.length > 0) {
      console.info(`Cleaned up ${users_to_remove.length} old buffers`);
    }

    return users_to_remove.length;
  }

  /**
   * Get buffer statistics.
   */
  get_stats(): Record<string, any> {
    const stats = { ...this._stats };

    let total_buffered_measurements = 0;
    for (const buffer of this.buffers.values()) {
      total_buffered_measurements += buffer.measurements.length;
    }

    return {
      ...stats,
      active_buffers: this.buffers.size,
      total_buffered_measurements,
      ready_for_processing: this.get_ready_buffers().length,
      config: {
        buffer_hours: this.buffer_hours,
        max_buffer_measurements: this.max_buffer_measurements,
        trigger_mode: this.trigger_mode
      }
    };
  }

  /**
   * Create empty buffer for user.
   */
  private _create_user_buffer(user_id: string): void {
    this.buffers.set(user_id, {
      measurements: [],
      first_timestamp: null,
      last_timestamp: null,
      created_at: new Date()
    });
    this._stats.buffers_created += 1;
    console.debug(`Created buffer for user ${user_id}`);
  }

  /**
   * Enforce buffer size limits by removing oldest measurements.
   */
  private _enforce_buffer_limits(user_id: string): void {
    const buffer_data = this.buffers.get(user_id)!;
    const measurements = buffer_data.measurements;

    if (measurements.length > this.max_buffer_measurements) {
      // Sort by timestamp and keep most recent
      measurements.sort((a, b) => {
        const ts_a = a.timestamp as Date;
        const ts_b = b.timestamp as Date;
        return ts_a.getTime() - ts_b.getTime();
      });
      buffer_data.measurements = measurements.slice(-this.max_buffer_measurements);

      // Update first timestamp
      if (buffer_data.measurements.length > 0) {
        buffer_data.first_timestamp = buffer_data.measurements[0].timestamp as Date;
      }

      console.debug(
        `Enforced buffer limit for user ${user_id}, kept ${buffer_data.measurements.length} measurements`
      );
    }
  }

  /**
   * Check if buffer should be triggered for processing.
   */
  private _check_buffer_trigger(user_id: string): TriggerResult {
    const buffer_data = this.buffers.get(user_id)!;
    const measurements = buffer_data.measurements;

    if (measurements.length === 0 || !buffer_data.first_timestamp) {
      return { should_trigger: false, reason: 'empty_buffer' };
    }

    // Check minimum measurements requirement for meaningful analysis
    const outlier_config = this.config.outlier_detection || {};
    const min_measurements = outlier_config.min_measurements_for_analysis || 5;

    if (measurements.length < min_measurements) {
      return {
        should_trigger: false,
        reason: 'insufficient_measurements',
        measurement_count: measurements.length,
        min_required: min_measurements
      };
    }

    // Calculate buffer age
    const buffer_age = new Date().getTime() - buffer_data.first_timestamp.getTime();
    const age_hours = buffer_age / (3600 * 1000);

    if (this.trigger_mode === 'time_based') {
      // Trigger when buffer reaches configured age AND has enough measurements
      if (age_hours >= this.buffer_hours) {
        return {
          should_trigger: true,
          reason: 'time_based_trigger',
          buffer_age_hours: age_hours,
          trigger_threshold_hours: this.buffer_hours,
          measurement_count: measurements.length
        };
      }
    } else if (this.trigger_mode === 'measurement_count') {
      // Trigger when buffer reaches measurement limit
      if (measurements.length >= this.max_buffer_measurements) {
        return {
          should_trigger: true,
          reason: 'measurement_count_trigger',
          measurement_count: measurements.length,
          trigger_threshold: this.max_buffer_measurements
        };
      }
    }

    return {
      should_trigger: false,
      reason: 'threshold_not_reached',
      buffer_age_hours: age_hours,
      measurement_count: measurements.length
    };
  }

  /**
   * Force trigger a buffer for immediate processing (for testing/debugging).
   */
  force_trigger_buffer(user_id: string): boolean {
    if (this.buffers.has(user_id) && this.buffers.get(user_id)!.measurements.length > 0) {
      console.info(`Force triggered buffer for user ${user_id}`);
      return true;
    }
    return false;
  }

  /**
   * Clean up buffer resources.
   *
   * Clears all buffers and resets internal state.
   * Called by BufferFactory when removing instances.
   */
  cleanup(): void {
    // Log summary before cleanup
    let total_measurements = 0;
    for (const buffer of this.buffers.values()) {
      total_measurements += buffer.measurements.length;
    }

    if (total_measurements > 0) {
      console.info(
        `Cleaning up ReplayBuffer with ${this.buffers.size} users, ` +
          `${total_measurements} total measurements`
      );
    }

    // Clear all buffers
    this.buffers.clear();

    // Reset stats
    this._stats.total_measurements_buffered = 0;
    this._stats.buffers_triggered = 0;

    console.debug('ReplayBuffer cleanup complete');
  }
}
