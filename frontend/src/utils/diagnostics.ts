/**
 * Diagnostics System - Enterprise-grade performance and execution tracking
 * 
 * Provides comprehensive logging, performance monitoring, and execution tracing
 * to identify bottlenecks, infinite loops, and resource issues.
 */

export interface DiagnosticEntry {
  id: string;
  timestamp: number;
  component: string;
  method: string;
  phase: 'start' | 'end' | 'error';
  duration?: number;
  metadata?: any;
  stackTrace?: string;
}

export interface PerformanceMetric {
  name: string;
  count: number;
  totalTime: number;
  avgTime: number;
  minTime: number;
  maxTime: number;
  lastTime: number;
}

export class DiagnosticsSystem {
  private static instance: DiagnosticsSystem;
  private entries: DiagnosticEntry[] = [];
  private performanceMetrics: Map<string, PerformanceMetric> = new Map();
  private activeTraces: Map<string, number> = new Map();
  private isEnabled: boolean = true;
  private maxEntries: number = 1000;
  private slowThresholdMs: number = 100;
  private infiniteLoopThresholdMs: number = 5000;
  private watchdogTimers: Map<string, NodeJS.Timeout> = new Map();
  
  private constructor() {
    // Enable diagnostics in development
    this.isEnabled = import.meta.env.MODE === 'development';
    
    // Expose to window for debugging
    if (typeof window !== 'undefined') {
      (window as any).__diagnostics = this;
    }
  }
  
  static getInstance(): DiagnosticsSystem {
    if (!DiagnosticsSystem.instance) {
      DiagnosticsSystem.instance = new DiagnosticsSystem();
    }
    return DiagnosticsSystem.instance;
  }
  
  /**
   * Start tracing a method execution
   */
  trace(component: string, method: string, metadata?: any): string {
    if (!this.isEnabled) return '';
    
    const traceId = `${component}.${method}-${Date.now()}-${Math.random()}`;
    const startTime = performance.now();
    
    // Store active trace
    this.activeTraces.set(traceId, startTime);
    
    // Log start
    this.addEntry({
      id: traceId,
      timestamp: Date.now(),
      component,
      method,
      phase: 'start',
      metadata
    });
    
    // Set watchdog timer to detect infinite loops
    const watchdogId = setTimeout(() => {
      console.error(`[DIAGNOSTICS] INFINITE LOOP DETECTED in ${component}.${method}`, {
        duration: performance.now() - startTime,
        metadata,
        stackTrace: new Error().stack
      });
      
      // Force stop the trace
      this.endTrace(traceId, new Error('Infinite loop detected'));
      
      // Emit critical event
      window.dispatchEvent(new CustomEvent('diagnostic-critical', {
        detail: { component, method, reason: 'infinite-loop' }
      }));
    }, this.infiniteLoopThresholdMs);
    
    this.watchdogTimers.set(traceId, watchdogId);
    
    // Log to console in dev
    console.log(`[TRACE START] ${component}.${method}`, metadata);
    
    return traceId;
  }
  
  /**
   * End tracing a method execution
   */
  endTrace(traceId: string, error?: Error): void {
    if (!this.isEnabled || !traceId) return;
    
    const startTime = this.activeTraces.get(traceId);
    if (!startTime) return;
    
    const endTime = performance.now();
    const duration = endTime - startTime;
    
    // Clear watchdog
    const watchdogId = this.watchdogTimers.get(traceId);
    if (watchdogId) {
      clearTimeout(watchdogId);
      this.watchdogTimers.delete(traceId);
    }
    
    // Extract component and method from traceId
    const [componentMethod] = traceId.split('-');
    const [component, method] = componentMethod.split('.');
    
    // Log end or error
    this.addEntry({
      id: traceId,
      timestamp: Date.now(),
      component,
      method,
      phase: error ? 'error' : 'end',
      duration,
      metadata: error ? { error: error.message, stack: error.stack } : undefined
    });
    
    // Update performance metrics
    this.updateMetrics(componentMethod, duration);
    
    // Clean up
    this.activeTraces.delete(traceId);
    
    // Log slow operations
    if (duration > this.slowThresholdMs) {
      console.warn(`[SLOW OPERATION] ${component}.${method} took ${duration.toFixed(2)}ms`);
    }
    
    // Log to console
    if (error) {
      console.error(`[TRACE ERROR] ${component}.${method} (${duration.toFixed(2)}ms)`, error);
    } else {
      console.log(`[TRACE END] ${component}.${method} (${duration.toFixed(2)}ms)`);
    }
  }
  
  /**
   * Add a diagnostic entry
   */
  private addEntry(entry: DiagnosticEntry): void {
    this.entries.push(entry);
    
    // Limit entries to prevent memory leak
    if (this.entries.length > this.maxEntries) {
      this.entries = this.entries.slice(-this.maxEntries / 2);
    }
  }
  
  /**
   * Update performance metrics
   */
  private updateMetrics(key: string, duration: number): void {
    const existing = this.performanceMetrics.get(key);
    
    if (existing) {
      existing.count++;
      existing.totalTime += duration;
      existing.avgTime = existing.totalTime / existing.count;
      existing.minTime = Math.min(existing.minTime, duration);
      existing.maxTime = Math.max(existing.maxTime, duration);
      existing.lastTime = duration;
    } else {
      this.performanceMetrics.set(key, {
        name: key,
        count: 1,
        totalTime: duration,
        avgTime: duration,
        minTime: duration,
        maxTime: duration,
        lastTime: duration
      });
    }
  }
  
  /**
   * Get performance report
   */
  getPerformanceReport(): PerformanceMetric[] {
    return Array.from(this.performanceMetrics.values())
      .sort((a, b) => b.totalTime - a.totalTime);
  }
  
  /**
   * Get slow operations
   */
  getSlowOperations(thresholdMs: number = 50): DiagnosticEntry[] {
    return this.entries
      .filter(e => e.phase === 'end' && e.duration && e.duration > thresholdMs)
      .sort((a, b) => (b.duration || 0) - (a.duration || 0));
  }
  
  /**
   * Get error entries
   */
  getErrors(): DiagnosticEntry[] {
    return this.entries.filter(e => e.phase === 'error');
  }
  
  /**
   * Get active traces (potential hanging operations)
   */
  getActiveTraces(): Array<{ id: string; duration: number }> {
    const now = performance.now();
    return Array.from(this.activeTraces.entries()).map(([id, startTime]) => ({
      id,
      duration: now - startTime
    }));
  }
  
  /**
   * Clear all diagnostics
   */
  clear(): void {
    this.entries = [];
    this.performanceMetrics.clear();
    this.activeTraces.clear();
    
    // Clear all watchdogs
    this.watchdogTimers.forEach(timer => clearTimeout(timer));
    this.watchdogTimers.clear();
  }
  
  /**
   * Export diagnostics data
   */
  export(): string {
    return JSON.stringify({
      timestamp: new Date().toISOString(),
      entries: this.entries,
      performanceMetrics: Array.from(this.performanceMetrics.values()),
      activeTraces: this.getActiveTraces(),
      errors: this.getErrors(),
      slowOperations: this.getSlowOperations()
    }, null, 2);
  }
  
  /**
   * Enable/disable diagnostics
   */
  setEnabled(enabled: boolean): void {
    this.isEnabled = enabled;
    console.log(`[DIAGNOSTICS] ${enabled ? 'Enabled' : 'Disabled'}`);
  }
  
  /**
   * Check if any operations are hanging
   */
  checkForHangs(): Array<{ id: string; duration: number }> {
    const hangs = this.getActiveTraces().filter(trace => trace.duration > 1000);
    if (hangs.length > 0) {
      console.error('[DIAGNOSTICS] Hanging operations detected:', hangs);
    }
    return hangs;
  }
}

/**
 * Diagnostic decorator for class methods
 */
export function Diagnostic(component: string) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function (...args: any[]) {
      const diagnostics = DiagnosticsSystem.getInstance();
      const traceId = diagnostics.trace(component, propertyKey, { args: args.length });
      
      try {
        const result = await originalMethod.apply(this, args);
        diagnostics.endTrace(traceId);
        return result;
      } catch (error) {
        diagnostics.endTrace(traceId, error as Error);
        throw error;
      }
    };
    
    return descriptor;
  };
}

/**
 * Manual trace helper
 */
export function trace(component: string, method: string, fn: () => any): any {
  const diagnostics = DiagnosticsSystem.getInstance();
  const traceId = diagnostics.trace(component, method);
  
  try {
    const result = fn();
    if (result instanceof Promise) {
      return result
        .then(value => {
          diagnostics.endTrace(traceId);
          return value;
        })
        .catch(error => {
          diagnostics.endTrace(traceId, error);
          throw error;
        });
    }
    diagnostics.endTrace(traceId);
    return result;
  } catch (error) {
    diagnostics.endTrace(traceId, error as Error);
    throw error;
  }
}

export default DiagnosticsSystem;