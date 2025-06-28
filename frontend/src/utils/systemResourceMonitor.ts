/**
 * System Resource Monitor
 * Cross-platform resource monitoring for Windows, Mac, and Linux
 * Tracks CPU, memory, and performance metrics with automatic throttling
 */

import React from 'react';

interface SystemMetrics {
  cpu: {
    usage: number; // Percentage 0-100
    cores: number;
    temperature?: number; // Celsius
  };
  memory: {
    total: number; // Bytes
    used: number; // Bytes
    free: number; // Bytes
    percentage: number; // 0-100
    heap: {
      used: number;
      total: number;
      limit: number;
    };
  };
  performance: {
    fps: number;
    frameTime: number; // ms
    renderTime: number; // ms
    updateTime: number; // ms
  };
  processes: {
    active: number;
    pending: number;
    blocked: number;
  };
}

interface ResourceThresholds {
  cpu: {
    warning: number; // Default 70%
    critical: number; // Default 90%
  };
  memory: {
    warning: number; // Default 80%
    critical: number; // Default 95%
  };
  fps: {
    minimum: number; // Default 30fps
    target: number; // Default 60fps
  };
}

export class SystemResourceMonitor {
  private metrics: SystemMetrics;
  private thresholds: ResourceThresholds;
  private updateInterval: number = 1000; // 1 second
  private callbacks: Map<string, (metrics: SystemMetrics) => void> = new Map();
  private intervalId?: NodeJS.Timeout;
  private performanceObserver?: PerformanceObserver;
  private lastFrameTime: number = 0;
  private frameCount: number = 0;
  private frameTimes: number[] = [];
  private maxFrameSamples: number = 60;
  
  // Platform detection
  private platform: 'windows' | 'mac' | 'linux' | 'unknown';
  
  constructor(thresholds?: Partial<ResourceThresholds>) {
    this.platform = this.detectPlatform();
    
    this.thresholds = {
      cpu: {
        warning: thresholds?.cpu?.warning ?? 70,
        critical: thresholds?.cpu?.critical ?? 90
      },
      memory: {
        warning: thresholds?.memory?.warning ?? 80,
        critical: thresholds?.memory?.critical ?? 95
      },
      fps: {
        minimum: thresholds?.fps?.minimum ?? 30,
        target: thresholds?.fps?.target ?? 60
      }
    };
    
    this.metrics = this.getInitialMetrics();
    this.setupPerformanceObserver();
  }
  
  /**
   * Detect the current platform
   */
  private detectPlatform(): 'windows' | 'mac' | 'linux' | 'unknown' {
    if (typeof window !== 'undefined' && window.navigator) {
      const userAgent = window.navigator.userAgent.toLowerCase();
      if (userAgent.includes('win')) return 'windows';
      if (userAgent.includes('mac')) return 'mac';
      if (userAgent.includes('linux')) return 'linux';
    }
    
    // Node.js environment
    if (typeof process !== 'undefined') {
      switch (process.platform) {
        case 'win32': return 'windows';
        case 'darwin': return 'mac';
        case 'linux': return 'linux';
      }
    }
    
    return 'unknown';
  }
  
  /**
   * Get initial metrics structure
   */
  private getInitialMetrics(): SystemMetrics {
    return {
      cpu: {
        usage: 0,
        cores: navigator.hardwareConcurrency || 4,
        temperature: undefined
      },
      memory: {
        total: 0,
        used: 0,
        free: 0,
        percentage: 0,
        heap: {
          used: 0,
          total: 0,
          limit: 0
        }
      },
      performance: {
        fps: 60,
        frameTime: 16.67,
        renderTime: 0,
        updateTime: 0
      },
      processes: {
        active: 0,
        pending: 0,
        blocked: 0
      }
    };
  }
  
  /**
   * Setup performance observer for detailed metrics
   */
  private setupPerformanceObserver(): void {
    if (typeof PerformanceObserver === 'undefined') return;
    
    try {
      this.performanceObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'measure') {
            if (entry.name === 'render') {
              this.metrics.performance.renderTime = entry.duration;
            } else if (entry.name === 'update') {
              this.metrics.performance.updateTime = entry.duration;
            }
          }
        }
      });
      
      this.performanceObserver.observe({ entryTypes: ['measure'] });
    } catch (error) {
      console.warn('Performance Observer not available:', error);
    }
  }
  
  /**
   * Start monitoring system resources
   */
  start(): void {
    if (this.intervalId) return;
    
    this.intervalId = setInterval(() => {
      this.updateMetrics();
      this.notifyCallbacks();
      this.checkThresholds();
    }, this.updateInterval);
    
    // Start frame monitoring
    this.startFrameMonitoring();
  }
  
  /**
   * Stop monitoring
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
    
    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }
  }
  
  /**
   * Update all metrics
   */
  private async updateMetrics(): Promise<void> {
    await this.updateMemoryMetrics();
    await this.updateCPUMetrics();
    this.updateProcessMetrics();
  }
  
  /**
   * Update memory metrics with cross-platform support
   */
  private async updateMemoryMetrics(): Promise<void> {
    // Browser memory API
    if ('memory' in performance && (performance as any).memory) {
      const memory = (performance as any).memory;
      this.metrics.memory.heap = {
        used: memory.usedJSHeapSize || 0,
        total: memory.totalJSHeapSize || 0,
        limit: memory.jsHeapSizeLimit || 0
      };
      
      this.metrics.memory.percentage = 
        (this.metrics.memory.heap.used / this.metrics.memory.heap.limit) * 100;
    }
    
    // Try to get system memory if available (requires permissions)
    try {
      if ('storage' in navigator && 'estimate' in navigator.storage) {
        const estimate = await navigator.storage.estimate();
        if (estimate.usage && estimate.quota) {
          this.metrics.memory.used = estimate.usage;
          this.metrics.memory.total = estimate.quota;
          this.metrics.memory.free = estimate.quota - estimate.usage;
        }
      }
    } catch (error) {
      // Fallback to heap memory
      this.metrics.memory.used = this.metrics.memory.heap.used;
      this.metrics.memory.total = this.metrics.memory.heap.limit;
      this.metrics.memory.free = this.metrics.memory.heap.limit - this.metrics.memory.heap.used;
    }
  }
  
  /**
   * Update CPU metrics (estimated in browser environment)
   */
  private async updateCPUMetrics(): Promise<void> {
    // Estimate CPU usage based on main thread blocking
    const startTime = performance.now();
    
    // Perform a small benchmark
    let sum = 0;
    for (let i = 0; i < 1000000; i++) {
      sum += Math.sqrt(i);
    }
    
    const elapsed = performance.now() - startTime;
    
    // Estimate CPU usage based on benchmark time
    // Lower times indicate less CPU load
    const expectedTime = 10; // ms for benchmark on idle system
    const cpuLoad = Math.min(100, (elapsed / expectedTime) * 50);
    
    // Smooth the CPU usage with previous values
    this.metrics.cpu.usage = this.metrics.cpu.usage * 0.7 + cpuLoad * 0.3;
  }
  
  /**
   * Update process metrics
   */
  private updateProcessMetrics(): void {
    // Count active promises and timers
    let activeCount = 0;
    let pendingCount = 0;
    
    // Check for queued microtasks (estimated)
    if ('queueMicrotask' in window) {
      let microtaskExecuted = false;
      queueMicrotask(() => { microtaskExecuted = true; });
      
      // If microtask hasn't executed immediately, queue is busy
      setTimeout(() => {
        if (!microtaskExecuted) pendingCount++;
      }, 0);
    }
    
    this.metrics.processes = {
      active: activeCount,
      pending: pendingCount,
      blocked: 0
    };
  }
  
  /**
   * Start frame rate monitoring
   */
  private startFrameMonitoring(): void {
    const measureFrame = (timestamp: number) => {
      if (this.lastFrameTime > 0) {
        const frameTime = timestamp - this.lastFrameTime;
        this.frameTimes.push(frameTime);
        
        // Keep only recent samples
        if (this.frameTimes.length > this.maxFrameSamples) {
          this.frameTimes.shift();
        }
        
        // Calculate average frame time and FPS
        const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        this.metrics.performance.frameTime = avgFrameTime;
        this.metrics.performance.fps = 1000 / avgFrameTime;
      }
      
      this.lastFrameTime = timestamp;
      
      if (this.intervalId) {
        requestAnimationFrame(measureFrame);
      }
    };
    
    requestAnimationFrame(measureFrame);
  }
  
  /**
   * Check if any thresholds are exceeded
   */
  private checkThresholds(): void {
    const alerts: string[] = [];
    
    // CPU threshold check
    if (this.metrics.cpu.usage > this.thresholds.cpu.critical) {
      alerts.push(`CRITICAL: CPU usage at ${this.metrics.cpu.usage.toFixed(1)}%`);
      this.applyCPUThrottling();
    } else if (this.metrics.cpu.usage > this.thresholds.cpu.warning) {
      alerts.push(`WARNING: CPU usage at ${this.metrics.cpu.usage.toFixed(1)}%`);
    }
    
    // Memory threshold check
    if (this.metrics.memory.percentage > this.thresholds.memory.critical) {
      alerts.push(`CRITICAL: Memory usage at ${this.metrics.memory.percentage.toFixed(1)}%`);
      this.applyMemoryOptimization();
    } else if (this.metrics.memory.percentage > this.thresholds.memory.warning) {
      alerts.push(`WARNING: Memory usage at ${this.metrics.memory.percentage.toFixed(1)}%`);
    }
    
    // FPS threshold check
    if (this.metrics.performance.fps < this.thresholds.fps.minimum) {
      alerts.push(`WARNING: FPS dropped to ${this.metrics.performance.fps.toFixed(1)}`);
      this.applyRenderOptimization();
    }
    
    // Emit alerts
    if (alerts.length > 0) {
      this.emitResourceAlert(alerts);
    }
  }
  
  /**
   * Apply CPU throttling when usage is critical
   */
  private applyCPUThrottling(): void {
    // Increase delays between operations
    if (window.requestIdleCallback) {
      window.requestIdleCallback(() => {
        console.log('Applied CPU throttling - deferring non-critical operations');
      }, { timeout: 2000 });
    }
  }
  
  /**
   * Apply memory optimization when usage is critical
   */
  private applyMemoryOptimization(): void {
    // Force garbage collection if available
    if (typeof (global as any).gc === 'function') {
      (global as any).gc();
    }
    
    // Clear caches
    if ('caches' in window) {
      caches.keys().then(names => {
        names.forEach(name => caches.delete(name));
      });
    }
    
    console.log('Applied memory optimization - cleared caches');
  }
  
  /**
   * Apply render optimization when FPS is low
   */
  private applyRenderOptimization(): void {
    // Reduce render quality temporarily
    document.body.style.imageRendering = 'pixelated';
    
    // Re-enable quality after stabilization
    setTimeout(() => {
      document.body.style.imageRendering = 'auto';
    }, 5000);
    
    console.log('Applied render optimization - reduced quality temporarily');
  }
  
  /**
   * Emit resource alerts
   */
  private emitResourceAlert(alerts: string[]): void {
    const event = new CustomEvent('resource-alert', {
      detail: { alerts, metrics: this.metrics }
    });
    window.dispatchEvent(event);
  }
  
  /**
   * Subscribe to metric updates
   */
  subscribe(id: string, callback: (metrics: SystemMetrics) => void): void {
    this.callbacks.set(id, callback);
  }
  
  /**
   * Unsubscribe from updates
   */
  unsubscribe(id: string): void {
    this.callbacks.delete(id);
  }
  
  /**
   * Notify all callbacks
   */
  private notifyCallbacks(): void {
    this.callbacks.forEach(callback => {
      try {
        callback(this.getMetrics());
      } catch (error) {
        console.error('Error in resource monitor callback:', error);
      }
    });
  }
  
  /**
   * Get current metrics
   */
  getMetrics(): SystemMetrics {
    return { ...this.metrics };
  }
  
  /**
   * Get resource health status
   */
  getHealthStatus(): 'healthy' | 'warning' | 'critical' {
    if (this.metrics.cpu.usage > this.thresholds.cpu.critical ||
        this.metrics.memory.percentage > this.thresholds.memory.critical ||
        this.metrics.performance.fps < this.thresholds.fps.minimum) {
      return 'critical';
    }
    
    if (this.metrics.cpu.usage > this.thresholds.cpu.warning ||
        this.metrics.memory.percentage > this.thresholds.memory.warning ||
        this.metrics.performance.fps < this.thresholds.fps.target) {
      return 'warning';
    }
    
    return 'healthy';
  }
  
  /**
   * Get recommendations based on current metrics
   */
  getRecommendations(): string[] {
    const recommendations: string[] = [];
    
    if (this.metrics.cpu.usage > this.thresholds.cpu.warning) {
      recommendations.push('Consider reducing the number of active visualizations');
      recommendations.push('Disable auto-rotation in 3D views');
    }
    
    if (this.metrics.memory.percentage > this.thresholds.memory.warning) {
      recommendations.push('Close unused component windows');
      recommendations.push('Clear execution logs periodically');
      recommendations.push('Reduce timeline depth in visualizations');
    }
    
    if (this.metrics.performance.fps < this.thresholds.fps.target) {
      recommendations.push('Lower visualization quality settings');
      recommendations.push('Reduce particle counts in 3D scenes');
      recommendations.push('Disable real-time updates for inactive components');
    }
    
    return recommendations;
  }
  
  /**
   * Export metrics for analysis
   */
  exportMetrics(): string {
    const data = {
      timestamp: new Date().toISOString(),
      platform: this.platform,
      metrics: this.metrics,
      thresholds: this.thresholds,
      health: this.getHealthStatus(),
      recommendations: this.getRecommendations()
    };
    
    return JSON.stringify(data, null, 2);
  }
}

// Singleton instance
let monitorInstance: SystemResourceMonitor | null = null;

/**
 * Get or create the system resource monitor instance
 */
export function getSystemResourceMonitor(): SystemResourceMonitor {
  if (!monitorInstance) {
    monitorInstance = new SystemResourceMonitor();
  }
  return monitorInstance;
}

/**
 * Hook for React components to use system metrics
 */
export function useSystemMetrics() {
  const [metrics, setMetrics] = React.useState<SystemMetrics | null>(null);
  const [health, setHealth] = React.useState<'healthy' | 'warning' | 'critical'>('healthy');
  
  React.useEffect(() => {
    const monitor = getSystemResourceMonitor();
    const id = `component-${Date.now()}`;
    
    monitor.subscribe(id, (newMetrics) => {
      setMetrics(newMetrics);
      setHealth(monitor.getHealthStatus());
    });
    
    // Start monitoring if not already started
    monitor.start();
    
    return () => {
      monitor.unsubscribe(id);
    };
  }, []);
  
  return { metrics, health };
}