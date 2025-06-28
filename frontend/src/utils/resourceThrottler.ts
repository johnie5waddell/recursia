/**
 * Resource Throttler
 * Manages and limits resource usage across the application
 */

import React from 'react';
import { getSystemResourceMonitor } from './systemResourceMonitor';
import { getPerformanceOptimizer } from './performanceOptimizer';

interface ThrottleRule {
  resource: 'cpu' | 'memory' | 'network' | 'render';
  threshold: number;
  action: 'delay' | 'skip' | 'queue' | 'reduce';
  priority: number;
}

interface ThrottledTask {
  id: string;
  execute: () => Promise<any>;
  resource: string;
  priority: number;
  size: number;
  timestamp: number;
}

export class ResourceThrottler {
  private rules: ThrottleRule[] = [];
  private taskQueue: ThrottledTask[] = [];
  private activeResources: Map<string, number> = new Map();
  private resourceMonitor = getSystemResourceMonitor();
  private performanceOptimizer = getPerformanceOptimizer();
  private processingInterval?: NodeJS.Timeout;
  
  constructor() {
    this.setupDefaultRules();
    this.startProcessing();
  }
  
  /**
   * Setup default throttling rules
   */
  private setupDefaultRules(): void {
    this.addRule({
      resource: 'cpu',
      threshold: 80,
      action: 'delay',
      priority: 1
    });
    
    this.addRule({
      resource: 'memory',
      threshold: 85,
      action: 'queue',
      priority: 2
    });
    
    this.addRule({
      resource: 'render',
      threshold: 30, // FPS threshold
      action: 'reduce',
      priority: 3
    });
  }
  
  /**
   * Add throttling rule
   */
  addRule(rule: ThrottleRule): void {
    this.rules.push(rule);
    this.rules.sort((a, b) => b.priority - a.priority);
  }
  
  /**
   * Execute task with throttling
   */
  async execute<T>(
    task: () => Promise<T>,
    options: {
      resource: string;
      priority?: number;
      size?: number;
    }
  ): Promise<T> {
    const throttledTask: ThrottledTask = {
      id: `task-${Date.now()}-${Math.random()}`,
      execute: task,
      resource: options.resource,
      priority: options.priority ?? 0,
      size: options.size ?? 1,
      timestamp: Date.now()
    };
    
    // Check if should throttle
    const shouldThrottle = this.checkThrottling(throttledTask);
    
    if (shouldThrottle.throttle) {
      switch (shouldThrottle.action) {
        case 'delay':
          await this.delay(shouldThrottle.delay || 100);
          break;
        case 'queue':
          return this.queueTask(throttledTask);
        case 'skip':
          throw new Error('Task skipped due to resource constraints');
        case 'reduce':
          // Reduce quality/complexity before execution
          await this.reduceTaskComplexity(throttledTask);
          break;
      }
    }
    
    // Track resource usage
    this.trackResourceUsage(throttledTask.resource, throttledTask.size);
    
    try {
      return await task();
    } finally {
      this.releaseResource(throttledTask.resource, throttledTask.size);
    }
  }
  
  /**
   * Check if task should be throttled
   */
  private checkThrottling(task: ThrottledTask): {
    throttle: boolean;
    action?: 'delay' | 'skip' | 'queue' | 'reduce';
    delay?: number;
  } {
    const metrics = this.resourceMonitor.getMetrics();
    
    for (const rule of this.rules) {
      let currentUsage = 0;
      
      switch (rule.resource) {
        case 'cpu':
          currentUsage = metrics.cpu.usage;
          break;
        case 'memory':
          currentUsage = metrics.memory.percentage;
          break;
        case 'render':
          currentUsage = 100 - metrics.performance.fps; // Inverse for threshold comparison
          break;
      }
      
      if (currentUsage > rule.threshold) {
        return {
          throttle: true,
          action: rule.action,
          delay: this.calculateDelay(currentUsage, rule.threshold)
        };
      }
    }
    
    return { throttle: false };
  }
  
  /**
   * Calculate delay based on resource usage
   */
  private calculateDelay(current: number, threshold: number): number {
    const excess = current - threshold;
    return Math.min(1000, excess * 10); // Max 1 second delay
  }
  
  /**
   * Delay execution
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  /**
   * Queue task for later execution
   */
  private async queueTask<T>(task: ThrottledTask): Promise<T> {
    return new Promise((resolve, reject) => {
      const wrappedTask = {
        ...task,
        execute: async () => {
          try {
            const result = await task.execute();
            resolve(result);
            return result;
          } catch (error) {
            reject(error);
            throw error;
          }
        }
      };
      
      this.taskQueue.push(wrappedTask);
      this.taskQueue.sort((a, b) => b.priority - a.priority);
    });
  }
  
  /**
   * Reduce task complexity
   */
  private async reduceTaskComplexity(task: ThrottledTask): Promise<void> {
    // Apply quality reductions based on resource type
    switch (task.resource) {
      case 'render':
        // Reduce render quality
        document.body.style.imageRendering = 'pixelated';
        document.body.classList.add('reduced-quality');
        
        // Re-enable after task
        setTimeout(() => {
          document.body.style.imageRendering = 'auto';
          document.body.classList.remove('reduced-quality');
        }, 2000);
        break;
        
      case 'compute':
        // Reduce computation precision
        task.size = task.size * 0.5;
        break;
    }
  }
  
  /**
   * Track resource usage
   */
  private trackResourceUsage(resource: string, size: number): void {
    const current = this.activeResources.get(resource) || 0;
    this.activeResources.set(resource, current + size);
  }
  
  /**
   * Release resource
   */
  private releaseResource(resource: string, size: number): void {
    const current = this.activeResources.get(resource) || 0;
    this.activeResources.set(resource, Math.max(0, current - size));
  }
  
  /**
   * Start processing queued tasks
   */
  private startProcessing(): void {
    this.processingInterval = setInterval(() => {
      this.processQueue();
    }, 100);
  }
  
  /**
   * Process task queue
   */
  private async processQueue(): Promise<void> {
    if (this.taskQueue.length === 0) return;
    
    const metrics = this.resourceMonitor.getMetrics();
    
    // Check if resources are available
    if (metrics.cpu.usage < 70 && metrics.memory.percentage < 80) {
      const task = this.taskQueue.shift();
      if (task) {
        this.trackResourceUsage(task.resource, task.size);
        
        try {
          await task.execute();
        } catch (error) {
          console.error('Queued task error:', error);
        } finally {
          this.releaseResource(task.resource, task.size);
        }
      }
    }
  }
  
  /**
   * Get throttling statistics
   */
  getStats(): {
    queueLength: number;
    activeResources: Record<string, number>;
    rules: ThrottleRule[];
  } {
    return {
      queueLength: this.taskQueue.length,
      activeResources: Object.fromEntries(this.activeResources),
      rules: [...this.rules]
    };
  }
  
  /**
   * Clear task queue
   */
  clearQueue(): void {
    this.taskQueue = [];
  }
  
  /**
   * Stop throttler
   */
  stop(): void {
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
    }
    this.clearQueue();
  }
}

// Singleton instance
let throttlerInstance: ResourceThrottler | null = null;

export function getResourceThrottler(): ResourceThrottler {
  if (!throttlerInstance) {
    throttlerInstance = new ResourceThrottler();
  }
  return throttlerInstance;
}

/**
 * React hook for resource throttling
 */
export function useResourceThrottling() {
  const throttler = getResourceThrottler();
  
  const throttledExecute = React.useCallback(
    async <T,>(
      task: () => Promise<T>,
      resource: string = 'general',
      priority: number = 0
    ): Promise<T> => {
      return throttler.execute(task, { resource, priority });
    },
    [throttler]
  );
  
  return {
    execute: throttledExecute,
    getStats: () => throttler.getStats()
  };
}

/**
 * Decorator for automatic throttling
 */
export function Throttled(resource: string = 'general', priority: number = 0) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function (...args: any[]) {
      const throttler = getResourceThrottler();
      return throttler.execute(
        () => originalMethod.apply(this, args),
        { resource, priority }
      );
    };
    
    return descriptor;
  };
}