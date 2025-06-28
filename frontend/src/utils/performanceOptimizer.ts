/**
 * Performance Optimizer
 * Manages render cycles, state updates, and component optimization
 */

import React from 'react';
import { getSystemResourceMonitor } from './systemResourceMonitor';
import { MemoryLeakDetector } from './memoryLeakDetector';

interface OptimizationConfig {
  enableThrottling: boolean;
  enableBatching: boolean;
  enableLazyLoading: boolean;
  enableMemoization: boolean;
  targetFPS: number;
  maxRenderTime: number;
}

interface ComponentMetrics {
  renderCount: number;
  averageRenderTime: number;
  lastRenderTime: number;
  props: any;
  state: any;
}

export class PerformanceOptimizer {
  private config: OptimizationConfig;
  private componentMetrics: Map<string, ComponentMetrics> = new Map();
  private updateQueue: Map<string, any> = new Map();
  private batchTimeout?: NodeJS.Timeout;
  private frameScheduler: FrameScheduler;
  private resourceMonitor = getSystemResourceMonitor();
  private leakDetector = MemoryLeakDetector.getInstance();
  
  constructor(config?: Partial<OptimizationConfig>) {
    this.config = {
      enableThrottling: config?.enableThrottling ?? true,
      enableBatching: config?.enableBatching ?? true,
      enableLazyLoading: config?.enableLazyLoading ?? true,
      enableMemoization: config?.enableMemoization ?? true,
      targetFPS: config?.targetFPS ?? 60,
      maxRenderTime: config?.maxRenderTime ?? 16
    };
    
    this.frameScheduler = new FrameScheduler(this.config.targetFPS);
  }
  
  /**
   * Track component render
   */
  trackRender(componentName: string, renderTime: number, props?: any, state?: any): void {
    const existing = this.componentMetrics.get(componentName) || {
      renderCount: 0,
      averageRenderTime: 0,
      lastRenderTime: 0,
      props: {},
      state: {}
    };
    
    existing.renderCount++;
    existing.averageRenderTime = 
      (existing.averageRenderTime * (existing.renderCount - 1) + renderTime) / existing.renderCount;
    existing.lastRenderTime = renderTime;
    existing.props = props;
    existing.state = state;
    
    this.componentMetrics.set(componentName, existing);
    
    // Check for performance issues
    if (renderTime > this.config.maxRenderTime) {
      console.warn(`Slow render detected in ${componentName}: ${renderTime}ms`);
    }
  }
  
  /**
   * Batch state updates
   */
  batchUpdate(componentId: string, update: any): Promise<void> {
    if (!this.config.enableBatching) {
      return Promise.resolve(update());
    }
    
    return new Promise((resolve) => {
      this.updateQueue.set(componentId, { update, resolve });
      
      if (this.batchTimeout) {
        clearTimeout(this.batchTimeout);
      }
      
      this.batchTimeout = setTimeout(() => {
        this.processBatch();
      }, 0);
    });
  }
  
  /**
   * Process batched updates
   */
  private processBatch(): void {
    const updates = Array.from(this.updateQueue.values());
    this.updateQueue.clear();
    
    // Execute all updates in a single frame
    this.frameScheduler.schedule(() => {
      for (const { update, resolve } of updates) {
        try {
          update();
          resolve();
        } catch (error) {
          console.error('Batch update error:', error);
        }
      }
    });
  }
  
  /**
   * Throttle function execution
   */
  throttle<T extends (...args: any[]) => any>(
    fn: T,
    delay: number
  ): (...args: Parameters<T>) => void {
    if (!this.config.enableThrottling) {
      return fn;
    }
    
    let lastCall = 0;
    let timeout: NodeJS.Timeout | null = null;
    
    return (...args: Parameters<T>) => {
      const now = Date.now();
      
      if (now - lastCall >= delay) {
        lastCall = now;
        fn(...args);
      } else if (!timeout) {
        timeout = setTimeout(() => {
          lastCall = Date.now();
          fn(...args);
          timeout = null;
        }, delay - (now - lastCall));
      }
    };
  }
  
  /**
   * Debounce function execution
   */
  debounce<T extends (...args: any[]) => any>(
    fn: T,
    delay: number
  ): (...args: Parameters<T>) => void {
    let timeout: NodeJS.Timeout | null = null;
    
    return (...args: Parameters<T>) => {
      if (timeout) {
        clearTimeout(timeout);
      }
      
      timeout = setTimeout(() => {
        fn(...args);
        timeout = null;
      }, delay);
    };
  }
  
  /**
   * Create memoized selector
   */
  createSelector<T, R>(
    dependencies: ((state: T) => any)[],
    selector: (...args: any[]) => R
  ): (state: T) => R {
    if (!this.config.enableMemoization) {
      return (state: T) => {
        const deps = dependencies.map(dep => dep(state));
        return selector(...deps);
      };
    }
    
    let lastDeps: any[] = [];
    let lastResult: R;
    let initialized = false;
    
    return (state: T) => {
      const currentDeps = dependencies.map(dep => dep(state));
      
      if (!initialized || !this.shallowEqual(lastDeps, currentDeps)) {
        lastDeps = currentDeps;
        lastResult = selector(...currentDeps);
        initialized = true;
      }
      
      return lastResult;
    };
  }
  
  /**
   * Shallow equality check
   */
  private shallowEqual(a: any[], b: any[]): boolean {
    if (a.length !== b.length) return false;
    
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false;
    }
    
    return true;
  }
  
  /**
   * Get optimization recommendations
   */
  getRecommendations(): string[] {
    const recommendations: string[] = [];
    const metrics = this.resourceMonitor.getMetrics();
    
    // Check component metrics
    for (const [name, data] of this.componentMetrics) {
      if (data.renderCount > 100) {
        recommendations.push(`Component "${name}" has rendered ${data.renderCount} times - consider memoization`);
      }
      
      if (data.averageRenderTime > this.config.maxRenderTime) {
        recommendations.push(`Component "${name}" has slow average render time: ${data.averageRenderTime.toFixed(2)}ms`);
      }
    }
    
    // Check resource usage
    if (metrics.cpu.usage > 70) {
      recommendations.push('High CPU usage - consider throttling animations and updates');
    }
    
    if (metrics.memory.percentage > 80) {
      recommendations.push('High memory usage - consider lazy loading and virtualization');
    }
    
    // Check for memory leaks
    const leakSummary = this.leakDetector.getSummary();
    if (leakSummary.totalLeaks > 0) {
      recommendations.push(`${leakSummary.totalLeaks} memory leaks detected - run cleanup`);
    }
    
    return recommendations;
  }
  
  /**
   * Apply automatic optimizations
   */
  applyOptimizations(): void {
    const metrics = this.resourceMonitor.getMetrics();
    
    // Adjust frame rate based on performance
    if (metrics.performance.fps < 30) {
      this.frameScheduler.setTargetFPS(30);
      console.log('Reduced target FPS to 30 for better performance');
    }
    
    // Enable aggressive throttling under high load
    if (metrics.cpu.usage > 80) {
      this.config.enableThrottling = true;
      this.config.maxRenderTime = 32; // Allow longer renders
    }
    
    // Clean up memory if needed
    if (metrics.memory.percentage > 90) {
      this.leakDetector.cleanup();
      
      // Clear component metrics for GC'd components
      const activeComponents = new Set<string>();
      document.querySelectorAll('[data-component-name]').forEach(el => {
        activeComponents.add(el.getAttribute('data-component-name')!);
      });
      
      for (const name of this.componentMetrics.keys()) {
        if (!activeComponents.has(name)) {
          this.componentMetrics.delete(name);
        }
      }
    }
  }
  
  /**
   * Get performance report
   */
  getReport(): {
    slowComponents: Array<{ name: string; avgTime: number; count: number }>;
    memoryLeaks: number;
    recommendations: string[];
    metrics: any;
  } {
    const slowComponents = Array.from(this.componentMetrics.entries())
      .filter(([_, data]) => data.averageRenderTime > this.config.maxRenderTime)
      .map(([name, data]) => ({
        name,
        avgTime: data.averageRenderTime,
        count: data.renderCount
      }))
      .sort((a, b) => b.avgTime - a.avgTime);
    
    return {
      slowComponents,
      memoryLeaks: this.leakDetector.getSummary().totalLeaks,
      recommendations: this.getRecommendations(),
      metrics: this.resourceMonitor.getMetrics()
    };
  }
}

/**
 * Frame Scheduler for smooth animations
 */
class FrameScheduler {
  private targetFPS: number;
  private frameTime: number;
  private lastFrameTime: number = 0;
  private tasks: (() => void)[] = [];
  private rafId?: number;
  
  constructor(targetFPS: number = 60) {
    this.targetFPS = targetFPS;
    this.frameTime = 1000 / targetFPS;
  }
  
  setTargetFPS(fps: number): void {
    this.targetFPS = fps;
    this.frameTime = 1000 / fps;
  }
  
  schedule(task: () => void): void {
    this.tasks.push(task);
    
    if (!this.rafId) {
      this.rafId = requestAnimationFrame(this.processTasks.bind(this));
    }
  }
  
  private processTasks(timestamp: number): void {
    const deltaTime = timestamp - this.lastFrameTime;
    
    if (deltaTime >= this.frameTime) {
      const tasksToRun = this.tasks.splice(0);
      this.lastFrameTime = timestamp;
      
      for (const task of tasksToRun) {
        try {
          task();
        } catch (error) {
          console.error('Frame task error:', error);
        }
      }
    }
    
    if (this.tasks.length > 0) {
      this.rafId = requestAnimationFrame(this.processTasks.bind(this));
    } else {
      this.rafId = undefined;
    }
  }
}

// Global optimizer instance
let optimizerInstance: PerformanceOptimizer | null = null;

export function getPerformanceOptimizer(): PerformanceOptimizer {
  if (!optimizerInstance) {
    optimizerInstance = new PerformanceOptimizer();
  }
  return optimizerInstance;
}

/**
 * React performance hook
 */
export function usePerformanceOptimization(componentName: string) {
  const optimizer = getPerformanceOptimizer();
  const renderStartRef = React.useRef<number>(0);
  
  // Track render start
  renderStartRef.current = performance.now();
  
  React.useEffect(() => {
    // Track render completion
    const renderTime = performance.now() - renderStartRef.current;
    optimizer.trackRender(componentName, renderTime);
    
    // Track memory leaks
    const detector = MemoryLeakDetector.getInstance();
    detector.trackObject({ componentName }, 'react-component', componentName);
    
    return () => {
      // Cleanup on unmount
      detector.cleanup('object-reference');
    };
  });
  
  return {
    batchUpdate: (update: () => void) => optimizer.batchUpdate(componentName, update),
    throttle: optimizer.throttle.bind(optimizer),
    debounce: optimizer.debounce.bind(optimizer)
  };
}

/**
 * HOC for automatic performance optimization
 */
export function withPerformanceOptimization<P extends object>(
  Component: React.ComponentType<P>,
  componentName: string
): React.ComponentType<P> {
  const MemoizedComponent = React.memo((props: P) => {
    const { batchUpdate } = usePerformanceOptimization(componentName);
    
    // Add performance props
    const enhancedProps = {
      ...props,
      batchUpdate,
      'data-component-name': componentName
    };
    
    return React.createElement(Component, enhancedProps);
  });
  
  return MemoizedComponent as React.ComponentType<P>;
}