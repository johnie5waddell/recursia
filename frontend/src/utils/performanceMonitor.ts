/**
 * Performance Monitor Utility
 * 
 * Provides real-time performance metrics and memory usage tracking
 * to help identify performance bottlenecks and memory leaks.
 */

import { useEffect, useRef, useCallback } from 'react';

export interface PerformanceMetrics {
  fps: number;
  memoryUsedMB: number;
  memoryLimitMB: number;
  renderTime: number;
  updateTime: number;
  activeComponents: number;
  warnings: string[];
}

export class PerformanceMonitor {
  private frameCount = 0;
  private lastTime = performance.now();
  private fps = 0;
  private renderTimes: number[] = [];
  private updateTimes: number[] = [];
  private sampleSize = 60; // Keep last 60 samples
  
  /**
   * Start monitoring a frame
   */
  startFrame(): void {
    this.frameCount++;
    const now = performance.now();
    const delta = now - this.lastTime;
    
    if (delta >= 1000) {
      this.fps = Math.round((this.frameCount * 1000) / delta);
      this.frameCount = 0;
      this.lastTime = now;
    }
  }
  
  /**
   * Record render time
   */
  recordRenderTime(time: number): void {
    this.renderTimes.push(time);
    if (this.renderTimes.length > this.sampleSize) {
      this.renderTimes.shift();
    }
  }
  
  /**
   * Record update time
   */
  recordUpdateTime(time: number): void {
    this.updateTimes.push(time);
    if (this.updateTimes.length > this.sampleSize) {
      this.updateTimes.shift();
    }
  }
  
  /**
   * Get current performance metrics
   */
  getMetrics(): PerformanceMetrics {
    const warnings: string[] = [];
    
    // Calculate averages
    const avgRenderTime = this.renderTimes.length > 0
      ? this.renderTimes.reduce((a, b) => a + b, 0) / this.renderTimes.length
      : 0;
      
    const avgUpdateTime = this.updateTimes.length > 0
      ? this.updateTimes.reduce((a, b) => a + b, 0) / this.updateTimes.length
      : 0;
    
    // Memory info (if available)
    let memoryUsedMB = 0;
    let memoryLimitMB = 0;
    
    if ('memory' in performance && (performance as any).memory) {
      const memory = (performance as any).memory;
      memoryUsedMB = Math.round(memory.usedJSHeapSize / 1048576);
      memoryLimitMB = Math.round(memory.jsHeapSizeLimit / 1048576);
      
      // Memory warnings
      const memoryUsagePercent = (memoryUsedMB / memoryLimitMB) * 100;
      if (memoryUsagePercent > 90) {
        warnings.push(`Critical memory usage: ${memoryUsagePercent.toFixed(1)}%`);
      } else if (memoryUsagePercent > 75) {
        warnings.push(`High memory usage: ${memoryUsagePercent.toFixed(1)}%`);
      }
    }
    
    // Performance warnings
    if (this.fps < 30 && this.fps > 0) {
      warnings.push(`Low FPS: ${this.fps}`);
    }
    
    if (avgRenderTime > 16.67) {
      warnings.push(`Slow render: ${avgRenderTime.toFixed(1)}ms`);
    }
    
    if (avgUpdateTime > 8) {
      warnings.push(`Slow updates: ${avgUpdateTime.toFixed(1)}ms`);
    }
    
    return {
      fps: this.fps,
      memoryUsedMB,
      memoryLimitMB,
      renderTime: avgRenderTime,
      updateTime: avgUpdateTime,
      activeComponents: 0, // To be set by consumer
      warnings
    };
  }
  
  /**
   * Create a performance timing wrapper
   */
  static time<T>(label: string, fn: () => T): T {
    const start = performance.now();
    const result = fn();
    const duration = performance.now() - start;
    
    if (duration > 16.67) {
      console.warn(`[Performance] ${label} took ${duration.toFixed(2)}ms`);
    }
    
    return result;
  }
  
  /**
   * Create an async performance timing wrapper
   */
  static async timeAsync<T>(label: string, fn: () => Promise<T>): Promise<T> {
    const start = performance.now();
    const result = await fn();
    const duration = performance.now() - start;
    
    if (duration > 16.67) {
      console.warn(`[Performance] ${label} took ${duration.toFixed(2)}ms`);
    }
    
    return result;
  }
}

// Singleton instance
export const performanceMonitor = new PerformanceMonitor();

/**
 * React hook for performance monitoring
 */
export function usePerformanceMonitor(componentName: string) {
  const renderStart = useRef<number>();
  
  useEffect(() => {
    renderStart.current = performance.now();
    
    return () => {
      if (renderStart.current) {
        const renderTime = performance.now() - renderStart.current;
        performanceMonitor.recordRenderTime(renderTime);
        
        if (renderTime > 50) {
          console.warn(`[Performance] ${componentName} render took ${renderTime.toFixed(2)}ms`);
        }
      }
    };
  });
  
  const timeUpdate = useCallback((updateName: string, fn: () => void) => {
    const start = performance.now();
    fn();
    const duration = performance.now() - start;
    performanceMonitor.recordUpdateTime(duration);
    
    if (duration > 16.67) {
      console.warn(`[Performance] ${componentName}.${updateName} took ${duration.toFixed(2)}ms`);
    }
  }, [componentName]);
  
  return {
    timeUpdate,
    metrics: performanceMonitor.getMetrics()
  };
}

// For use in the browser console
if (typeof window !== 'undefined') {
  (window as any).performanceMonitor = performanceMonitor;
}