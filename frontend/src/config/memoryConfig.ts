/**
 * Memory Configuration
 * Centralized configuration for memory-sensitive parameters
 */

/**
 * Get memory-safe grid size based on available memory
 */
export function getMemorySafeGridSize(): number {
  // CRITICAL: Keep grid sizes very small to prevent freezing
  // Even 12^3 = 1,728 cells is causing issues
  
  // Check if we have access to memory info
  if ('memory' in performance && (performance as any).memory) {
    const memory = (performance as any).memory;
    const availableBytes = memory.jsHeapSizeLimit - memory.usedJSHeapSize;
    const availableMB = availableBytes / (1024 * 1024);
    
    console.log(`Available memory: ${availableMB.toFixed(0)}MB`);
    
    // Ultra-conservative grid sizes for stability
    if (availableMB < 1000) {
      console.warn('Low memory detected, using minimal grid size');
      return 3; // 3^3 = 27 cells (ultra minimal)
    } else if (availableMB < 2000) {
      return 4; // 4^3 = 64 cells
    } else if (availableMB < 3000) {
      return 5; // 5^3 = 125 cells
    } else if (availableMB < 4000) {
      return 6; // 6^3 = 216 cells
    } else {
      // Maximum grid size for stability
      return 8; // 8^3 = 512 cells (absolute max)
    }
  }
  
  // Default to ultra-safe value
  console.log('Memory API not available, using minimal grid size');
  return 4; // 4^3 = 64 cells
}

/**
 * Memory configuration constants
 */
export const MEMORY_CONFIG = {
  // Grid sizes for different memory profiles
  MINIMAL_GRID_SIZE: 4,
  SAFE_GRID_SIZE: 6,
  DEFAULT_GRID_SIZE: 8,
  PERFORMANCE_GRID_SIZE: 12,
  
  // Worker pool sizes
  MIN_WORKERS: 1,
  DEFAULT_WORKERS: 2,
  MAX_WORKERS: 4,
  
  // Buffer sizes
  HISTORY_BUFFER_SIZE: 100,
  SNAPSHOT_LIMIT: 10,
  
  // Update rates
  SAFE_UPDATE_INTERVAL: 50, // ms
  PERFORMANCE_UPDATE_INTERVAL: 16, // ms
  
  // Memory thresholds
  CRITICAL_MEMORY_THRESHOLD: 0.9, // 90% usage
  WARNING_MEMORY_THRESHOLD: 0.7, // 70% usage
  
  // Cleanup intervals
  GC_INTERVAL: 30000, // 30 seconds
  CACHE_CLEANUP_INTERVAL: 60000 // 60 seconds
};

/**
 * Get optimal worker count based on cores and memory
 */
export function getOptimalWorkerCount(): number {
  const cores = navigator.hardwareConcurrency || 4;
  const gridSize = getMemorySafeGridSize();
  
  // Reduce workers if using larger grids
  if (gridSize >= 20) {
    return Math.min(cores - 1, 2); // Leave one core free
  } else if (gridSize >= 16) {
    return Math.min(cores - 1, 3);
  } else {
    return Math.min(cores, MEMORY_CONFIG.MAX_WORKERS);
  }
}

/**
 * Memory monitoring utilities
 */
export class MemoryMonitor {
  private static instance: MemoryMonitor;
  private callbacks: Array<(usage: number) => void> = [];
  private interval?: number;
  
  static getInstance(): MemoryMonitor {
    if (!MemoryMonitor.instance) {
      MemoryMonitor.instance = new MemoryMonitor();
    }
    return MemoryMonitor.instance;
  }
  
  start(): void {
    if (this.interval) return;
    
    this.interval = window.setInterval(() => {
      const usage = this.getMemoryUsage();
      this.callbacks.forEach(cb => cb(usage));
      
      if (usage > MEMORY_CONFIG.CRITICAL_MEMORY_THRESHOLD) {
        console.error('Critical memory usage detected:', (usage * 100).toFixed(1) + '%');
        this.triggerEmergencyCleanup();
      } else if (usage > MEMORY_CONFIG.WARNING_MEMORY_THRESHOLD) {
        console.warn('High memory usage:', (usage * 100).toFixed(1) + '%');
      }
    }, 5000);
  }
  
  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = undefined;
    }
  }
  
  onMemoryChange(callback: (usage: number) => void): () => void {
    this.callbacks.push(callback);
    return () => {
      this.callbacks = this.callbacks.filter(cb => cb !== callback);
    };
  }
  
  getMemoryUsage(): number {
    if ('memory' in performance && (performance as any).memory) {
      const memory = (performance as any).memory;
      return memory.usedJSHeapSize / memory.jsHeapSizeLimit;
    }
    return 0.5; // Default to 50% if unknown
  }
  
  private triggerEmergencyCleanup(): void {
    // Force garbage collection if available
    if (typeof (globalThis as any).gc === 'function') {
      (globalThis as any).gc();
    }
    
    // Dispatch custom event for components to clean up
    window.dispatchEvent(new CustomEvent('memory-critical', {
      detail: { usage: this.getMemoryUsage() }
    }));
  }
}