/**
 * Enterprise-Grade Memory Manager
 * Comprehensive memory management solution with automatic cleanup,
 * resource tracking, and performance optimization
 */

import * as React from 'react';
import { EventEmitter } from './EventEmitter';

/**
 * Resource types that can be tracked and managed
 */
export enum ResourceType {
  COMPONENT = 'component',
  WORKER = 'worker',
  CANVAS = 'canvas',
  WEBGL = 'webgl',
  TIMER = 'timer',
  SUBSCRIPTION = 'subscription',
  EVENT_LISTENER = 'event_listener',
  DOM_NODE = 'dom_node',
  OBSERVABLE = 'observable',
  ANIMATION_FRAME = 'animation_frame',
  WEBSOCKET = 'websocket',
  MEDIA_STREAM = 'media_stream'
}

/**
 * Memory usage thresholds for triggering cleanup
 */
interface MemoryThresholds {
  warning: number;  // MB
  critical: number; // MB
  maxResources: number;
}

/**
 * Tracked resource interface
 */
interface TrackedResource {
  id: string;
  type: ResourceType;
  component?: string;
  description?: string;
  size: number;
  created: number;
  lastAccessed: number;
  cleanup: () => void;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

/**
 * Memory statistics interface
 */
export interface MemoryStats {
  total: number;
  used: number;
  available: number;
  percentUsed: number;
  resourceCount: number;
  resourcesByType: Record<ResourceType, number>;
  lastGC: number;
  gcCount: number;
}

/**
 * Cleanup strategy interface
 */
interface CleanupStrategy {
  shouldCleanup(resource: TrackedResource, stats: MemoryStats): boolean;
  getPriority(resource: TrackedResource): number;
}

/**
 * Default cleanup strategies
 */
const CLEANUP_STRATEGIES: Record<string, CleanupStrategy> = {
  /**
   * Age-based cleanup - remove old resources
   */
  age: {
    shouldCleanup: (resource, stats) => {
      const age = Date.now() - resource.created;
      const ageThreshold = resource.priority === 'critical' ? 600000 : 300000; // 10min or 5min
      return age > ageThreshold && stats.percentUsed > 70;
    },
    getPriority: (resource) => {
      const age = Date.now() - resource.created;
      return age / 60000; // Priority increases with age in minutes
    }
  },

  /**
   * Access-based cleanup - remove least recently used
   */
  lru: {
    shouldCleanup: (resource, stats) => {
      const idleTime = Date.now() - resource.lastAccessed;
      const idleThreshold = resource.priority === 'critical' ? 300000 : 120000; // 5min or 2min
      return idleTime > idleThreshold && stats.percentUsed > 60;
    },
    getPriority: (resource) => {
      const idleTime = Date.now() - resource.lastAccessed;
      return idleTime / 60000; // Priority increases with idle time
    }
  },

  /**
   * Size-based cleanup - remove large resources first
   */
  size: {
    shouldCleanup: (resource, stats) => {
      return resource.size > 1048576 && stats.percentUsed > 80; // 1MB
    },
    getPriority: (resource) => {
      return resource.size / 1048576; // Priority based on MB
    }
  },

  /**
   * Type-based cleanup - prioritize certain resource types
   */
  type: {
    shouldCleanup: (resource, stats) => {
      const lowPriorityTypes = [ResourceType.DOM_NODE, ResourceType.EVENT_LISTENER];
      return lowPriorityTypes.includes(resource.type) && stats.percentUsed > 75;
    },
    getPriority: (resource) => {
      const priorities: Record<ResourceType, number> = {
        [ResourceType.DOM_NODE]: 10,
        [ResourceType.EVENT_LISTENER]: 9,
        [ResourceType.TIMER]: 8,
        [ResourceType.ANIMATION_FRAME]: 7,
        [ResourceType.SUBSCRIPTION]: 6,
        [ResourceType.OBSERVABLE]: 5,
        [ResourceType.CANVAS]: 4,
        [ResourceType.WEBGL]: 3,
        [ResourceType.WORKER]: 2,
        [ResourceType.WEBSOCKET]: 1,
        [ResourceType.MEDIA_STREAM]: 1,
        [ResourceType.COMPONENT]: 0
      };
      return priorities[resource.type] || 5;
    }
  }
};

/**
 * Enterprise-grade memory manager implementation
 */
export class MemoryManager extends EventEmitter {
  private resources: Map<string, TrackedResource> = new Map();
  private thresholds: MemoryThresholds;
  private strategies: CleanupStrategy[];
  private monitorInterval?: number;
  private gcInterval?: number;
  private stats: MemoryStats;
  private isMonitoring: boolean = false;
  private performanceObserver?: PerformanceObserver;
  private finalizationRegistry: FinalizationRegistry<string>;

  constructor(thresholds: Partial<MemoryThresholds> = {}) {
    super();
    
    this.thresholds = {
      warning: thresholds.warning || 512, // 512MB
      critical: thresholds.critical || 1024, // 1GB
      maxResources: thresholds.maxResources || 10000
    };

    this.strategies = [
      CLEANUP_STRATEGIES.lru,
      CLEANUP_STRATEGIES.age,
      CLEANUP_STRATEGIES.size,
      CLEANUP_STRATEGIES.type
    ];

    this.stats = {
      total: 0,
      used: 0,
      available: 0,
      percentUsed: 0,
      resourceCount: 0,
      resourcesByType: {} as Record<ResourceType, number>,
      lastGC: Date.now(),
      gcCount: 0
    };

    // Setup finalization registry for automatic cleanup
    this.finalizationRegistry = new FinalizationRegistry((resourceId: string) => {
      this.untrack(resourceId);
    });

    this.setupPerformanceMonitoring();
  }

  /**
   * Setup performance monitoring
   */
  private setupPerformanceMonitoring(): void {
    if ('PerformanceObserver' in window) {
      try {
        this.performanceObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.entryType === 'measure' && entry.name.startsWith('memory-')) {
              this.emit('performance', {
                name: entry.name,
                duration: entry.duration,
                startTime: entry.startTime
              });
            }
          }
        });
        this.performanceObserver.observe({ entryTypes: ['measure'] });
      } catch (e) {
        console.warn('Performance monitoring not available:', e);
      }
    }
  }

  /**
   * Track a resource for memory management
   */
  track(
    id: string,
    type: ResourceType,
    cleanup: () => void,
    options: {
      component?: string;
      description?: string;
      size?: number;
      priority?: TrackedResource['priority'];
      weakRef?: any;
    } = {}
  ): void {
    const resource: TrackedResource = {
      id,
      type,
      component: options.component,
      description: options.description,
      size: options.size || this.estimateSize(type),
      created: Date.now(),
      lastAccessed: Date.now(),
      cleanup,
      priority: options.priority || 'medium'
    };

    this.resources.set(id, resource);

    // Register for automatic cleanup if weak reference provided
    if (options.weakRef) {
      this.finalizationRegistry.register(options.weakRef, id);
    }

    // Update stats
    this.updateResourceStats();

    // Check if cleanup needed
    if (this.resources.size > this.thresholds.maxResources) {
      this.performCleanup('resource_limit');
    }

    this.emit('resource:tracked', { id, type, component: options.component });
  }

  /**
   * Untrack a resource
   */
  untrack(id: string): void {
    const resource = this.resources.get(id);
    if (resource) {
      this.resources.delete(id);
      this.updateResourceStats();
      this.emit('resource:untracked', { id, type: resource.type });
    }
  }

  /**
   * Access a resource (updates last accessed time)
   */
  access(id: string): void {
    const resource = this.resources.get(id);
    if (resource) {
      resource.lastAccessed = Date.now();
    }
  }

  /**
   * Clean up a specific resource
   */
  cleanup(id: string): boolean {
    const resource = this.resources.get(id);
    if (resource) {
      try {
        performance.mark(`memory-cleanup-start-${id}`);
        resource.cleanup();
        this.untrack(id);
        performance.mark(`memory-cleanup-end-${id}`);
        performance.measure(
          `memory-cleanup-${id}`,
          `memory-cleanup-start-${id}`,
          `memory-cleanup-end-${id}`
        );
        return true;
      } catch (error) {
        console.error(`Failed to cleanup resource ${id}:`, error);
        this.emit('error', { type: 'cleanup_failed', id, error });
        return false;
      }
    }
    return false;
  }

  /**
   * Perform automatic cleanup based on strategies
   */
  private performCleanup(reason: string): void {
    performance.mark('memory-gc-start');
    
    const stats = this.getStats();
    const resourcesToClean: TrackedResource[] = [];

    // Identify resources to clean
    for (const resource of this.resources.values()) {
      for (const strategy of this.strategies) {
        if (strategy.shouldCleanup(resource, stats)) {
          resourcesToClean.push(resource);
          break;
        }
      }
    }

    // Sort by cleanup priority
    resourcesToClean.sort((a, b) => {
      const priorityA = Math.max(...this.strategies.map(s => s.getPriority(a)));
      const priorityB = Math.max(...this.strategies.map(s => s.getPriority(b)));
      return priorityB - priorityA;
    });

    // Clean up resources
    let cleanedCount = 0;
    let freedMemory = 0;
    const targetCount = Math.min(
      resourcesToClean.length,
      Math.ceil(this.resources.size * 0.2) // Clean up to 20% of resources
    );

    for (let i = 0; i < targetCount; i++) {
      const resource = resourcesToClean[i];
      if (this.cleanup(resource.id)) {
        cleanedCount++;
        freedMemory += resource.size;
      }
    }

    this.stats.lastGC = Date.now();
    this.stats.gcCount++;

    performance.mark('memory-gc-end');
    performance.measure('memory-gc', 'memory-gc-start', 'memory-gc-end');

    this.emit('gc:completed', {
      reason,
      cleanedCount,
      freedMemory,
      duration: performance.getEntriesByName('memory-gc').pop()?.duration || 0
    });
  }

  /**
   * Start memory monitoring
   */
  startMonitoring(intervalMs: number = 30000): void {
    if (this.isMonitoring) return;

    this.isMonitoring = true;
    
    // Monitor memory usage
    this.monitorInterval = window.setInterval(() => {
      this.updateMemoryStats();
      this.checkMemoryPressure();
    }, intervalMs);

    // Periodic garbage collection
    this.gcInterval = window.setInterval(() => {
      const stats = this.getStats();
      if (stats.percentUsed > 60) {
        this.performCleanup('periodic');
      }
    }, intervalMs * 2);

    this.emit('monitoring:started');
  }

  /**
   * Stop memory monitoring
   */
  stopMonitoring(): void {
    if (!this.isMonitoring) return;

    this.isMonitoring = false;

    if (this.monitorInterval) {
      clearInterval(this.monitorInterval);
      this.monitorInterval = undefined;
    }

    if (this.gcInterval) {
      clearInterval(this.gcInterval);
      this.gcInterval = undefined;
    }

    this.emit('monitoring:stopped');
  }

  /**
   * Update memory statistics
   */
  private updateMemoryStats(): void {
    if ('memory' in performance && (performance as any).memory) {
      const memory = (performance as any).memory;
      this.stats.total = memory.jsHeapSizeLimit / 1048576; // Convert to MB
      this.stats.used = memory.usedJSHeapSize / 1048576;
      this.stats.available = this.stats.total - this.stats.used;
      this.stats.percentUsed = (this.stats.used / this.stats.total) * 100;
    } else {
      // Fallback estimation based on tracked resources
      this.stats.used = Array.from(this.resources.values())
        .reduce((sum, r) => sum + r.size, 0) / 1048576;
      this.stats.total = this.thresholds.critical;
      this.stats.available = this.stats.total - this.stats.used;
      this.stats.percentUsed = (this.stats.used / this.stats.total) * 100;
    }
  }

  /**
   * Update resource statistics
   */
  private updateResourceStats(): void {
    this.stats.resourceCount = this.resources.size;
    this.stats.resourcesByType = {} as Record<ResourceType, number>;

    for (const resource of this.resources.values()) {
      this.stats.resourcesByType[resource.type] = 
        (this.stats.resourcesByType[resource.type] || 0) + 1;
    }
  }

  /**
   * Check memory pressure and trigger cleanup if needed
   */
  private checkMemoryPressure(): void {
    const stats = this.getStats();

    if (stats.used > this.thresholds.critical) {
      this.emit('memory:critical', stats);
      this.performCleanup('critical_pressure');
    } else if (stats.used > this.thresholds.warning) {
      this.emit('memory:warning', stats);
      this.performCleanup('warning_pressure');
    }
  }

  /**
   * Estimate resource size based on type
   */
  private estimateSize(type: ResourceType): number {
    const estimates: Record<ResourceType, number> = {
      [ResourceType.COMPONENT]: 50000, // 50KB
      [ResourceType.WORKER]: 1048576, // 1MB
      [ResourceType.CANVAS]: 4194304, // 4MB
      [ResourceType.WEBGL]: 8388608, // 8MB
      [ResourceType.TIMER]: 1000, // 1KB
      [ResourceType.SUBSCRIPTION]: 5000, // 5KB
      [ResourceType.EVENT_LISTENER]: 1000, // 1KB
      [ResourceType.DOM_NODE]: 10000, // 10KB
      [ResourceType.OBSERVABLE]: 5000, // 5KB
      [ResourceType.ANIMATION_FRAME]: 1000, // 1KB
      [ResourceType.WEBSOCKET]: 100000, // 100KB
      [ResourceType.MEDIA_STREAM]: 10485760 // 10MB
    };

    return estimates[type] || 10000; // Default 10KB
  }

  /**
   * Get current memory statistics
   */
  getStats(): MemoryStats {
    this.updateMemoryStats();
    return { ...this.stats };
  }

  /**
   * Get detailed resource information
   */
  getResourceDetails(): Array<{
    id: string;
    type: ResourceType;
    component?: string;
    size: string;
    age: string;
    idle: string;
    priority: string;
  }> {
    const now = Date.now();
    return Array.from(this.resources.values()).map(resource => ({
      id: resource.id,
      type: resource.type,
      component: resource.component,
      size: `${(resource.size / 1024).toFixed(2)} KB`,
      age: `${Math.floor((now - resource.created) / 1000)} s`,
      idle: `${Math.floor((now - resource.lastAccessed) / 1000)} s`,
      priority: resource.priority
    }));
  }

  /**
   * Force garbage collection (if available)
   */
  forceGC(): void {
    this.performCleanup('manual');
    
    // Try browser GC if available
    if (typeof (globalThis as any).gc === 'function') {
      try {
        (globalThis as any).gc();
      } catch (e) {
        // Ignore
      }
    }
  }

  /**
   * Cleanup all resources
   */
  dispose(): void {
    this.stopMonitoring();
    
    // Clean up all resources
    for (const resource of this.resources.values()) {
      try {
        resource.cleanup();
      } catch (e) {
        console.error(`Failed to cleanup resource ${resource.id}:`, e);
      }
    }
    
    this.resources.clear();
    
    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }
    
    this.removeAllListeners();
  }
}

// Singleton instance
let memoryManager: MemoryManager | null = null;

/**
 * Get the global memory manager instance
 */
export function getMemoryManager(): MemoryManager {
  if (!memoryManager) {
    memoryManager = new MemoryManager();
    memoryManager.startMonitoring();
  }
  return memoryManager;
}

/**
 * React hook for memory management
 */
export function useMemoryManager(
  componentName: string,
  options: { priority?: TrackedResource['priority'] } = {}
): {
  track: (id: string, type: ResourceType, cleanup: () => void, opts?: any) => void;
  untrack: (id: string) => void;
  cleanup: (id: string) => boolean;
} {
  const manager = getMemoryManager();
  const resourceIds = React.useRef<Set<string>>(new Set());

  React.useEffect(() => {
    // Cleanup on unmount
    return () => {
      for (const id of resourceIds.current) {
        manager.cleanup(id);
      }
      resourceIds.current.clear();
    };
  }, []);

  return {
    track: (id: string, type: ResourceType, cleanup: () => void, opts?: any) => {
      manager.track(id, type, cleanup, {
        component: componentName,
        priority: options.priority,
        ...opts
      });
      resourceIds.current.add(id);
    },
    untrack: (id: string) => {
      manager.untrack(id);
      resourceIds.current.delete(id);
    },
    cleanup: (id: string) => {
      const result = manager.cleanup(id);
      resourceIds.current.delete(id);
      return result;
    }
  };
}