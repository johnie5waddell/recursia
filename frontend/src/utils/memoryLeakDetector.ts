/**
 * Memory Leak Detector
 * Identifies and prevents memory leaks in the application
 * Tracks object references, event listeners, and DOM nodes
 */

import React from 'react';

interface LeakReport {
  id: string;
  type: 'event-listener' | 'dom-node' | 'object-reference' | 'timer' | 'observable';
  description: string;
  size: number;
  stackTrace?: string;
  timestamp: number;
}

interface TrackedObject {
  ref: WeakRef<any>;
  type: string;
  size: number;
  created: number;
  component?: string;
}

export class MemoryLeakDetector {
  private leaks: Map<string, LeakReport> = new Map();
  private trackedObjects: Map<string, TrackedObject> = new Map();
  private eventListenerMap: WeakMap<EventTarget, Map<string, Function[]>> = new WeakMap();
  private timers: Set<ReturnType<typeof setTimeout>> = new Set();
  private observers: Set<MutationObserver | IntersectionObserver | ResizeObserver> = new Set();
  private checkInterval: number = 30000; // 30 seconds
  private intervalId?: ReturnType<typeof setInterval>;
  private finalizationRegistry: FinalizationRegistry<string>;
  private enabled: boolean = false; // Disabled by default to prevent crashes
  
  constructor() {
    // Setup finalization registry to detect when objects are garbage collected
    this.finalizationRegistry = new FinalizationRegistry((heldValue: string) => {
      this.trackedObjects.delete(heldValue);
    });
    
    // Don't setup hooks automatically - wait for explicit enable
    // this.setupGlobalHooks();
  }
  
  /**
   * Setup global hooks to track potential memory leaks
   */
  private setupGlobalHooks(): void {
    // Hook into addEventListener
    this.hookEventListeners();
    
    // Hook into timers
    this.hookTimers();
    
    // Hook into observers
    this.hookObservers();
  }
  
  /**
   * Hook into event listener methods to track them
   */
  private hookEventListeners(): void {
    if (!this.enabled) return;
    
    const originalAddEventListener = EventTarget.prototype.addEventListener;
    const originalRemoveEventListener = EventTarget.prototype.removeEventListener;
    
    EventTarget.prototype.addEventListener = function(
      type: string,
      listener: EventListenerOrEventListenerObject | null,
      options?: boolean | AddEventListenerOptions
    ): void {
      try {
        // Track the listener
        const detector = MemoryLeakDetector.getInstance();
        if (detector.enabled && listener && this && typeof this === 'object') {
          detector.trackEventListener(this, type, listener);
        }
      } catch (error) {
        console.error('[MemoryLeakDetector] Error in addEventListener wrapper:', error);
      }
      
      // Call original method
      return originalAddEventListener.call(this, type, listener, options);
    };
    
    EventTarget.prototype.removeEventListener = function(
      type: string,
      listener: EventListenerOrEventListenerObject | null,
      options?: boolean | EventListenerOptions
    ): void {
      try {
        // Untrack the listener
        const detector = MemoryLeakDetector.getInstance();
        if (detector.enabled && listener && this && typeof this === 'object') {
          detector.untrackEventListener(this, type, listener);
        }
      } catch (error) {
        console.error('[MemoryLeakDetector] Error in removeEventListener wrapper:', error);
      }
      
      // Call original method
      return originalRemoveEventListener.call(this, type, listener, options);
    };
  }
  
  /**
   * Hook into timer functions
   */
  private hookTimers(): void {
    if (!this.enabled) return;
    
    const originalSetTimeout = window.setTimeout;
    const originalSetInterval = window.setInterval;
    const originalClearTimeout = window.clearTimeout;
    const originalClearInterval = window.clearInterval;
    
    window.setTimeout = ((...args: Parameters<typeof setTimeout>) => {
      const id = originalSetTimeout.apply(window, args);
      const detector = MemoryLeakDetector.getInstance();
      if (detector.enabled) {
        detector.timers.add(id);
      }
      return id;
    }) as typeof setTimeout;
    
    window.setInterval = ((...args: Parameters<typeof setInterval>) => {
      const id = originalSetInterval.apply(window, args);
      const detector = MemoryLeakDetector.getInstance();
      if (detector.enabled) {
        detector.timers.add(id);
      }
      return id;
    }) as typeof setInterval;
    
    window.clearTimeout = ((id?: Parameters<typeof clearTimeout>[0]): void => {
      if (id !== undefined) {
        const detector = MemoryLeakDetector.getInstance();
        if (detector.enabled) {
          detector.timers.delete(id as ReturnType<typeof setTimeout>);
        }
      }
      return originalClearTimeout.call(window, id);
    }) as typeof clearTimeout;
    
    window.clearInterval = ((id?: Parameters<typeof clearInterval>[0]): void => {
      if (id !== undefined) {
        const detector = MemoryLeakDetector.getInstance();
        if (detector.enabled) {
          detector.timers.delete(id as ReturnType<typeof setTimeout>);
        }
      }
      return originalClearInterval.call(window, id);
    }) as typeof clearInterval;
  }
  
  /**
   * Hook into observer APIs
   */
  private hookObservers(): void {
    if (!this.enabled) return;
    
    // MutationObserver
    const OriginalMutationObserver = window.MutationObserver;
    window.MutationObserver = class extends OriginalMutationObserver {
      constructor(callback: MutationCallback) {
        super(callback);
        const detector = MemoryLeakDetector.getInstance();
        if (detector.enabled) {
          detector.observers.add(this);
        }
      }
      
      disconnect(): void {
        const detector = MemoryLeakDetector.getInstance();
        if (detector.enabled) {
          detector.observers.delete(this);
        }
        super.disconnect();
      }
    };
    
    // IntersectionObserver
    if (window.IntersectionObserver) {
      const OriginalIntersectionObserver = window.IntersectionObserver;
      window.IntersectionObserver = class extends OriginalIntersectionObserver {
        constructor(callback: IntersectionObserverCallback, options?: IntersectionObserverInit) {
          super(callback, options);
          const detector = MemoryLeakDetector.getInstance();
          if (detector.enabled) {
            detector.observers.add(this);
          }
        }
        
        disconnect(): void {
          const detector = MemoryLeakDetector.getInstance();
          if (detector.enabled) {
            detector.observers.delete(this);
          }
          super.disconnect();
        }
      };
    }
    
    // ResizeObserver
    if (window.ResizeObserver) {
      const OriginalResizeObserver = window.ResizeObserver;
      window.ResizeObserver = class extends OriginalResizeObserver {
        constructor(callback: ResizeObserverCallback) {
          super(callback);
          const detector = MemoryLeakDetector.getInstance();
          if (detector.enabled) {
            detector.observers.add(this);
          }
        }
        
        disconnect(): void {
          const detector = MemoryLeakDetector.getInstance();
          if (detector.enabled) {
            detector.observers.delete(this);
          }
          super.disconnect();
        }
      };
    }
  }
  
  /**
   * Track an event listener
   */
  private trackEventListener(
    target: EventTarget,
    type: string,
    listener: EventListenerOrEventListenerObject | Function
  ): void {
    // WeakMap requires object keys - validate target
    if (!target || typeof target !== 'object') {
      console.warn('[MemoryLeakDetector] Invalid event target:', target);
      return;
    }
    
    try {
      if (!this.eventListenerMap.has(target)) {
        this.eventListenerMap.set(target, new Map());
      }
      
      const targetMap = this.eventListenerMap.get(target)!;
      if (!targetMap.has(type)) {
        targetMap.set(type, []);
      }
      
      targetMap.get(type)!.push(listener as Function);
    } catch (error) {
      console.error('[MemoryLeakDetector] Error tracking event listener:', error);
    }
  }
  
  /**
   * Untrack an event listener
   */
  private untrackEventListener(
    target: EventTarget,
    type: string,
    listener: EventListenerOrEventListenerObject | Function
  ): void {
    // WeakMap requires object keys - validate target
    if (!target || typeof target !== 'object') {
      return;
    }
    
    try {
      const targetMap = this.eventListenerMap.get(target);
      if (!targetMap) return;
      
      const listeners = targetMap.get(type);
      if (!listeners) return;
      
      const index = listeners.indexOf(listener as Function);
      if (index > -1) {
        listeners.splice(index, 1);
      }
      
      // Clean up empty structures
      if (listeners.length === 0) {
        targetMap.delete(type);
      }
      
      if (targetMap.size === 0) {
        this.eventListenerMap.delete(target);
      }
    } catch (error) {
      console.error('[MemoryLeakDetector] Error untracking event listener:', error);
    }
  }
  
  /**
   * Track an object for leak detection
   */
  trackObject(obj: any, type: string, component?: string): void {
    const id = `obj-${Date.now()}-${Math.random()}`;
    const size = this.estimateObjectSize(obj);
    
    this.trackedObjects.set(id, {
      ref: new WeakRef(obj),
      type,
      size,
      created: Date.now(),
      component
    });
    
    // Register for garbage collection notification
    this.finalizationRegistry.register(obj, id);
  }
  
  /**
   * Estimate object size in bytes
   */
  private estimateObjectSize(obj: any): number {
    const seen = new WeakSet();
    
    const sizeof = (object: any): number => {
      if (object === null || object === undefined) return 0;
      if (typeof object === 'boolean') return 4;
      if (typeof object === 'number') return 8;
      if (typeof object === 'string') return object.length * 2;
      
      if (typeof object === 'object') {
        if (seen.has(object)) return 0;
        seen.add(object);
        
        let size = 0;
        
        if (Array.isArray(object)) {
          for (const item of object) {
            size += sizeof(item);
          }
        } else {
          for (const key in object) {
            if (object.hasOwnProperty(key)) {
              size += sizeof(key) + sizeof(object[key]);
            }
          }
        }
        
        return size;
      }
      
      return 0;
    };
    
    return sizeof(obj);
  }
  
  /**
   * Start leak detection
   */
  start(): void {
    if (!this.enabled || this.intervalId) return;
    
    this.intervalId = setInterval(() => {
      this.detectLeaks();
    }, this.checkInterval);
    
    // Initial check
    this.detectLeaks();
  }
  
  /**
   * Stop leak detection
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
  }
  
  /**
   * Detect potential memory leaks
   */
  private detectLeaks(): void {
    // Check for orphaned event listeners
    this.checkOrphanedListeners();
    
    // Check for long-lived objects
    this.checkLongLivedObjects();
    
    // Check for active timers
    this.checkActiveTimers();
    
    // Check for active observers
    this.checkActiveObservers();
    
    // Check DOM for detached nodes
    this.checkDetachedNodes();
  }
  
  /**
   * Check for orphaned event listeners
   */
  private checkOrphanedListeners(): void {
    // This is handled by the WeakMap automatically
    // Listeners on garbage collected targets will be removed
  }
  
  /**
   * Check for objects that have lived too long
   */
  private checkLongLivedObjects(): void {
    const now = Date.now();
    const maxAge = 300000; // 5 minutes
    
    for (const [id, tracked] of this.trackedObjects) {
      const obj = tracked.ref.deref();
      
      if (!obj) {
        // Object was garbage collected, remove tracking
        this.trackedObjects.delete(id);
        continue;
      }
      
      const age = now - tracked.created;
      if (age > maxAge) {
        this.reportLeak({
          id,
          type: 'object-reference',
          description: `Long-lived ${tracked.type} object in ${tracked.component || 'unknown'} component`,
          size: tracked.size,
          timestamp: now
        });
      }
    }
  }
  
  /**
   * Check for active timers
   */
  private checkActiveTimers(): void {
    if (this.timers.size > 100) {
      this.reportLeak({
        id: 'timers',
        type: 'timer',
        description: `Excessive active timers: ${this.timers.size}`,
        size: this.timers.size * 8,
        timestamp: Date.now()
      });
    }
  }
  
  /**
   * Check for active observers
   */
  private checkActiveObservers(): void {
    if (this.observers.size > 50) {
      this.reportLeak({
        id: 'observers',
        type: 'observable',
        description: `Excessive active observers: ${this.observers.size}`,
        size: this.observers.size * 100,
        timestamp: Date.now()
      });
    }
  }
  
  /**
   * Check for detached DOM nodes
   */
  private checkDetachedNodes(): void {
    const allNodes = document.querySelectorAll('*');
    let detachedCount = 0;
    
    allNodes.forEach(node => {
      if (!document.body.contains(node) && node.parentNode) {
        detachedCount++;
      }
    });
    
    if (detachedCount > 100) {
      this.reportLeak({
        id: 'dom-nodes',
        type: 'dom-node',
        description: `Detached DOM nodes found: ${detachedCount}`,
        size: detachedCount * 50,
        timestamp: Date.now()
      });
    }
  }
  
  /**
   * Report a memory leak
   */
  private reportLeak(leak: LeakReport): void {
    this.leaks.set(leak.id, leak);
    
    // Emit event
    const event = new CustomEvent('memory-leak-detected', {
      detail: leak
    });
    window.dispatchEvent(event);
    
    console.warn('Memory leak detected:', leak);
  }
  
  /**
   * Get all detected leaks
   */
  getLeaks(): LeakReport[] {
    return Array.from(this.leaks.values());
  }
  
  /**
   * Clear a specific leak after it's been fixed
   */
  clearLeak(id: string): void {
    this.leaks.delete(id);
  }
  
  /**
   * Get memory leak summary
   */
  getSummary(): {
    totalLeaks: number;
    totalSize: number;
    byType: Record<string, number>;
  } {
    const leaks = this.getLeaks();
    const byType: Record<string, number> = {};
    
    let totalSize = 0;
    
    for (const leak of leaks) {
      totalSize += leak.size;
      byType[leak.type] = (byType[leak.type] || 0) + 1;
    }
    
    return {
      totalLeaks: leaks.length,
      totalSize,
      byType
    };
  }
  
  /**
   * Clean up specific types of leaks
   */
  cleanup(type?: LeakReport['type']): void {
    if (!type) {
      // Clean up everything
      this.cleanupTimers();
      this.cleanupObservers();
      this.cleanupObjects();
    } else {
      switch (type) {
        case 'timer':
          this.cleanupTimers();
          break;
        case 'observable':
          this.cleanupObservers();
          break;
        case 'object-reference':
          this.cleanupObjects();
          break;
      }
    }
  }
  
  /**
   * Clean up all timers
   */
  private cleanupTimers(): void {
    for (const id of this.timers) {
      clearTimeout(id);
      clearInterval(id);
    }
    this.timers.clear();
  }
  
  /**
   * Clean up all observers
   */
  private cleanupObservers(): void {
    for (const observer of this.observers) {
      observer.disconnect();
    }
    this.observers.clear();
  }
  
  /**
   * Clean up tracked objects
   */
  private cleanupObjects(): void {
    // Force garbage collection if available (only in Node.js with --expose-gc flag)
    if (typeof globalThis !== 'undefined' && (globalThis as any).gc && typeof (globalThis as any).gc === 'function') {
      try {
        (globalThis as any).gc();
      } catch (e) {
        // Ignore GC errors in browser environment
      }
    }
    
    // Clear tracked objects
    this.trackedObjects.clear();
  }
  
  /**
   * Enable memory leak detection
   */
  enable(): void {
    if (this.enabled) return;
    
    console.log('[MemoryLeakDetector] Enabling memory leak detection');
    this.enabled = true;
    this.setupGlobalHooks();
    this.start();
  }
  
  /**
   * Disable memory leak detection
   */
  disable(): void {
    if (!this.enabled) return;
    
    console.log('[MemoryLeakDetector] Disabling memory leak detection');
    this.enabled = false;
    this.stop();
    // Note: We can't unhook the global methods once hooked
  }
  
  /**
   * Check if memory leak detection is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }
  
  // Singleton instance
  private static instance: MemoryLeakDetector;
  
  static getInstance(): MemoryLeakDetector {
    if (!MemoryLeakDetector.instance) {
      MemoryLeakDetector.instance = new MemoryLeakDetector();
    }
    return MemoryLeakDetector.instance;
  }
}

/**
 * React hook for memory leak detection
 */
export function useMemoryLeakDetector(componentName: string) {
  React.useEffect(() => {
    const detector = MemoryLeakDetector.getInstance();
    
    // Only track if enabled
    if (!detector.isEnabled()) {
      return;
    }
    
    detector.start();
    
    // Track component mount
    const mountTime = Date.now();
    
    return () => {
      if (!detector.isEnabled()) {
        return;
      }
      
      // Check for leaks on unmount
      const unmountTime = Date.now();
      const lifetime = unmountTime - mountTime;
      
      if (lifetime < 1000) {
        // Component unmounted too quickly, might indicate rapid re-renders
        console.warn(`Component ${componentName} had short lifetime: ${lifetime}ms`);
      }
    };
  }, [componentName]);
}

/**
 * Decorator for automatic leak detection in classes
 */
export function DetectLeaks(constructor: Function) {
  const original = constructor.prototype.componentWillUnmount || function() {};
  
  constructor.prototype.componentWillUnmount = function() {
    const detector = MemoryLeakDetector.getInstance();
    detector.cleanup();
    original.call(this);
  };
}