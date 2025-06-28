/**
 * Memory-Managed Component Utilities
 * Higher-order components and hooks for automatic memory management
 */

import React, { Component, ComponentType, FC, useEffect, useRef, useCallback } from 'react';
import { getMemoryManager, ResourceType, useMemoryManager } from './memoryManager';

/**
 * Props injected by memory management HOC
 */
export interface MemoryManagedProps {
  memoryManager: {
    track: (id: string, type: ResourceType, cleanup: () => void, opts?: any) => void;
    untrack: (id: string) => void;
    cleanup: (id: string) => boolean;
  };
}

/**
 * Higher-order component for class components with memory management
 */
export function withMemoryManagement<P extends object>(
  WrappedComponent: ComponentType<P>,
  componentName?: string
): ComponentType<Omit<P, keyof MemoryManagedProps>> {
  return class MemoryManagedComponent extends Component<Omit<P, keyof MemoryManagedProps>> {
    private memoryManager = getMemoryManager();
    private resourceIds = new Set<string>();
    private timers = new Set<number>();
    private intervals = new Set<number>();
    private animationFrames = new Set<number>();
    private eventListeners: Array<{
      target: EventTarget;
      type: string;
      listener: EventListener;
      options?: boolean | AddEventListenerOptions;
    }> = [];

    componentDidMount() {
      // Track component mount
      this.memoryManager.track(
        `component-${componentName || WrappedComponent.name}-${Date.now()}`,
        ResourceType.COMPONENT,
        () => this.cleanupResources(),
        {
          component: componentName || WrappedComponent.name,
          description: 'React component lifecycle'
        }
      );
    }

    componentWillUnmount() {
      this.cleanupResources();
    }

    /**
     * Enhanced setTimeout with automatic cleanup
     */
    setTimeout = (handler: TimerHandler, timeout?: number): number => {
      const id = window.setTimeout(() => {
        this.timers.delete(id);
        if (typeof handler === 'function') {
          handler();
        }
      }, timeout);
      
      this.timers.add(id);
      
      const resourceId = `timer-${id}`;
      this.memoryManager.track(
        resourceId,
        ResourceType.TIMER,
        () => {
          window.clearTimeout(id);
          this.timers.delete(id);
        },
        {
          component: componentName || WrappedComponent.name,
          description: `Timer ${id} (${timeout}ms)`
        }
      );
      this.resourceIds.add(resourceId);
      
      return id;
    };

    /**
     * Enhanced setInterval with automatic cleanup
     */
    setInterval = (handler: TimerHandler, timeout?: number): number => {
      const id = window.setInterval(handler, timeout);
      this.intervals.add(id);
      
      const resourceId = `interval-${id}`;
      this.memoryManager.track(
        resourceId,
        ResourceType.TIMER,
        () => {
          window.clearInterval(id);
          this.intervals.delete(id);
        },
        {
          component: componentName || WrappedComponent.name,
          description: `Interval ${id} (${timeout}ms)`
        }
      );
      this.resourceIds.add(resourceId);
      
      return id;
    };

    /**
     * Enhanced requestAnimationFrame with automatic cleanup
     */
    requestAnimationFrame = (callback: FrameRequestCallback): number => {
      const id = window.requestAnimationFrame((time) => {
        this.animationFrames.delete(id);
        callback(time);
      });
      
      this.animationFrames.add(id);
      
      const resourceId = `raf-${id}`;
      this.memoryManager.track(
        resourceId,
        ResourceType.ANIMATION_FRAME,
        () => {
          window.cancelAnimationFrame(id);
          this.animationFrames.delete(id);
        },
        {
          component: componentName || WrappedComponent.name,
          description: `Animation frame ${id}`
        }
      );
      this.resourceIds.add(resourceId);
      
      return id;
    };

    /**
     * Enhanced addEventListener with automatic cleanup
     */
    addEventListener = (
      target: EventTarget,
      type: string,
      listener: EventListener,
      options?: boolean | AddEventListenerOptions
    ): void => {
      target.addEventListener(type, listener, options);
      this.eventListeners.push({ target, type, listener, options });
      
      const resourceId = `listener-${type}-${Date.now()}`;
      this.memoryManager.track(
        resourceId,
        ResourceType.EVENT_LISTENER,
        () => {
          target.removeEventListener(type, listener, options);
        },
        {
          component: componentName || WrappedComponent.name,
          description: `Event listener for ${type}`
        }
      );
      this.resourceIds.add(resourceId);
    };

    /**
     * Manual resource tracking
     */
    track = (id: string, type: ResourceType, cleanup: () => void, opts?: any): void => {
      this.memoryManager.track(id, type, cleanup, {
        component: componentName || WrappedComponent.name,
        ...opts
      });
      this.resourceIds.add(id);
    };

    /**
     * Manual resource untracking
     */
    untrack = (id: string): void => {
      this.memoryManager.untrack(id);
      this.resourceIds.delete(id);
    };

    /**
     * Manual resource cleanup
     */
    cleanup = (id: string): boolean => {
      const result = this.memoryManager.cleanup(id);
      this.resourceIds.delete(id);
      return result;
    };

    /**
     * Clean up all resources
     */
    private cleanupResources(): void {
      // Clear timers
      for (const id of this.timers) {
        window.clearTimeout(id);
      }
      this.timers.clear();

      // Clear intervals
      for (const id of this.intervals) {
        window.clearInterval(id);
      }
      this.intervals.clear();

      // Cancel animation frames
      for (const id of this.animationFrames) {
        window.cancelAnimationFrame(id);
      }
      this.animationFrames.clear();

      // Remove event listeners
      for (const { target, type, listener, options } of this.eventListeners) {
        target.removeEventListener(type, listener, options);
      }
      this.eventListeners = [];

      // Clean up tracked resources
      for (const id of this.resourceIds) {
        this.memoryManager.cleanup(id);
      }
      this.resourceIds.clear();
    }

    render() {
      const memoryManagerProp = {
        track: this.track,
        untrack: this.untrack,
        cleanup: this.cleanup
      };

      return (
        <WrappedComponent
          {...(this.props as P)}
          memoryManager={memoryManagerProp}
        />
      );
    }
  };
}

/**
 * Hook for managed timers
 */
export function useManagedTimers(componentName: string) {
  const { track } = useMemoryManager(componentName);
  const timersRef = useRef<Set<number>>(new Set());

  const managedSetTimeout = useCallback((handler: TimerHandler, timeout?: number): number => {
    const id = window.setTimeout(() => {
      timersRef.current.delete(id);
      if (typeof handler === 'function') {
        handler();
      }
    }, timeout);
    
    timersRef.current.add(id);
    track(`timer-${id}`, ResourceType.TIMER, () => {
      window.clearTimeout(id);
      timersRef.current.delete(id);
    });
    
    return id;
  }, [track]);

  const managedSetInterval = useCallback((handler: TimerHandler, timeout?: number): number => {
    const id = window.setInterval(handler, timeout);
    timersRef.current.add(id);
    
    track(`interval-${id}`, ResourceType.TIMER, () => {
      window.clearInterval(id);
      timersRef.current.delete(id);
    });
    
    return id;
  }, [track]);

  const clearManagedTimeout = useCallback((id?: number): void => {
    if (id !== undefined) {
      window.clearTimeout(id);
      timersRef.current.delete(id);
    }
  }, []);

  const clearManagedInterval = useCallback((id?: number): void => {
    if (id !== undefined) {
      window.clearInterval(id);
      timersRef.current.delete(id);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      for (const id of timersRef.current) {
        window.clearTimeout(id);
        window.clearInterval(id);
      }
      timersRef.current.clear();
    };
  }, []);

  return {
    setTimeout: managedSetTimeout,
    setInterval: managedSetInterval,
    clearTimeout: clearManagedTimeout,
    clearInterval: clearManagedInterval
  };
}

/**
 * Hook for managed event listeners
 */
export function useManagedEventListeners(componentName: string) {
  const { track } = useMemoryManager(componentName);
  const listenersRef = useRef<Array<{
    target: EventTarget;
    type: string;
    listener: EventListener;
    options?: boolean | AddEventListenerOptions;
  }>>([]);

  const addEventListener = useCallback((
    target: EventTarget,
    type: string,
    listener: EventListener,
    options?: boolean | AddEventListenerOptions
  ): void => {
    target.addEventListener(type, listener, options);
    listenersRef.current.push({ target, type, listener, options });
    
    track(`listener-${type}-${Date.now()}`, ResourceType.EVENT_LISTENER, () => {
      target.removeEventListener(type, listener, options);
    });
  }, [track]);

  const removeEventListener = useCallback((
    target: EventTarget,
    type: string,
    listener: EventListener,
    options?: boolean | AddEventListenerOptions
  ): void => {
    target.removeEventListener(type, listener, options);
    listenersRef.current = listenersRef.current.filter(
      l => !(l.target === target && l.type === type && l.listener === listener)
    );
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      for (const { target, type, listener, options } of listenersRef.current) {
        target.removeEventListener(type, listener, options);
      }
      listenersRef.current = [];
    };
  }, []);

  return {
    addEventListener,
    removeEventListener
  };
}

/**
 * Hook for managed RAF (requestAnimationFrame)
 */
export function useManagedRAF(componentName: string) {
  const { track } = useMemoryManager(componentName);
  const rafRef = useRef<Set<number>>(new Set());

  const requestAnimationFrame = useCallback((callback: FrameRequestCallback): number => {
    const id = window.requestAnimationFrame((time) => {
      rafRef.current.delete(id);
      callback(time);
    });
    
    rafRef.current.add(id);
    track(`raf-${id}`, ResourceType.ANIMATION_FRAME, () => {
      window.cancelAnimationFrame(id);
      rafRef.current.delete(id);
    });
    
    return id;
  }, [track]);

  const cancelAnimationFrame = useCallback((id?: number): void => {
    if (id !== undefined) {
      window.cancelAnimationFrame(id);
      rafRef.current.delete(id);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      for (const id of rafRef.current) {
        window.cancelAnimationFrame(id);
      }
      rafRef.current.clear();
    };
  }, []);

  return {
    requestAnimationFrame,
    cancelAnimationFrame
  };
}

/**
 * Hook for WebGL resource management
 */
type WebGLResource = WebGLBuffer | WebGLTexture | WebGLProgram | WebGLShader | WebGLFramebuffer | WebGLRenderbuffer;

export function useManagedWebGL(componentName: string) {
  const { track } = useMemoryManager(componentName);
  const glResourcesRef = useRef<Map<string, WebGLResource>>(new Map());

  const trackWebGLResource = useCallback((
    id: string,
    resource: WebGLResource,
    gl: WebGLRenderingContext | WebGL2RenderingContext,
    type: 'buffer' | 'texture' | 'program' | 'shader' | 'framebuffer' | 'renderbuffer'
  ): void => {
    glResourcesRef.current.set(id, resource);
    
    const cleanup = () => {
      switch (type) {
        case 'buffer':
          gl.deleteBuffer(resource as WebGLBuffer);
          break;
        case 'texture':
          gl.deleteTexture(resource as WebGLTexture);
          break;
        case 'program':
          gl.deleteProgram(resource as WebGLProgram);
          break;
        case 'shader':
          gl.deleteShader(resource as WebGLShader);
          break;
        case 'framebuffer':
          gl.deleteFramebuffer(resource as WebGLFramebuffer);
          break;
        case 'renderbuffer':
          gl.deleteRenderbuffer(resource as WebGLRenderbuffer);
          break;
      }
      glResourcesRef.current.delete(id);
    };
    
    track(`webgl-${type}-${id}`, ResourceType.WEBGL, cleanup, {
      description: `WebGL ${type}`,
      size: type === 'texture' ? 4194304 : 65536 // Estimate sizes
    });
  }, [track]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      glResourcesRef.current.clear();
    };
  }, []);

  return {
    trackWebGLResource
  };
}

/**
 * Hook for Worker management
 */
export function useManagedWorker(componentName: string) {
  const { track } = useMemoryManager(componentName);
  const workersRef = useRef<Map<string, Worker>>(new Map());

  const createWorker = useCallback((scriptURL: string | URL, options?: WorkerOptions): Worker => {
    const worker = new Worker(scriptURL, options);
    const id = `worker-${Date.now()}-${Math.random()}`;
    
    workersRef.current.set(id, worker);
    track(id, ResourceType.WORKER, () => {
      worker.terminate();
      workersRef.current.delete(id);
    }, {
      description: `Worker: ${scriptURL}`,
      size: 1048576 // 1MB estimate
    });
    
    return worker;
  }, [track]);

  const terminateWorker = useCallback((worker: Worker): void => {
    for (const [id, w] of workersRef.current) {
      if (w === worker) {
        worker.terminate();
        workersRef.current.delete(id);
        break;
      }
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      for (const worker of workersRef.current.values()) {
        worker.terminate();
      }
      workersRef.current.clear();
    };
  }, []);

  return {
    createWorker,
    terminateWorker
  };
}