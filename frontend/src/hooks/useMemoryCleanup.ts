/**
 * Memory Cleanup Hooks
 * Comprehensive memory management hooks for React components
 */

import { useEffect, useRef, useCallback, DependencyList } from 'react';
import { useMemoryManager, ResourceType } from '../utils/memoryManager';
import {
  useManagedTimers,
  useManagedEventListeners,
  useManagedRAF,
  useManagedWebGL,
  useManagedWorker
} from '../utils/MemoryManagedComponent';

/**
 * Enhanced useEffect with automatic cleanup tracking
 */
export function useCleanupEffect(
  effect: () => (() => void) | void,
  deps: DependencyList,
  componentName: string
) {
  const { track } = useMemoryManager(componentName);
  const cleanupRef = useRef<(() => void) | void>();

  useEffect(() => {
    // Track the effect
    const effectId = `effect-${componentName}-${Date.now()}`;
    track(effectId, ResourceType.SUBSCRIPTION, () => {
      if (cleanupRef.current) {
        cleanupRef.current();
      }
    });

    // Run the effect
    cleanupRef.current = effect();

    // Return cleanup
    return () => {
      if (cleanupRef.current) {
        cleanupRef.current();
      }
    };
  }, deps);
}

/**
 * Enhanced useCallback with memory tracking
 */
export function useTrackedCallback<T extends (...args: any[]) => any>(
  callback: T,
  deps: DependencyList,
  componentName: string
): T {
  const { track, untrack } = useMemoryManager(componentName);
  const callbackId = useRef<string>();

  const trackedCallback = useCallback((...args: Parameters<T>) => {
    // Track callback execution
    const executionId = `callback-exec-${Date.now()}`;
    track(executionId, ResourceType.SUBSCRIPTION, () => {}, {
      description: 'Callback execution',
      size: 1000
    });

    try {
      return callback(...args);
    } finally {
      // Immediate cleanup
      untrack(executionId);
    }
  }, deps) as T;

  useEffect(() => {
    // Track the callback itself
    callbackId.current = `callback-${componentName}-${Date.now()}`;
    track(callbackId.current, ResourceType.SUBSCRIPTION, () => {}, {
      description: 'Tracked callback',
      size: 5000
    });

    return () => {
      if (callbackId.current) {
        untrack(callbackId.current);
      }
    };
  }, [trackedCallback]);

  return trackedCallback;
}

/**
 * Comprehensive memory cleanup hook
 */
export function useMemoryCleanup(componentName: string) {
  const memoryManager = useMemoryManager(componentName);
  const timers = useManagedTimers(componentName);
  const eventListeners = useManagedEventListeners(componentName);
  const raf = useManagedRAF(componentName);
  const webgl = useManagedWebGL(componentName);
  const worker = useManagedWorker(componentName);

  // Track component lifecycle
  useEffect(() => {
    memoryManager.track(
      `component-${componentName}`,
      ResourceType.COMPONENT,
      () => {},
      {
        description: `React component: ${componentName}`,
        priority: 'high'
      }
    );

    return () => {
      memoryManager.cleanup(`component-${componentName}`);
    };
  }, []);

  return {
    ...memoryManager,
    ...timers,
    ...eventListeners,
    ...raf,
    ...webgl,
    ...worker,
    
    // Additional utility methods
    trackResource: (id: string, cleanup: () => void, size?: number) => {
      memoryManager.track(id, ResourceType.COMPONENT, cleanup, {
        size,
        description: `Custom resource in ${componentName}`
      });
    },
    
    trackCanvas: (canvas: HTMLCanvasElement, id: string) => {
      memoryManager.track(id, ResourceType.CANVAS, () => {
        // Clear canvas
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        // Remove from DOM if needed
        if (canvas.parentNode) {
          canvas.parentNode.removeChild(canvas);
        }
      }, {
        size: canvas.width * canvas.height * 4, // RGBA
        description: `Canvas ${canvas.width}x${canvas.height}`
      });
    },
    
    trackWebSocket: (ws: WebSocket, id: string) => {
      memoryManager.track(id, ResourceType.WEBSOCKET, () => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.close();
        }
      }, {
        description: `WebSocket to ${ws.url}`
      });
    },
    
    trackMediaStream: (stream: MediaStream, id: string) => {
      memoryManager.track(id, ResourceType.MEDIA_STREAM, () => {
        stream.getTracks().forEach(track => track.stop());
      }, {
        size: 10485760, // 10MB estimate
        description: `MediaStream with ${stream.getTracks().length} tracks`
      });
    }
  };
}

/**
 * Hook for tracking fetch requests
 */
export function useFetch(componentName: string) {
  const { track, untrack } = useMemoryManager(componentName);
  const abortControllers = useRef<Map<string, AbortController>>(new Map());

  const trackedFetch = useCallback(async (
    url: RequestInfo,
    options?: RequestInit & { id?: string }
  ): Promise<Response> => {
    const controller = new AbortController();
    const fetchId = options?.id || `fetch-${Date.now()}`;
    
    // Track abort controller
    abortControllers.current.set(fetchId, controller);
    track(fetchId, ResourceType.SUBSCRIPTION, () => {
      controller.abort();
      abortControllers.current.delete(fetchId);
    }, {
      description: `Fetch request to ${url}`
    });

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      
      return response;
    } finally {
      // Cleanup after completion
      untrack(fetchId);
      abortControllers.current.delete(fetchId);
    }
  }, [track, untrack]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Abort all pending requests
      for (const controller of abortControllers.current.values()) {
        controller.abort();
      }
      abortControllers.current.clear();
    };
  }, []);

  return {
    fetch: trackedFetch,
    abortFetch: (id: string) => {
      const controller = abortControllers.current.get(id);
      if (controller) {
        controller.abort();
        abortControllers.current.delete(id);
        untrack(id);
      }
    }
  };
}

/**
 * Hook for managing subscription cleanup
 */
export function useSubscription<T>(componentName: string) {
  const { track, untrack } = useMemoryManager(componentName);
  const subscriptions = useRef<Map<string, () => void>>(new Map());

  const subscribe = useCallback((
    id: string,
    cleanup: () => void,
    description?: string
  ) => {
    subscriptions.current.set(id, cleanup);
    track(`subscription-${id}`, ResourceType.SUBSCRIPTION, cleanup, {
      description: description || `Subscription ${id}`
    });
  }, [track]);

  const unsubscribe = useCallback((id: string) => {
    const cleanup = subscriptions.current.get(id);
    if (cleanup) {
      cleanup();
      subscriptions.current.delete(id);
      untrack(`subscription-${id}`);
    }
  }, [untrack]);

  // Cleanup all on unmount
  useEffect(() => {
    return () => {
      for (const [id, cleanup] of subscriptions.current) {
        cleanup();
      }
      subscriptions.current.clear();
    };
  }, []);

  return {
    subscribe,
    unsubscribe,
    unsubscribeAll: () => {
      for (const [id, cleanup] of subscriptions.current) {
        cleanup();
        untrack(`subscription-${id}`);
      }
      subscriptions.current.clear();
    }
  };
}

/**
 * Hook for managing ResizeObserver with cleanup
 */
export function useResizeObserver(
  callback: ResizeObserverCallback,
  componentName: string
) {
  const { track } = useMemoryManager(componentName);
  const observerRef = useRef<ResizeObserver>();
  const elementsRef = useRef<Set<Element>>(new Set());

  useEffect(() => {
    observerRef.current = new ResizeObserver(callback);
    
    track('resize-observer', ResourceType.OBSERVABLE, () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    }, {
      description: 'ResizeObserver'
    });

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [callback]);

  const observe = useCallback((element: Element) => {
    if (observerRef.current && element) {
      observerRef.current.observe(element);
      elementsRef.current.add(element);
    }
  }, []);

  const unobserve = useCallback((element: Element) => {
    if (observerRef.current && element) {
      observerRef.current.unobserve(element);
      elementsRef.current.delete(element);
    }
  }, []);

  return {
    observe,
    unobserve,
    disconnect: () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
        elementsRef.current.clear();
      }
    }
  };
}

/**
 * Hook for managing IntersectionObserver with cleanup
 */
export function useIntersectionObserver(
  callback: IntersectionObserverCallback,
  options: IntersectionObserverInit | undefined,
  componentName: string
) {
  const { track } = useMemoryManager(componentName);
  const observerRef = useRef<IntersectionObserver>();
  const elementsRef = useRef<Set<Element>>(new Set());

  useEffect(() => {
    observerRef.current = new IntersectionObserver(callback, options);
    
    track('intersection-observer', ResourceType.OBSERVABLE, () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    }, {
      description: 'IntersectionObserver'
    });

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [callback, options]);

  const observe = useCallback((element: Element) => {
    if (observerRef.current && element) {
      observerRef.current.observe(element);
      elementsRef.current.add(element);
    }
  }, []);

  const unobserve = useCallback((element: Element) => {
    if (observerRef.current && element) {
      observerRef.current.unobserve(element);
      elementsRef.current.delete(element);
    }
  }, []);

  return {
    observe,
    unobserve,
    disconnect: () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
        elementsRef.current.clear();
      }
    }
  };
}