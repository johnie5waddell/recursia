/**
 * Execution Cache Hook
 * Prevents duplicate executions and manages execution state
 * Helps avoid race conditions and duplicate API calls
 */

import { useRef, useCallback } from 'react';

interface ExecutionCacheEntry {
  code: string;
  timestamp: number;
  result?: any;
  inProgress: boolean;
}

const CACHE_DURATION = 1000; // 1 second - prevent rapid duplicate executions
const MAX_CACHE_SIZE = 10;

export function useExecutionCache() {
  const cacheRef = useRef<Map<string, ExecutionCacheEntry>>(new Map());
  const executionQueueRef = useRef<Set<string>>(new Set());

  /**
   * Generate cache key from code
   */
  const getCacheKey = useCallback((code: string): string => {
    // Simple hash function for code
    let hash = 0;
    for (let i = 0; i < code.length; i++) {
      const char = code.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return `exec_${hash}_${code.length}`;
  }, []);

  /**
   * Check if execution is in progress or recently completed
   */
  const isExecutionCached = useCallback((code: string): boolean => {
    const key = getCacheKey(code);
    const entry = cacheRef.current.get(key);
    
    if (!entry) return false;
    
    // Check if in progress
    if (entry.inProgress) {
      // Execution already in progress
      return true;
    }
    
    // Check if recently executed
    const age = Date.now() - entry.timestamp;
    if (age < CACHE_DURATION) {
      // Recently executed - using cached result
      return true;
    }
    
    return false;
  }, [getCacheKey]);

  /**
   * Get cached result if available
   */
  const getCachedResult = useCallback((code: string): any | null => {
    const key = getCacheKey(code);
    const entry = cacheRef.current.get(key);
    
    if (!entry || entry.inProgress) return null;
    
    const age = Date.now() - entry.timestamp;
    if (age < CACHE_DURATION) {
      return entry.result;
    }
    
    return null;
  }, [getCacheKey]);

  /**
   * Mark execution as started
   */
  const markExecutionStart = useCallback((code: string): boolean => {
    const key = getCacheKey(code);
    
    // Check if already in progress
    if (executionQueueRef.current.has(key)) {
      // Execution already queued
      return false;
    }
    
    // Add to queue
    executionQueueRef.current.add(key);
    
    // Update cache
    cacheRef.current.set(key, {
      code,
      timestamp: Date.now(),
      inProgress: true
    });
    
    // Limit cache size
    if (cacheRef.current.size > MAX_CACHE_SIZE) {
      const oldestKey = Array.from(cacheRef.current.keys())[0];
      cacheRef.current.delete(oldestKey);
    }
    
    return true;
  }, [getCacheKey]);

  /**
   * Mark execution as completed
   */
  const markExecutionComplete = useCallback((code: string, result: any) => {
    const key = getCacheKey(code);
    
    // Remove from queue
    executionQueueRef.current.delete(key);
    
    // Update cache
    cacheRef.current.set(key, {
      code,
      timestamp: Date.now(),
      result,
      inProgress: false
    });
  }, [getCacheKey]);

  /**
   * Mark execution as failed
   */
  const markExecutionFailed = useCallback((code: string) => {
    const key = getCacheKey(code);
    
    // Remove from queue
    executionQueueRef.current.delete(key);
    
    // Remove from cache to allow retry
    cacheRef.current.delete(key);
  }, [getCacheKey]);

  /**
   * Clear all cache
   */
  const clearCache = useCallback(() => {
    cacheRef.current.clear();
    executionQueueRef.current.clear();
  }, []);

  /**
   * Wrap an async execution function with caching
   */
  const wrapExecution = useCallback(<T>(
    executeFunction: (code: string) => Promise<T>
  ) => {
    return async (code: string): Promise<T> => {
      // Check cache first
      const cachedResult = getCachedResult(code);
      if (cachedResult !== null) {
        // Returning cached result
        return cachedResult;
      }
      
      // Check if already executing
      if (isExecutionCached(code)) {
        // Wait a bit and check again
        await new Promise(resolve => setTimeout(resolve, 100));
        const result = getCachedResult(code);
        if (result !== null) {
          return result;
        }
        // If still no result, throw to prevent hanging
        throw new Error('Execution already in progress');
      }
      
      // Mark as started
      if (!markExecutionStart(code)) {
        throw new Error('Failed to start execution - already in queue');
      }
      
      try {
        // Execute
        const result = await executeFunction(code);
        
        // Cache result
        markExecutionComplete(code, result);
        
        return result;
      } catch (error) {
        // Mark as failed
        markExecutionFailed(code);
        throw error;
      }
    };
  }, [getCachedResult, isExecutionCached, markExecutionStart, markExecutionComplete, markExecutionFailed]);

  return {
    isExecutionCached,
    getCachedResult,
    markExecutionStart,
    markExecutionComplete,
    markExecutionFailed,
    clearCache,
    wrapExecution
  };
}