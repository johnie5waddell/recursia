/**
 * Engine State Management Hook
 * Manages backend readiness and connection state with proper initialization
 * Prevents race conditions during first-run execution
 */

import { useState, useEffect, useRef, useCallback } from 'react';

interface EngineState {
  isInitialized: boolean;
  isHealthy: boolean;
  lastHealthCheck: number;
  initializationAttempts: number;
  error: string | null;
}

const HEALTH_CHECK_INTERVAL = 5000; // 5 seconds
const INITIALIZATION_TIMEOUT = 30000; // 30 seconds
const MAX_INIT_ATTEMPTS = 10;

export function useEngineState() {
  const [engineState, setEngineState] = useState<EngineState>({
    isInitialized: false,
    isHealthy: false,
    lastHealthCheck: 0,
    initializationAttempts: 0,
    error: null
  });

  const initTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const healthCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const initializingRef = useRef(false);

  /**
   * Check if backend is healthy
   */
  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);

      const response = await fetch('http://localhost:8080/api/health', {
        method: 'GET',
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        return data.status === 'healthy' || data.status === 'ok';
      }
      return false;
    } catch (error) {
      // Network errors or timeouts
      return false;
    }
  }, []);

  /**
   * Initialize backend connection with retries
   */
  const initializeBackend = useCallback(async () => {
    if (initializingRef.current) {
      console.log('[EngineState] Already initializing, skipping...');
      return;
    }

    initializingRef.current = true;
    console.log('[EngineState] Starting backend initialization...');

    let attempts = 0;
    let initialized = false;

    while (attempts < MAX_INIT_ATTEMPTS && !initialized) {
      attempts++;
      console.log(`[EngineState] Initialization attempt ${attempts}/${MAX_INIT_ATTEMPTS}`);

      // Check health
      const isHealthy = await checkHealth();
      
      if (isHealthy) {
        console.log('[EngineState] Backend is healthy!');
        
        // Try a simple execution to ensure full readiness
        try {
          const testResponse = await fetch('http://localhost:8080/api/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              code: 'print "Engine initialization test";',
              options: {}
            }),
            signal: AbortSignal.timeout(5000)
          });

          if (testResponse.ok) {
            const result = await testResponse.json();
            if (result.success) {
              console.log('[EngineState] Backend fully initialized and ready');
              initialized = true;
              setEngineState({
                isInitialized: true,
                isHealthy: true,
                lastHealthCheck: Date.now(),
                initializationAttempts: attempts,
                error: null
              });
            }
          }
        } catch (testError) {
          // Test execution failed
        }
      }

      if (!initialized) {
        // Wait before retry with exponential backoff
        const delay = Math.min(1000 * Math.pow(1.5, attempts - 1), 5000);
        // Waiting before retry
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    if (!initialized) {
      const errorMsg = 'Failed to initialize backend after maximum attempts';
      console.error(`[EngineState] ${errorMsg}`);
      setEngineState(prev => ({
        ...prev,
        error: errorMsg,
        initializationAttempts: attempts
      }));
    }

    initializingRef.current = false;
  }, [checkHealth]);

  /**
   * Periodic health check
   */
  const performHealthCheck = useCallback(async () => {
    if (!engineState.isInitialized) return;

    const isHealthy = await checkHealth();
    setEngineState(prev => ({
      ...prev,
      isHealthy,
      lastHealthCheck: Date.now(),
      error: isHealthy ? null : 'Backend health check failed'
    }));

    if (!isHealthy) {
      // Backend health check failed, reinitializing
      // Try to reinitialize
      initializeBackend();
    }
  }, [checkHealth, engineState.isInitialized, initializeBackend]);

  /**
   * Wait for engine to be ready
   */
  const waitForReady = useCallback(async (timeout: number = 10000): Promise<boolean> => {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      if (engineState.isInitialized && engineState.isHealthy) {
        return true;
      }
      
      // If not initialized yet, trigger initialization
      if (!engineState.isInitialized && !initializingRef.current) {
        initializeBackend();
      }
      
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    return false;
  }, [engineState.isInitialized, engineState.isHealthy, initializeBackend]);

  /**
   * Force reinitialization
   */
  const reinitialize = useCallback(() => {
    console.log('[EngineState] Force reinitializing backend...');
    setEngineState({
      isInitialized: false,
      isHealthy: false,
      lastHealthCheck: 0,
      initializationAttempts: 0,
      error: null
    });
    initializeBackend();
  }, [initializeBackend]);

  // Initialize on mount
  useEffect(() => {
    initializeBackend();

    // Set up health check interval
    healthCheckIntervalRef.current = setInterval(performHealthCheck, HEALTH_CHECK_INTERVAL);

    // Cleanup
    return () => {
      if (initTimeoutRef.current) {
        clearTimeout(initTimeoutRef.current);
      }
      if (healthCheckIntervalRef.current) {
        clearInterval(healthCheckIntervalRef.current);
      }
    };
  }, []); // Only run on mount

  return {
    ...engineState,
    waitForReady,
    reinitialize,
    checkHealth
  };
}