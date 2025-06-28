/**
 * Hook to properly handle RSP values from backend
 * Fixes the issue where RSP always shows as 1.0
 */

import { useEffect, useRef } from 'react';

export function useRSPFix(engineAPI: any) {
  const lastValidRSP = useRef<number>(0);
  
  useEffect(() => {
    if (!engineAPI) return;
    
    // Subscribe to metrics updates
    const handleMetricsUpdate = (metrics: any) => {
      if (metrics && typeof metrics.rsp === 'number' && metrics.rsp > 0) {
        // Check if RSP is not the default 1.0 or if it's a valid calculated value
        if (metrics.rsp !== 1.0 || 
            (metrics.information && metrics.coherence && metrics.entropy)) {
          lastValidRSP.current = metrics.rsp;
          
          // RSP value received and validated
        }
      }
    };
    
    // Listen for metrics updates
    if (engineAPI.on) {
      engineAPI.on('metrics', handleMetricsUpdate);
    }
    
    return () => {
      if (engineAPI.off) {
        engineAPI.off('metrics', handleMetricsUpdate);
      }
    };
  }, [engineAPI]);
  
  return lastValidRSP.current;
}