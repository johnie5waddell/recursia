/**
 * Custom hook for subscribing to real-time OSH Quantum Engine data
 * Ensures all components receive continuous updates from the simulated universe
 */

import { useState, useEffect, useRef } from 'react';
import { OSHQuantumEngine } from '../engines/OSHQuantumEngine';

export interface EngineDataSnapshot {
  // RSP Engine Data
  rsp: number;
  information: number;
  coherence: number;
  entropy: number;
  recursionDepth: number;
  
  // Memory Field Data
  memoryFragments: any[];
  fieldCoherence: number;
  fieldEntropy: number;
  totalCoherence: number;
  averageCoherence: number;
  
  // Observer Data
  observers: any[];
  observerCount: number;
  averageFocus: number;
  
  // Wavefunction Data
  wavefunctionAmplitudes: number[];
  collapseEvents: number;
  measurementResults: any[];
  
  // Error Reduction Data
  errorRate: number;
  quantumVolume: number;
  rmcsActive: boolean;
  iccActive: boolean;
  coflActive: boolean;
  
  // Performance Data
  fps: number;
  cpuUsage: number;
  memoryUsage: number;
  updateTime: number;
  
  // Timestamp
  timestamp: number;
}

interface UseEngineDataOptions {
  updateInterval?: number;
  includeHistory?: boolean;
  historySize?: number;
}

export function useEngineData(
  engine: OSHQuantumEngine | null,
  options: UseEngineDataOptions = {}
): {
  data: EngineDataSnapshot | null;
  history: EngineDataSnapshot[];
  isConnected: boolean;
  error: Error | null;
} {
  const {
    updateInterval = 50, // 20 FPS by default
    includeHistory = false,
    historySize = 100
  } = options;
  
  const [data, setData] = useState<EngineDataSnapshot | null>(null);
  const [history, setHistory] = useState<EngineDataSnapshot[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const historyRef = useRef<EngineDataSnapshot[]>([]);
  
  useEffect(() => {
    if (!engine) {
      setIsConnected(false);
      return;
    }
    
    let intervalId: NodeJS.Timeout;
    let lastUpdateTime = performance.now();
    
    const collectData = (): EngineDataSnapshot | null => {
      try {
        const currentTime = performance.now();
        const updateTime = currentTime - lastUpdateTime;
        lastUpdateTime = currentTime;
        
        // Ensure all engine subsystems are available
        if (!engine.rspEngine || !engine.memoryFieldEngine || !engine.observerEngine || 
            !engine.errorReductionPlatform) {
          return null;
        }
        
        // Collect data from all engine subsystems
        const rspState = engine.rspEngine.getState();
        const memoryField = engine.memoryFieldEngine.getField();
        const memoryMetrics = engine.memoryFieldEngine.getMetrics();
        const observers = engine.observerEngine.getAllObservers();
        // Create synthetic wavefunction from already fetched memory field
        const wavefunction = engine.createSyntheticWavefunctionState ? 
          engine.createSyntheticWavefunctionState(memoryField) : null;
        const errorMetrics = engine.errorReductionPlatform.getMetrics();
        const perfMetrics = engine.getPerformanceMetrics();
        
        // Validate data exists
        if (!rspState || !memoryField || !wavefunction) {
          return null;
        }
        
        // Calculate derived metrics
        const observerCoherences = observers.map(o => o.coherence);
        const averageFocus = observerCoherences.length > 0 
          ? observerCoherences.reduce((a, b) => a + b, 0) / observerCoherences.length 
          : 0;
        
        // Create snapshot
        const snapshot: EngineDataSnapshot = {
          // RSP Data
          rsp: isFinite(rspState.rsp) ? rspState.rsp : 0,
          information: isFinite(rspState.information) ? rspState.information : 0,
          coherence: isFinite(rspState.coherence) ? rspState.coherence : 0,
          entropy: isFinite(rspState.entropy) ? rspState.entropy : 0,
          recursionDepth: Math.floor(Math.log2(rspState.information * rspState.coherence + 1)),
          
          // Memory Field Data
          memoryFragments: memoryField.fragments || [],
          fieldCoherence: memoryMetrics.coherence,
          fieldEntropy: memoryMetrics.entropy,
          totalCoherence: memoryField.totalCoherence || 0,
          averageCoherence: memoryField.averageCoherence || 0,
          
          // Observer Data
          observers: observers,
          observerCount: observers.length,
          averageFocus: averageFocus,
          
          // Wavefunction Data
          wavefunctionAmplitudes: wavefunction.amplitude ? wavefunction.amplitude.map(c => {
            if (c && typeof c.magnitude === 'function') {
              return c.magnitude();
            } else if (c && typeof c.real === 'number' && typeof c.imag === 'number') {
              return Math.sqrt(c.real * c.real + c.imag * c.imag);
            }
            return 0;
          }) : [],
          collapseEvents: observers.filter(o => o.memoryParticipation > 0.5).length,
          measurementResults: observers.map(o => ({ observerId: o.name, value: o.focus })),
          
          // Error Reduction Data
          errorRate: errorMetrics?.effectiveErrorRate || 0,
          quantumVolume: errorMetrics?.quantumVolume || 0,
          rmcsActive: errorMetrics?.effectiveErrorRate < 0.001,
          iccActive: rspState.coherence > 0.8,
          coflActive: memoryMetrics.coherence > 0.7,
          
          // Performance Data
          fps: perfMetrics.fps,
          cpuUsage: perfMetrics.cpu,
          memoryUsage: perfMetrics.memory,
          updateTime: updateTime,
          
          // Timestamp
          timestamp: Date.now()
        };
        
        return snapshot;
      } catch (err) {
        console.error('Error collecting engine data:', err);
        setError(err as Error);
        return null;
      }
    };
    
    const updateData = () => {
      const snapshot = collectData();
      if (snapshot) {
        setData(snapshot);
        setIsConnected(true);
        setError(null);
        
        // Update history if requested
        if (includeHistory) {
          historyRef.current = [...historyRef.current, snapshot].slice(-historySize);
          setHistory(historyRef.current);
        }
      }
    };
    
    // Initial update
    updateData();
    
    // Set up interval updates
    intervalId = setInterval(updateData, updateInterval);
    
    return () => {
      clearInterval(intervalId);
    };
  }, [engine, updateInterval, includeHistory, historySize]);
  
  return {
    data,
    history,
    isConnected,
    error
  };
}

/**
 * Hook for subscribing to specific engine metrics
 */
export function useEngineMetric<T>(
  engine: OSHQuantumEngine | null,
  selector: (data: EngineDataSnapshot) => T,
  updateInterval: number = 100
): T | null {
  const { data } = useEngineData(engine, { updateInterval });
  return data ? selector(data) : null;
}

/**
 * Hook for tracking engine events
 */
export function useEngineEvents(
  engine: OSHQuantumEngine | null,
  eventTypes: string[] = []
): Array<{ type: string; timestamp: number; data: any }> {
  const [events, setEvents] = useState<Array<{ type: string; timestamp: number; data: any }>>([]);
  
  useEffect(() => {
    if (!engine) return;
    
    const listeners: Array<() => void> = [];
    
    // Subscribe to specific event types
    eventTypes.forEach(eventType => {
      const listener = (data: any) => {
        setEvents(prev => [...prev, {
          type: eventType,
          timestamp: Date.now(),
          data
        }].slice(-100)); // Keep last 100 events
      };
      
      // Note: This assumes the engine has an event emitter interface
      // You may need to adjust based on actual engine implementation
      // engine.on(eventType, listener);
      // listeners.push(() => engine.off(eventType, listener));
    });
    
    return () => {
      listeners.forEach(cleanup => cleanup());
    };
  }, [engine, eventTypes]);
  
  return events;
}