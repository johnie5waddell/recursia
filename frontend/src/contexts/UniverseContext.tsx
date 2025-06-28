import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import OSHQuantumEngine from '../engines/OSHQuantumEngine';

interface UniverseState {
  timestamp: number;
  initialized: boolean;
  memoryField: {
    fragments: number;
    totalCoherence: number;
    strain: number;
    field: any;
  };
  wavefunction: {
    dimensions: number;
    coherence: number;
    state: any;
  };
  rsp: {
    value: number;
    information: number;
    complexity: number;
    observerDensity: number;
    sustainabilityScore: number;
  };
  performance: {
    fps: number;
    memoryUsage: number;
    updateTime: number;
  };
  engines: Record<string, boolean>;
}

interface UniverseContextValue {
  engine: OSHQuantumEngine | null;
  isSimulating: boolean;
  universeState: UniverseState | null;
  lastUpdate: number;
  error: string | null;
}

const UniverseContext = createContext<UniverseContextValue | null>(null);

export const useUniverse = () => {
  const context = useContext(UniverseContext);
  if (!context) {
    throw new Error('useUniverse must be used within a UniverseProvider');
  }
  return context;
};

interface UniverseProviderProps {
  children: ReactNode;
  engine: OSHQuantumEngine | null;
  isSimulating: boolean;
}

export const UniverseProvider: React.FC<UniverseProviderProps> = ({ 
  children, 
  engine, 
  isSimulating 
}) => {
  const [universeState, setUniverseState] = useState<UniverseState | null>(null);
  const [lastUpdate, setLastUpdate] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!engine) {
      setUniverseState(null);
      return;
    }

    // Add a small delay to ensure engine is fully initialized
    const timeoutId = setTimeout(() => {
      // Get initial state
      try {
        if (engine && typeof engine.getState === 'function') {
          const state = engine.getState();
          setUniverseState(state);
          setError(state.error || null);
        } else {
          console.warn('[UniverseContext] Engine does not have getState method yet');
        }
      } catch (err) {
        console.error('Failed to get initial universe state:', err);
        console.error('Error details:', {
          message: err instanceof Error ? err.message : 'Unknown error',
          stack: err instanceof Error ? err.stack : undefined,
          engine: engine,
          engineType: engine?.constructor?.name
        });
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    }, 100);

    // Listen for universe updates
    const handleUniverseUpdate = (event: CustomEvent) => {
      try {
        const { metrics, engineState } = event.detail;
        const timestamp = engineState?.timestamp || metrics?.timestamp || Date.now();
        setLastUpdate(timestamp);
        
        // Create a complete universe state from the data
        if (engineState || metrics) {
          const newState: UniverseState = {
            timestamp,
            initialized: true,
            memoryField: {
              fragments: engineState?.memoryField?.fragments || 0,  // memory_fragments removed
              totalCoherence: engineState?.memoryField?.totalCoherence || metrics?.coherence || 0.95,
              strain: metrics?.strain || metrics?.memory_strain || 0,
              field: null
            },
            wavefunction: {
              dimensions: 3,
              coherence: metrics?.coherence || 0.95,
              state: engineState?.wavefunction || null
            },
            rsp: {
              value: engineState?.rsp?.value || metrics?.rsp || 0,
              information: metrics?.information || metrics?.integrated_information || 0,
              complexity: metrics?.complexity || 1,
              observerDensity: metrics?.observer_count || 0,
              sustainabilityScore: metrics?.temporal_stability || 0.95
            },
            performance: {
              fps: engineState?.performance?.fps || metrics?.fps || 60,
              memoryUsage: engineState?.performance?.memoryUsage || metrics?.resources?.memory || 0,
              updateTime: Date.now() - timestamp
            },
            engines: {
              universe: true,
              running: metrics?.universe_running || false,
              mode: metrics?.universe_mode || 'standard',
              time: metrics?.universe_time || 0,
              iterations: metrics?.iteration_count || 0
            }
          };
          
          setUniverseState(newState);
          setError(null);
          
          // Enhanced logging for debugging universe state
          if (metrics?.universe_running && metrics?.iteration_count % 50 === 0) {
            console.log('[UniverseContext] 🔄 Universe state sync:', {
              time: metrics.universe_time?.toFixed(3),
              iterations: metrics.iteration_count,
              phi: metrics.phi,
              rsp: newState.rsp.value,
              integrated_info: metrics.integrated_information,
              observer_count: newState.rsp.observerDensity,
              coherence: newState.memoryField.totalCoherence,
              emergence: metrics.emergence_index,
              from_engine_state: !!engineState,
              from_metrics: !!metrics
            });
          }
        }
      } catch (err) {
        console.error('Failed to handle universe update:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    };

    // Subscribe to universe events
    window.addEventListener('universeUpdate', handleUniverseUpdate as EventListener);
    
    // Also listen for start/stop events
    const handleUniverseStart = (event?: CustomEvent) => {
      setError(null);
    };
    
    const handleUniverseStop = () => {
      // Handle stop event
    };

    window.addEventListener('universeStarted', handleUniverseStart);
    window.addEventListener('universeStopped', handleUniverseStop);

    return () => {
      clearTimeout(timeoutId);
      window.removeEventListener('universeUpdate', handleUniverseUpdate as EventListener);
      window.removeEventListener('universeStarted', handleUniverseStart);
      window.removeEventListener('universeStopped', handleUniverseStop);
    };
  }, [engine]);

  const value: UniverseContextValue = {
    engine,
    isSimulating,
    universeState,
    lastUpdate,
    error
  };

  return (
    <UniverseContext.Provider value={value}>
      {children}
    </UniverseContext.Provider>
  );
};

export default UniverseContext;