/**
 * useEngineAPI Hook - Enhanced Version
 * Enterprise-grade API integration for OSH Quantum Engine
 * Provides real-time physics simulation data with WebSocket support
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { toast } from 'react-hot-toast';
import { 
  ServerMessage, 
  ClientMessageType,
  MetricsData,
  QuantumState,
  isMetricsUpdate,
  isStatesMessage,
  isErrorMessage,
  isConnectionMessage,
  isUniverseStartedMessage,
  isExecutionCompleteMessage,
  isExecutionLogMessage,
  isTimelineEventMessage
} from '../types/websocket';

// Extend window for debugging
declare global {
  interface Window {
    lastUniverseMetrics?: any;
  }
}

// MetricsData interface is now imported from types/websocket.ts

// QuantumState interface is now imported from types/websocket.ts

interface ExecutionResult {
  success: boolean;
  message?: string;
  output?: string | string[];
  outputs?: string[]; // Add outputs property
  error?: string;
  errors?: Array<{
    type: string;
    message: string;
    line?: number;
    column?: number;
    filename?: string;
    timestamp?: string;
  }> | string[];
  warnings?: string[];
  measurements?: Array<{
    state: string;
    qubit: number;
    outcome: number;
    probability: number;
    timestamp: number;
  }>;
  iterations?: number;
  total_execution_time?: number;
  metrics?: {
    rsp?: number;
    coherence?: number;
    coherence_std?: number;
    entropy_std?: number;
    recursion_depth?: number;
    information_curvature?: number;
    entropy?: number;
    information?: number;
    depth?: number;
    strain?: number;
    focus?: number;
    observer_count?: number;
    state_count?: number;
    measurement_count?: number;
    field_energy?: number;
    field_coherence?: number;
    observer_focus?: number;
    integrated_information?: number;
    phi?: number;
    complexity?: number;
    entropy_flux?: number;
    emergence_index?: number;
    temporal_stability?: number;
    memory_field_coupling?: number;
    conservation_law?: number;
  };
}

interface CompilationResult {
  success: boolean;
  ast?: any;
  errors?: string[];
  warnings?: string[];
}

// Timeline event interface
export interface TimelineEvent {
  id: string;
  timestamp: number;
  type: 'collapse' | 'entanglement' | 'divergence' | 'attractor' | 'measurement' | 'execution' | 'error' | 'warning' | 'info';
  description: string;
  data: any;
  level?: 'info' | 'warning' | 'error' | 'success';
  source?: string;
  duration?: number;
}

// Simulation snapshot interface
export interface SimulationSnapshot {
  timestamp: number;
  simulationTime: number;
  metrics: MetricsData;
  states: Record<string, QuantumState>;
  events: TimelineEvent[];
  executionLog: Array<{ timestamp: number; message: string; level: string }>;
}

interface StatesData {
  [key: string]: QuantumState;
}

interface LogEntry {
  timestamp: number;
  message: string;
  level: string;
}

// Hook return interface
export interface EngineAPIHook {
  // Connection state
  isConnected: boolean;
  wsStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  error: string | null;
  
  // Data
  states: StatesData;
  metrics: MetricsData;
  executionLog: LogEntry[];
  simulationSnapshots: SimulationSnapshot[];
  isSimulationPaused: boolean;
  simulationTime: number;
  
  // Actions
  execute: (code: string, iterations?: number) => Promise<ExecutionResult>;
  compile: (code: string, target?: string) => Promise<CompilationResult>;
  disconnect: () => void;
  reconnect: () => void;
  clearExecutionLog: () => void;
  pauseSimulation: () => void;
  resumeSimulation: () => void;
  seekSimulation: (time: number) => void;
  clearSnapshots: () => void;
  
  // Universe control
  startUniverseSimulation: (mode?: string) => void;
  stopUniverseSimulation: () => void;
  setUniverseMode: (mode: string) => void;
  updateUniverseParameters: (params: any) => void;
}

// API configuration with proper browser environment handling
const getApiBaseUrl = (): string => {
  // Check for environment variable or use default
  const envApiUrl = import.meta.env?.VITE_API_URL;
  if (envApiUrl) {
    return envApiUrl;
  }
  
  // Always use direct backend URL for consistency
  // The backend has CORS configured to allow all origins
  return 'http://localhost:8080';
};

const getWebSocketUrl = (): string => {
  const envWsUrl = import.meta.env?.VITE_WS_URL;
  if (envWsUrl) {
    return envWsUrl;
  }
  
  // In development, use relative path to leverage Vite's proxy
  if (import.meta.env.DEV) {
    // Use relative WebSocket path that will be proxied by Vite
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws`;
  }
  
  // In production, use the API base URL
  const apiBase = getApiBaseUrl();
  if (apiBase && apiBase.startsWith('http')) {
    return apiBase.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws';
  }
  
  // Fallback to direct connection
  return 'ws://localhost:8080/ws';
};

const API_BASE_URL = getApiBaseUrl();
const WS_URL = getWebSocketUrl();


/**
 * Enhanced Engine API Hook
 * Connects to the continuously running physics simulation backend
 */
export function useEngineAPI(): EngineAPIHook {
  // State management - Initialize with default values instead of null
  const [metrics, setMetrics] = useState<MetricsData>({
    // Core OSH metrics with reasonable defaults
    rsp: 0,
    coherence: 0.95,
    entropy: 0.1,
    information: 0,
    
    // Additional metrics
    strain: 0,
    phi: 0,
    emergence_index: 0,
    field_energy: 0,
    temporal_stability: 0.95,
    observer_influence: 0,
    memory_field_coupling: 0,
    
    // System metrics
    recursion_depth: 0,
    observer_focus: 0,
    focus: 0,
    information_curvature: 0.001,
    integrated_information: 0,
    complexity: 1,
    entropy_flux: 0,
    conservation_law: 1,
    
    // Counts
    observer_count: 0,
    state_count: 0,
    depth: 0,
    
    // Performance
    fps: 60,
    error: 0.001,
    quantum_volume: 0,
    
    // Derivatives
    drsp_dt: 0,
    di_dt: 0,
    dc_dt: 0,
    de_dt: 0,
    acceleration: 0,
    
    // Other
    measurement_count: 0,
    timestamp: Date.now()
  });
  const [states, setStates] = useState<StatesData>({});
  const [isConnected, setIsConnected] = useState(false);
  const [wsStatus, setWsStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const [error, setError] = useState<string | null>(null);
  
  // Timeline state
  const [executionLog, setExecutionLog] = useState<LogEntry[]>([]);
  const [simulationSnapshots, setSimulationSnapshots] = useState<SimulationSnapshot[]>([]);
  const [isSimulationPaused, setIsSimulationPaused] = useState(false);
  const [simulationTime, setSimulationTime] = useState(0);
  const maxSnapshots = 1000; // Keep last 1000 snapshots
  const snapshotInterval = 100; // Record snapshot every 100ms
  const lastSnapshotTimeRef = useRef(0);
  const currentEventsRef = useRef<TimelineEvent[]>([]);
  const lastMetricsUpdateRef = useRef(0);
  const metricsUpdateThrottleMs = 50; // Throttle metrics updates to max 20Hz
  
  // WebSocket management
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const messageQueueRef = useRef<ClientMessageType[]>([]);
  
  /**
   * Process and normalize metrics data from various backend sources
   */
  const processMetrics = useCallback((rawMetrics: Partial<MetricsData> & Record<string, any>): MetricsData => {
    // Handle different metric formats and field names from backend
    const processed: MetricsData = {
      // Core metrics with proper fallbacks - use nullish coalescing to preserve zeros
      rsp: rawMetrics.rsp ?? rawMetrics.recursive_simulation_potential ?? 0,
      coherence: rawMetrics.coherence ?? 0.95,
      entropy: rawMetrics.entropy ?? rawMetrics.entropy_flux ?? 0.1,
      information: rawMetrics.information ?? rawMetrics.integrated_information ?? rawMetrics.information_density ?? 0,
      
      // Additional metrics
      strain: rawMetrics.strain ?? rawMetrics.memory_strain ?? 0,
      phi: rawMetrics.phi ?? 0,
      emergence_index: rawMetrics.emergence_index ?? (rawMetrics.phi ? rawMetrics.phi / 15.0 : 0),
      field_energy: rawMetrics.field_energy ?? 0,
      temporal_stability: rawMetrics.temporal_stability ?? 0.95,
      observer_influence: rawMetrics.observer_influence ?? 0,
      memory_field_coupling: rawMetrics.memory_field_coupling ?? rawMetrics.memory_strain ?? 0,
      
      // System metrics
      recursion_depth: rawMetrics.recursion_depth ?? rawMetrics.recursive_depth ?? 0,
      observer_focus: rawMetrics.observer_focus ?? rawMetrics.focus ?? rawMetrics.observer_influence ?? 0,
      focus: rawMetrics.focus ?? rawMetrics.observer_focus ?? rawMetrics.observer_influence ?? 0,
      information_curvature: rawMetrics.information_curvature ?? 0.001,
      integrated_information: rawMetrics.integrated_information ?? rawMetrics.information ?? rawMetrics.information_density ?? 0,
      complexity: rawMetrics.complexity ?? rawMetrics.kolmogorov_complexity ?? 1,
      entropy_flux: rawMetrics.entropy_flux ?? rawMetrics.entanglement_entropy ?? 0,
      conservation_law: rawMetrics.conservation_law ?? (1 - (rawMetrics.conservation_violation ?? 0)),
      
      // Counts
      observer_count: rawMetrics.observer_count ?? 0,
      state_count: rawMetrics.state_count ?? 0,
      depth: rawMetrics.depth ?? rawMetrics.recursion_depth ?? rawMetrics.recursive_depth ?? 0,
      
      // Performance
      fps: rawMetrics.fps ?? 60,
      error: rawMetrics.error ?? 0.001,
      quantum_volume: rawMetrics.quantum_volume ?? 0,
      
      // Derivatives
      drsp_dt: rawMetrics.drsp_dt ?? 0,
      di_dt: rawMetrics.di_dt ?? 0,
      dc_dt: rawMetrics.dc_dt ?? 0,
      de_dt: rawMetrics.de_dt ?? 0,
      acceleration: rawMetrics.acceleration ?? 0,
      
      // Other
      measurement_count: rawMetrics.measurement_count ?? 0,
      timestamp: rawMetrics.timestamp ?? Date.now(),
      // memory_fragments: rawMetrics.memory_fragments ?? [],  // Removed
      resources: rawMetrics.resources,
      
      // Dynamic universe metrics
      universe_time: rawMetrics.universe_time ?? 0,
      iteration_count: rawMetrics.iteration_count ?? 0,
      num_entanglements: rawMetrics.num_entanglements ?? 0,
      universe_mode: rawMetrics.universe_mode ?? 'Standard Universe',
      universe_running: rawMetrics.universe_running ?? false,
      
      // Additional metrics for components
      phase: rawMetrics.phase ?? (rawMetrics.universe_time ? (rawMetrics.universe_time * 0.1) % (2 * Math.PI) : 0),
      gravitational_anomaly: rawMetrics.gravitational_anomaly ?? 0,
      consciousness_probability: rawMetrics.consciousness_probability ?? 0,
      memory_strain: rawMetrics.memory_strain ?? rawMetrics.strain ?? 0,
      
      // Pause state
      is_paused: rawMetrics.is_paused ?? false
    };
    
    // Enhanced logging for universe mode
    if (rawMetrics.universe_running) {
      // Log critical metrics every 50 iterations
      if (rawMetrics.iteration_count && rawMetrics.iteration_count % 50 === 0) {
        console.log('[useEngineAPI] 🌌 Universe metrics (iter ' + rawMetrics.iteration_count + '):', {
          phi: rawMetrics.phi?.toFixed(6) ?? '0',
          integrated_information: rawMetrics.integrated_information?.toFixed(6) ?? '0',
          rsp: rawMetrics.rsp?.toFixed(6) ?? '0',
          observer_count: rawMetrics.observer_count ?? 0,
          state_count: rawMetrics.state_count ?? 0,
          num_entanglements: rawMetrics.num_entanglements ?? 0,  // Add entanglement logging
          coherence: rawMetrics.coherence?.toFixed(6) ?? '0',
          emergence_index: rawMetrics.emergence_index?.toFixed(6) ?? '0',
          observer_influence: rawMetrics.observer_influence?.toFixed(6) ?? '0'
        });
      }
      
      // Critical metrics check - no need to warn during normal operation
    }
    
    return processed;
  }, []);
  
  /**
   * Add entry to execution log
   */
  const addToExecutionLog = useCallback((message: string, level: string = 'info') => {
    const entry: LogEntry = {
      timestamp: Date.now(),
      message,
      level
    };
    setExecutionLog(prev => [...prev.slice(-999), entry]); // Keep last 1000 entries
  }, []);
  
  /**
   * Record simulation snapshot
   */
  const recordSnapshot = useCallback((metricsData: MetricsData, simTime: number) => {
    const snapshot: SimulationSnapshot = {
      timestamp: Date.now(),
      simulationTime: simTime,
      metrics: { ...metricsData },
      states: { ...states },
      events: [...currentEventsRef.current],
      executionLog: [...executionLog].slice(-100) // Last 100 log entries
    };
    
    setSimulationSnapshots(prev => {
      const updated = [...prev, snapshot];
      // Keep only the last maxSnapshots
      if (updated.length > maxSnapshots) {
        return updated.slice(-maxSnapshots);
      }
      return updated;
    });
    
    // Clear current events for next snapshot
    currentEventsRef.current = [];
  }, [states, executionLog]);
  
  /**
   * Connect to WebSocket for real-time updates
   */
  const connectWebSocket = useCallback(() => {
    // Don't connect if already connected or connecting
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }
    
    // Clean up any existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    try {
      // Use the dynamically determined WebSocket URL
      const finalUrl = getWebSocketUrl();
      
      console.log('[useEngineAPI] Connecting to WebSocket:', finalUrl);
      const ws = new WebSocket(finalUrl);
      wsRef.current = ws; // IMPORTANT: Store the WebSocket reference immediately!
      setWsStatus('connecting');
      
      ws.onopen = () => {
        setIsConnected(true);
        setWsStatus('connected');
        setError(null);
        reconnectAttemptsRef.current = 0;
        
        // Send any queued messages
        while (messageQueueRef.current.length > 0) {
          const msg = messageQueueRef.current.shift();
          ws.send(JSON.stringify(msg));
        }
        
        // Disabled auto-start - let user control universe simulation
        // setTimeout(() => {
        //   console.log('[useEngineAPI] Auto-starting universe simulation...');
        //   ws.send(JSON.stringify({ type: 'start_universe', mode: 'standard' }));
        // }, 2000);
      };
      
      ws.onerror = (error) => {
        setError('WebSocket connection error');
        setWsStatus('error');
      };
      
      ws.onclose = (event) => {
        setIsConnected(false);
        setWsStatus('disconnected');
        wsRef.current = null;
        
        // Attempt reconnection if not explicitly closed
        if (event.code !== 1000 && reconnectAttemptsRef.current < 5) {
          reconnectAttemptsRef.current++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000);
          reconnectTimeoutRef.current = setTimeout(connectWebSocket, delay);
        }
      };
      
      ws.onmessage = (event) => {
        try {
          const message: ServerMessage = JSON.parse(event.data);
          
          
          // Handle different message types with type guards
          switch (message.type) {
            case 'metrics':
            case 'metrics_update':
              
              // Ensure we have valid data before updating
              if (isMetricsUpdate(message) && message.data) {
                // Enhanced debug logging for universe mode
                if (message.data.universe_running) {
                  const timestamp = new Date(message.data.timestamp * 1000).toISOString();
                  console.log(`[useEngineAPI] 🌌 Universe Update @ ${timestamp}:`, {
                    universe_time: message.data.universe_time,
                    iteration_count: message.data.iteration_count,
                    timestamp: message.data.timestamp,
                    phi: message.data.phi,
                    rsp: message.data.rsp,
                    coherence: message.data.coherence,
                    observer_count: message.data.observer_count,
                    state_count: message.data.state_count,
                    time_derivatives: {
                      drsp_dt: message.data.drsp_dt,
                      di_dt: message.data.di_dt,
                      dc_dt: message.data.dc_dt,
                      de_dt: message.data.de_dt
                    },
                    universe_running: message.data.universe_running
                  });
                  
                  // Debug observer count specifically
                  if (message.data.observer_count === 0 && message.data.iteration_count > 5) {
                    console.warn('[useEngineAPI] ⚠️ Observer count is 0 after 5 iterations!', {
                      raw_observer_count: message.data.observer_count,
                      iteration_count: message.data.iteration_count
                    });
                  }
                  
                  // Check if metrics are changing
                  if (window.lastUniverseMetrics) {
                    const lastMetrics = window.lastUniverseMetrics;
                    if (lastMetrics.timestamp === message.data.timestamp) {
                      console.warn('[useEngineAPI] ⚠️ Same timestamp as last update!');
                    }
                    if (lastMetrics.iteration_count === message.data.iteration_count) {
                      console.warn('[useEngineAPI] ⚠️ Same iteration count as last update!');
                    }
                    if (lastMetrics.phi === message.data.phi && lastMetrics.rsp === message.data.rsp) {
                      console.warn('[useEngineAPI] ⚠️ PHI and RSP values unchanged!');
                    }
                  }
                  window.lastUniverseMetrics = message.data;
                }
                
                const processed = processMetrics(message.data);
                
                // Debug processed observer count - only log when it changes or is non-zero
                if (message.data.universe_running && message.data.observer_count > 0) {
                  console.log('[useEngineAPI] 🎉 Observer count is now non-zero!', {
                    raw_observer_count: message.data.observer_count,
                    processed_observer_count: processed.observer_count,
                    iteration: message.data.iteration_count
                  });
                }
                
                // Throttle metrics updates to prevent infinite loops
                const now = Date.now();
                if (now - lastMetricsUpdateRef.current >= metricsUpdateThrottleMs) {
                  setMetrics(processed);
                  lastMetricsUpdateRef.current = now;
                }
                
                // Update pause state if present in metrics
                if (message.data.is_paused !== undefined) {
                  setIsSimulationPaused(message.data.is_paused);
                }
                
                // Update simulation time
                if (message.data.timestamp) {
                  setSimulationTime(message.data.timestamp);
                }
                
                // Record snapshot if interval has passed and not paused
                if (!isSimulationPaused) {
                  const now = Date.now();
                  if (now - lastSnapshotTimeRef.current >= snapshotInterval) {
                    recordSnapshot(processed, message.data.timestamp || now);
                    lastSnapshotTimeRef.current = now;
                  }
                }
                
                // Emit universe update event if universe is running
                if (message.data.universe_running) {
                  const universeEvent = new CustomEvent('universeUpdate', {
                    detail: {
                      metrics: processed,
                      engineState: {
                        timestamp: message.data.timestamp || Date.now(),
                        performance: {
                          fps: processed.fps,
                          memoryUsage: processed.resources?.memory || 0
                        },
                        memoryField: {
                          fragments: 0,  // memory_fragments removed
                          totalCoherence: processed.coherence
                        },
                        rsp: {
                          value: processed.rsp
                        },
                        wavefunction: {
                          amplitude: Math.sqrt(processed.coherence),
                          phase: processed.phase || 0
                        },
                        universe: {
                          time: message.data.universe_time || 0,
                          iteration: message.data.iteration_count || 0,
                          mode: message.data.universe_mode || 'standard',
                          running: message.data.universe_running || false,
                          entanglements: message.data.num_entanglements || 0
                        }
                      }
                    }
                  });
                  window.dispatchEvent(universeEvent);
                }
              } else {
                console.warn('[useEngineAPI] Invalid metrics data received from WebSocket:', message.data);
              }
              break;
              
            case 'execution_complete':
              // Refresh states after execution
              fetchStates();
              if (isExecutionCompleteMessage(message) && message.data?.message) {
                toast.success(message.data.message);
                // Add to execution log
                addToExecutionLog(message.data.message, 'success');
              }
              break;
              
            case 'execution_log':
            case 'log':
              if (isExecutionLogMessage(message) && message.data?.message) {
                addToExecutionLog(message.data.message, message.data.level || 'info');
              }
              break;
              
            case 'timeline_event':
              if (isTimelineEventMessage(message) && message.data) {
                const event: TimelineEvent = {
                  id: `event-${Date.now()}-${Math.random()}`,
                  timestamp: message.data.timestamp || Date.now(),
                  type: message.data.type || 'info' as any,
                  description: message.data.description || '',
                  data: message.data.data,
                  level: message.data.level as any,
                  source: message.data.source,
                  duration: message.data.duration
                };
                currentEventsRef.current.push(event);
              }
              break;
              
            case 'simulation_paused':
              setIsSimulationPaused(true);
              break;
              
            case 'simulation_resumed':
              setIsSimulationPaused(false);
              break;
              
            case 'universe_started':
              addToExecutionLog('Universe simulation started', 'success');
              
              // Update metrics with universe running state
              if (isUniverseStartedMessage(message) && message.data) {
                // Force update metrics to reflect universe running state
                setMetrics(prev => ({
                  ...prev,
                  universe_running: message.data.running || true,
                  universe_mode: message.data.mode || 'Standard Universe',
                  iteration_count: message.data.iteration_count || 0,
                  universe_time: message.data.universe_time || 0
                }));
                
                // Also emit a custom event for other components
                window.dispatchEvent(new CustomEvent('universeStartConfirmed', { 
                  detail: message.data 
                }));
              }
              break;
              
            case 'universe_stopped':
              addToExecutionLog('Universe simulation stopped', 'info');
              break;
              
            case 'connected':
            case 'connection':
              break;
              
            case 'echo':
              // Handle echo messages without logging
              break;
              
            case 'pong':
              // Handle pong response - connection is alive
              break;
              
            case 'error':
              console.error('[WebSocket] Backend error:', message.data);
              if (isErrorMessage(message)) {
                toast.error(message.data?.message || 'Backend error');
              }
              break;
              
            case 'universe_mode_changed':
              // Handle universe mode change confirmation
              console.log('[WebSocket] Universe mode changed:', message.data);
              break;
              
            case 'universe_params_updated':
              // Handle universe parameters update confirmation
              console.log('[WebSocket] Universe parameters updated:', message.data);
              break;
              
            default:
              // Silently ignore unhandled message types
          }
        } catch (err) {
          console.error('[WebSocket] Failed to parse message:', err);
        }
      };
      
      // Remove duplicate handlers - they were already set above
    } catch (err) {
      setWsStatus('error');
      setError('Failed to connect to backend');
      toast.error('Failed to connect to backend');
    }
  }, [processMetrics, recordSnapshot]);
  
  /**
   * Fetch quantum states
   */
  const fetchStates = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/states`);
      if (response.ok) {
        const data = await response.json();
        setStates(data.states || {});
      }
    } catch (err) {
      console.error('Failed to fetch states:', err);
    }
  }, []);
  
  // Temporarily disabled to debug issue
  // const engineState = useEngineState();
  // const executionCache = useExecutionCache();

  /**
   * Execute Recursia code with retry logic
   */
  const execute = useCallback(async (code: string, iterations: number = 1, retryCount: number = 0): Promise<ExecutionResult> => {
    // Simple pre-flight check without the broken health check
    if (!isConnected && retryCount === 0) {
      // Don't block execution, just log the warning
    }
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000); // 120 second timeout to match test expectations
      
      const response = await fetch(`${API_BASE_URL}/api/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          code, 
          options: { 
            timeout: 120.0,  // Pass timeout to backend
            debug: false 
          }, 
          iterations 
        }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      let result;
      try {
        result = await response.json();
      } catch (jsonError) {
        // If JSON parsing fails, try to get text
        const text = await response.text();
        console.error('Failed to parse JSON response:', text);
        throw new Error(`Invalid JSON response from server: ${text.substring(0, 200)}`);
      }
      
      // Check if the HTTP response itself failed
      if (!response.ok) {
        // Enhanced error reporting for 500 errors
        if (response.status === 500) {
          const errorDetail = result.detail || result.error || 'Internal server error';
          console.error('Server error details:', result);
          
          // Check for specific error patterns
          if (typeof errorDetail === 'object' && errorDetail.errors) {
            throw new Error(`Compilation failed: ${errorDetail.errors.join(', ')}`);
          } else if (typeof errorDetail === 'string') {
            throw new Error(errorDetail);
          } else {
            throw new Error(`Server error: ${JSON.stringify(errorDetail).substring(0, 200)}`);
          }
        }
        throw new Error(result.error || `HTTP ${response.status}: ${response.statusText}`);
      }
      
      // For successful HTTP responses, return the result as-is
      // The caller will handle success/failure based on result.success
      
      // Update states after execution
      await fetchStates();
      
      // Update metrics if available
      if (result.metrics) {
        const processed = processMetrics(result.metrics);
        setMetrics(processed);
        
        // Record execution event
        const event: TimelineEvent = {
          id: `exec-${Date.now()}-${Math.random()}`,
          timestamp: Date.now(),
          type: 'execution',
          description: `Executed ${iterations} iteration(s)`,
          data: { code, iterations, metrics: result.metrics },
          level: 'info',
          source: 'api'
        };
        currentEventsRef.current.push(event);
        
        // Force snapshot after execution
        recordSnapshot(processed, Date.now());
        
        // Also fetch latest metrics from backend to ensure sync
        // This ensures UI updates immediately even if WebSocket is disconnected
        setTimeout(async () => {
          try {
            const metricsRes = await fetch(`${API_BASE_URL}/api/metrics`);
            if (metricsRes.ok) {
              const latestMetrics = await metricsRes.json();
              const processedLatest = processMetrics(latestMetrics);
              setMetrics(processedLatest);
            }
          } catch (err) {
            // Failed to refresh metrics after execution
          }
        }, 100); // Small delay to ensure backend has updated
      }
      
      // Process output array to string if needed
      if (Array.isArray(result.output)) {
        result.output = result.output.join('\n');
      }
      
      // Return the result exactly as received from the API
      // The QuantumOSHStudio component will handle success/failure display
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // Check if it's a temporary error that we should retry
      const isTemporaryError = 
        errorMessage.includes('Failed to fetch') ||
        errorMessage.includes('NetworkError') ||
        errorMessage.includes('HTTP 500') ||
        errorMessage.includes('HTTP 502') ||
        errorMessage.includes('HTTP 503') ||
        errorMessage.includes('HTTP 504') ||
        errorMessage.includes('AbortError');
      
      // Retry logic for temporary errors
      if (isTemporaryError && retryCount < 3) {
        const delay = Math.min(1000 * Math.pow(2, retryCount), 5000); // Exponential backoff, max 5 seconds
        
        // Don't show error toast for retries
        await new Promise(resolve => setTimeout(resolve, delay));
        return execute(code, iterations, retryCount + 1);
      }
      
      // Only show toast for final failure
      if (errorMessage.includes('HTTP') || errorMessage.includes('network') || errorMessage.includes('Failed to fetch')) {
        if (retryCount > 0) {
          toast.error(`Connection error after ${retryCount} retries: ${errorMessage}`);
        } else {
          toast.error(`Connection error: ${errorMessage}`);
        }
      }
      
      return {
        success: false,
        error: errorMessage,
        errors: [errorMessage],
        warnings: []
      };
    }
  }, [fetchStates, isConnected, processMetrics]);
  
  /**
   * Compile Recursia code
   */
  const compile = useCallback(async (code: string, target: string = 'quantum_simulator'): Promise<CompilationResult> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/compile`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code, target }),
      });
      
      if (!response.ok) {
        throw new Error('Compilation failed');
      }
      
      return await response.json();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      toast.error(`Compilation failed: ${errorMessage}`);
      return {
        success: false,
        errors: [errorMessage],
      };
    }
  }, []);
  
  /**
   * Send message through WebSocket
   */
  const sendMessage = useCallback((message: ClientMessageType): boolean => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    } else {
      // Queue message for when connection is established
      messageQueueRef.current.push(message);
      return false;
    }
  }, []);
  
  /**
   * Initialize connection and fetch initial data
   */
  useEffect(() => {
    // Initial data fetch
    const fetchInitialData = async () => {
      try {
        // Fetch metrics with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
        
        try {
          const metricsRes = await fetch(`${API_BASE_URL}/api/metrics`, {
            signal: controller.signal
          });
          clearTimeout(timeoutId);
          
          if (metricsRes.ok) {
            const metricsData = await metricsRes.json();
            const processed = processMetrics(metricsData);
            setMetrics(processed);
          }
        } catch (fetchErr) {
          if (fetchErr instanceof Error && fetchErr.name === 'AbortError') {
            // Metrics fetch timed out
          } else {
            // Failed to fetch metrics
          }
        }
        
        // Fetch states (non-blocking)
        fetchStates().catch(err => {
          // Failed to fetch states
        });
      } catch (err) {
        console.error('Failed to fetch initial data:', err);
        // Only show error if not already shown
        if (!isConnected) {
          toast.error('Backend API not available. Some features may be limited.');
        }
      }
    };
    
    fetchInitialData();
    
    // Connect WebSocket after a delay to ensure page is fully loaded
    const connectTimeout = setTimeout(() => {
      if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
        connectWebSocket();
      }
    }, 500);
    
    // Send periodic pings to keep connection alive
    const pingInterval = setInterval(() => {
      sendMessage({ type: 'ping' });
    }, 30000);
    
    // Periodic metrics fetch as fallback for WebSocket
    // ONLY fetch if WebSocket is disconnected to avoid overwriting real-time data
    const metricsInterval = setInterval(async () => {
      // Only poll if WebSocket is NOT connected
      if (!isConnected && wsRef.current?.readyState !== WebSocket.OPEN) {
        try {
          const response = await fetch(`${API_BASE_URL}/api/metrics`);
          if (response.ok) {
            const metricsData = await response.json();
            const processed = processMetrics(metricsData);
            setMetrics(processed);
          }
        } catch (err) {
          // Silently fail - WebSocket is primary method
        }
      }
    }, 10000); // Every 10 seconds as fallback
    
    // Cleanup
    return () => {
      clearTimeout(connectTimeout);
      clearInterval(pingInterval);
      clearInterval(metricsInterval);
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []); // Empty dependencies to run only on mount
  
  /**
   * Pause simulation
   */
  const pauseSimulation = useCallback(() => {
    setIsSimulationPaused(true);
    sendMessage({ type: 'pause_simulation' });
    addToExecutionLog('Simulation paused', 'info');
  }, [sendMessage, addToExecutionLog]);
  
  /**
   * Resume simulation
   */
  const resumeSimulation = useCallback(() => {
    setIsSimulationPaused(false);
    sendMessage({ type: 'resume_simulation' });
    addToExecutionLog('Simulation resumed', 'info');
  }, [sendMessage, addToExecutionLog]);
  
  /**
   * Seek to specific simulation time
   */
  const seekSimulation = useCallback((time: number) => {
    // Find closest snapshot
    const snapshot = simulationSnapshots.reduce((closest, current) => {
      const currentDiff = Math.abs(current.simulationTime - time);
      const closestDiff = Math.abs(closest.simulationTime - time);
      return currentDiff < closestDiff ? current : closest;
    }, simulationSnapshots[0]);
    
    if (snapshot) {
      // Restore state from snapshot
      setMetrics(snapshot.metrics);
      setStates(snapshot.states);
      setSimulationTime(snapshot.simulationTime);
      sendMessage({ type: 'seek_simulation', data: { time: snapshot.simulationTime } });
      addToExecutionLog(`Seeked to time ${snapshot.simulationTime.toFixed(2)}`, 'info');
    }
  }, [simulationSnapshots, sendMessage, addToExecutionLog]);
  
  /**
   * Clear execution log
   */
  const clearExecutionLog = useCallback(() => {
    setExecutionLog([]);
  }, []);
  
  /**
   * Clear snapshots
   */
  const clearSnapshots = useCallback(() => {
    setSimulationSnapshots([]);
    currentEventsRef.current = [];
    lastSnapshotTimeRef.current = 0;
  }, []);
  
  /**
   * Disconnect WebSocket
   */
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
    setWsStatus('disconnected');
  }, []);
  
  /**
   * Reconnect WebSocket
   */
  const reconnect = useCallback(() => {
    disconnect();
    setTimeout(connectWebSocket, 100);
  }, [disconnect, connectWebSocket]);
  
  /**
   * Start universe simulation
   */
  const startUniverseSimulation = useCallback((mode: string = 'standard') => {
    // Ensure WebSocket is connected before sending
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      // Force reconnect if not connected
      connectWebSocket();
      // Queue the message to be sent after connection
      messageQueueRef.current.push({ type: 'start_universe', data: { mode } });
      addToExecutionLog('Connecting to backend...', 'info');
      
      // Set up a one-time listener to retry when connected
      const checkConnection = setInterval(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          clearInterval(checkConnection);
          sendMessage({ type: 'start_universe', data: { mode } });
          addToExecutionLog(`Starting universe simulation in ${mode} mode`, 'info');
        }
      }, 100);
      
      // Clear the interval after 5 seconds if still not connected
      setTimeout(() => clearInterval(checkConnection), 5000);
      return;
    }
    
    const sent = sendMessage({ type: 'start_universe', data: { mode } });
    
    addToExecutionLog(`Starting universe simulation in ${mode} mode`, 'info');
    
    // Emit window event for UniverseContext
    window.dispatchEvent(new CustomEvent('universeStarted', { detail: { mode } }));
  }, [sendMessage, addToExecutionLog, connectWebSocket]);
  
  /**
   * Stop universe simulation
   */
  const stopUniverseSimulation = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      sendMessage({ type: 'stop_universe' });
      addToExecutionLog('Stopping universe simulation', 'info');
    }
    
    // Emit window event for UniverseContext
    window.dispatchEvent(new CustomEvent('universeStopped'));
  }, [sendMessage, addToExecutionLog]);
  
  /**
   * Set universe mode
   */
  const setUniverseMode = useCallback((mode: string) => {
    sendMessage({ type: 'set_universe_mode', data: { mode } });
    addToExecutionLog(`Universe mode changed to: ${mode}`, 'info');
  }, [sendMessage, addToExecutionLog]);
  
  /**
   * Update universe parameters
   */
  const updateUniverseParameters = useCallback((params: any) => {
    sendMessage({ type: 'update_universe_params', data: { params } });
    addToExecutionLog('Universe parameters updated', 'info');
  }, [sendMessage, addToExecutionLog]);
  
  return {
    // Connection state
    isConnected,
    wsStatus,
    error,
    
    // Data
    states,
    metrics,
    executionLog,
    simulationSnapshots,
    isSimulationPaused,
    simulationTime,
    
    // Actions
    execute,
    compile,
    disconnect,
    reconnect,
    clearExecutionLog,
    pauseSimulation,
    resumeSimulation,
    seekSimulation,
    clearSnapshots,
    
    // Universe control
    startUniverseSimulation,
    stopUniverseSimulation,
    setUniverseMode,
    updateUniverseParameters
  };
}