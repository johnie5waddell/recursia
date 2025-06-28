/**
 * TypeScript type definitions for WebSocket communication
 * Ensures type safety for all WebSocket messages
 */

// Base message structure
export interface WebSocketMessage<T = unknown> {
  type: string;
  data?: T;
}

// Specific message types from server
export interface ConnectionMessage extends WebSocketMessage<{
  id: number;
  message: string;
}> {
  type: 'connection';
}

export interface MetricsUpdateMessage extends WebSocketMessage<MetricsData> {
  type: 'metrics_update';
}

export interface StatesMessage extends WebSocketMessage<Record<string, QuantumState>> {
  type: 'states';
}

export interface ErrorMessage extends WebSocketMessage<{
  message: string;
  code?: string;
}> {
  type: 'error';
}

export interface SimulationPausedMessage extends WebSocketMessage<{
  paused: boolean;
}> {
  type: 'simulation_paused';
}

export interface SimulationResumedMessage extends WebSocketMessage<{
  paused: boolean;
}> {
  type: 'simulation_resumed';
}

export interface UniverseStartedMessage extends WebSocketMessage<{
  mode: string;
  running: boolean;
  iteration_count: number;
  universe_time: number;
}> {
  type: 'universe_started';
}

export interface UniverseStatsMessage extends WebSocketMessage<any> {
  type: 'universe_stats';
}

export interface ExecutionCompleteMessage extends WebSocketMessage<{
  message?: string;
}> {
  type: 'execution_complete';
}

export interface ExecutionLogMessage extends WebSocketMessage<{
  message?: string;
  level?: string;
}> {
  type: 'execution_log' | 'log';
}

export interface TimelineEventMessage extends WebSocketMessage<{
  timestamp?: number;
  type?: string;
  description?: string;
  data?: any;
  level?: string;
  source?: string;
  duration?: number;
}> {
  type: 'timeline_event';
}

export interface UniverseModeChangedMessage extends WebSocketMessage<{
  mode: string;
  success: boolean;
}> {
  type: 'universe_mode_changed';
}

export interface UniverseParamsUpdatedMessage extends WebSocketMessage<{
  success: boolean;
  params?: Record<string, any>;
  error?: string;
}> {
  type: 'universe_params_updated';
}

// Client message types
export interface ClientMessage<T = unknown> {
  type: string;
  data?: T;
}

export interface PingMessage extends ClientMessage {
  type: 'ping';
}

export interface GetMetricsMessage extends ClientMessage {
  type: 'get_metrics';
}

export interface GetStatesMessage extends ClientMessage {
  type: 'get_states';
}

export interface PauseSimulationMessage extends ClientMessage {
  type: 'pause_simulation';
}

export interface ResumeSimulationMessage extends ClientMessage {
  type: 'resume_simulation';
}

export interface SeekSimulationMessage extends ClientMessage<{
  time: number;
}> {
  type: 'seek_simulation';
}

export interface StartUniverseMessage extends ClientMessage<{
  mode?: string;
}> {
  type: 'start_universe';
}

export interface StopUniverseMessage extends ClientMessage {
  type: 'stop_universe';
}

export interface SetUniverseModeMessage extends ClientMessage<{
  mode: string;
}> {
  type: 'set_universe_mode';
}

export interface UpdateUniverseParamsMessage extends ClientMessage<{
  params: Record<string, any>;
}> {
  type: 'update_universe_params';
}

// Union type of all server messages
export type ServerMessage = 
  | ConnectionMessage
  | MetricsUpdateMessage
  | StatesMessage
  | ErrorMessage
  | SimulationPausedMessage
  | SimulationResumedMessage
  | UniverseStartedMessage
  | UniverseStatsMessage
  | ExecutionCompleteMessage
  | ExecutionLogMessage
  | TimelineEventMessage
  | UniverseModeChangedMessage
  | UniverseParamsUpdatedMessage
  | WebSocketMessage<unknown>; // Fallback for unknown message types

// Union type of all client messages
export type ClientMessageType = 
  | PingMessage
  | GetMetricsMessage
  | GetStatesMessage
  | PauseSimulationMessage
  | ResumeSimulationMessage
  | SeekSimulationMessage
  | StartUniverseMessage
  | StopUniverseMessage
  | SetUniverseModeMessage
  | UpdateUniverseParamsMessage;

// Type guards for runtime type checking
export function isMetricsUpdate(msg: ServerMessage): msg is MetricsUpdateMessage {
  return msg.type === 'metrics_update' && msg.data !== undefined;
}

export function isStatesMessage(msg: ServerMessage): msg is StatesMessage {
  return msg.type === 'states' && msg.data !== undefined;
}

export function isErrorMessage(msg: ServerMessage): msg is ErrorMessage {
  return msg.type === 'error';
}

export function isConnectionMessage(msg: ServerMessage): msg is ConnectionMessage {
  return msg.type === 'connection';
}

export function isUniverseStartedMessage(msg: ServerMessage): msg is UniverseStartedMessage {
  return msg.type === 'universe_started';
}

export function isExecutionCompleteMessage(msg: ServerMessage): msg is ExecutionCompleteMessage {
  return msg.type === 'execution_complete';
}

export function isExecutionLogMessage(msg: ServerMessage): msg is ExecutionLogMessage {
  return msg.type === 'execution_log' || msg.type === 'log';
}

export function isTimelineEventMessage(msg: ServerMessage): msg is TimelineEventMessage {
  return msg.type === 'timeline_event';
}

// Re-export types from other files
export interface MetricsData {
  // Core OSH metrics
  rsp: number;
  coherence: number;
  entropy: number;
  information: number;
  
  // Additional metrics
  strain: number;
  phi: number;
  emergence_index: number;
  field_energy: number;
  temporal_stability: number;
  observer_influence: number;
  memory_field_coupling: number;
  
  // Dynamic universe metrics
  universe_time?: number;
  iteration_count?: number;
  num_entanglements?: number;
  
  // System status
  observer_count: number;
  state_count: number;
  recursion_depth: number;
  observer_focus: number;
  focus?: number;
  depth?: number;
  
  // Additional metrics
  information_curvature: number;
  integrated_information: number;
  complexity: number;
  entropy_flux: number;
  conservation_law: number;
  
  // Performance
  fps: number;
  error: number;
  quantum_volume: number;
  
  // Time derivatives
  drsp_dt: number;
  di_dt: number;
  dc_dt: number;
  de_dt: number;
  acceleration: number;
  
  // Conservation verification
  conservation_verified?: boolean;
  conservation_ratio?: number;
  conservation_error?: number;
  conservation_message?: string;
  
  // Memory fragments - removed
  // memory_fragments?: Array<{
  //   coherence: number;
  //   size: number;
  //   coupling_strength: number;
  //   position: [number, number, number];
  //   phase: number;
  // }>;
  
  // Resources
  resources?: {
    memory?: number;
    cpu?: number;
    gpu?: number;
    healthy?: boolean;
  };
  
  // Additional fields
  measurement_count?: number;
  universe_mode?: string;
  universe_running?: boolean;
  phase?: number;
  gravitational_anomaly?: number;
  consciousness_probability?: number;
  memory_strain?: number;
  is_paused?: boolean;
  
  timestamp: number;
}

export interface QuantumState {
  name: string;
  num_qubits: number;
  qubit_count?: number; // Alternative name for num_qubits
  state_vector?: { real: number; imag: number }[];
  measurements?: any[];
  coherence?: number; // Top-level for compatibility
  entropy?: number; // Top-level for compatibility
  type?: string; // State type
  properties: {
    coherence?: number;
    entropy?: number;
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