/**
 * Backend Metrics Synchronization Utility
 * 
 * Ensures frontend engines use metrics from the unified backend API
 * instead of calculating locally. This maintains consistency between
 * the VM calculations and frontend display.
 */

import { OSHQuantumEngine } from '../engines/OSHQuantumEngine';

export interface BackendMetrics {
  // Core OSH metrics from VM
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
  
  // Derivatives
  drsp_dt: number;
  di_dt: number;
  dc_dt: number;
  de_dt: number;
  acceleration: number;
  
  // System state
  observer_count: number;
  state_count: number;
  recursion_depth: number;
  observer_focus: number;
  
  // Conservation law
  conservation_verified?: boolean;
  conservation_ratio?: number;
  conservation_error?: number;
  
  // Memory field - removed
  // memory_fragments?: Array<{
  //   coherence: number;
  //   size: number;
  //   coupling_strength: number;
  //   position: [number, number, number];
  //   phase: number;
  // }>;
  
  // Additional fields
  integrated_information: number;
  complexity: number;
  entropy_flux: number;
  conservation_law: number;
  information_curvature: number;
  quantum_volume: number;
  error: number;
  fps: number;
  timestamp: number;
  
  // Aliases for compatibility
  focus?: number;
  depth?: number;
}

/**
 * Synchronize backend metrics with frontend engines
 * This ensures all engines use the same metric values calculated by the VM
 */
export function syncEnginesWithBackend(
  engine: OSHQuantumEngine | null,
  metrics: BackendMetrics | null
): void {
  if (!engine || !metrics) {
    return;
  }
  
  console.log('[BackendMetricsSync] Synchronizing engines with backend metrics');
  
  // Update RSP Engine
  if (engine.rspEngine?.updateFromBackend) {
    try {
      engine.rspEngine.updateFromBackend(metrics);
    } catch (e) {
      console.warn('[BackendMetricsSync] Failed to update RSP engine:', e);
    }
  }
  
  // Update Error Reduction Platform
  if (engine.errorReductionPlatform?.updateErrorRateFromSystemMetrics) {
    try {
      engine.errorReductionPlatform.updateErrorRateFromSystemMetrics(metrics);
    } catch (e) {
      console.warn('[BackendMetricsSync] Failed to update error reduction platform:', e);
    }
  }
  
  // Update Observer Engine
  if (engine.observerEngine && metrics.observer_focus !== undefined) {
    try {
      engine.observerEngine.setGlobalFocus(metrics.observer_focus);
    } catch (e) {
      console.warn('[BackendMetricsSync] Failed to update observer engine:', e);
    }
  }
  
  // Update Simulation Harness metrics
  if ((engine.simulationHarness as any)?.updateMetrics) {
    try {
      const simulationMetrics = {
        entropy: metrics.entropy,
        coherence: metrics.coherence,
        information: metrics.information,
        complexity: metrics.complexity || metrics.information,
        observer_count: metrics.observer_count,
        state_count: metrics.state_count,
        total_entanglement: metrics.phi || 0,
        average_fidelity: 1 - metrics.error,
        decoherence_rate: metrics.entropy_flux || 0,
        consciousness_emergence: metrics.emergence_index || 0
      };
      (engine.simulationHarness as any).updateMetrics(simulationMetrics);
    } catch (e) {
      console.warn('[BackendMetricsSync] Failed to update simulation harness:', e);
    }
  }
  
  // Update ML Observer
  if ((engine.mlObserver as any)?.updateMetrics) {
    try {
      (engine.mlObserver as any).updateMetrics({
        coherence: metrics.coherence,
        entropy: metrics.entropy,
        observerFocus: metrics.observer_focus,
        quantumVolume: metrics.quantum_volume
      });
    } catch (e) {
      console.warn('[BackendMetricsSync] Failed to update ML observer:', e);
    }
  }
  
  // Update Curvature Generator
  if ((engine.curvatureGenerator as any)?.updateFromMetrics) {
    try {
      (engine.curvatureGenerator as any).updateFromMetrics({
        informationDensity: metrics.information,
        curvature: metrics.information_curvature,
        strain: metrics.strain
      });
    } catch (e) {
      console.warn('[BackendMetricsSync] Failed to update curvature generator:', e);
    }
  }
  
  // Update Tensor Field Engine
  if ((engine.tensorField as any)?.updateFieldMetrics) {
    try {
      (engine.tensorField as any).updateFieldMetrics({
        fieldStrength: metrics.field_energy,
        coherence: metrics.coherence,
        entanglement: metrics.phi
      });
    } catch (e) {
      console.warn('[BackendMetricsSync] Failed to update tensor field:', e);
    }
  }
}

/**
 * Check if backend metrics are valid and complete
 */
export function areBackendMetricsValid(metrics: any): metrics is BackendMetrics {
  if (!metrics || typeof metrics !== 'object') {
    return false;
  }
  
  // Check for required core metrics
  const requiredFields = ['rsp', 'coherence', 'entropy', 'information'];
  for (const field of requiredFields) {
    if (typeof metrics[field] !== 'number' || !isFinite(metrics[field])) {
      return false;
    }
  }
  
  return true;
}

/**
 * Get display-ready metrics with fallbacks
 * Prioritizes backend metrics but provides sensible defaults
 */
export function getDisplayMetrics(
  backendMetrics: BackendMetrics | null,
  localMetrics?: Partial<BackendMetrics>
): BackendMetrics {
  // If we have valid backend metrics, use them
  if (backendMetrics && areBackendMetricsValid(backendMetrics)) {
    return backendMetrics;
  }
  
  // Otherwise, merge local metrics with defaults
  const now = Date.now();
  return {
    // Core metrics
    rsp: localMetrics?.rsp || 0,
    coherence: localMetrics?.coherence || 0.95,
    entropy: localMetrics?.entropy || 0.05,
    information: localMetrics?.information || 10,
    
    // Additional metrics
    strain: localMetrics?.strain || 0,
    phi: localMetrics?.phi || 0,
    emergence_index: localMetrics?.emergence_index || 0,
    field_energy: localMetrics?.field_energy || 1,
    temporal_stability: localMetrics?.temporal_stability || 0.99,
    observer_influence: localMetrics?.observer_influence || 0.1,
    memory_field_coupling: localMetrics?.memory_field_coupling || 0.5,
    
    // Derivatives
    drsp_dt: localMetrics?.drsp_dt || 0,
    di_dt: localMetrics?.di_dt || 0,
    dc_dt: localMetrics?.dc_dt || 0,
    de_dt: localMetrics?.de_dt || 0,
    acceleration: localMetrics?.acceleration || 0,
    
    // System state
    observer_count: localMetrics?.observer_count || 0,
    state_count: localMetrics?.state_count || 0,
    recursion_depth: localMetrics?.recursion_depth || 0,
    observer_focus: localMetrics?.observer_focus || 0,
    
    // Additional fields
    integrated_information: localMetrics?.integrated_information || 0,
    complexity: localMetrics?.complexity || 10,
    entropy_flux: localMetrics?.entropy_flux || 0.05,
    conservation_law: localMetrics?.conservation_law || 0,
    information_curvature: localMetrics?.information_curvature || 0,
    quantum_volume: localMetrics?.quantum_volume || 0,
    error: localMetrics?.error || 0.001,
    fps: localMetrics?.fps || 60,
    timestamp: localMetrics?.timestamp || now,
    
    // Aliases
    focus: localMetrics?.focus || localMetrics?.observer_focus || 0,
    depth: localMetrics?.depth || localMetrics?.recursion_depth || 0
  };
}