/**
 * Extended Engine Type Definitions
 * Provides comprehensive type definitions for all engine interfaces
 */

import { Complex } from '../utils/complex';
import { MemoryField as BaseMemoryField, MemoryFragment as BaseMemoryFragment } from '../engines/MemoryFieldEngine';

/**
 * Extended MemoryField interface with additional properties
 * Used by various engines that need extra field information
 */
export interface ExtendedMemoryField extends BaseMemoryField {
  informationFlow?: number;
  coherence?: number;
  strain?: number;
  recursiveDepth?: number;
  energyDensity?: number;
  quantumPotential?: number;
}

/**
 * Extended MemoryFragment interface with additional properties
 */
export interface ExtendedMemoryFragment extends BaseMemoryFragment {
  amplitude?: number;
  phase?: number;
  couplingStrength?: number;
  wavefunction?: Complex[];
  energyLevel?: number;
}

/**
 * Engine update result interface
 */
export interface EngineUpdateResult {
  success: boolean;
  data?: any;
  error?: string;
  timestamp: number;
}

/**
 * Base engine interface that all engines should implement
 */
export interface BaseEngine {
  update(deltaTime: number, context?: any): EngineUpdateResult | void;
  reset?(): void;
  getState?(): any;
}

/**
 * Type guards for extended types
 */
export function isExtendedMemoryField(field: any): field is ExtendedMemoryField {
  return field && typeof field === 'object' && 'fragments' in field;
}

export function isExtendedMemoryFragment(fragment: any): fragment is ExtendedMemoryFragment {
  return fragment && typeof fragment === 'object' && 'id' in fragment && 'position' in fragment;
}

/**
 * Utility to convert base types to extended types
 */
export function toExtendedMemoryField(field: BaseMemoryField): ExtendedMemoryField {
  return {
    ...field,
    informationFlow: field.totalEntropy || 1,
    coherence: field.averageCoherence || 0.5,
    strain: 0.1,
    recursiveDepth: Math.min(20, field.fragments.length),
    energyDensity: field.totalCoherence || 1,
    quantumPotential: 0.5
  };
}

export function toExtendedMemoryFragment(fragment: BaseMemoryFragment): ExtendedMemoryFragment {
  return {
    ...fragment,
    amplitude: fragment.coherence || 0.5,
    phase: 0,
    couplingStrength: 0.5,
    wavefunction: fragment.state,
    energyLevel: fragment.entropy || 0.5
  };
}

/**
 * Engine orchestrator result type
 */
export interface OrchestratorResult {
  updated: boolean;
  engineResults: Map<string, EngineUpdateResult>;
  timestamp: number;
}

/**
 * Wavefunction state interface
 */
export interface WavefunctionState {
  amplitude: Complex[];
  grid: Complex[][][];
  gridSize: number;
  time: number;
  totalProbability: number;
  coherenceField?: number[][][];
  phaseField?: number[][][];
}