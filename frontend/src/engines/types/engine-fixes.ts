/**
 * Type fixes and extensions for engine compatibility
 * Provides missing types and interfaces to resolve compilation errors
 */

import { Complex } from '../../utils/complex';
import { Observer as BaseObserver } from '../ObserverEngine';

/**
 * Extended Observer interface with additional properties
 * Used by engines that need position information
 */
export interface ExtendedObserver extends BaseObserver {
  position?: [number, number, number];
  coherenceInfluence?: number;
}

/**
 * Morphic field interface
 */
export interface MorphicField {
  id: string;
  resonance: number;
  morphicResonance?: number;
  field: Complex[][];
  timestamp: number;
}

/**
 * Enhanced memory fragment with decay property
 */
export interface EnhancedMemoryFragment {
  id: string;
  position: [number, number, number];
  state: Complex[];
  coherence: number;
  timestamp: number;
  entropy?: number;
  strain?: number;
  parentFragments?: string[];
  childFragments?: string[];
  decay?: number;
}

/**
 * Enhanced wavefunction state
 */
export interface EnhancedWavefunctionState {
  amplitude: Complex[];
  amplitudes?: Complex[]; // Alias for compatibility
  grid: Complex[][][];
  gridSize: number;
  time: number;
  totalProbability: number;
  coherenceField: number[][][];
  phaseField: number[][][];
  coherence?: number;
}

/**
 * Type guards
 */
export function isExtendedObserver(observer: any): observer is ExtendedObserver {
  return observer && 
         typeof observer.id === 'string' &&
         typeof observer.coherence === 'number' &&
         Array.isArray(observer.focus);
}

export function hasPosition(observer: any): observer is ExtendedObserver {
  return observer && Array.isArray(observer.position) && observer.position.length === 3;
}

/**
 * Utility functions for type conversions
 */
export function toExtendedObserver(observer: BaseObserver): ExtendedObserver {
  return {
    ...observer,
    position: observer.focus, // Use focus as position by default
    coherenceInfluence: observer.coherence * 0.5
  };
}

export function createComplexFromNumber(value: number): Complex {
  return new Complex(value, 0);
}

export function createComplexArray(real: number[], imag?: number[]): Complex[] {
  return real.map((r, i) => new Complex(r, imag?.[i] || 0));
}