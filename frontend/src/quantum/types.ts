/**
 * Quantum Types Module
 * Comprehensive type definitions for quantum computing operations
 * Essential for type safety and mathematical rigor
 */

import { Complex } from '../utils/complex';

/**
 * Quantum state representation
 */
export interface QuantumState {
  amplitudes: Complex[];
  numQubits: number;
  normalization: number;
  coherence?: number;  // For gravitational wave component compatibility
  entropy?: number;    // For gravitational wave component compatibility
  energy?: number;     // For engine integration
  type?: string;       // For state classification
}

/**
 * Quantum gate definition
 */
export interface QuantumGate {
  name: string;
  matrix: Complex[][];
  qubits: number[];
  parameters?: number[];
}

/**
 * Quantum circuit
 */
export interface QuantumCircuit {
  gates: QuantumGate[];
  qubits: number;
  measurements: Measurement[];
}

/**
 * Measurement specification
 */
export interface Measurement {
  qubit: number;
  basis: 'Z' | 'X' | 'Y';
  outcome?: 0 | 1;
  probability?: number;
}

/**
 * Quantum error metrics
 */
export interface QuantumErrorMetrics {
  current_error_rate: number;
  average_fidelity: number;
  quantum_volume: number;
  coherence_time: number;
  gate_error_rates: Map<string, number>;
  readout_error: number;
  crosstalk: number[][];
}

/**
 * Quantum hardware configuration
 */
export interface QuantumHardwareConfig {
  backend: 'simulator' | 'ibmq' | 'rigetti' | 'ionq' | 'google';
  numQubits: number;
  connectivity: number[][];
  gateSet: string[];
  coherenceTime: number;
  gateTime: number;
}

/**
 * Entanglement specification
 */
export interface EntanglementSpec {
  qubits: number[];
  type: 'bell' | 'ghz' | 'w' | 'custom';
  strength: number;
}

/**
 * Quantum algorithm result
 */
export interface QuantumResult {
  state: QuantumState;
  measurements: Measurement[];
  metadata: {
    executionTime: number;
    shots: number;
    backend: string;
  };
}

/**
 * Quantum field properties
 */
export interface QuantumField {
  dimension: number;
  operators: Map<string, Complex[][]>;
  vacuumEnergy: number;
  propagator: (x: number[], y: number[]) => Complex;
}

/**
 * Topological quantum properties
 */
export interface TopologicalProperties {
  chernNumber: number;
  berryPhase: number;
  topologicalInvariant: number;
  anyonicStatistics?: 'fermionic' | 'bosonic' | 'anyonic';
}

/**
 * Quantum error correction code
 */
export interface QECCode {
  name: string;
  logicalQubits: number;
  physicalQubits: number;
  distance: number;
  threshold: number;
  syndromeTable: Map<string, string>;
}

/**
 * Quantum channel (noise model)
 */
export interface QuantumChannel {
  type: 'depolarizing' | 'amplitude_damping' | 'phase_damping' | 'custom';
  parameters: number[];
  krauzOperators: Complex[][][];
}

/**
 * Helper type for quantum operations
 */
export type QuantumOperator = Complex[][];

/**
 * Quantum register for holding qubits
 */
export interface QuantumRegister {
  qubits: number;
  size: number; // Added for compatibility (same as qubits)
  state: QuantumState;
  entanglements: Map<number, number[]>;
}

/**
 * Error correction metrics
 */
export interface ErrorCorrectionMetrics {
  errorRate: number;
  fidelity: number;
  threshold: number;
  correctedErrors: number;
  detectedErrors: number;
}

/**
 * Error syndrome for quantum error correction
 */
export interface ErrorSyndrome {
  syndrome: number[];
  errorType: 'bit_flip' | 'phase_flip' | 'both' | 'none';
  location: number[];
  weight: number;
}

/**
 * Curvature tensor for information geometry visualization
 * Represents the Ricci curvature tensor R_μν and related quantities
 * as per OSH theory (R_μν ~ α∇_μ∇_ν I)
 */
export interface CurvatureTensor {
  position: [number, number, number];  // 3D spatial position
  ricci: number[][];                   // Ricci tensor components R_μν
  scalar: number;                      // Scalar curvature R = g^μν R_μν
  information: number;                 // Information density I(x,t) in bits
  timestamp?: number;                  // Time of calculation
  fieldStrength?: number;              // Field strength magnitude
}

/**
 * Stabilizer code for quantum error correction
 */
export interface StabilizerCode {
  generators: QuantumOperator[];
  logicalOperators: {
    X: QuantumOperator[];
    Z: QuantumOperator[];
  };
  n: number; // physical qubits
  k: number; // logical qubits
  d: number; // distance
}

/**
 * Quantum operation (gate or measurement)
 */
export interface QuantumOperation {
  type: 'gate' | 'measurement' | 'reset' | 'barrier';
  targets: number[];
  qubits: number[]; // Added for compatibility
  gate?: QuantumGate;
  measurement?: Measurement;
}

/**
 * General error metrics
 */
export interface ErrorMetrics {
  totalErrors?: number;
  errorRate?: number;
  errorTypes?: Map<string, number>;
  timestamp: number;
  logicalErrorRate: number;
  physicalErrorRate: number;
  gateErrorRates: Map<string, number>;
  coherenceTime: number;
  readoutFidelity: number;
}

/**
 * Pauli matrices
 */
export const PauliMatrices = {
  I: [
    [new Complex(1, 0), new Complex(0, 0)],
    [new Complex(0, 0), new Complex(1, 0)]
  ],
  X: [
    [new Complex(0, 0), new Complex(1, 0)],
    [new Complex(1, 0), new Complex(0, 0)]
  ],
  Y: [
    [new Complex(0, 0), new Complex(0, -1)],
    [new Complex(0, 1), new Complex(0, 0)]
  ],
  Z: [
    [new Complex(1, 0), new Complex(0, 0)],
    [new Complex(0, 0), new Complex(-1, 0)]
  ]
};

/**
 * Common quantum gates
 */
export const QuantumGates = {
  H: [ // Hadamard
    [new Complex(1/Math.sqrt(2), 0), new Complex(1/Math.sqrt(2), 0)],
    [new Complex(1/Math.sqrt(2), 0), new Complex(-1/Math.sqrt(2), 0)]
  ],
  T: [ // T gate
    [new Complex(1, 0), new Complex(0, 0)],
    [new Complex(0, 0), new Complex(1/Math.sqrt(2), 1/Math.sqrt(2))]
  ],
  S: [ // S gate
    [new Complex(1, 0), new Complex(0, 0)],
    [new Complex(0, 0), new Complex(0, 1)]
  ]
};

/**
 * Quantum state utilities
 */
export class QuantumStateUtils {
  static createBellState(type: 'phi+' | 'phi-' | 'psi+' | 'psi-'): QuantumState {
    const s = 1 / Math.sqrt(2);
    let amplitudes: Complex[];
    
    switch (type) {
      case 'phi+':
        amplitudes = [new Complex(s, 0), new Complex(0, 0), new Complex(0, 0), new Complex(s, 0)];
        break;
      case 'phi-':
        amplitudes = [new Complex(s, 0), new Complex(0, 0), new Complex(0, 0), new Complex(-s, 0)];
        break;
      case 'psi+':
        amplitudes = [new Complex(0, 0), new Complex(s, 0), new Complex(s, 0), new Complex(0, 0)];
        break;
      case 'psi-':
        amplitudes = [new Complex(0, 0), new Complex(s, 0), new Complex(-s, 0), new Complex(0, 0)];
        break;
    }
    
    return {
      amplitudes,
      numQubits: 2,
      normalization: 1
    };
  }
  
  static fidelity(state1: QuantumState, state2: QuantumState): number {
    if (state1.amplitudes.length !== state2.amplitudes.length) {
      throw new Error('States must have same dimension');
    }
    
    let overlap = new Complex(0, 0);
    for (let i = 0; i < state1.amplitudes.length; i++) {
      overlap = overlap.add(
        state1.amplitudes[i].conjugate().multiply(state2.amplitudes[i])
      );
    }
    
    return overlap.magnitude() * overlap.magnitude();
  }
  
  static entropy(state: QuantumState): number {
    let entropy = 0;
    
    for (const amplitude of state.amplitudes) {
      const prob = amplitude.magnitude() * amplitude.magnitude();
      if (prob > 0) {
        entropy -= prob * Math.log2(prob);
      }
    }
    
    return entropy;
  }
}