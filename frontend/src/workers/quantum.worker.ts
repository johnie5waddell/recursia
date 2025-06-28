/**
 * Quantum Computation Web Worker
 * Handles heavy quantum calculations off the main thread
 */

import { Complex } from '../utils/complex';
import { WavefunctionSimulator } from '../engines/WavefunctionSimulator';
import { EntropyCoherenceSolver } from '../engines/EntropyCoherenceSolver';

// Worker message types
interface WorkerRequest {
  id: string;
  type: 'evolve_wavefunction' | 'calculate_entropy' | 'calculate_coherence' | 'tensor_product';
  data: any;
}

interface WorkerResponse {
  id: string;
  type: string;
  result: any;
  error?: string;
}

// Initialize engines
const wavefunctionSim = new WavefunctionSimulator();
const entropySolver = new EntropyCoherenceSolver();

// Message handler
self.addEventListener('message', async (event: MessageEvent<WorkerRequest>) => {
  const { id, type, data } = event.data;
  
  try {
    let result: any;
    
    switch (type) {
      case 'evolve_wavefunction':
        result = await evolveWavefunction(data);
        break;
        
      case 'calculate_entropy':
        result = calculateEntropy(data.state);
        break;
        
      case 'calculate_coherence':
        result = calculateCoherence(data.state);
        break;
        
      case 'tensor_product':
        result = calculateTensorProduct(data.state1, data.state2);
        break;
        
      default:
        throw new Error(`Unknown operation: ${type}`);
    }
    
    const response: WorkerResponse = {
      id,
      type,
      result
    };
    
    self.postMessage(response);
    
  } catch (error) {
    const response: WorkerResponse = {
      id,
      type,
      result: null,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
    
    self.postMessage(response);
  }
});

/**
 * Evolve wavefunction for given time
 */
async function evolveWavefunction(data: {
  deltaTime: number;
  potential?: (x: number, y: number, z: number, t: number) => number;
}): Promise<any> {
  wavefunctionSim.propagate(data.deltaTime);
  return wavefunctionSim.getState();
}

/**
 * Calculate entropy of quantum state
 */
function calculateEntropy(state: any[]): number {
  let entropy = 0;
  
  for (const amplitude of state) {
    // Handle both Complex objects and plain objects with real/imag properties
    let mag: number;
    if (amplitude.magnitude && typeof amplitude.magnitude === 'function') {
      mag = amplitude.magnitude();
    } else if (amplitude.real !== undefined && amplitude.imag !== undefined) {
      mag = Math.sqrt(amplitude.real * amplitude.real + amplitude.imag * amplitude.imag);
    } else {
      mag = 0;
    }
    
    const prob = mag * mag;
    if (prob > 0) {
      entropy -= prob * Math.log2(prob);
    }
  }
  
  return entropy;
}

/**
 * Calculate coherence of quantum state
 */
function calculateCoherence(state: any[]): number {
  const densityMatrix = calculateDensityMatrix(state);
  let coherence = 0;
  
  // Sum off-diagonal elements
  for (let i = 0; i < densityMatrix.length; i++) {
    for (let j = 0; j < densityMatrix[i].length; j++) {
      if (i !== j) {
        const elem = densityMatrix[i][j];
        let mag: number;
        if (elem.magnitude && typeof elem.magnitude === 'function') {
          mag = elem.magnitude();
        } else if (elem.real !== undefined && elem.imag !== undefined) {
          mag = Math.sqrt(elem.real * elem.real + elem.imag * elem.imag);
        } else {
          mag = 0;
        }
        coherence += mag;
      }
    }
  }
  
  return coherence / (state.length * (state.length - 1));
}

/**
 * Calculate density matrix from state vector
 */
function calculateDensityMatrix(state: any[]): any[][] {
  const n = state.length;
  const density: any[][] = [];
  
  for (let i = 0; i < n; i++) {
    density[i] = [];
    for (let j = 0; j < n; j++) {
      const a = state[i];
      const b = state[j];
      
      // Handle complex multiplication manually
      let real1 = a.real !== undefined ? a.real : 0;
      let imag1 = a.imag !== undefined ? a.imag : 0;
      let real2 = b.real !== undefined ? b.real : 0;
      let imag2 = b.imag !== undefined ? -b.imag : 0; // conjugate
      
      // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
      density[i][j] = {
        real: real1 * real2 - imag1 * imag2,
        imag: real1 * imag2 + imag1 * real2
      };
    }
  }
  
  return density;
}

/**
 * Calculate tensor product of two states
 */
function calculateTensorProduct(state1: any[], state2: any[]): any[] {
  const result: any[] = [];
  
  for (const a1 of state1) {
    for (const a2 of state2) {
      // Handle complex multiplication manually
      let real1 = a1.real !== undefined ? a1.real : 0;
      let imag1 = a1.imag !== undefined ? a1.imag : 0;
      let real2 = a2.real !== undefined ? a2.real : 0;
      let imag2 = a2.imag !== undefined ? a2.imag : 0;
      
      // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
      result.push({
        real: real1 * real2 - imag1 * imag2,
        imag: real1 * imag2 + imag1 * real2
      });
    }
  }
  
  return result;
}

// Export for TypeScript
export type { WorkerRequest, WorkerResponse };