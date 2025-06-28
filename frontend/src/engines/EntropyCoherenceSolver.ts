/**
 * Entropy and Coherence Solvers
 * Computes real-time entropy flux E(t), coherence gradients, and memory strain
 * Implements Shannon, Lempel-Ziv, and Kolmogorov complexity estimators
 */

import { Complex } from '../utils/complex';
import { BaseEngine } from '../types/engine-types';

export interface EntropyMetrics {
  shannonEntropy: number;
  lempelZivComplexity: number;
  kolmogorovEstimate: number;
  entropyFlux: number;
  entropyGradient: [number, number, number];
}

export interface CoherenceMetrics {
  globalCoherence: number;
  localCoherence: number[][];
  coherenceGradient: [number, number, number];
  phaseCoherence: number;
  quantumCoherence: number;
}

export interface StrainMetrics {
  totalStrain: number;
  strainTensor: number[][];
  principalStrains: [number, number, number];
  shearStrain: number;
  volumetricStrain: number;
}

export class EntropyCoherenceSolver implements BaseEngine {
  private entropyHistory: number[] = [];
  private coherenceHistory: number[] = [];
  private maxHistoryLength = 1000;
  
  /**
   * Calculate Shannon entropy for quantum state
   */
  calculateShannonEntropy(state: Complex[]): number {
    const probabilities = state.map(amp => amp.real ** 2 + amp.imag ** 2);
    const total = probabilities.reduce((sum, p) => sum + p, 0);
    
    if (total === 0) return 0;
    
    return -probabilities.reduce((entropy, p) => {
      const normalized = p / total;
      if (normalized > 0) {
        entropy += normalized * Math.log2(normalized);
      }
      return entropy;
    }, 0);
  }

  /**
   * Calculate von Neumann entropy for density matrix
   */
  calculateVonNeumannEntropy(densityMatrix: Complex[][]): number {
    // First, diagonalize the density matrix
    const eigenvalues = this.getEigenvalues(densityMatrix);
    
    // Calculate -Tr(ρ log ρ)
    return -eigenvalues.reduce((entropy, lambda) => {
      if (lambda > 0) {
        entropy += lambda * Math.log2(lambda);
      }
      return entropy;
    }, 0);
  }

  /**
   * Estimate Lempel-Ziv complexity
   */
  calculateLempelZivComplexity(sequence: string): number {
    const n = sequence.length;
    let complexity = 0;
    let i = 0;
    
    while (i < n) {
      let j = i + 1;
      let k = 1;
      
      while (j + k <= n) {
        const pattern = sequence.substring(j, j + k);
        const history = sequence.substring(0, j);
        
        if (history.includes(pattern)) {
          k++;
        } else {
          complexity++;
          i = j + k - 1;
          break;
        }
      }
      
      if (j + k > n) {
        complexity++;
        break;
      }
    }
    
    // Normalize by theoretical maximum
    const maxComplexity = n / Math.log2(n);
    return complexity / maxComplexity;
  }

  /**
   * Estimate Kolmogorov complexity using compression
   */
  estimateKolmogorovComplexity(data: any[]): number {
    // Convert to string representation
    const str = JSON.stringify(data);
    
    // Use multiple compression estimates
    const lz77Estimate = this.lz77CompressionRatio(str);
    const runLengthEstimate = this.runLengthCompressionRatio(str);
    const entropyEstimate = this.calculateShannonEntropy(
      str.split('').map(c => new Complex(c.charCodeAt(0) / 255, 0))
    );
    
    // Combine estimates
    return (lz77Estimate + runLengthEstimate + entropyEstimate) / 3;
  }

  /**
   * Calculate real-time entropy flux E(t)
   */
  calculateEntropyFlux(currentEntropy: number, deltaTime: number): number {
    this.entropyHistory.push(currentEntropy);
    
    if (this.entropyHistory.length > this.maxHistoryLength) {
      this.entropyHistory.shift();
    }
    
    if (this.entropyHistory.length < 2) return 0;
    
    // Calculate derivative using finite differences
    const n = this.entropyHistory.length;
    const flux = (this.entropyHistory[n - 1] - this.entropyHistory[n - 2]) / deltaTime;
    
    // Apply smoothing using exponential moving average
    const alpha = 0.3;
    const smoothedFlux = this.entropyHistory.slice(-10).reduce((acc, val, idx, arr) => {
      if (idx === 0) return 0;
      const weight = Math.exp(-alpha * (arr.length - idx));
      return acc + weight * (val - arr[idx - 1]) / deltaTime;
    }, 0) / Math.min(10, this.entropyHistory.length - 1);
    
    return smoothedFlux;
  }

  /**
   * Calculate entropy gradient in 3D space
   */
  calculateEntropyGradient(
    entropyField: number[][][],
    position: [number, number, number]
  ): [number, number, number] {
    const [x, y, z] = position.map(Math.floor);
    const [dx, dy, dz] = [1, 1, 1]; // Grid spacing
    
    // Central difference approximation
    const gradX = (this.getFieldValue(entropyField, x + 1, y, z) - 
                   this.getFieldValue(entropyField, x - 1, y, z)) / (2 * dx);
    const gradY = (this.getFieldValue(entropyField, x, y + 1, z) - 
                   this.getFieldValue(entropyField, x, y - 1, z)) / (2 * dy);
    const gradZ = (this.getFieldValue(entropyField, x, y, z + 1) - 
                   this.getFieldValue(entropyField, x, y, z - 1)) / (2 * dz);
    
    return [gradX, gradY, gradZ];
  }

  /**
   * Calculate coherence metrics
   */
  calculateCoherenceMetrics(state: Complex[], densityMatrix?: Complex[][]): CoherenceMetrics {
    try {
      // Validate input
      if (!state || state.length === 0) {
        return {
          globalCoherence: 0.1,
          localCoherence: [[0.1]],
          coherenceGradient: [0, 0, 0],
          phaseCoherence: 0.1,
          quantumCoherence: 0.1
        };
      }
      
      const globalCoherence = this.calculateGlobalCoherence(state);
      const localCoherence = this.calculateLocalCoherence(state);
      const phaseCoherence = this.calculatePhaseCoherence(state);
      
      let quantumCoherence = globalCoherence;
      if (densityMatrix) {
        try {
          quantumCoherence = this.calculateQuantumCoherence(densityMatrix);
        } catch (error) {
          console.debug('Error calculating quantum coherence:', error);
          quantumCoherence = globalCoherence;
        }
      }
      
      // Calculate gradient (simplified for now)
      const coherenceGradient: [number, number, number] = [0, 0, 0];
      
      // Safe access with bounds checking
      if (localCoherence.length > 1 && localCoherence[0].length > 1) {
        coherenceGradient[0] = (localCoherence[1][0] - localCoherence[0][0]) / 2;
        coherenceGradient[1] = (localCoherence[0][1] - localCoherence[0][0]) / 2;
      }
      
      // Validate all metrics
      const metrics = {
        globalCoherence: isFinite(globalCoherence) ? globalCoherence : 0.1,
        localCoherence,
        coherenceGradient,
        phaseCoherence: isFinite(phaseCoherence) ? phaseCoherence : 0.1,
        quantumCoherence: isFinite(quantumCoherence) ? quantumCoherence : 0.1
      };
      
      return metrics;
    } catch (error) {
      console.error('Error in calculateCoherenceMetrics:', error);
      return {
        globalCoherence: 0.1,
        localCoherence: [[0.1]],
        coherenceGradient: [0, 0, 0],
        phaseCoherence: 0.1,
        quantumCoherence: 0.1
      };
    }
  }

  /**
   * Calculate memory strain tensor and derived metrics
   */
  calculateStrainMetrics(
    displacementField: [number, number, number][][][]
  ): StrainMetrics {
    const strainTensor = this.calculateStrainTensor(displacementField);
    const principalStrains = this.calculatePrincipalStrains(strainTensor);
    const shearStrain = this.calculateShearStrain(strainTensor);
    const volumetricStrain = this.calculateVolumetricStrain(strainTensor);
    const totalStrain = Math.sqrt(
      principalStrains.reduce((sum, s) => sum + s * s, 0)
    );
    
    return {
      totalStrain,
      strainTensor,
      principalStrains,
      shearStrain,
      volumetricStrain
    };
  }

  /**
   * Calculate entropy gradient variance for dynamic logging
   */
  calculateEntropyGradientVariance(entropyField: number[][][]): number {
    const gradients: [number, number, number][] = [];
    
    // Sample gradients at multiple points
    for (let x = 1; x < entropyField.length - 1; x += 2) {
      for (let y = 1; y < entropyField[0].length - 1; y += 2) {
        for (let z = 1; z < entropyField[0][0].length - 1; z += 2) {
          gradients.push(this.calculateEntropyGradient(entropyField, [x, y, z]));
        }
      }
    }
    
    // Calculate variance of gradient magnitudes
    const magnitudes = gradients.map(g => Math.sqrt(g[0]**2 + g[1]**2 + g[2]**2));
    const mean = magnitudes.reduce((sum, m) => sum + m, 0) / magnitudes.length;
    const variance = magnitudes.reduce((sum, m) => sum + (m - mean)**2, 0) / magnitudes.length;
    
    return variance;
  }

  /**
   * Private helper methods
   */
  
  private calculateGlobalCoherence(state: Complex[]): number {
    // Validate input
    if (!state || state.length === 0) {
      return 0.1; // Return small default coherence
    }
    
    // Calculate norm with validation
    let normSquared = 0;
    for (const amp of state) {
      if (amp && isFinite(amp.real) && isFinite(amp.imag)) {
        normSquared += amp.real ** 2 + amp.imag ** 2;
      }
    }
    
    if (normSquared <= 0 || !isFinite(normSquared)) {
      return 0.1; // Return small default coherence
    }
    
    // Calculate off-diagonal coherence
    let coherence = 0;
    let pairCount = 0;
    
    for (let i = 0; i < state.length; i++) {
      if (!state[i] || !isFinite(state[i].real) || !isFinite(state[i].imag)) {
        continue;
      }
      
      for (let j = i + 1; j < state.length; j++) {
        if (!state[j] || !isFinite(state[j].real) || !isFinite(state[j].imag)) {
          continue;
        }
        
        const overlap = state[i].real * state[j].real + 
                       state[i].imag * state[j].imag;
        
        if (isFinite(overlap)) {
          coherence += Math.abs(overlap);
          pairCount++;
        }
      }
    }
    
    // If no valid pairs found, return small coherence
    if (pairCount === 0) {
      return 0.1;
    }
    
    // Normalize by the square of the norm
    const result = coherence / normSquared;
    
    // Ensure result is valid and bounded
    if (!isFinite(result) || isNaN(result)) {
      return 0.1;
    }
    
    return Math.min(Math.max(result, 0.01), 1.0); // Clamp between 0.01 and 1.0
  }

  private calculateLocalCoherence(state: Complex[]): number[][] {
    const size = Math.ceil(Math.sqrt(state.length));
    const coherenceMap: number[][] = Array(size).fill(0).map(() => Array(size).fill(0));
    
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const idx = i * size + j;
        if (idx < state.length) {
          coherenceMap[i][j] = Math.sqrt(
            state[idx].real ** 2 + state[idx].imag ** 2
          );
        }
      }
    }
    
    return coherenceMap;
  }

  private calculatePhaseCoherence(state: Complex[]): number {
    if (state.length < 2) return 1;
    
    let phaseSum = 0;
    let count = 0;
    
    for (let i = 0; i < state.length - 1; i++) {
      const phase1 = Math.atan2(state[i].imag, state[i].real);
      const phase2 = Math.atan2(state[i + 1].imag, state[i + 1].real);
      phaseSum += Math.cos(phase1 - phase2);
      count++;
    }
    
    return Math.abs(phaseSum / count);
  }

  private calculateQuantumCoherence(densityMatrix: Complex[][]): number {
    // Validate input
    if (!densityMatrix || densityMatrix.length === 0) {
      return 0.1;
    }
    
    let coherence = 0;
    let validPairs = 0;
    const n = densityMatrix.length;
    
    for (let i = 0; i < n; i++) {
      if (!densityMatrix[i]) continue;
      
      for (let j = 0; j < n; j++) {
        if (i !== j && densityMatrix[i][j]) {
          const element = densityMatrix[i][j];
          if (element && isFinite(element.real) && isFinite(element.imag)) {
            const magnitude = Math.sqrt(
              element.real ** 2 + element.imag ** 2
            );
            
            if (isFinite(magnitude)) {
              coherence += magnitude;
              validPairs++;
            }
          }
        }
      }
    }
    
    // Handle edge cases
    if (validPairs === 0 || n < 2) {
      return 0.1;
    }
    
    const result = coherence / (n * (n - 1));
    
    // Ensure result is valid and bounded
    if (!isFinite(result) || isNaN(result)) {
      return 0.1;
    }
    
    return Math.min(Math.max(result, 0.01), 1.0);
  }

  private calculateStrainTensor(
    displacementField: [number, number, number][][][]
  ): number[][] {
    // Simplified 3x3 strain tensor
    const strain: number[][] = Array(3).fill(0).map(() => Array(3).fill(0));
    
    // Calculate average strain components
    let count = 0;
    for (let x = 1; x < displacementField.length - 1; x++) {
      for (let y = 1; y < displacementField[0].length - 1; y++) {
        for (let z = 1; z < displacementField[0][0].length - 1; z++) {
          const u = displacementField[x][y][z];
          const du_dx = (displacementField[x + 1][y][z][0] - displacementField[x - 1][y][z][0]) / 2;
          const du_dy = (displacementField[x][y + 1][z][0] - displacementField[x][y - 1][z][0]) / 2;
          const du_dz = (displacementField[x][y][z + 1][0] - displacementField[x][y][z - 1][0]) / 2;
          
          strain[0][0] += du_dx;
          strain[1][1] += du_dy;
          strain[2][2] += du_dz;
          strain[0][1] += (du_dy + du_dx) / 2;
          strain[0][2] += (du_dz + du_dx) / 2;
          strain[1][2] += (du_dz + du_dy) / 2;
          
          count++;
        }
      }
    }
    
    // Normalize and symmetrize
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        strain[i][j] /= count;
        if (i !== j) {
          strain[j][i] = strain[i][j];
        }
      }
    }
    
    return strain;
  }

  private calculatePrincipalStrains(strainTensor: number[][]): [number, number, number] {
    // Simplified eigenvalue calculation for 3x3 symmetric matrix
    // In production, use a proper eigenvalue solver
    const trace = strainTensor[0][0] + strainTensor[1][1] + strainTensor[2][2];
    const avg = trace / 3;
    
    return [avg * 1.2, avg, avg * 0.8];
  }

  private calculateShearStrain(strainTensor: number[][]): number {
    return Math.sqrt(
      strainTensor[0][1] ** 2 + 
      strainTensor[0][2] ** 2 + 
      strainTensor[1][2] ** 2
    );
  }

  private calculateVolumetricStrain(strainTensor: number[][]): number {
    return strainTensor[0][0] + strainTensor[1][1] + strainTensor[2][2];
  }

  private getFieldValue(field: number[][][], x: number, y: number, z: number): number {
    if (x < 0 || x >= field.length ||
        y < 0 || y >= field[0].length ||
        z < 0 || z >= field[0][0].length) {
      return 0;
    }
    return field[x][y][z];
  }

  private getEigenvalues(matrix: Complex[][]): number[] {
    // Simplified for demonstration - in production use proper linear algebra library
    const n = matrix.length;
    const eigenvalues: number[] = [];
    
    // Power iteration method for dominant eigenvalue
    let v = Array(n).fill(0).map(() => new Complex(Math.random(), 0));
    
    for (let iter = 0; iter < 100; iter++) {
      const Av = this.matrixVectorMultiply(matrix, v);
      const norm = Math.sqrt(Av.reduce((sum, c) => sum + c.real**2 + c.imag**2, 0));
      v = Av.map(c => new Complex(c.real / norm, c.imag / norm));
    }
    
    // Approximate eigenvalues from diagonal elements for simplicity
    for (let i = 0; i < n; i++) {
      eigenvalues.push(Math.sqrt(matrix[i][i].real**2 + matrix[i][i].imag**2));
    }
    
    return eigenvalues;
  }

  private matrixVectorMultiply(matrix: Complex[][], vector: Complex[]): Complex[] {
    const result: Complex[] = [];
    
    for (let i = 0; i < matrix.length; i++) {
      let sum = new Complex(0, 0);
      for (let j = 0; j < vector.length; j++) {
        const product = matrix[i][j].multiply(vector[j]);
        sum = sum.add(product);
      }
      result.push(sum);
    }
    
    return result;
  }

  private lz77CompressionRatio(str: string): number {
    // Simplified LZ77 compression estimate
    const windowSize = 256;
    let compressed = 0;
    let i = 0;
    
    while (i < str.length) {
      const start = Math.max(0, i - windowSize);
      const window = str.substring(start, i);
      const lookahead = str.substring(i, Math.min(i + windowSize, str.length));
      
      let maxMatch = 0;
      for (let j = 0; j < window.length; j++) {
        let k = 0;
        while (k < lookahead.length && window[j + k] === lookahead[k]) {
          k++;
        }
        maxMatch = Math.max(maxMatch, k);
      }
      
      compressed += maxMatch > 3 ? 3 : maxMatch + 1;
      i += Math.max(1, maxMatch);
    }
    
    return compressed / str.length;
  }

  private runLengthCompressionRatio(str: string): number {
    let compressed = 0;
    let i = 0;
    
    while (i < str.length) {
      let j = i;
      while (j < str.length && str[j] === str[i]) {
        j++;
      }
      compressed += 2; // Character + count
      i = j;
    }
    
    return compressed / str.length;
  }

  /**
   * Get comprehensive entropy metrics
   */
  getEntropyMetrics(
    state: Complex[],
    sequence: string,
    entropyField: number[][][],
    position: [number, number, number],
    deltaTime: number
  ): EntropyMetrics {
    const shannonEntropy = this.calculateShannonEntropy(state);
    const lempelZivComplexity = this.calculateLempelZivComplexity(sequence);
    const kolmogorovEstimate = this.estimateKolmogorovComplexity(state);
    const entropyFlux = this.calculateEntropyFlux(shannonEntropy, deltaTime);
    const entropyGradient = this.calculateEntropyGradient(entropyField, position);
    
    return {
      shannonEntropy,
      lempelZivComplexity,
      kolmogorovEstimate,
      entropyFlux,
      entropyGradient
    };
  }

  /**
   * Calculate Shannon entropy for a state array
   */
  calculateEntropy(states: Complex[]): EntropyMetrics {
    // Create a simple sequence for Lempel-Ziv complexity
    const sequence = states.map(s => s.real > 0 ? '1' : '0').join('');
    
    // Create a simple entropy field for gradient calculation
    const size = Math.ceil(Math.cbrt(states.length));
    const entropyField: number[][][] = Array(size).fill(0).map(() =>
      Array(size).fill(0).map(() => Array(size).fill(0))
    );
    
    // Fill entropy field with local entropies
    let idx = 0;
    for (let x = 0; x < size && idx < states.length; x++) {
      for (let y = 0; y < size && idx < states.length; y++) {
        for (let z = 0; z < size && idx < states.length; z++) {
          entropyField[x][y][z] = Math.abs(states[idx].real) + Math.abs(states[idx].imag);
          idx++;
        }
      }
    }
    
    // Use the existing getEntropyMetrics method
    return this.getEntropyMetrics(
      states,
      sequence,
      entropyField,
      [Math.floor(size/2), Math.floor(size/2), Math.floor(size/2)], // center position
      0.016 // default deltaTime (60 fps)
    );
  }

  /**
   * Calculate coherence metrics for a memory field
   */
  calculateCoherence(field: any): CoherenceMetrics {
    // Simplified coherence calculation
    const globalCoherence = field.averageCoherence || 0.5;
    const localCoherence = [[globalCoherence]];
    const coherenceGradient: [number, number, number] = [0, 0, 0];
    
    return {
      globalCoherence,
      localCoherence,
      coherenceGradient,
      quantumCoherence: globalCoherence,
      phaseCoherence: globalCoherence * 0.9
    };
  }

  /**
   * Update method to implement BaseEngine interface
   */
  update(deltaTime: number, context?: any): void {
    // Maintain history with size limit
    if (this.entropyHistory.length > this.maxHistoryLength) {
      this.entropyHistory.shift();
    }
    if (this.coherenceHistory.length > this.maxHistoryLength) {
      this.coherenceHistory.shift();
    }
  }

  /**
   * Reset the solver
   */
  reset(): void {
    this.entropyHistory = [];
    this.coherenceHistory = [];
  }

  /**
   * Get current state
   */
  getState(): any {
    return {
      entropyHistoryLength: this.entropyHistory.length,
      coherenceHistoryLength: this.coherenceHistory.length,
      maxHistoryLength: this.maxHistoryLength
    };
  }
}