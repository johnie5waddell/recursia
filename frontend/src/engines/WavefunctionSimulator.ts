/**
 * Wavefunction Simulator
 * Time-evolves system using Schrödinger-style propagation over memory-indexed grid
 * Supports visualization of uncollapsed base waveform and observer memory modulation
 */

import { Complex } from '../utils/complex';
import { FFT } from '../utils/fft';
import type { MemoryField, MemoryFragment } from './MemoryFieldEngine';
import type { Observer } from './ObserverEngine';

export interface WavefunctionState {
  amplitude: Complex[]; // Flattened 3D grid for compatibility
  grid: Complex[][][]; // 3D spatial grid
  gridSize: number;
  time: number;
  totalProbability: number;
  coherenceField: number[][][];
  phaseField: number[][][];
}

export interface GridParameters {
  sizeX: number;
  sizeY: number;
  sizeZ: number;
  spacing: number;
  boundaryCondition: 'periodic' | 'fixed' | 'absorbing';
}

export interface EvolutionParameters {
  mass: number;
  hbar: number;
  potential: (x: number, y: number, z: number, t: number) => number;
  memoryModulation: boolean;
  observerInfluence: boolean;
}

export class WavefunctionSimulator {
  private grid: Complex[][][] = [];
  private gridParams: GridParameters;
  private evolutionParams: EvolutionParameters;
  private time: number = 0;
  private memoryIndexedPotential: number[][][] = [];
  private frameCount: number = 0;
  
  constructor(
    gridParams: GridParameters = {
      sizeX: 16, // Reduced from 64 for memory safety
      sizeY: 16,
      sizeZ: 16,
      spacing: 0.1,
      boundaryCondition: 'periodic'
    },
    evolutionParams: EvolutionParameters = {
      mass: 1,
      hbar: 1,
      potential: (x, y, z, t) => 0,
      memoryModulation: true,
      observerInfluence: true
    }
  ) {
    this.gridParams = gridParams;
    this.evolutionParams = evolutionParams;
    this.initializeGrid();
    this.initializeMemoryPotential();
  }

  /**
   * Initialize the wavefunction grid
   */
  private initializeGrid(): void {
    const { sizeX, sizeY, sizeZ } = this.gridParams;
    
    this.grid = Array(sizeX).fill(null).map(() =>
      Array(sizeY).fill(null).map(() =>
        Array(sizeZ).fill(null).map(() => Complex.zero())
      )
    );
  }

  /**
   * Initialize memory-indexed potential
   */
  private initializeMemoryPotential(): void {
    const { sizeX, sizeY, sizeZ } = this.gridParams;
    
    this.memoryIndexedPotential = Array(sizeX).fill(null).map(() =>
      Array(sizeY).fill(null).map(() =>
        Array(sizeZ).fill(0)
      )
    );
  }

  /**
   * Initialize quantum state with given dimensions
   */
  initializeState(dimensions: number): void {
    // Ensure grid is properly initialized before setting wavepacket
    const { sizeX, sizeY, sizeZ, spacing } = this.gridParams;
    
    // Validate grid parameters
    if (!sizeX || !sizeY || !sizeZ || sizeX <= 0 || sizeY <= 0 || sizeZ <= 0) {
      console.error('WavefunctionSimulator: Invalid grid parameters', this.gridParams);
      return;
    }
    
    // First, ensure grid exists and is properly sized
    if (!this.grid || this.grid.length !== sizeX) {
      this.initializeGrid();
    }
    
    // Set a default Gaussian wavepacket centered at grid center
    const center: [number, number, number] = [
      sizeX * spacing / 2,
      sizeY * spacing / 2,
      sizeZ * spacing / 2
    ];
    const momentum: [number, number, number] = [0, 0, 0];
    const width = spacing * 5; // Wider initial wavepacket for stability
    
    // Use a safe initialization that won't cause recursion
    this.setSafeGaussianWavepacket(center, momentum, width);
  }
  
  /**
   * Set Gaussian wavepacket with recursion protection
   */
  private setSafeGaussianWavepacket(
    center: [number, number, number],
    momentum: [number, number, number],
    width: number
  ): void {
    const { sizeX, sizeY, sizeZ, spacing } = this.gridParams;
    
    // Calculate normalization factor for 3D Gaussian
    const norm = Math.pow(2 * Math.PI * width * width, -3/4);
    
    let totalProb = 0;
    
    // First pass: calculate the wavefunction and total probability
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          const x = i * spacing;
          const y = j * spacing;
          const z = k * spacing;
          
          // Gaussian envelope
          const dx = x - center[0];
          const dy = y - center[1];
          const dz = z - center[2];
          const r2 = dx*dx + dy*dy + dz*dz;
          const gaussian = Math.exp(-r2 / (4 * width * width));
          
          // Plane wave phase
          const phase = (momentum[0] * x + momentum[1] * y + momentum[2] * z) / this.evolutionParams.hbar;
          
          const real = norm * gaussian * Math.cos(phase);
          const imag = norm * gaussian * Math.sin(phase);
          
          this.grid[i][j][k] = new Complex(real, imag);
          totalProb += real * real + imag * imag;
        }
      }
    }
    
    // Second pass: normalize if needed
    totalProb *= spacing * spacing * spacing;
    const normFactor = Math.sqrt(totalProb);
    
    if (normFactor > 1e-10 && Math.abs(normFactor - 1.0) > 1e-6) {
      for (let i = 0; i < sizeX; i++) {
        for (let j = 0; j < sizeY; j++) {
          for (let k = 0; k < sizeZ; k++) {
            this.grid[i][j][k].real /= normFactor;
            this.grid[i][j][k].imag /= normFactor;
          }
        }
      }
    }
  }

  /**
   * Set initial wavefunction (Gaussian wave packet)
   */
  setGaussianWavepacket(
    center: [number, number, number],
    momentum: [number, number, number],
    width: number
  ): void {
    const { sizeX, sizeY, sizeZ, spacing } = this.gridParams;
    
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          const x = i * spacing;
          const y = j * spacing;
          const z = k * spacing;
          
          // Gaussian envelope
          const dx = x - center[0];
          const dy = y - center[1];
          const dz = z - center[2];
          const r2 = dx*dx + dy*dy + dz*dz;
          const gaussian = Math.exp(-r2 / (4 * width * width));
          
          // Plane wave phase
          const phase = (momentum[0] * x + momentum[1] * y + momentum[2] * z) / this.evolutionParams.hbar;
          
          // Normalization factor
          const norm = Math.pow(2 * Math.PI * width * width, -3/4);
          
          this.grid[i][j][k] = new Complex(
            norm * gaussian * Math.cos(phase),
            norm * gaussian * Math.sin(phase)
          );
        }
      }
    }
    
    this.normalizeWavefunction(true); // Skip reinit to prevent recursion
  }

  /**
   * Set superposition of eigenstates
   */
  setSuperpositionState(
    coefficients: Complex[],
    eigenstates: Complex[][][][]
  ): void {
    const { sizeX, sizeY, sizeZ } = this.gridParams;
    
    // Clear grid
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          this.grid[i][j][k] = Complex.zero();
        }
      }
    }
    
    // Add weighted eigenstates
    coefficients.forEach((coeff, n) => {
      if (n < eigenstates.length) {
        for (let i = 0; i < sizeX; i++) {
          for (let j = 0; j < sizeY; j++) {
            for (let k = 0; k < sizeZ; k++) {
              const eigenValue = eigenstates[n][i][j][k];
              const contribution = coeff.multiply(eigenValue);
              this.grid[i][j][k] = this.grid[i][j][k].add(contribution);
            }
          }
        }
      }
    });
    
    this.normalizeWavefunction();
  }

  /**
   * Time evolution using split-operator method
   */
  evolve(deltaTime: number, memoryField?: MemoryField, observers?: Observer[]): void {
    // For performance, reduce computational load
    const reducedDeltaTime = Math.min(deltaTime, 0.01); // Cap delta time
    
    // Update memory-indexed potential if provided (only every few frames)
    if (memoryField && this.evolutionParams.memoryModulation && this.frameCount % 5 === 0) {
      this.updateMemoryPotential(memoryField);
    }
    
    // Apply observer influence if enabled (only every few frames)
    if (observers && this.evolutionParams.observerInfluence && this.frameCount % 3 === 0) {
      this.applyObserverModulation(observers);
    }
    
    // Simplified evolution for performance
    // Only do full split-operator every few frames
    if (this.frameCount % 2 === 0) {
      // Full split-operator method: U(dt) = exp(-iV*dt/2ℏ) * exp(-iT*dt/ℏ) * exp(-iV*dt/2ℏ)
      
      // Step 1: Apply potential operator for dt/2
      this.applyPotentialOperator(reducedDeltaTime / 2);
      
      // Step 2: Apply kinetic operator (in momentum space)
      this.applyKineticOperator(reducedDeltaTime);
      
      // Step 3: Apply potential operator for dt/2
      this.applyPotentialOperator(reducedDeltaTime / 2);
    } else {
      // Simplified evolution - just potential operator
      this.applyPotentialOperator(reducedDeltaTime);
    }
    
    // Update time
    this.time += reducedDeltaTime;
    this.frameCount++;
    
    // Ensure normalization (only every few frames)
    if (this.frameCount % 10 === 0) {
      this.normalizeWavefunction();
    }
  }

  /**
   * Apply potential operator exp(-iV*dt/ℏ)
   */
  private applyPotentialOperator(deltaTime: number): void {
    const { sizeX, sizeY, sizeZ, spacing } = this.gridParams;
    const { hbar, potential } = this.evolutionParams;
    
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          const x = i * spacing;
          const y = j * spacing;
          const z = k * spacing;
          
          // Total potential = external + memory-indexed
          const V = potential(x, y, z, this.time) + this.memoryIndexedPotential[i][j][k];
          
          // Phase rotation
          const phase = -V * deltaTime / hbar;
          const cos_phase = Math.cos(phase);
          const sin_phase = Math.sin(phase);
          
          const oldReal = this.grid[i][j][k].real;
          const oldImag = this.grid[i][j][k].imag;
          
          this.grid[i][j][k].real = oldReal * cos_phase - oldImag * sin_phase;
          this.grid[i][j][k].imag = oldReal * sin_phase + oldImag * cos_phase;
        }
      }
    }
  }

  /**
   * Apply kinetic operator using FFT
   */
  private applyKineticOperator(deltaTime: number): void {
    const { sizeX, sizeY, sizeZ, spacing } = this.gridParams;
    const { mass, hbar } = this.evolutionParams;
    
    // Skip FFT for small time steps or use simplified version
    if (deltaTime < 0.001 || this.frameCount % 4 !== 0) {
      // Simplified kinetic operator - approximate with local finite differences
      this.applySimplifiedKineticOperator(deltaTime);
      return;
    }
    
    try {
      // Transform to momentum space
      const momentumGrid = this.fft3D(this.grid);
      
      // Apply kinetic energy operator in momentum space
      const dk = 2 * Math.PI / (spacing * sizeX); // Assume cubic grid
      
      for (let i = 0; i < sizeX; i++) {
        for (let j = 0; j < sizeY; j++) {
          for (let k = 0; k < sizeZ; k++) {
            // Momentum components (centered)
            const kx = (i < sizeX/2 ? i : i - sizeX) * dk;
            const ky = (j < sizeY/2 ? j : j - sizeY) * dk;
            const kz = (k < sizeZ/2 ? k : k - sizeZ) * dk;
            
            // Kinetic energy
            const E_kinetic = (hbar * hbar * (kx*kx + ky*ky + kz*kz)) / (2 * mass);
            
            // Phase rotation
            const phase = -E_kinetic * deltaTime / hbar;
            const cos_phase = Math.cos(phase);
            const sin_phase = Math.sin(phase);
            
            const oldReal = momentumGrid[i][j][k].real;
            const oldImag = momentumGrid[i][j][k].imag;
            
            momentumGrid[i][j][k].real = oldReal * cos_phase - oldImag * sin_phase;
            momentumGrid[i][j][k].imag = oldReal * sin_phase + oldImag * cos_phase;
          }
        }
      }
      
      // Transform back to position space
      this.grid = this.ifft3D(momentumGrid);
    } catch (error) {
      console.warn('WavefunctionSimulator: FFT failed, using simplified kinetic operator', error);
      this.applySimplifiedKineticOperator(deltaTime);
    }
  }
  
  /**
   * Simplified kinetic operator for performance
   */
  private applySimplifiedKineticOperator(deltaTime: number): void {
    // Simple diffusion-like operator as approximation
    const { sizeX, sizeY, sizeZ } = this.gridParams;
    const { mass, hbar } = this.evolutionParams;
    const diffusionCoeff = hbar / (2 * mass) * deltaTime;
    
    // Apply simple smoothing as kinetic energy approximation
    const tempGrid = this.copyGrid(this.grid);
    
    for (let i = 1; i < sizeX - 1; i++) {
      for (let j = 1; j < sizeY - 1; j++) {
        for (let k = 1; k < sizeZ - 1; k++) {
          // Average with neighbors (simplified Laplacian)
          const neighbors = [
            tempGrid[i-1][j][k], tempGrid[i+1][j][k],
            tempGrid[i][j-1][k], tempGrid[i][j+1][k],
            tempGrid[i][j][k-1], tempGrid[i][j][k+1]
          ];
          
          let avgReal = tempGrid[i][j][k].real;
          let avgImag = tempGrid[i][j][k].imag;
          
          neighbors.forEach(neighbor => {
            avgReal += diffusionCoeff * (neighbor.real - tempGrid[i][j][k].real) / 6;
            avgImag += diffusionCoeff * (neighbor.imag - tempGrid[i][j][k].imag) / 6;
          });
          
          this.grid[i][j][k] = new Complex(avgReal, avgImag);
        }
      }
    }
  }

  /**
   * Update memory-indexed potential based on memory field
   */
  private updateMemoryPotential(memoryField: MemoryField): void {
    const { sizeX, sizeY, sizeZ, spacing } = this.gridParams;
    
    // Clear potential
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          this.memoryIndexedPotential[i][j][k] = 0;
        }
      }
    }
    
    // Add contribution from each memory fragment
    memoryField.fragments.forEach(fragment => {
      const [fx, fy, fz] = fragment.position;
      
      // Map to grid indices
      const cx = Math.floor(fx / spacing);
      const cy = Math.floor(fy / spacing);
      const cz = Math.floor(fz / spacing);
      
      // Add Gaussian potential well/barrier based on coherence
      const strength = (fragment.coherence - 0.5) * 10; // Positive for high coherence
      const width = 5; // Grid points
      
      for (let i = Math.max(0, cx - width); i < Math.min(sizeX, cx + width); i++) {
        for (let j = Math.max(0, cy - width); j < Math.min(sizeY, cy + width); j++) {
          for (let k = Math.max(0, cz - width); k < Math.min(sizeZ, cz + width); k++) {
            const r2 = (i-cx)*(i-cx) + (j-cy)*(j-cy) + (k-cz)*(k-cz);
            this.memoryIndexedPotential[i][j][k] += strength * Math.exp(-r2 / (width*width));
          }
        }
      }
    });
    
    // Add strain field contribution
    const strainScale = 0.1;
    for (let i = 0; i < Math.min(sizeX, memoryField.strainTensor.length); i++) {
      for (let j = 0; j < Math.min(sizeY, memoryField.strainTensor[0].length); j++) {
        for (let k = 0; k < Math.min(sizeZ, memoryField.strainTensor[0][0].length); k++) {
          this.memoryIndexedPotential[i][j][k] += strainScale * memoryField.strainTensor[i][j][k];
        }
      }
    }
  }

  /**
   * Apply observer modulation to wavefunction
   */
  private applyObserverModulation(observers: Observer[]): void {
    const { sizeX, sizeY, sizeZ, spacing } = this.gridParams;
    
    // Ensure observers is an array
    if (!Array.isArray(observers)) {
      return;
    }
    
    observers.forEach(observer => {
      // Validate observer has required properties
      if (!observer || !observer.focus || !Array.isArray(observer.focus) || observer.focus.length !== 3) {
        console.debug('Invalid observer structure:', observer);
        return;
      }
      
      const [ox, oy, oz] = observer.focus;
      
      // Map to grid indices
      const cx = Math.floor(ox / spacing);
      const cy = Math.floor(oy / spacing);
      const cz = Math.floor(oz / spacing);
      
      // Observer creates local coherence enhancement
      const influence = observer.coherence;
      const radius = 10;
      
      for (let i = Math.max(0, cx - radius); i < Math.min(sizeX, cx + radius); i++) {
        for (let j = Math.max(0, cy - radius); j < Math.min(sizeY, cy + radius); j++) {
          for (let k = Math.max(0, cz - radius); k < Math.min(sizeZ, cz + radius); k++) {
            const r = Math.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy) + (k-cz)*(k-cz));
            if (r < radius) {
              // Phase modulation based on observer phase
              const phaseShift = observer.phase * (1 - r/radius) * influence;
              const cos_phase = Math.cos(phaseShift);
              const sin_phase = Math.sin(phaseShift);
              
              const oldReal = this.grid[i][j][k].real;
              const oldImag = this.grid[i][j][k].imag;
              
              this.grid[i][j][k].real = oldReal * cos_phase - oldImag * sin_phase;
              this.grid[i][j][k].imag = oldReal * sin_phase + oldImag * cos_phase;
            }
          }
        }
      }
    });
  }

  /**
   * Normalize wavefunction
   */
  private normalizeWavefunction(skipReinit: boolean = false): void {
    const totalProb = this.calculateTotalProbability();
    const norm = Math.sqrt(totalProb);
    
    if (norm > 1e-10) { // Use small threshold instead of zero
      const { sizeX, sizeY, sizeZ } = this.gridParams;
      for (let i = 0; i < sizeX; i++) {
        for (let j = 0; j < sizeY; j++) {
          for (let k = 0; k < sizeZ; k++) {
            this.grid[i][j][k].real /= norm;
            this.grid[i][j][k].imag /= norm;
          }
        }
      }
    } else if (!skipReinit) {
      // If wavefunction is essentially zero and we're not skipping reinitialization
      console.debug('WavefunctionSimulator: Wavefunction norm is near zero, reinitializing');
      const { sizeX, sizeY, sizeZ, spacing } = this.gridParams;
      
      // Validate grid parameters first
      if (!sizeX || !sizeY || !sizeZ || sizeX <= 0 || sizeY <= 0 || sizeZ <= 0) {
        console.error('WavefunctionSimulator: Cannot reinitialize - invalid grid parameters');
        return;
      }
      
      // Re-initialize with a stable Gaussian wavepacket
      const center: [number, number, number] = [
        sizeX * spacing / 2,
        sizeY * spacing / 2,
        sizeZ * spacing / 2
      ];
      
      // Use setSafeGaussianWavepacket to avoid recursion
      this.setSafeGaussianWavepacket(center, [0, 0, 0], spacing * 3);
    }
  }

  /**
   * Calculate total probability
   */
  calculateTotalProbability(): number {
    const { sizeX, sizeY, sizeZ, spacing } = this.gridParams;
    let total = 0;
    
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          const psi = this.grid[i][j][k];
          total += (psi.real * psi.real + psi.imag * psi.imag);
        }
      }
    }
    
    return total * spacing * spacing * spacing;
  }

  /**
   * Get probability density at a point
   */
  getProbabilityDensity(x: number, y: number, z: number): number {
    const { spacing } = this.gridParams;
    const i = Math.floor(x / spacing);
    const j = Math.floor(y / spacing);
    const k = Math.floor(z / spacing);
    
    if (this.isValidIndex(i, j, k)) {
      const psi = this.grid[i][j][k];
      return psi.real * psi.real + psi.imag * psi.imag;
    }
    
    return 0;
  }

  /**
   * Get current wavefunction state
   */
  getState(): WavefunctionState {
    const { sizeX, sizeY, sizeZ } = this.gridParams;
    
    // Calculate coherence field
    const coherenceField = Array(sizeX).fill(null).map(() =>
      Array(sizeY).fill(null).map(() =>
        Array(sizeZ).fill(0)
      )
    );
    
    // Calculate phase field
    const phaseField = Array(sizeX).fill(null).map(() =>
      Array(sizeY).fill(null).map(() =>
        Array(sizeZ).fill(0)
      )
    );
    
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          const psi = this.grid[i][j][k];
          const magnitude = Math.sqrt(psi.real * psi.real + psi.imag * psi.imag);
          
          coherenceField[i][j][k] = magnitude;
          phaseField[i][j][k] = Math.atan2(psi.imag, psi.real);
        }
      }
    }
    
    // Create flattened amplitude array for compatibility
    const amplitude: Complex[] = [];
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          amplitude.push(this.grid[i][j][k]);
        }
      }
    }

    return {
      amplitude,
      grid: this.grid,
      gridSize: sizeX, // Assume cubic grid
      time: this.time,
      totalProbability: this.calculateTotalProbability(),
      coherenceField,
      phaseField
    };
  }

  /**
   * Get slice of wavefunction for 2D visualization
   */
  getSlice(
    axis: 'x' | 'y' | 'z',
    position: number
  ): { real: number[][]; imag: number[][]; probability: number[][] } {
    const { sizeX, sizeY, sizeZ, spacing } = this.gridParams;
    const index = Math.floor(position / spacing);
    
    let real: number[][];
    let imag: number[][];
    let probability: number[][];
    
    switch (axis) {
      case 'x':
        if (index < 0 || index >= sizeX) {
          return { real: [], imag: [], probability: [] };
        }
        real = Array(sizeY).fill(null).map(() => Array(sizeZ).fill(0));
        imag = Array(sizeY).fill(null).map(() => Array(sizeZ).fill(0));
        probability = Array(sizeY).fill(null).map(() => Array(sizeZ).fill(0));
        
        for (let j = 0; j < sizeY; j++) {
          for (let k = 0; k < sizeZ; k++) {
            const psi = this.grid[index][j][k];
            real[j][k] = psi.real;
            imag[j][k] = psi.imag;
            probability[j][k] = psi.real * psi.real + psi.imag * psi.imag;
          }
        }
        break;
        
      case 'y':
        if (index < 0 || index >= sizeY) {
          return { real: [], imag: [], probability: [] };
        }
        real = Array(sizeX).fill(null).map(() => Array(sizeZ).fill(0));
        imag = Array(sizeX).fill(null).map(() => Array(sizeZ).fill(0));
        probability = Array(sizeX).fill(null).map(() => Array(sizeZ).fill(0));
        
        for (let i = 0; i < sizeX; i++) {
          for (let k = 0; k < sizeZ; k++) {
            const psi = this.grid[i][index][k];
            real[i][k] = psi.real;
            imag[i][k] = psi.imag;
            probability[i][k] = psi.real * psi.real + psi.imag * psi.imag;
          }
        }
        break;
        
      case 'z':
        if (index < 0 || index >= sizeZ) {
          return { real: [], imag: [], probability: [] };
        }
        real = Array(sizeX).fill(null).map(() => Array(sizeY).fill(0));
        imag = Array(sizeX).fill(null).map(() => Array(sizeY).fill(0));
        probability = Array(sizeX).fill(null).map(() => Array(sizeY).fill(0));
        
        for (let i = 0; i < sizeX; i++) {
          for (let j = 0; j < sizeY; j++) {
            const psi = this.grid[i][j][index];
            real[i][j] = psi.real;
            imag[i][j] = psi.imag;
            probability[i][j] = psi.real * psi.real + psi.imag * psi.imag;
          }
        }
        break;
    }
    
    return { real, imag, probability };
  }

  /**
   * Helper methods
   */
  
  private isValidIndex(i: number, j: number, k: number): boolean {
    const { sizeX, sizeY, sizeZ } = this.gridParams;
    return i >= 0 && i < sizeX && j >= 0 && j < sizeY && k >= 0 && k < sizeZ;
  }

  /**
   * 3D FFT using proper Cooley-Tukey algorithm
   */
  private fft3D(grid: Complex[][][]): Complex[][][] {
    // Apply boundary conditions first
    const gridCopy = this.copyGrid(grid);
    this.applyBoundaryConditions(gridCopy);
    
    try {
      return FFT.fft3D(gridCopy);
    } catch (error) {
      // Fallback to padding if grid size is not power of 2
      return this.fft3DWithPadding(gridCopy);
    }
  }

  private ifft3D(grid: Complex[][][]): Complex[][][] {
    try {
      return FFT.ifft3D(grid);
    } catch (error) {
      // Fallback implementation
      return this.ifft3DWithPadding(grid);
    }
  }

  private copyGrid(grid: Complex[][][]): Complex[][][] {
    return grid.map(plane => 
      plane.map(row => 
        row.map(c => c.clone())
      )
    );
  }

  private fft3DWithPadding(grid: Complex[][][]): Complex[][][] {
    const { sizeX, sizeY, sizeZ } = this.gridParams;
    
    // Find next power of 2 for each dimension
    const paddedX = 1 << Math.ceil(Math.log2(sizeX));
    const paddedY = 1 << Math.ceil(Math.log2(sizeY));
    const paddedZ = 1 << Math.ceil(Math.log2(sizeZ));
    
    // Create padded grid
    const paddedGrid: Complex[][][] = Array(paddedX).fill(null).map(() =>
      Array(paddedY).fill(null).map(() =>
        Array(paddedZ).fill(null).map(() => Complex.zero())
      )
    );
    
    // Copy original data
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          paddedGrid[i][j][k] = grid[i][j][k];
        }
      }
    }
    
    // Transform padded grid
    const transformed = FFT.fft3D(paddedGrid);
    
    // Extract original size
    const result: Complex[][][] = Array(sizeX).fill(null).map(() =>
      Array(sizeY).fill(null).map(() =>
        Array(sizeZ).fill(null).map(() => Complex.zero())
      )
    );
    
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          result[i][j][k] = transformed[i][j][k];
        }
      }
    }
    
    return result;
  }

  private ifft3DWithPadding(grid: Complex[][][]): Complex[][][] {
    // Similar padding approach for inverse FFT
    const { sizeX, sizeY, sizeZ } = this.gridParams;
    
    const paddedX = 1 << Math.ceil(Math.log2(sizeX));
    const paddedY = 1 << Math.ceil(Math.log2(sizeY));
    const paddedZ = 1 << Math.ceil(Math.log2(sizeZ));
    
    const paddedGrid: Complex[][][] = Array(paddedX).fill(null).map(() =>
      Array(paddedY).fill(null).map(() =>
        Array(paddedZ).fill(null).map(() => Complex.zero())
      )
    );
    
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          paddedGrid[i][j][k] = grid[i][j][k];
        }
      }
    }
    
    const transformed = FFT.ifft3D(paddedGrid);
    
    const result: Complex[][][] = Array(sizeX).fill(null).map(() =>
      Array(sizeY).fill(null).map(() =>
        Array(sizeZ).fill(null).map(() => Complex.zero())
      )
    );
    
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          result[i][j][k] = transformed[i][j][k];
        }
      }
    }
    
    return result;
  }

  private applyBoundaryConditions(grid: Complex[][][]): void {
    const { sizeX, sizeY, sizeZ, boundaryCondition } = this.gridParams;
    
    switch (boundaryCondition) {
      case 'fixed':
        // Zero at boundaries
        for (let i = 0; i < sizeX; i++) {
          for (let j = 0; j < sizeY; j++) {
            grid[i][j][0] = new Complex(0, 0);
            grid[i][j][sizeZ - 1] = new Complex(0, 0);
          }
        }
        for (let i = 0; i < sizeX; i++) {
          for (let k = 0; k < sizeZ; k++) {
            grid[i][0][k] = new Complex(0, 0);
            grid[i][sizeY - 1][k] = new Complex(0, 0);
          }
        }
        for (let j = 0; j < sizeY; j++) {
          for (let k = 0; k < sizeZ; k++) {
            grid[0][j][k] = new Complex(0, 0);
            grid[sizeX - 1][j][k] = new Complex(0, 0);
          }
        }
        break;
        
      case 'periodic':
        // Already handled by FFT
        break;
        
      case 'absorbing':
        // Gradual absorption at boundaries
        const absorbWidth = 5;
        for (let i = 0; i < sizeX; i++) {
          for (let j = 0; j < sizeY; j++) {
            for (let k = 0; k < sizeZ; k++) {
              let factor = 1.0;
              
              // X boundaries
              if (i < absorbWidth) factor *= i / absorbWidth;
              if (i >= sizeX - absorbWidth) factor *= (sizeX - 1 - i) / absorbWidth;
              
              // Y boundaries
              if (j < absorbWidth) factor *= j / absorbWidth;
              if (j >= sizeY - absorbWidth) factor *= (sizeY - 1 - j) / absorbWidth;
              
              // Z boundaries
              if (k < absorbWidth) factor *= k / absorbWidth;
              if (k >= sizeZ - absorbWidth) factor *= (sizeZ - 1 - k) / absorbWidth;
              
              grid[i][j][k].real *= factor;
              grid[i][j][k].imag *= factor;
            }
          }
        }
        break;
    }
  }

  /**
   * Extract possible collapse outcomes
   */
  extractCollapseOutcomes(numOutcomes: number): Complex[][] {
    // Simplified: return slices of the current wavefunction
    const outcomes: Complex[][] = [];
    const { sizeX, sizeY, sizeZ } = this.gridParams;
    const totalSize = sizeX * sizeY * sizeZ;
    
    for (let n = 0; n < numOutcomes; n++) {
      const outcome: Complex[] = [];
      
      // Extract a 1D representation focusing on different regions
      const offset = Math.floor(n * totalSize / numOutcomes);
      
      for (let idx = 0; idx < Math.min(100, totalSize); idx++) {
        const globalIdx = (offset + idx * 7) % totalSize;
        const i = Math.floor(globalIdx / (sizeY * sizeZ));
        const j = Math.floor((globalIdx % (sizeY * sizeZ)) / sizeZ);
        const k = globalIdx % sizeZ;
        
        outcome.push(this.grid[i][j][k].clone());
      }
      
      outcomes.push(outcome);
    }
    
    return outcomes;
  }

  /**
   * Additional methods for compatibility with existing codebase
   */
  
  // Remove duplicate method - already implemented above

  propagate(deltaTime: number, potential?: Complex[]): void {
    // If potential array is provided, convert to function
    if (potential) {
      const { sizeX, sizeY, sizeZ } = this.gridParams;
      const originalPotential = this.evolutionParams.potential;
      
      this.evolutionParams.potential = (x, y, z, t) => {
        const i = Math.floor(x / this.gridParams.spacing);
        const j = Math.floor(y / this.gridParams.spacing);
        const k = Math.floor(z / this.gridParams.spacing);
        const index = i * sizeY * sizeZ + j * sizeZ + k;
        
        if (index >= 0 && index < potential.length) {
          return potential[index].real; // Use real part as potential
        }
        return originalPotential(x, y, z, t);
      };
    }
    
    this.evolve(deltaTime);
    
    // Restore original potential if it was modified
    if (potential) {
      // The potential was temporarily modified, restore if needed
    }
  }

  setState(state: { amplitude: Complex[]; gridSize: number }): void {
    const { sizeX, sizeY, sizeZ } = this.gridParams;
    
    // Validate input
    if (state.amplitude.length !== sizeX * sizeY * sizeZ) {
      throw new Error(`State amplitude length ${state.amplitude.length} doesn't match grid size ${sizeX * sizeY * sizeZ}`);
    }
    
    // Convert flattened array back to 3D grid
    let index = 0;
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        for (let k = 0; k < sizeZ; k++) {
          this.grid[i][j][k] = state.amplitude[index++];
        }
      }
    }
    
    this.normalizeWavefunction();
  }

  /**
   * Get current time
   */
  getTime(): number {
    return this.time;
  }

  /**
   * Reset simulator to initial state
   */
  reset(): void {
    this.time = 0;
    this.initializeGrid();
    this.initializeMemoryPotential();
  }

  /**
   * Get grid parameters
   */
  getGridParams(): GridParameters {
    return { ...this.gridParams };
  }

  /**
   * Get evolution parameters
   */
  getEvolutionParams(): EvolutionParameters {
    return { ...this.evolutionParams };
  }

  /**
   * Set evolution parameters
   */
  setEvolutionParams(params: Partial<EvolutionParameters>): void {
    this.evolutionParams = { ...this.evolutionParams, ...params };
  }
}