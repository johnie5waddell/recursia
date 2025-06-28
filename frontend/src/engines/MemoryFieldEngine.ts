/**
 * Memory Field Engine - Core OSH Implementation
 * Maintains recursive memory history M(t) with fragment merging,
 * coherence wave propagation, and defragmentation
 */

import { Complex } from '../utils/complex';

export interface MemoryFragment {
  id: string;
  position: [number, number, number];
  state: Complex[];
  coherence: number;
  timestamp: number;
  entropy?: number;
  strain?: number;
  parentFragments?: string[];
  childFragments?: string[];
  metadata?: {
    originalName?: string;
    connections?: number;
    size?: number;
    couplingStrength?: number;
  };
}

export interface MemoryField {
  fragments: MemoryFragment[];
  totalCoherence: number;
  averageCoherence: number;
  lastDefragmentation: number;
  coherenceMatrix?: number[][];
  strainTensor?: number[][][];
  totalEntropy?: number;
  timeIndex?: number;
}

export interface CoherenceWave {
  origin: [number, number, number];
  amplitude: number;
  frequency: number;
  phase: number;
  timestamp: number;
}

export class MemoryFieldEngine {
  private currentField!: MemoryField; // Will be initialized in constructor
  private memoryHistory: MemoryField[] = [];
  private activeWaves: CoherenceWave[] = [];
  private fragmentationThreshold = 0.3;
  private mergeThreshold = 0.8;
  private maxHistoryDepth = 100; // Reduced from 1000 to prevent excessive memory usage
  private coherenceDecayRate = 0.01;
  private readonly MAX_FRAGMENTS = 25000; // Hard limit to prevent memory issues
  private readonly SAFE_FRAGMENT_LIMIT = 1000; // Trigger cleanup above this
  private readonly MIN_FRAGMENT_LIMIT = 100; // Minimum to keep after aggressive cleanup
  
  constructor() {
    this.initializeField();
  }

  /**
   * Initialize empty memory field
   */
  private initializeField(): void {
    // Create initial fragments for a non-zero field
    const initialFragments: MemoryFragment[] = [];
    
    // Add a few seed fragments to ensure non-zero entropy
    const seedCount = Math.min(3, this.MIN_FRAGMENT_LIMIT / 10); // Ensure we don't start with too many
    for (let i = 0; i < seedCount; i++) {
      const position: [number, number, number] = [
        Math.random() * 10 - 5,
        Math.random() * 10 - 5,
        Math.random() * 10 - 5
      ];
      
      // Create simple initial state
      const state = [new Complex(0.5, 0.5), new Complex(0.5, -0.5)];
      
      initialFragments.push({
        id: `init_frag_${i}`,
        position,
        state,
        coherence: 0.5,
        timestamp: Date.now(),
        entropy: 0.5,
        strain: 0,
        parentFragments: [],
        childFragments: []
      });
    }
    
    this.currentField = {
      fragments: initialFragments,
      totalCoherence: 1.5,
      averageCoherence: 0.5,
      lastDefragmentation: Date.now(),
      coherenceMatrix: this.createCoherenceMatrix(16),
      strainTensor: this.createStrainTensor(8, 8, 8),
      totalEntropy: 1.5,
      timeIndex: 0
    };
    this.memoryHistory.push({ ...this.currentField });
  }

  /**
   * Get current memory field state M(t)
   */
  getField(): MemoryField {
    // Ensure we always return a valid field
    if (!this.currentField || !this.currentField.fragments) {
      this.initializeField();
    }
    return { ...this.currentField };
  }

  /**
   * Update memory field to M(t+1)
   */
  updateField(deltaTime: number): MemoryField {
    // Create new field state
    const newField: MemoryField = {
      fragments: [...this.currentField.fragments],
      totalCoherence: 0,
      averageCoherence: 0,
      lastDefragmentation: this.currentField.lastDefragmentation,
      coherenceMatrix: this.evolveCoherenceMatrix(this.currentField.coherenceMatrix || [], deltaTime),
      strainTensor: this.evolveStrainTensor(this.currentField.strainTensor || [], deltaTime),
      totalEntropy: 0,
      timeIndex: (this.currentField.timeIndex || 0) + 1
    };

    // Process fragment evolution
    this.processFragmentEvolution(newField, deltaTime);
    
    // Propagate coherence waves
    this.propagateCoherenceWaves(newField, deltaTime);
    
    // Check for fragment merging
    this.checkFragmentMerging(newField);
    
    // Perform defragmentation if needed
    if (this.needsDefragmentation(newField)) {
      this.defragmentMemory(newField);
      newField.lastDefragmentation = Date.now();
    }
    
    // Calculate field statistics
    this.calculateFieldStatistics(newField);
    
    // Update current field
    this.currentField = newField;
    
    // Add to history and trim if needed
    this.memoryHistory.push({ ...newField });
    if (this.memoryHistory.length > this.maxHistoryDepth) {
      this.memoryHistory.shift();
    }
    
    return { ...newField };
  }

  /**
   * Update method for consistency with other engines
   */
  update(deltaTime: number): void {
    this.updateField(deltaTime);
    
    // Check fragment count and cleanup if needed
    if (this.currentField.fragments.length > this.SAFE_FRAGMENT_LIMIT) {
      console.warn(`[MemoryFieldEngine] Fragment count (${this.currentField.fragments.length}) exceeds safe limit, triggering cleanup`);
      this.cleanup();
    }
    
    // Periodic cleanup every 100 updates
    if (this.currentField.timeIndex && this.currentField.timeIndex % 100 === 0) {
      this.cleanup();
    }
  }

  /**
   * Add new memory fragment
   */
  addFragment(state: Complex[], position: [number, number, number]): string {
    // Check if we've hit the hard limit
    if (this.currentField.fragments.length >= this.MAX_FRAGMENTS) {
      console.error(`[MemoryFieldEngine] Fragment limit reached (${this.MAX_FRAGMENTS}). Rejecting new fragment.`);
      // Trigger aggressive cleanup
      this.aggressiveCleanup();
      return '';
    }
    
    // Trigger cleanup if we're above safe limit
    if (this.currentField.fragments.length > this.SAFE_FRAGMENT_LIMIT) {
      this.cleanup();
    }
    
    const id = `frag_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const fragment: MemoryFragment = {
      id,
      position: [...position] as [number, number, number],
      state: state.map(c => c.clone()),
      coherence: this.calculateCoherence(state),
      timestamp: Date.now(),
      entropy: this.calculateEntropy(state),
      strain: 0,
      parentFragments: [],
      childFragments: []
    };
    
    this.currentField.fragments.push(fragment);
    
    // Generate coherence wave from new fragment
    this.generateCoherenceWave(position, fragment.coherence);
    
    // Update field statistics
    this.calculateFieldStatistics(this.currentField);
    
    return id;
  }

  /**
   * Remove fragment by ID
   */
  removeFragment(id: string): boolean {
    const index = this.currentField.fragments.findIndex(f => f.id === id);
    if (index !== -1) {
      this.currentField.fragments.splice(index, 1);
      this.calculateFieldStatistics(this.currentField);
      return true;
    }
    return false;
  }

  /**
   * Process fragment evolution over time with coherence conservation
   */
  private processFragmentEvolution(field: MemoryField, deltaTime: number): void {
    // Safety check: ensure we don't have too many fragments
    if (field.fragments.length > this.MAX_FRAGMENTS) {
      console.error(`[MemoryFieldEngine] Fragment count ${field.fragments.length} exceeds maximum ${this.MAX_FRAGMENTS}`);
      this.aggressiveCleanup();
      return;
    }
    
    // Calculate total coherence before evolution (conservation principle)
    const totalCoherenceBefore = field.fragments.reduce((sum, f) => sum + f.coherence, 0);
    
    field.fragments.forEach(fragment => {
      // Modified coherence decay with quantum error correction
      const baseDecay = Math.exp(-this.coherenceDecayRate * deltaTime);
      const neighborBoost = this.calculateNeighborCoherenceBoost(fragment, field.fragments);
      const waveBoost = this.calculateWaveCoherenceBoost(fragment, field);
      
      // Apply decay with corrections
      fragment.coherence *= baseDecay * (1 + neighborBoost + waveBoost);
      
      // Update strain based on neighboring fragments
      fragment.strain = this.calculateFragmentStrain(fragment, field.fragments);
      
      // Evolve quantum state with error-correcting perturbations
      fragment.state = fragment.state.map((c, i) => {
        // Reduced noise for high-coherence fragments
        const noise = 0.001 * deltaTime * (1 - fragment.coherence);
        
        // Add error correction term that preserves phase relationships
        const errorCorrection = this.calculateErrorCorrection(fragment, i);
        
        return new Complex(
          c.real + (Math.random() - 0.5) * noise + errorCorrection.real,
          c.imag + (Math.random() - 0.5) * noise + errorCorrection.imag
        );
      });
      
      // Normalize to preserve probability
      fragment.state = Complex.normalize(fragment.state);
      
      // Recalculate coherence after evolution
      fragment.coherence = Math.max(0, this.calculateCoherence(fragment.state));
    });
    
    // Enforce coherence conservation with redistribution
    const totalCoherenceAfter = field.fragments.reduce((sum, f) => sum + f.coherence, 0);
    if (totalCoherenceAfter > 0 && totalCoherenceBefore > 0) {
      const conservationFactor = totalCoherenceBefore / totalCoherenceAfter;
      field.fragments.forEach(f => {
        f.coherence = Math.min(1, f.coherence * conservationFactor);
      });
    }
  }
  
  /**
   * Calculate coherence boost from neighboring fragments
   */
  private calculateNeighborCoherenceBoost(fragment: MemoryFragment, allFragments: MemoryFragment[]): number {
    let boost = 0;
    const neighborRadius = 2.0;
    
    allFragments.forEach(neighbor => {
      if (neighbor.id !== fragment.id) {
        const distance = this.calculateDistance(fragment.position, neighbor.position);
        if (distance < neighborRadius) {
          // Nearby high-coherence fragments provide stability
          const influence = neighbor.coherence * Math.exp(-distance / neighborRadius);
          boost += influence * 0.1;
        }
      }
    });
    
    return Math.min(0.5, boost); // Cap at 50% boost
  }
  
  /**
   * Calculate coherence boost from wave interactions
   */
  private calculateWaveCoherenceBoost(fragment: MemoryFragment, field: MemoryField): number {
    let boost = 0;
    
    this.activeWaves.forEach(wave => {
      const distance = this.calculateDistance(fragment.position, wave.origin);
      if (distance < wave.amplitude * 10) {
        // Constructive interference boosts coherence
        const phase = wave.phase - distance * 0.1;
        const interference = Math.cos(phase) * wave.amplitude;
        if (interference > 0) {
          boost += interference * 0.05;
        }
      }
    });
    
    return Math.min(0.3, boost);
  }
  
  /**
   * Calculate quantum error correction term
   */
  private calculateErrorCorrection(fragment: MemoryFragment, stateIndex: number): Complex {
    // Three-qubit phase flip code for error correction
    const state = fragment.state;
    if (stateIndex >= 0 && stateIndex < state.length - 2) {
      const a0 = state[stateIndex];
      const a1 = state[stateIndex + 1];
      const a2 = state[stateIndex + 2];
      
      // Syndrome detection
      const syndrome1 = a0.multiply(a1.conjugate()).phase();
      const syndrome2 = a1.multiply(a2.conjugate()).phase();
      
      // Error correction based on syndrome
      if (Math.abs(syndrome1) > Math.PI / 2) {
        // Phase error detected
        return new Complex(0, -syndrome1 * 0.1);
      }
    }
    
    return new Complex(0, 0);
  }

  /**
   * Propagate coherence waves through the field
   */
  private propagateCoherenceWaves(field: MemoryField, deltaTime: number): void {
    // Update existing waves
    this.activeWaves = this.activeWaves.filter(wave => {
      wave.phase += wave.frequency * deltaTime;
      wave.amplitude *= Math.exp(-0.1 * deltaTime); // Decay
      
      // Remove old waves (older than 30 seconds)
      const waveAge = Date.now() - wave.timestamp;
      if (waveAge > 30000) {
        return false;
      }
      
      return wave.amplitude > 0.01; // Remove weak waves
    });

    // Apply wave effects to fragments
    field.fragments.forEach(fragment => {
      this.activeWaves.forEach(wave => {
        const distance = this.calculateDistance(fragment.position, wave.origin);
        if (distance < 10) { // Wave interaction range
          const waveEffect = wave.amplitude * Math.cos(wave.phase - distance * 0.1);
          fragment.coherence += waveEffect * 0.01;
          fragment.coherence = Math.max(0, Math.min(1, fragment.coherence));
        }
      });
    });
    
    // Additional cleanup: limit total waves to prevent memory growth
    const maxActiveWaves = 50;
    if (this.activeWaves.length > maxActiveWaves) {
      // Sort by amplitude (keep strongest waves)
      this.activeWaves.sort((a, b) => b.amplitude - a.amplitude);
      this.activeWaves = this.activeWaves.slice(0, maxActiveWaves);
    }
  }

  /**
   * Check and perform fragment merging
   */
  private checkFragmentMerging(field: MemoryField): void {
    for (let i = 0; i < field.fragments.length; i++) {
      for (let j = i + 1; j < field.fragments.length; j++) {
        const frag1 = field.fragments[i];
        const frag2 = field.fragments[j];
        
        const distance = this.calculateDistance(frag1.position, frag2.position);
        const coherenceProduct = frag1.coherence * frag2.coherence;
        
        if (distance < 2.0 && coherenceProduct > this.mergeThreshold) {
          // Merge fragments
          const mergedFragment = this.mergeFragments(frag1, frag2);
          
          // Replace fragments with merged version
          field.fragments.splice(j, 1); // Remove second fragment first
          field.fragments.splice(i, 1, mergedFragment); // Replace first with merged
          
          // Start over since indices have changed
          return this.checkFragmentMerging(field);
        }
      }
    }
  }

  /**
   * Merge two fragments into one
   */
  private mergeFragments(frag1: MemoryFragment, frag2: MemoryFragment): MemoryFragment {
    // Calculate merged position (weighted by coherence)
    const totalCoherence = frag1.coherence + frag2.coherence;
    const weight1 = frag1.coherence / totalCoherence;
    const weight2 = frag2.coherence / totalCoherence;
    
    const mergedPosition: [number, number, number] = [
      frag1.position[0] * weight1 + frag2.position[0] * weight2,
      frag1.position[1] * weight1 + frag2.position[1] * weight2,
      frag1.position[2] * weight1 + frag2.position[2] * weight2
    ];

    // Merge quantum states using tensor product
    const mergedState = Complex.tensorProduct(frag1.state, frag2.state);
    
    return {
      id: `merged_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      position: mergedPosition,
      state: mergedState,
      coherence: Math.min(1, frag1.coherence + frag2.coherence),
      timestamp: Date.now(),
      entropy: this.calculateEntropy(mergedState),
      strain: 0,
      parentFragments: [frag1.id, frag2.id],
      childFragments: []
    };
  }

  /**
   * Check if defragmentation is needed
   */
  private needsDefragmentation(field: MemoryField): boolean {
    const timeSinceLastDefrag = Date.now() - field.lastDefragmentation;
    const fragmentCount = field.fragments.length;
    const lowCoherenceCount = field.fragments.filter(f => f.coherence < this.fragmentationThreshold).length;
    
    return timeSinceLastDefrag > 60000 || // Every minute
           fragmentCount > this.SAFE_FRAGMENT_LIMIT || // Too many fragments
           lowCoherenceCount > fragmentCount * 0.5; // Too many low-coherence fragments
  }

  /**
   * Enforce fragment limit - called automatically but can be called manually
   */
  enforceFragmentLimit(): void {
    if (!this.currentField || !this.currentField.fragments) return;
    
    const currentCount = this.currentField.fragments.length;
    
    if (currentCount > this.MAX_FRAGMENTS) {
      console.error(`[MemoryFieldEngine] Fragment count ${currentCount} exceeds maximum ${this.MAX_FRAGMENTS}, forcing aggressive cleanup`);
      this.aggressiveCleanup();
    } else if (currentCount > this.SAFE_FRAGMENT_LIMIT) {
      console.warn(`[MemoryFieldEngine] Fragment count ${currentCount} exceeds safe limit ${this.SAFE_FRAGMENT_LIMIT}, running cleanup`);
      this.cleanup();
    }
  }
  
  /**
   * Perform memory defragmentation
   */
  private defragmentMemory(field: MemoryField): void {
    // Remove very low coherence fragments
    field.fragments = field.fragments.filter(f => f.coherence > 0.1);
    
    // Consolidate nearby fragments with similar states
    const consolidated: MemoryFragment[] = [];
    const processed = new Set<string>();
    
    field.fragments.forEach(fragment => {
      if (processed.has(fragment.id)) return;
      
      const similar = field.fragments.filter(other => 
        !processed.has(other.id) &&
        this.calculateDistance(fragment.position, other.position) < 1.0 &&
        this.calculateStateOverlap(fragment.state, other.state) > 0.8
      );
      
      if (similar.length > 1) {
        // Merge similar fragments
        let mergedFragment = similar[0];
        for (let i = 1; i < similar.length; i++) {
          mergedFragment = this.mergeFragments(mergedFragment, similar[i]);
        }
        consolidated.push(mergedFragment);
        similar.forEach(f => processed.add(f.id));
      } else {
        consolidated.push(fragment);
        processed.add(fragment.id);
      }
    });
    
    field.fragments = consolidated;
  }

  /**
   * Generate coherence wave from position
   */
  private generateCoherenceWave(origin: [number, number, number], amplitude: number): void {
    // Don't generate waves with very low amplitude
    if (amplitude < 0.05) return;
    
    const wave: CoherenceWave = {
      origin: [...origin] as [number, number, number],
      amplitude: amplitude * 0.5,
      frequency: 1.0 + Math.random() * 2.0,
      phase: Math.random() * 2 * Math.PI,
      timestamp: Date.now()
    };
    
    this.activeWaves.push(wave);
    
    // More aggressive cleanup - limit number of active waves
    const maxWaves = 50;
    if (this.activeWaves.length > maxWaves) {
      // Remove oldest waves first
      this.activeWaves.sort((a, b) => b.timestamp - a.timestamp);
      this.activeWaves = this.activeWaves.slice(0, maxWaves);
    }
  }

  /**
   * Calculate field statistics
   */
  private calculateFieldStatistics(field: MemoryField): void {
    if (field.fragments.length === 0) {
      field.totalCoherence = 0;
      field.averageCoherence = 0;
      field.totalEntropy = 0;
      return;
    }
    
    field.totalCoherence = field.fragments.reduce((sum, f) => sum + f.coherence, 0);
    field.averageCoherence = field.totalCoherence / field.fragments.length;
    field.totalEntropy = field.fragments.reduce((sum, f) => sum + (f.entropy || 0), 0);
  }

  /**
   * Calculate coherence of quantum state
   */
  private calculateCoherence(state: Complex[]): number {
    if (state.length === 0) return 0;
    
    // Calculate coherence as inverse of purity
    let purity = 0;
    for (let i = 0; i < state.length; i++) {
      const prob = state[i].magnitude() * state[i].magnitude();
      purity += prob * prob;
    }
    
    return 1 - purity;
  }

  /**
   * Calculate entropy of quantum state
   */
  private calculateEntropy(state: Complex[]): number {
    if (state.length === 0) return 0;
    
    let entropy = 0;
    for (const amplitude of state) {
      const prob = amplitude.magnitude() * amplitude.magnitude();
      if (prob > 0) {
        entropy -= prob * Math.log2(prob);
      }
    }
    
    return entropy;
  }

  /**
   * Calculate fragment strain based on neighbors
   */
  private calculateFragmentStrain(fragment: MemoryFragment, allFragments: MemoryFragment[]): number {
    let strain = 0;
    
    allFragments.forEach(other => {
      if (other.id !== fragment.id) {
        const distance = this.calculateDistance(fragment.position, other.position);
        if (distance < 5.0) {
          // Strain increases with proximity and coherence difference
          const coherenceDiff = Math.abs(fragment.coherence - other.coherence);
          strain += coherenceDiff / (distance + 0.1);
        }
      }
    });
    
    return Math.min(1, strain);
  }

  /**
   * Calculate distance between two positions
   */
  private calculateDistance(pos1: [number, number, number], pos2: [number, number, number]): number {
    const dx = pos1[0] - pos2[0];
    const dy = pos1[1] - pos2[1];
    const dz = pos1[2] - pos2[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  /**
   * Calculate overlap between two quantum states
   */
  private calculateStateOverlap(state1: Complex[], state2: Complex[]): number {
    const minLength = Math.min(state1.length, state2.length);
    if (minLength === 0) return 0;
    
    let overlap = 0;
    for (let i = 0; i < minLength; i++) {
      const product = state1[i].conjugate().multiply(state2[i]);
      overlap += product.real; // Real part of inner product
    }
    
    return Math.abs(overlap) / minLength;
  }

  /**
   * Create coherence matrix
   */
  private createCoherenceMatrix(size: number): number[][] {
    return Array(size).fill(null).map(() => Array(size).fill(0));
  }

  /**
   * Create strain tensor
   */
  private createStrainTensor(x: number, y: number, z: number): number[][][] {
    return Array(x).fill(null).map(() =>
      Array(y).fill(null).map(() =>
        Array(z).fill(0)
      )
    );
  }

  /**
   * Evolve coherence matrix
   */
  private evolveCoherenceMatrix(matrix: number[][], deltaTime: number): number[][] {
    if (!matrix || matrix.length === 0) return this.createCoherenceMatrix(16);
    
    return matrix.map(row =>
      row.map(value => value * Math.exp(-0.01 * deltaTime))
    );
  }

  /**
   * Evolve strain tensor
   */
  private evolveStrainTensor(tensor: number[][][], deltaTime: number): number[][][] {
    if (!tensor || tensor.length === 0) return this.createStrainTensor(8, 8, 8);
    
    return tensor.map(plane =>
      plane.map(row =>
        row.map(value => value * Math.exp(-0.05 * deltaTime))
      )
    );
  }

  /**
   * Get memory history
   */
  getHistory(): MemoryField[] {
    return [...this.memoryHistory];
  }

  /**
   * Get active waves
   */
  getActiveWaves(): CoherenceWave[] {
    return [...this.activeWaves];
  }

  /**
   * Reset memory field
   */
  reset(): void {
    this.memoryHistory = [];
    this.activeWaves = [];
    this.initializeField();
  }

  /**
   * Clean up old data to prevent memory leaks
   * Should be called periodically or when memory usage is high
   */
  cleanup(): void {
    // Clean up old waves
    const now = Date.now();
    this.activeWaves = this.activeWaves.filter(wave => {
      const age = now - wave.timestamp;
      return age < 30000 && wave.amplitude > 0.01;
    });
    
    // Limit waves to max
    if (this.activeWaves.length > 50) {
      this.activeWaves.sort((a, b) => b.amplitude - a.amplitude);
      this.activeWaves = this.activeWaves.slice(0, 50);
    }
    
    // Clean up memory history
    if (this.memoryHistory.length > this.maxHistoryDepth) {
      this.memoryHistory = this.memoryHistory.slice(-this.maxHistoryDepth);
    }
    
    // Clean up low coherence fragments
    if (this.currentField && this.currentField.fragments.length > this.SAFE_FRAGMENT_LIMIT / 2) {
      // More aggressive cleanup threshold
      const coherenceThreshold = this.currentField.fragments.length > this.SAFE_FRAGMENT_LIMIT ? 0.1 : 0.05;
      const maxToKeep = Math.min(this.SAFE_FRAGMENT_LIMIT / 2, this.currentField.fragments.length);
      
      this.currentField.fragments = this.currentField.fragments
        .filter(f => f.coherence > coherenceThreshold)
        .sort((a, b) => b.coherence - a.coherence)
        .slice(0, maxToKeep);
      this.calculateFieldStatistics(this.currentField);
    }
  }
  
  /**
   * Aggressive cleanup for emergency situations
   */
  private aggressiveCleanup(): void {
    console.warn('[MemoryFieldEngine] Performing aggressive cleanup due to fragment limit');
    
    if (!this.currentField || !this.currentField.fragments) return;
    
    // Keep only the most coherent fragments
    this.currentField.fragments = this.currentField.fragments
      .sort((a, b) => b.coherence - a.coherence)
      .slice(0, this.MIN_FRAGMENT_LIMIT);
    
    // Clear waves
    this.activeWaves = this.activeWaves.slice(0, 10);
    
    // Clear most of history
    this.memoryHistory = this.memoryHistory.slice(-10);
    
    // Force defragmentation
    this.defragmentMemory(this.currentField);
    this.currentField.lastDefragmentation = Date.now();
    
    // Recalculate statistics
    this.calculateFieldStatistics(this.currentField);
    
    console.log(`[MemoryFieldEngine] Reduced fragments to ${this.currentField.fragments.length}`);
  }

  /**
   * Get fragment by ID
   */
  getFragment(id: string): MemoryFragment | undefined {
    return this.currentField.fragments.find(f => f.id === id);
  }

  /**
   * Get fragments in spatial region
   */
  getFragmentsInRegion(center: [number, number, number], radius: number): MemoryFragment[] {
    return this.currentField.fragments.filter(fragment => 
      this.calculateDistance(fragment.position, center) <= radius
    );
  }

  /**
   * Get visualization data for 3D rendering
   */
  getVisualizationData(): {
    fragments: MemoryFragment[];
    waves: CoherenceWave[];
    metrics: {
      totalCoherence: number;
      averageCoherence: number;
      fragmentCount: number;
      entropy: number;
    };
  } {
    // Log if fragment count is getting high
    if (this.currentField.fragments.length > this.SAFE_FRAGMENT_LIMIT) {
      console.warn(`[MemoryFieldEngine] High fragment count in visualization: ${this.currentField.fragments.length}`);
    }
    
    return {
      fragments: [...this.currentField.fragments],
      waves: [...this.activeWaves],
      metrics: {
        totalCoherence: this.currentField.totalCoherence,
        averageCoherence: this.currentField.averageCoherence,
        fragmentCount: this.currentField.fragments.length,
        entropy: this.currentField.totalEntropy || 0
      }
    };
  }

  /**
   * Get current field metrics
   */
  getMetrics(): {
    coherence: number;
    entropy: number;
    fragmentCount: number;
    waveCount: number;
  } {
    // Ensure field is initialized
    if (!this.currentField || !this.currentField.fragments) {
      this.initializeField();
    }
    
    // Calculate metrics with validation
    const coherence = (this.currentField?.averageCoherence !== undefined && isFinite(this.currentField.averageCoherence)) ? 
      this.currentField.averageCoherence : 0.1;
    
    const entropy = (this.currentField?.totalEntropy !== undefined && isFinite(this.currentField.totalEntropy) && this.currentField.totalEntropy > 0) ? 
      this.currentField.totalEntropy : 1.0;
    
    return {
      coherence: Math.max(coherence, 0.01), // Ensure minimum coherence
      entropy: Math.max(entropy, 0.1), // Ensure minimum entropy
      fragmentCount: this.currentField.fragments.length,
      waveCount: this.activeWaves.length
    };
  }

  /**
   * Get current field state
   */
  getCurrentField(): MemoryField {
    // Ensure we always return a valid field
    if (!this.currentField || !this.currentField.fragments) {
      this.initializeField();
    }
    return { ...this.currentField };
  }
  
  /**
   * Get current fragment count
   */
  getFragmentCount(): number {
    return this.currentField?.fragments?.length || 0;
  }

  /**
   * Set the strain value for the memory field
   * Updates strain across all fragments and the strain tensor
   * @param strain - The strain value to apply (0.0 to 1.0)
   */
  setStrain(strain: number): void {
    // Validate strain value
    const validStrain = Math.max(0, Math.min(1, strain || 0));
    
    // Update strain in all fragments
    if (this.currentField && this.currentField.fragments) {
      this.currentField.fragments.forEach(fragment => {
        fragment.strain = validStrain;
      });
    }
    
    // Update strain tensor if it exists
    if (this.currentField && this.currentField.strainTensor) {
      const strainMultiplier = 1 + validStrain;
      this.currentField.strainTensor = this.currentField.strainTensor.map(plane =>
        plane.map(row =>
          row.map(value => value * strainMultiplier)
        )
      );
    }
    
    // Trigger coherence wave if strain is high
    if (validStrain > 0.7) {
      // Use generateCoherenceWave method which has proper cleanup
      this.generateCoherenceWave([0, 0, 0], validStrain * 2);
    }
    
    // Update field coherence based on strain
    if (this.currentField) {
      const coherenceImpact = 1 - (validStrain * 0.3); // High strain reduces coherence
      this.currentField.averageCoherence *= coherenceImpact;
      this.currentField.totalCoherence = this.currentField.fragments.reduce(
        (sum, f) => sum + (f.coherence * coherenceImpact), 0
      );
    }
  }

  /**
   * Calculate strain at a specific position
   */
  calculateStrainAt(position: { x: number; y: number; z: number }): number {
    let totalStrain = 0;
    let contributingFragments = 0;
    
    // Calculate strain based on nearby fragment density and coherence variance
    this.currentField.fragments.forEach(fragment => {
      const distance = this.calculateDistance(
        fragment.position,
        [position.x, position.y, position.z]
      );
      
      if (distance < 20) { // Consider fragments within radius
        const influence = Math.exp(-distance / 10);
        const fragmentStrain = fragment.strain || 0;
        totalStrain += fragmentStrain * influence;
        contributingFragments++;
      }
    });
    
    // Normalize by contributing fragments
    return contributingFragments > 0 ? totalStrain / contributingFragments : 0;
  }
  
  /**
   * Get fragment limits
   */
  getFragmentLimits(): { current: number; safe: number; max: number; min: number } {
    return {
      current: this.currentField.fragments.length,
      safe: this.SAFE_FRAGMENT_LIMIT,
      max: this.MAX_FRAGMENTS,
      min: this.MIN_FRAGMENT_LIMIT
    };
  }

}