/**
 * Coherence Field Locking Engine
 * Military-grade quantum coherence stabilization at macro scales
 * Enables deterministic control over high-coherence regions
 */

import { Complex } from '../utils/complex';
import { MemoryField } from './MemoryFieldEngine';
import { Observer } from './ObserverEngine';
import { ExtendedObserver, toExtendedObserver, hasPosition, createComplexFromNumber } from './types/engine-fixes';

export interface CoherenceLockConfiguration {
  lockingStrength: number; // 0-1, military spec requires >0.95
  spatialResolution: number; // Grid points per unit volume
  temporalCoherence: number; // Microseconds of lock duration
  observerDensity: number; // Observers per cubic meter
  fieldHarmonics: number[]; // Resonant frequencies for locking
}

export interface LockingField {
  id: string;
  center: [number, number, number];
  radius: number;
  coherence: number;
  stability: number;
  observers: Observer[];
  fieldTensor: Complex[][][];
  harmonicResonance: Map<number, number>;
  lockTimestamp: number;
  decayRate: number;
}

export class CoherenceFieldLockingEngine {
  private activeFields: Map<string, LockingField> = new Map();
  private fieldInteractions: Map<string, Set<string>> = new Map();
  private globalCoherenceMatrix: Complex[][];
  private lockingPotential: number = 0;
  
  constructor(private config: CoherenceLockConfiguration) {
    console.log(`[CoherenceFieldLockingEngine] Constructor started, spatial resolution: ${config.spatialResolution}`);
    const startTime = performance.now();
    this.globalCoherenceMatrix = this.initializeCoherenceMatrix();
    console.log(`[CoherenceFieldLockingEngine] Constructor completed in ${(performance.now() - startTime).toFixed(2)}ms`);
  }
  
  private initializeCoherenceMatrix(): Complex[][] {
    const size = this.config.spatialResolution;
    console.log(`[CoherenceFieldLockingEngine] Initializing coherence matrix ${size}x${size} = ${size * size} elements`);
    
    if (size > 50) {
      console.warn(`[CoherenceFieldLockingEngine] Large matrix size (${size}x${size}), using lazy initialization`);
      // Return empty matrix for lazy initialization
      return [];
    }
    
    const matrix: Complex[][] = [];
    
    for (let i = 0; i < size; i++) {
      matrix[i] = [];
      for (let j = 0; j < size; j++) {
        // Initialize with quantum vacuum fluctuations
        const phase = Math.random() * 2 * Math.PI;
        const amplitude = 1e-15 * Math.sqrt(-Math.log(Math.random()));
        matrix[i][j] = new Complex(
          amplitude * Math.cos(phase),
          amplitude * Math.sin(phase)
        );
      }
    }
    
    return matrix;
  }
  
  /**
   * Create a coherence-locked field region
   */
  createLockingField(
    position: [number, number, number],
    radius: number,
    targetCoherence: number
  ): LockingField {
    const fieldId = `field_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Initialize field tensor with standing wave pattern
    const gridSize = Math.ceil(radius * 2 * this.config.spatialResolution);
    const fieldTensor: Complex[][][] = [];
    
    for (let x = 0; x < gridSize; x++) {
      fieldTensor[x] = [];
      for (let y = 0; y < gridSize; y++) {
        fieldTensor[x][y] = [];
        for (let z = 0; z < gridSize; z++) {
          const r = Math.sqrt(
            Math.pow(x - gridSize/2, 2) + 
            Math.pow(y - gridSize/2, 2) + 
            Math.pow(z - gridSize/2, 2)
          ) / this.config.spatialResolution;
          
          if (r <= radius) {
            // Bessel function for spherical standing wave
            const bessel = this.sphericalBessel(r * Math.PI / radius);
            const phase = 0; // Locked phase
            fieldTensor[x][y][z] = new Complex(
              bessel * targetCoherence,
              0
            );
          } else {
            fieldTensor[x][y][z] = new Complex(0, 0);
          }
        }
      }
    }
    
    // Deploy observer array for field maintenance
    const observers = this.deployObserverArray(position, radius);
    
    // Calculate harmonic resonances
    const harmonicResonance = new Map<number, number>();
    this.config.fieldHarmonics.forEach(freq => {
      harmonicResonance.set(freq, this.calculateResonanceStrength(freq, radius));
    });
    
    const field: LockingField = {
      id: fieldId,
      center: position,
      radius,
      coherence: targetCoherence,
      stability: this.config.lockingStrength,
      observers,
      fieldTensor,
      harmonicResonance,
      lockTimestamp: Date.now(),
      decayRate: (1 - this.config.lockingStrength) / this.config.temporalCoherence
    };
    
    this.activeFields.set(fieldId, field);
    this.updateFieldInteractions();
    
    return field;
  }
  
  /**
   * Deploy quantum observer array for coherence maintenance
   */
  private deployObserverArray(
    center: [number, number, number],
    radius: number
  ): Observer[] {
    const observers: Observer[] = [];
    const count = Math.ceil(this.config.observerDensity * (4/3) * Math.PI * Math.pow(radius, 3));
    
    // Fibonacci sphere distribution for optimal coverage
    const phi = Math.PI * (3 - Math.sqrt(5)); // Golden angle
    
    for (let i = 0; i < count; i++) {
      const y = 1 - (i / (count - 1)) * 2;
      const radiusAtY = Math.sqrt(1 - y * y);
      const theta = phi * i;
      
      const x = Math.cos(theta) * radiusAtY;
      const z = Math.sin(theta) * radiusAtY;
      
      const observerPos: [number, number, number] = [
        center[0] + x * radius,
        center[1] + y * radius,
        center[2] + z * radius
      ];
      
      observers.push({
        id: `obs_${i}`,
        name: `LockObserver_${i}`,
        coherence: this.config.lockingStrength,
        focus: observerPos,
        phase: 0, // Phase-locked
        collapseThreshold: 0.99, // High threshold for stability
        memoryParticipation: 0.8,
        entangledObservers: [],
        observationHistory: []
      } as ExtendedObserver);
    }
    
    return observers;
  }
  
  /**
   * Update field evolution with military-grade precision
   */
  updateFields(deltaTime: number): Map<string, LockingField> {
    this.activeFields.forEach((field, id) => {
      // Apply coherence decay
      const decayFactor = Math.exp(-field.decayRate * deltaTime);
      field.coherence *= decayFactor;
      
      // Quantum error correction via observer consensus
      const observerCorrection = this.calculateObserverConsensus(field.observers);
      field.coherence = Math.min(1, field.coherence + observerCorrection);
      
      // Update field tensor with wave propagation
      this.propagateFieldTensor(field, deltaTime);
      
      // Check field interactions
      const interactions = this.fieldInteractions.get(id);
      if (interactions && interactions.size > 0) {
        interactions.forEach(otherId => {
          const otherField = this.activeFields.get(otherId);
          if (otherField) {
            this.processFieldInterference(field, otherField, deltaTime);
          }
        });
      }
      
      // Remove fields below threshold
      if (field.coherence < 0.1) {
        this.activeFields.delete(id);
        this.fieldInteractions.delete(id);
      }
    });
    
    this.updateGlobalCoherence();
    return this.activeFields;
  }
  
  /**
   * Calculate observer consensus for error correction
   */
  private calculateObserverConsensus(observers: Observer[]): number {
    if (observers.length === 0) return 0;
    
    // Weighted voting based on observer coherence influence
    let totalCorrection = 0;
    let totalWeight = 0;
    
    observers.forEach(observer => {
      const extObserver = toExtendedObserver(observer);
      const focusMagnitude = Math.sqrt(observer.focus[0]**2 + observer.focus[1]**2 + observer.focus[2]**2);
      const weight = focusMagnitude * (extObserver.coherenceInfluence || 0.5);
      const localCoherence = this.measureLocalCoherence(extObserver.position || observer.focus);
      const correction = (1 - localCoherence) * weight;
      
      totalCorrection += correction;
      totalWeight += weight;
    });
    
    return totalWeight > 0 ? totalCorrection / totalWeight : 0;
  }
  
  /**
   * Measure local coherence at a position
   */
  private measureLocalCoherence(position: [number, number, number]): number {
    let totalCoherence = 0;
    let totalWeight = 0;
    
    this.activeFields.forEach(field => {
      const distance = Math.sqrt(
        Math.pow(position[0] - field.center[0], 2) +
        Math.pow(position[1] - field.center[1], 2) +
        Math.pow(position[2] - field.center[2], 2)
      );
      
      if (distance <= field.radius) {
        const weight = 1 - (distance / field.radius);
        totalCoherence += field.coherence * weight;
        totalWeight += weight;
      }
    });
    
    return totalWeight > 0 ? totalCoherence / totalWeight : 0;
  }
  
  /**
   * Process quantum interference between locked fields
   */
  private processFieldInterference(
    field1: LockingField,
    field2: LockingField,
    deltaTime: number
  ): void {
    const distance = Math.sqrt(
      Math.pow(field1.center[0] - field2.center[0], 2) +
      Math.pow(field1.center[1] - field2.center[1], 2) +
      Math.pow(field1.center[2] - field2.center[2], 2)
    );
    
    const overlap = (field1.radius + field2.radius) - distance;
    
    if (overlap > 0) {
      // Constructive interference if phases align
      const phaseAlignment = this.calculatePhaseAlignment(field1, field2);
      const interferenceStrength = overlap / (field1.radius + field2.radius) * phaseAlignment;
      
      // Update coherence based on interference
      const coherenceBoost = interferenceStrength * this.config.lockingStrength * deltaTime;
      field1.coherence = Math.min(1, field1.coherence + coherenceBoost);
      field2.coherence = Math.min(1, field2.coherence + coherenceBoost);
      
      // Synchronize phases for stronger locking
      if (phaseAlignment > 0.8) {
        this.synchronizeFieldPhases(field1, field2);
      }
    }
  }
  
  /**
   * Calculate phase alignment between fields
   */
  private calculatePhaseAlignment(field1: LockingField, field2: LockingField): number {
    // Sample field tensors at overlap region
    let totalAlignment = 0;
    let sampleCount = 0;
    
    const samplePoints = 10;
    for (let i = 0; i < samplePoints; i++) {
      const t = i / (samplePoints - 1);
      const samplePos: [number, number, number] = [
        field1.center[0] * (1 - t) + field2.center[0] * t,
        field1.center[1] * (1 - t) + field2.center[1] * t,
        field1.center[2] * (1 - t) + field2.center[2] * t
      ];
      
      const amp1 = this.sampleFieldTensor(field1, samplePos);
      const amp2 = this.sampleFieldTensor(field2, samplePos);
      
      if (amp1 && amp2) {
        const phase1 = Math.atan2(amp1.imag, amp1.real);
        const phase2 = Math.atan2(amp2.imag, amp2.real);
        const alignment = Math.cos(phase1 - phase2);
        
        totalAlignment += alignment;
        sampleCount++;
      }
    }
    
    return sampleCount > 0 ? totalAlignment / sampleCount : 0;
  }
  
  /**
   * Sample field tensor at arbitrary position
   */
  private sampleFieldTensor(
    field: LockingField,
    position: [number, number, number]
  ): Complex | null {
    const relPos = [
      position[0] - field.center[0],
      position[1] - field.center[1],
      position[2] - field.center[2]
    ];
    
    const gridPos = relPos.map(p => 
      Math.floor((p + field.radius) * this.config.spatialResolution)
    );
    
    if (gridPos.some((p, i) => p < 0 || p >= field.fieldTensor.length)) {
      return null;
    }
    
    return field.fieldTensor[gridPos[0]][gridPos[1]][gridPos[2]];
  }
  
  /**
   * Synchronize phases between overlapping fields
   */
  private synchronizeFieldPhases(field1: LockingField, field2: LockingField): void {
    // Average the phases in overlap region
    const overlapCenter: [number, number, number] = [
      (field1.center[0] + field2.center[0]) / 2,
      (field1.center[1] + field2.center[1]) / 2,
      (field1.center[2] + field2.center[2]) / 2
    ];
    
    // Adjust observer phases to maintain synchronization
    [...field1.observers, ...field2.observers].forEach(observer => {
      const pos = hasPosition(observer) ? observer.position : observer.focus;
      const distToOverlap = Math.sqrt(
        Math.pow(pos![0] - overlapCenter[0], 2) +
        Math.pow(pos![1] - overlapCenter[1], 2) +
        Math.pow(pos![2] - overlapCenter[2], 2)
      );
      
      if (distToOverlap < (field1.radius + field2.radius) / 2) {
        observer.phase = 0; // Reset to synchronized phase
      }
    });
  }
  
  /**
   * Propagate field tensor using modified Schrödinger equation
   */
  private propagateFieldTensor(field: LockingField, deltaTime: number): void {
    const size = field.fieldTensor.length;
    const newTensor: Complex[][][] = [];
    
    // Apply quantum propagation with coherence locking
    for (let x = 0; x < size; x++) {
      newTensor[x] = [];
      for (let y = 0; y < size; y++) {
        newTensor[x][y] = [];
        for (let z = 0; z < size; z++) {
          const current = field.fieldTensor[x][y][z];
          
          // Laplacian for kinetic energy
          const laplacian = this.calculateLaplacian(field.fieldTensor, x, y, z);
          
          // Modified Hamiltonian with coherence locking term
          const hamiltonian = laplacian.multiply(createComplexFromNumber(-0.5)).add(
            current.multiply(createComplexFromNumber(field.coherence * this.config.lockingStrength))
          );
          
          // Time evolution
          const evolution = hamiltonian.multiply(new Complex(0, -deltaTime));
          newTensor[x][y][z] = current.add(evolution);
          
          // Normalize to maintain coherence
          const norm = newTensor[x][y][z].magnitude();
          if (norm > 0) {
            newTensor[x][y][z] = newTensor[x][y][z].multiply(createComplexFromNumber(1 / norm)).multiply(createComplexFromNumber(field.coherence));
          }
        }
      }
    }
    
    field.fieldTensor = newTensor;
  }
  
  /**
   * Calculate discrete Laplacian for field propagation
   */
  private calculateLaplacian(
    tensor: Complex[][][],
    x: number,
    y: number,
    z: number
  ): Complex {
    const size = tensor.length;
    let laplacian = tensor[x][y][z].multiply(createComplexFromNumber(-6));
    
    // Add neighboring contributions
    const neighbors = [
      [x-1, y, z], [x+1, y, z],
      [x, y-1, z], [x, y+1, z],
      [x, y, z-1], [x, y, z+1]
    ];
    
    neighbors.forEach(([nx, ny, nz]) => {
      if (nx >= 0 && nx < size && ny >= 0 && ny < size && nz >= 0 && nz < size) {
        laplacian = laplacian.add(tensor[nx][ny][nz]);
      }
    });
    
    return laplacian.multiply(createComplexFromNumber(1 / (this.config.spatialResolution * this.config.spatialResolution)));
  }
  
  /**
   * Update field interaction map
   */
  private updateFieldInteractions(): void {
    this.fieldInteractions.clear();
    
    const fields = Array.from(this.activeFields.values());
    for (let i = 0; i < fields.length; i++) {
      for (let j = i + 1; j < fields.length; j++) {
        const distance = Math.sqrt(
          Math.pow(fields[i].center[0] - fields[j].center[0], 2) +
          Math.pow(fields[i].center[1] - fields[j].center[1], 2) +
          Math.pow(fields[i].center[2] - fields[j].center[2], 2)
        );
        
        if (distance < fields[i].radius + fields[j].radius) {
          if (!this.fieldInteractions.has(fields[i].id)) {
            this.fieldInteractions.set(fields[i].id, new Set());
          }
          if (!this.fieldInteractions.has(fields[j].id)) {
            this.fieldInteractions.set(fields[j].id, new Set());
          }
          
          this.fieldInteractions.get(fields[i].id)!.add(fields[j].id);
          this.fieldInteractions.get(fields[j].id)!.add(fields[i].id);
        }
      }
    }
  }
  
  /**
   * Update global coherence matrix
   */
  private updateGlobalCoherence(): void {
    const size = this.config.spatialResolution;
    
    // Ensure matrix is properly initialized
    if (!this.globalCoherenceMatrix || this.globalCoherenceMatrix.length === 0) {
      this.globalCoherenceMatrix = [];
      for (let i = 0; i < size; i++) {
        this.globalCoherenceMatrix[i] = [];
      }
    }
    
    for (let i = 0; i < size; i++) {
      // Ensure row exists
      if (!this.globalCoherenceMatrix[i]) {
        this.globalCoherenceMatrix[i] = [];
      }
      
      for (let j = 0; j < size; j++) {
        // Map matrix position to 3D space
        const position: [number, number, number] = [
          i / this.config.spatialResolution,
          j / this.config.spatialResolution,
          0 // 2D projection for visualization
        ];
        
        const localCoherence = this.measureLocalCoherence(position);
        const amplitude = Math.sqrt(localCoherence);
        
        this.globalCoherenceMatrix[i][j] = new Complex(amplitude, 0);
      }
    }
    
    // Calculate total locking potential
    this.lockingPotential = this.calculateLockingPotential();
  }
  
  /**
   * Calculate total quantum locking potential
   */
  private calculateLockingPotential(): number {
    let totalPotential = 0;
    
    this.activeFields.forEach(field => {
      // Potential is proportional to volume * coherence^2 * stability
      const volume = (4/3) * Math.PI * Math.pow(field.radius, 3);
      const potential = volume * Math.pow(field.coherence, 2) * field.stability;
      totalPotential += potential;
    });
    
    // Add interaction terms
    this.fieldInteractions.forEach((interactions, fieldId) => {
      const field = this.activeFields.get(fieldId);
      if (field) {
        interactions.forEach(otherId => {
          const other = this.activeFields.get(otherId);
          if (other) {
            const interactionPotential = field.coherence * other.coherence * 
              this.calculatePhaseAlignment(field, other);
            totalPotential += interactionPotential;
          }
        });
      }
    });
    
    return totalPotential;
  }
  
  /**
   * Spherical Bessel function for standing wave patterns
   */
  private sphericalBessel(x: number): number {
    if (x === 0) return 1;
    return Math.sin(x) / x;
  }
  
  /**
   * Calculate resonance strength at given frequency
   */
  private calculateResonanceStrength(frequency: number, radius: number): number {
    // Resonance occurs when wavelength fits integer times in sphere
    const wavelength = 3e8 / frequency; // c / f
    const modesInRadius = radius / wavelength;
    const resonanceQuality = 1 - Math.abs(modesInRadius - Math.round(modesInRadius));
    
    return resonanceQuality * this.config.lockingStrength;
  }
  
  /**
   * Get metrics for monitoring
   */
  getMetrics(): {
    activeFieldCount: number;
    totalCoherence: number;
    lockingPotential: number;
    averageStability: number;
    interactionCount: number;
  } {
    let totalCoherence = 0;
    let totalStability = 0;
    
    this.activeFields.forEach(field => {
      totalCoherence += field.coherence;
      totalStability += field.stability;
    });
    
    const fieldCount = this.activeFields.size;
    
    return {
      activeFieldCount: fieldCount,
      totalCoherence,
      lockingPotential: this.lockingPotential,
      averageStability: fieldCount > 0 ? totalStability / fieldCount : 0,
      interactionCount: this.fieldInteractions.size
    };
  }
  
  /**
   * Export field configuration for analysis
   */
  exportFieldConfiguration(): any {
    const fields = Array.from(this.activeFields.values()).map(field => ({
      id: field.id,
      center: field.center,
      radius: field.radius,
      coherence: field.coherence,
      stability: field.stability,
      observerCount: field.observers.length,
      harmonics: Array.from(field.harmonicResonance.entries()),
      age: Date.now() - field.lockTimestamp
    }));
    
    return {
      timestamp: new Date().toISOString(),
      config: this.config,
      fields,
      metrics: this.getMetrics(),
      globalCoherenceMatrix: this.globalCoherenceMatrix.map(row => 
        row.map(c => ({ real: c.real, imag: c.imag }))
      )
    };
  }
}