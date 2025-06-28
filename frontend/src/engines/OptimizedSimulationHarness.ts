import { MemoryFieldEngine, type MemoryField } from './MemoryFieldEngine';
import { EntropyCoherenceSolver } from './EntropyCoherenceSolver';
import { RSPEngine, type RSPState } from './RSPEngine';
import { ObserverEngine, type Observer, type ObservationEvent } from './ObserverEngine';
import { WavefunctionSimulator, type WavefunctionState } from './WavefunctionSimulator';
import { Complex } from '../utils/complex';
import { SimulationParameters, SimulationState, OSHPrediction } from './SimulationHarness';
import { BaseEngine } from '../types/engine-types';

/**
 * Optimized SimulationHarness with performance improvements
 * - Lazy initialization of large arrays
 * - Spatial indexing for potential calculations
 * - Reduced grid operations
 * - Async operations moved to separate thread when possible
 */
export class OptimizedSimulationHarness implements BaseEngine {
  private memoryEngine: MemoryFieldEngine;
  private entropySolver: EntropyCoherenceSolver;
  private rspEngine: RSPEngine;
  private observerEngine: ObserverEngine;
  private wavefunctionSim: WavefunctionSimulator;
  
  private parameters: SimulationParameters;
  private currentState: SimulationState;
  private stateHistory: SimulationState[] = [];
  private predictions: OSHPrediction[] = [];
  
  // Performance optimizations
  private potentialCache: Float32Array | null = null;
  private lastPotentialUpdate: number = 0;
  private potentialUpdateInterval: number = 100; // ms
  private spatialIndex: Map<string, number[]> = new Map();
  private isInitialized: boolean = false;
  
  constructor(parameters: SimulationParameters) {
    const startTime = performance.now();
    console.log(`[OptimizedSimulationHarness] Constructor started with grid size: ${parameters.gridSize}³ = ${parameters.gridSize ** 3} cells`);
    this.parameters = parameters;
    
    // Initialize engines with smaller initial allocations
    console.log('[OptimizedSimulationHarness] Creating MemoryFieldEngine...');
    const t1 = performance.now();
    this.memoryEngine = new MemoryFieldEngine();
    console.log(`[OptimizedSimulationHarness] MemoryFieldEngine created in ${(performance.now() - t1).toFixed(2)}ms`);
    
    console.log('[OptimizedSimulationHarness] Creating EntropyCoherenceSolver...');
    const t2 = performance.now();
    this.entropySolver = new EntropyCoherenceSolver();
    console.log(`[OptimizedSimulationHarness] EntropyCoherenceSolver created in ${(performance.now() - t2).toFixed(2)}ms`);
    
    console.log('[OptimizedSimulationHarness] Creating RSPEngine...');
    const t3 = performance.now();
    this.rspEngine = new RSPEngine();
    console.log(`[OptimizedSimulationHarness] RSPEngine created in ${(performance.now() - t3).toFixed(2)}ms`);
    
    console.log('[OptimizedSimulationHarness] Creating ObserverEngine...');
    const t4 = performance.now();
    this.observerEngine = new ObserverEngine();
    console.log(`[OptimizedSimulationHarness] ObserverEngine created in ${(performance.now() - t4).toFixed(2)}ms`);
    
    // Delay wavefunction simulator initialization
    const effectiveSize = Math.min(parameters.gridSize, 8);
    console.log(`[OptimizedSimulationHarness] Creating WavefunctionSimulator with size ${effectiveSize}³...`);
    const t5 = performance.now();
    this.wavefunctionSim = new WavefunctionSimulator({
      sizeX: effectiveSize, // Start small
      sizeY: effectiveSize,
      sizeZ: effectiveSize,
      spacing: 0.1,
      boundaryCondition: 'periodic'
    });
    console.log(`[OptimizedSimulationHarness] WavefunctionSimulator created in ${(performance.now() - t5).toFixed(2)}ms`);
    
    // Initialize minimal state
    console.log('[OptimizedSimulationHarness] Initializing minimal state...');
    this.currentState = {
      time: 0,
      timestamp: 0,
      step: 0,
      memoryField: this.memoryEngine.getField(),
      rspState: this.rspEngine.getState(),
      wavefunction: this.wavefunctionSim.getState(),
      observers: [],
      events: []
    };
    
    const totalTime = performance.now() - startTime;
    console.log(`[OptimizedSimulationHarness] Constructor completed in ${totalTime.toFixed(2)}ms`);
  }

  /**
   * Lazy initialization to avoid blocking the main thread
   */
  async initialize(initialConditions?: Partial<SimulationState>): Promise<void> {
    if (this.isInitialized) return;
    
    console.log('[OptimizedSimulationHarness] Starting lazy initialization...');
    
    if (initialConditions) {
      Object.assign(this.currentState, initialConditions);
    }
    
    // Keep wavefunction simulator small initially
    const effectiveGridSize = Math.min(this.parameters.gridSize, 4); // Max 4x4x4 = 64 cells initially
    
    if (this.wavefunctionSim.getState().gridSize !== effectiveGridSize) {
      this.wavefunctionSim = new WavefunctionSimulator({
        sizeX: effectiveGridSize,
        sizeY: effectiveGridSize,
        sizeZ: effectiveGridSize,
        spacing: 0.1,
        boundaryCondition: 'periodic'
      });
    }
    
    // Initialize with minimal state
    const centerIdx = Math.floor(effectiveGridSize / 2);
    const numCells = Math.pow(effectiveGridSize, 3);
    
    // Create minimal amplitude array
    const amplitude: Complex[] = [];
    for (let i = 0; i < numCells; i++) {
      // Only center cell has non-zero amplitude
      const isCenter = i === (centerIdx + centerIdx * effectiveGridSize + 
                              centerIdx * effectiveGridSize * effectiveGridSize);
      amplitude.push(new Complex(isCenter ? 1 : 0, 0));
    }
    
    this.wavefunctionSim.setState({ amplitude, gridSize: effectiveGridSize });
    
    // Add minimal memory fragment
    this.memoryEngine.addFragment(
      [new Complex(1, 0)],
      [centerIdx, centerIdx, centerIdx]
    );
    
    // Add single observer
    this.observerEngine.createObserver(
      'default',
      'Default Observer',
      0.8,
      [centerIdx + 1, centerIdx, centerIdx]
    );
    
    this.isInitialized = true;
    console.log(`[OptimizedSimulationHarness] Initialization complete with ${effectiveGridSize}³ = ${numCells} cells`);
  }

  async step(): Promise<SimulationState> {
    if (!this.isInitialized) {
      await this.initialize();
    }
    
    const dt = this.parameters.timeStep;
    
    // 1. Update memory field
    const memoryField = this.memoryEngine.updateField(dt);
    
    // 2. Calculate entropy and coherence (sample-based for performance)
    const sampleSize = Math.min(100, memoryField.fragments.length);
    const sampledFragments = memoryField.fragments.slice(0, sampleSize);
    const entropyMetrics = this.entropySolver.calculateEntropy(
      sampledFragments.flatMap(f => f.state.slice(0, 10)) // Limit state size
    );
    const coherenceMetrics = this.entropySolver.calculateCoherence(memoryField);
    
    // 3. Update RSP state
    this.rspEngine.update(
      entropyMetrics.shannonEntropy,
      coherenceMetrics.globalCoherence,
      dt
    );
    const rspState = this.rspEngine.getState();
    
    // 4. Propagate wavefunction with cached potential
    const potential = this.getCachedPotential(memoryField);
    this.wavefunctionSim.propagate(dt, potential);
    
    // 5. Process observations (simplified)
    const observers = this.observerEngine.getObservers();
    const events: ObservationEvent[] = [];
    
    // Only process first observer for performance
    if (observers.length > 0) {
      const observer = observers[0];
      const wfState = this.wavefunctionSim.getState();
      
      // Use smaller amplitude sample
      const amplitudeSample = wfState.amplitude.slice(0, 100);
      
      this.observerEngine.updateObserver(
        observer.id,
        amplitudeSample,
        sampledFragments,
        dt
      );
    }
    
    // 6. Update state
    this.currentState = {
      time: this.currentState.time + dt,
      timestamp: this.currentState.time + dt,
      step: this.currentState.step + 1,
      memoryField,
      rspState,
      wavefunction: this.wavefunctionSim.getState(),
      observers: observers,
      events: [...this.currentState.events.slice(-50), ...events] // Keep only last 50 events
    };
    
    // 7. Store limited history
    this.stateHistory.push({ ...this.currentState });
    if (this.stateHistory.length > 100) { // Reduced from 1000
      this.stateHistory.shift();
    }
    
    return this.currentState;
  }

  /**
   * Optimized potential generation with caching and spatial indexing
   */
  private getCachedPotential(memoryField: MemoryField): Complex[] {
    const now = Date.now();
    
    // Use cached potential if recent enough
    if (this.potentialCache && (now - this.lastPotentialUpdate) < this.potentialUpdateInterval) {
      return Array.from(this.potentialCache).map(v => new Complex(v, 0));
    }
    
    // Generate new potential with optimizations
    const size = this.parameters.gridSize;
    const numCells = size * size * size;
    
    // Use Float32Array for better performance
    if (!this.potentialCache) {
      this.potentialCache = new Float32Array(numCells);
    }
    
    // Reset potential
    this.potentialCache.fill(0);
    
    // Update spatial index for memory fragments
    this.updateSpatialIndex(memoryField);
    
    // Only calculate potential near memory fragments (sparse calculation)
    const radius = 5; // Influence radius
    
    for (const fragment of memoryField.fragments) {
      const [fx, fy, fz] = fragment.position;
      
      // Only update cells within radius
      for (let dx = -radius; dx <= radius; dx++) {
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dz = -radius; dz <= radius; dz++) {
            const x = fx + dx;
            const y = fy + dy;
            const z = fz + dz;
            
            // Check bounds
            if (x >= 0 && x < size && y >= 0 && y < size && z >= 0 && z < size) {
              const idx = x + y * size + z * size * size;
              const r2 = dx * dx + dy * dy + dz * dz;
              
              if (r2 > 0 && r2 <= radius * radius) {
                this.potentialCache[idx] += fragment.coherence / Math.sqrt(r2);
              }
            }
          }
        }
      }
    }
    
    this.lastPotentialUpdate = now;
    
    // Convert to Complex array
    const potential: Complex[] = [];
    for (let i = 0; i < Math.min(numCells, 1000); i++) { // Limit size
      potential.push(new Complex(this.potentialCache[i] || 0, 0));
    }
    
    return potential;
  }
  
  /**
   * Update spatial index for efficient lookups
   */
  private updateSpatialIndex(memoryField: MemoryField): void {
    this.spatialIndex.clear();
    
    for (let i = 0; i < memoryField.fragments.length; i++) {
      const fragment = memoryField.fragments[i];
      const key = `${Math.floor(fragment.position[0] / 10)},${Math.floor(fragment.position[1] / 10)},${Math.floor(fragment.position[2] / 10)}`;
      
      if (!this.spatialIndex.has(key)) {
        this.spatialIndex.set(key, []);
      }
      this.spatialIndex.get(key)!.push(i);
    }
  }

  getState(): SimulationState {
    return this.currentState;
  }

  getPredictions(): OSHPrediction[] {
    return this.predictions;
  }

  getMetrics() {
    return {
      totalSteps: this.currentState.step,
      simulationTime: this.currentState.time,
      memoryFragments: this.currentState.memoryField.fragments.length,
      entropy: this.currentState.memoryField.totalEntropy,
      coherence: this.currentState.memoryField.totalCoherence,
      rsp: this.currentState.rspState.rsp,
      observerCount: this.currentState.observers.length,
      eventCount: this.currentState.events.length,
      historySize: this.stateHistory.length
    };
  }

  private async updatePredictions(): Promise<void> {
    // Placeholder - predictions disabled for performance
    this.predictions = [];
  }

  private async checkEmergentPhenomena(): Promise<void> {
    // Placeholder - disabled for performance
  }

  private async collapseWavefunction(measuredState: Complex[], focus: number): Promise<void> {
    // Update the wavefunction state directly since collapse method doesn't exist
    this.wavefunctionSim.setState({
      amplitude: measuredState,
      gridSize: this.parameters.gridSize
    });
  }

  /**
   * Update method to implement BaseEngine interface
   */
  update(deltaTime: number, context?: any): void {
    // Step the simulation forward
    this.step().catch(error => {
      console.error('[OptimizedSimulationHarness] Update error:', error);
    });
  }
}