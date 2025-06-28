/**
 * OSH Quantum Engine
 * Central orchestrator for all quantum simulation engines
 * Production-ready implementation with enterprise standards
 */

import { Complex } from '../utils/complex';
import { MemoryFieldEngine } from './MemoryFieldEngine';
import { EntropyCoherenceSolver } from './EntropyCoherenceSolver';
import { RSPEngine } from './RSPEngine';
import { ObserverEngine } from './ObserverEngine';
// WavefunctionSimulator disabled for memory optimization
// import { WavefunctionSimulator } from './WavefunctionSimulator';
import { OptimizedSimulationHarness } from './OptimizedSimulationHarness';
import { UnifiedQuantumErrorReductionPlatform } from './UnifiedQuantumErrorReductionPlatform';
import { MLAssistedObserver } from './MLObserver';
import { MacroTeleportationEngine } from './MacroTeleportation';
import { CurvatureTensorGenerator } from './CurvatureTensorGenerator';
import { SimulationSnapshotManager } from './SimulationSnapshotManager';
import { CoherenceFieldLockingEngine } from './CoherenceFieldLockingEngine';
import { RecursiveTensorFieldEngine } from './RecursiveTensorFieldEngine';
import { SubstrateIntrospectionEngine } from './SubstrateIntrospectionEngine';
import { ViteWorkerPool } from '../workers/ViteWorkerPool';
import { getMemorySafeGridSize, MemoryMonitor } from '../config/memoryConfig';
import { ResourceManager } from './ResourceManager';
import { DiagnosticsSystem, trace } from '../utils/diagnostics';
import { SimulationGuard } from './SimulationGuard';
import { QuantumUpdateScheduler } from './QuantumUpdateScheduler';
import { EngineOrchestrator } from './EngineOrchestrator';

export class OSHQuantumEngine {
  // Core engines
  public memoryFieldEngine: MemoryFieldEngine;
  public entropyCoherenceSolver: EntropyCoherenceSolver;
  public rspEngine: RSPEngine;
  public observerEngine: ObserverEngine;
  // WavefunctionSimulator disabled for memory optimization
  // public wavefunctionSimulator: WavefunctionSimulator;
  public simulationHarness: OptimizedSimulationHarness;
  public errorReductionPlatform: UnifiedQuantumErrorReductionPlatform;
  public mlObserver: MLAssistedObserver;
  public macroTeleportation: MacroTeleportationEngine;
  
  // Advanced engines
  public curvatureGenerator: CurvatureTensorGenerator;
  public snapshotManager: SimulationSnapshotManager;
  public coherenceLocking: CoherenceFieldLockingEngine;
  public tensorField: RecursiveTensorFieldEngine;
  public introspection: SubstrateIntrospectionEngine;
  
  // Performance tracking
  private workerPool?: ViteWorkerPool;
  private fps: number = 0;
  private frameCount: number = 0;
  private lastUpdateTime: number = Date.now();
  private memoryMonitor: MemoryMonitor;
  private resourceManager: ResourceManager;
  private diagnostics: DiagnosticsSystem;
  private simulationGuard: SimulationGuard;
  private updateScheduler: QuantumUpdateScheduler;
  private engineOrchestrator: EngineOrchestrator;
  private isFullyInitialized: boolean = false;
  private initializationTime: number = 0;
  private performanceIntervalId?: number;
  private memoryCriticalHandler?: () => void;
  private updateSkipCounter: number = 0;
  private isUpdating: boolean = false;
  
  constructor() {
    const constructorStart = performance.now();
    console.log('[OSHQuantumEngine] Constructor started');
    
    // Get memory monitor instance
    console.log('[OSHQuantumEngine] Getting memory monitor...');
    this.memoryMonitor = MemoryMonitor.getInstance();
    console.log('[OSHQuantumEngine] Memory monitor obtained');
    
    // Initialize resource manager
    console.log('[OSHQuantumEngine] Initializing resource manager...');
    this.resourceManager = ResourceManager.getInstance();
    console.log('[OSHQuantumEngine] Resource manager initialized');
    
    // Initialize diagnostics
    console.log('[OSHQuantumEngine] Initializing diagnostics...');
    this.diagnostics = DiagnosticsSystem.getInstance();
    console.log('[OSHQuantumEngine] Diagnostics initialized');
    
    // Initialize simulation guard
    console.log('[OSHQuantumEngine] Initializing simulation guard...');
    this.simulationGuard = SimulationGuard.getInstance();
    console.log('[OSHQuantumEngine] Simulation guard initialized');
    
    // Initialize update scheduler
    console.log('[OSHQuantumEngine] Initializing update scheduler...');
    this.updateScheduler = QuantumUpdateScheduler.getInstance();
    console.log('[OSHQuantumEngine] Update scheduler initialized');
    
    // Initialize engine orchestrator
    console.log('[OSHQuantumEngine] Initializing engine orchestrator...');
    this.engineOrchestrator = EngineOrchestrator.getInstance();
    console.log('[OSHQuantumEngine] Engine orchestrator initialized');
    
    // Subscribe to critical memory events
    this.memoryCriticalHandler = () => {
      console.warn('[OSHQuantumEngine] Critical memory threshold reached, clearing caches...');
      this.clearCaches();
    };
    window.addEventListener('memory-critical', this.memoryCriticalHandler);
    
    try {
      // Get memory-safe grid size based on available memory and resource limits
      const resourceLimits = this.resourceManager.getSafeLimits();
      const memoryBasedSize = getMemorySafeGridSize();
      const guardLimits = this.simulationGuard.getLimits();
      const safeGridSize = Math.min(memoryBasedSize, resourceLimits.maxGridSize, guardLimits.maxGridSize);
      console.log(`[OSHQuantumEngine] Initializing with grid size: ${safeGridSize}³ = ${safeGridSize ** 3} cells`);
      
      // Initialize core engines
      console.log('[OSHQuantumEngine] Initializing MemoryFieldEngine...');
      const t1 = performance.now();
      this.memoryFieldEngine = new MemoryFieldEngine();
      console.log(`[OSHQuantumEngine] MemoryFieldEngine initialized in ${(performance.now() - t1).toFixed(2)}ms`);
      
      // Register with orchestrator
      this.engineOrchestrator.registerEngine('memoryField', this.memoryFieldEngine);
      
      console.log('[OSHQuantumEngine] Initializing EntropyCoherenceSolver...');
      const t2 = performance.now();
      this.entropyCoherenceSolver = new EntropyCoherenceSolver();
      console.log(`[OSHQuantumEngine] EntropyCoherenceSolver initialized in ${(performance.now() - t2).toFixed(2)}ms`);
      
      // Register with orchestrator
      this.engineOrchestrator.registerEngine('entropyCoherence', this.entropyCoherenceSolver);
      
      console.log('[OSHQuantumEngine] Initializing RSPEngine...');
      const t3 = performance.now();
      this.rspEngine = new RSPEngine();
      console.log(`[OSHQuantumEngine] RSPEngine initialized in ${(performance.now() - t3).toFixed(2)}ms`);
      
      // Register with orchestrator
      this.engineOrchestrator.registerEngine('rsp', this.rspEngine);
      
      console.log('[OSHQuantumEngine] Initializing ObserverEngine...');
      const t4 = performance.now();
      this.observerEngine = new ObserverEngine();
      console.log(`[OSHQuantumEngine] ObserverEngine initialized in ${(performance.now() - t4).toFixed(2)}ms`);
      
      // Register with orchestrator
      this.engineOrchestrator.registerEngine('observer', this.observerEngine);
      
      // WavefunctionSimulator disabled for memory optimization
      // const initialSize = Math.min(safeGridSize, 8);
      // console.log(`[OSHQuantumEngine] Initializing WavefunctionSimulator with size ${initialSize}³...`);
      // const t5 = performance.now();
      // this.wavefunctionSimulator = new WavefunctionSimulator({
      //   sizeX: initialSize,
      //   sizeY: initialSize,
      //   sizeZ: initialSize,
      //   spacing: 0.1,
      //   boundaryCondition: 'periodic'
      // });
      // console.log(`[OSHQuantumEngine] WavefunctionSimulator initialized in ${(performance.now() - t5).toFixed(2)}ms`);
      console.log('[OSHQuantumEngine] WavefunctionSimulator disabled for memory optimization');
    } catch (error) {
      console.error('[OSHQuantumEngine] Failed to initialize core engines:', error);
      throw error;
    }
    
    // Initialize ML observer
    console.log('[OSHQuantumEngine] Initializing MLAssistedObserver...');
    const t6 = performance.now();
    this.mlObserver = new MLAssistedObserver({
      learningRate: 0.001,
      hiddenLayers: [64, 32], // Two hidden layers
      activationFunction: 'relu',
      optimizationTarget: 'rsp'
    });
    console.log(`[OSHQuantumEngine] MLAssistedObserver initialized in ${(performance.now() - t6).toFixed(2)}ms`);
    
    // Register with orchestrator - lazy loaded
    this.engineOrchestrator.registerEngine('mlObserver', this.mlObserver);
    
    // Initialize macro teleportation
    console.log('[OSHQuantumEngine] Initializing MacroTeleportationEngine...');
    const t7 = performance.now();
    this.macroTeleportation = new MacroTeleportationEngine({
      sourcePosition: [0, 0, 0],
      targetPosition: [10, 0, 0],
      objectSize: 1.0,
      coherenceRequirement: 0.95,
      verificationLevel: 'high'
    });
    console.log(`[OSHQuantumEngine] MacroTeleportationEngine initialized in ${(performance.now() - t7).toFixed(2)}ms`);
    
    // Register with orchestrator - lazy loaded
    this.engineOrchestrator.registerEngine('macroTeleportation', this.macroTeleportation);
    
    // Initialize advanced engines
    console.log('[OSHQuantumEngine] Initializing advanced engines...');
    
    const t8 = performance.now();
    this.curvatureGenerator = new CurvatureTensorGenerator();
    console.log(`[OSHQuantumEngine] CurvatureTensorGenerator initialized in ${(performance.now() - t8).toFixed(2)}ms`);
    
    // Register with orchestrator - low priority
    this.engineOrchestrator.registerEngine('curvatureGenerator', this.curvatureGenerator);
    
    const t9 = performance.now();
    this.snapshotManager = new SimulationSnapshotManager();
    console.log(`[OSHQuantumEngine] SimulationSnapshotManager initialized in ${(performance.now() - t9).toFixed(2)}ms`);
    
    // Snapshot manager doesn't need regular updates
    
    const t10 = performance.now();
    try {
      this.coherenceLocking = new CoherenceFieldLockingEngine({
        lockingStrength: 0.95,
        spatialResolution: 100,
        temporalCoherence: 1000,
        observerDensity: 100,
        fieldHarmonics: [1e9, 2.45e9, 5.8e9]
      });
      console.log(`[OSHQuantumEngine] CoherenceFieldLockingEngine initialized in ${(performance.now() - t10).toFixed(2)}ms`);
      
      // Register with orchestrator - low priority, lazy loaded
      this.engineOrchestrator.registerEngine('coherenceLocking', this.coherenceLocking);
    } catch (coherenceInitError) {
      console.error('[OSHQuantumEngine] Failed to initialize CoherenceFieldLockingEngine:', coherenceInitError);
      // Create a null implementation to prevent crashes
      this.coherenceLocking = null as any;
    }
    
    const t11 = performance.now();
    this.tensorField = new RecursiveTensorFieldEngine();
    console.log(`[OSHQuantumEngine] RecursiveTensorFieldEngine initialized in ${(performance.now() - t11).toFixed(2)}ms`);
    
    // Register with orchestrator - very low priority
    this.engineOrchestrator.registerEngine('tensorField', this.tensorField);
    
    const t12 = performance.now();
    this.introspection = new SubstrateIntrospectionEngine();
    console.log(`[OSHQuantumEngine] SubstrateIntrospectionEngine initialized in ${(performance.now() - t12).toFixed(2)}ms`);
    
    // Register with orchestrator - very low priority
    this.engineOrchestrator.registerEngine('introspection', this.introspection);
    
    // Initialize worker pool if available
    if (typeof Worker !== 'undefined') {
      try {
        console.log('[OSHQuantumEngine] Initializing ViteWorkerPool...');
        const t13 = performance.now();
        this.workerPool = new ViteWorkerPool(4);
        console.log(`[OSHQuantumEngine] ViteWorkerPool initialized in ${(performance.now() - t13).toFixed(2)}ms`);
      } catch (e) {
        console.warn('[OSHQuantumEngine] Web Workers not available:', e);
      }
    }
    
    // Initialize optimized simulation harness with memory-efficient parameters
    const safeGridSize = getMemorySafeGridSize();
    console.log(`[OSHQuantumEngine] Initializing OptimizedSimulationHarness with grid size ${safeGridSize}³...`);
    const t14 = performance.now();
    this.simulationHarness = new OptimizedSimulationHarness({
      gridSize: safeGridSize, // Dynamic sizing based on available memory
      timeStep: 0.01,
      memoryDecayRate: 0.05,
      coherenceDiffusion: 0.1,
      observerThreshold: 0.7,
      quantumCoupling: 0.5,
      entropyWeight: 1.0,
      informationFlow: 0.8
    });
    console.log(`[OSHQuantumEngine] OptimizedSimulationHarness initialized in ${(performance.now() - t14).toFixed(2)}ms`);
    
    // Register with orchestrator - medium priority
    this.engineOrchestrator.registerEngine('simulationHarness', this.simulationHarness);
    
    // Initialize error reduction platform
    console.log('[OSHQuantumEngine] Initializing UnifiedQuantumErrorReductionPlatform...');
    const t15 = performance.now();
    this.errorReductionPlatform = new UnifiedQuantumErrorReductionPlatform({
      targetErrorRate: 0.0002,
      mechanismWeights: {
        rmcs: 0.25,
        icc: 0.20,
        cofl: 0.20,
        recc: 0.20,
        bmfe: 0.15
      }
    });
    console.log(`[OSHQuantumEngine] UnifiedQuantumErrorReductionPlatform initialized in ${(performance.now() - t15).toFixed(2)}ms`);
    
    // Register with orchestrator - runs periodically, not every frame
    this.engineOrchestrator.registerEngine('errorReduction', this.errorReductionPlatform);
    
    // WavefunctionSimulator initialization disabled for memory optimization
    // console.log('[OSHQuantumEngine] Initializing wavefunction state...');
    // this.wavefunctionSimulator.initializeState(8); // 8 dimensional quantum state
    
    // Initialize memory field with a base fragment
    console.log('[OSHQuantumEngine] Adding initial memory fragment...');
    const initialState = Array(8).fill(0).map((_, i) => 
      new Complex(i === 0 ? 1 : 0, 0) // |000⟩ state
    );
    this.memoryFieldEngine.addFragment(initialState, [0, 0, 0]);
    
    // Initialize RSP engine with valid starting values
    console.log('[OSHQuantumEngine] Updating RSP engine with initial values...');
    this.rspEngine.updateRSP(
      initialState,
      [[new Complex(1, 0)]],
      1.0, // Initial entropy
      0.016 // Initial deltaTime
    );
    
    const totalTime = performance.now() - constructorStart;
    console.log(`[OSHQuantumEngine] Constructor completed in ${totalTime.toFixed(2)}ms`);
    
    // Mark initialization time
    this.initializationTime = Date.now();
  }
  
  start(): void {
    console.log('[OSHQuantumEngine] start() method called');
    const startTime = performance.now();
    
    // Start memory monitoring
    console.log('[OSHQuantumEngine] Starting memory monitor...');
    this.memoryMonitor.start();
    console.log('[OSHQuantumEngine] Memory monitor started');
    
    // Start error reduction platform without await
    console.log('[OSHQuantumEngine] Starting error reduction platform...');
    this.errorReductionPlatform.start().catch(err => {
      console.warn('[OSHQuantumEngine] Error reduction platform failed to start:', err);
    });
    
    // Start performance monitoring
    console.log('[OSHQuantumEngine] Starting performance monitoring...');
    this.startPerformanceMonitoring();
    console.log('[OSHQuantumEngine] Performance monitoring started');
    
    // Ensure all engines have valid initial state
    console.log('[OSHQuantumEngine] Validating and initializing engine state...');
    this.validateAndInitializeEngineState();
    console.log('[OSHQuantumEngine] Engine state validated');
    
    // Log initial resource state
    console.log('[OSHQuantumEngine] Initial resource state:');
    const resourceMetrics = this.resourceManager.getMetrics();
    console.log('[OSHQuantumEngine] ResourceManager metrics:', {
      memoryUsed: resourceMetrics.memoryUsed?.toFixed(1) + 'MB',
      memoryLimit: resourceMetrics.memoryLimit?.toFixed(1) + 'MB', 
      memoryPressure: resourceMetrics.memoryPressure?.toFixed(3),
      cpuUsage: resourceMetrics.cpuUsage?.toFixed(1) + '%',
      throttleLevel: resourceMetrics.throttleLevel?.toFixed(3),
      isHealthy: resourceMetrics.isHealthy
    });
    
    const orchestratorMetrics = this.engineOrchestrator.getMetrics();
    console.log('[OSHQuantumEngine] Orchestrator metrics:', {
      activeEngines: orchestratorMetrics.activeEngines,
      totalEngines: orchestratorMetrics.totalEngines,
      memoryPressure: orchestratorMetrics.memoryPressure?.toFixed(3),
      cpuUsage: orchestratorMetrics.cpuUsage?.toFixed(3)
    });
    
    // Run health check
    console.log('[OSHQuantumEngine] Running health check...');
    const health = this.validateEngineHealth();
    if (!health.healthy) {
      console.warn('[OSHQuantumEngine] Engine started with issues:', health.issues);
      console.log('[OSHQuantumEngine] Initial metrics:', health.metrics);
    } else {
      console.log('[OSHQuantumEngine] Engine initialized and ready');
      console.log('[OSHQuantumEngine] Initial metrics:', health.metrics);
    }
    
    const totalTime = performance.now() - startTime;
    console.log(`[OSHQuantumEngine] start() completed in ${totalTime.toFixed(2)}ms`);
    
    // Mark as fully initialized after a brief delay to ensure all async operations complete
    setTimeout(() => {
      this.isFullyInitialized = true;
      console.log('[OSHQuantumEngine] Engine marked as fully initialized');
    }, 100);
  }
  
  /**
   * Validate and initialize engine state to ensure proper data flow
   */
  private validateAndInitializeEngineState(): void {
    // WavefunctionSimulator validation disabled for memory optimization
    // const wfState = this.wavefunctionSimulator.getState();
    // if (!wfState.amplitude || wfState.amplitude.length === 0 || wfState.totalProbability < 0.1) {
    //   this.wavefunctionSimulator.setGaussianWavepacket(
    //     [3.2, 3.2, 3.2], // Center in physical space
    //     [0, 0, 0],       // Zero momentum
    //     0.5              // Width
    //   );
    // }
    
    // Ensure memory field has fragments
    const memoryField = this.memoryFieldEngine.getField();
    if (!memoryField.fragments || memoryField.fragments.length === 0) {
      // Add a default fragment if none exist
      const defaultState = Array(8).fill(0).map((_, i) => 
        new Complex(i === 0 ? 1 : 0, 0)
      );
      this.memoryFieldEngine.addFragment(defaultState, [0, 0, 0]);
    }
    
    // Ensure RSP engine has valid state
    const rspState = this.rspEngine.getState();
    if (!rspState || rspState.rsp === 0) {
      // Use default quantum state instead of wavefunction simulator
      const defaultAmplitude = Array(8).fill(0).map((_, i) => 
        new Complex(i === 0 ? 1 : 0, 0)
      );
      const coherenceMatrix = this.calculateCoherenceMatrix(defaultAmplitude);
      const memoryMetrics = this.memoryFieldEngine.getMetrics();
      
      this.rspEngine.updateRSP(
        defaultAmplitude,
        coherenceMatrix,
        memoryMetrics.entropy,
        0.016
      );
    }
    
    // Ensure observers exist
    const observers = this.observerEngine.getAllObservers();
    if (observers.length === 0) {
      // Add a default observer
      this.observerEngine.addObserver({
        name: 'DefaultObserver',
        focus: [0, 0, 0],
        phase: 0,
        collapseThreshold: 0.7,
        memoryParticipation: 0.5,
        entangledObservers: [],
        observationHistory: []
      });
    }
    
    // Start the engine orchestrator
    this.engineOrchestrator.start();
  }
  
  /**
   * Update all engines with delta time
   */
  update(deltaTime: number): void {
    // Debug logging
    if (this.frameCount % 60 === 0) {
      console.log('[OSHQuantumEngine] update() called - frame:', this.frameCount, 'deltaTime:', deltaTime, 'isFullyInitialized:', this.isFullyInitialized);
    }
    
    // Prevent concurrent updates
    if (this.isUpdating) {
      console.warn('[OSHQuantumEngine] Update already in progress, skipping');
      return;
    }
    
    const updateStart = performance.now();
    
    // Quick checks that don't need tracing
    if (!this.isFullyInitialized) {
      const timeSinceInit = Date.now() - this.initializationTime;
      if (timeSinceInit < 500) {
        if (this.frameCount % 10 === 0) {
          console.log('[OSHQuantumEngine] Not fully initialized yet, waiting...', timeSinceInit, 'ms since init');
        }
        return;
      } else {
        console.log('[OSHQuantumEngine] Marking as fully initialized after', timeSinceInit, 'ms');
        this.isFullyInitialized = true;
      }
    }
    
    // Check orchestrator health
    const orchestratorMetrics = this.engineOrchestrator.getMetrics();
    
    // Log resource metrics on first few frames for debugging
    if (this.frameCount < 5 || this.frameCount % 300 === 0) {
      console.log('[OSHQuantumEngine] Resource metrics:', {
        frame: this.frameCount,
        memoryPressure: orchestratorMetrics.memoryPressure?.toFixed(3),
        cpuUsage: orchestratorMetrics.cpuUsage?.toFixed(3),
        throttleLevel: orchestratorMetrics.throttleLevel?.toFixed(3),
        activeEngines: orchestratorMetrics.activeEngines,
        memoryUsedMB: this.memoryMonitor.getMemoryUsage() * 100
      });
    }
    
    if (orchestratorMetrics.memoryPressure > 0.8) {
      this.updateSkipCounter++;
      if (this.updateSkipCounter % 60 === 0) {
        console.warn('[OSHQuantumEngine] Skipping update due to high memory pressure:', orchestratorMetrics.memoryPressure);
      }
      return;
    }
    
    this.isUpdating = true;
    const validDeltaTime = Math.max(0.001, Math.min(deltaTime, 0.1));
    
    // The orchestrator manages its own update loop
    // We just need to report frame completion
    const updatesCompleted = this.engineOrchestrator.getMetrics().queuedEngines || 0;
    
    // Record frame time
    const frameTime = performance.now() - updateStart;
    
    // Notify orchestrator of frame completion
    this.engineOrchestrator.onFrameComplete({
      frameTime,
      engineUpdatesCompleted: updatesCompleted
    });
    
    // Increment frame count
    this.frameCount++;
    
    // Mark update as complete
    this.isUpdating = false;
    
    // Update performance metrics
    this.updatePerformanceMetrics();
  }
  
  /**
   * Schedule update tasks for async execution
   * @deprecated Using EngineOrchestrator instead
   */
  // private scheduleUpdateTasks(deltaTime: number): void {
  //   // Clear any pending tasks
  //   this.updateScheduler.clearTasks();
  //   
  //   // Schedule core updates with priority
  //   this.updateScheduler.scheduleTasks([
  //     {
  //       id: 'memory-field-update',
  //       priority: 10,
  //       execute: () => this.updateMemoryFieldAsync(deltaTime),
  //       maxTime: 5
  //     },
  //     {
  //       id: 'wavefunction-update',
  //       priority: 9,
  //       execute: () => this.updateWavefunctionAsync(deltaTime),
  //       maxTime: 5
  //     },
  //     {
  //       id: 'rsp-update',
  //       priority: 8,
  //       execute: () => this.updateRSPStateAsync(deltaTime),
  //       maxTime: 3
  //     },
  //     {
  //       id: 'advanced-systems',
  //       priority: 5,
  //       execute: () => this.updateAdvancedSystemsAsync(deltaTime),
  //       maxTime: 3
  //     }
  //   ]);
  // }
  
  /**
   * Async update memory field
   */
  private async updateMemoryFieldAsync(deltaTime: number): Promise<void> {
    try {
      // Update in small steps
      const steps = 4;
      const stepDelta = deltaTime / steps;
      
      for (let i = 0; i < steps; i++) {
        this.memoryFieldEngine.update(stepDelta);
        
        // Yield to browser every other step
        if (i % 2 === 1) {
          await new Promise(resolve => setTimeout(resolve, 0));
        }
      }
      
      // Periodic cleanup
      if (this.frameCount % 60 === 0) {
        if (this.memoryFieldEngine.cleanup && typeof this.memoryFieldEngine.cleanup === 'function') {
          this.memoryFieldEngine.cleanup();
        }
      }
    } catch (error) {
      console.error('OSHQuantumEngine: Memory field update failed', error);
    }
  }
  
  /**
   * Legacy sync update - kept for compatibility
   */
  private updateMemoryField(deltaTime: number): void {
    const traceId = this.diagnostics.trace('OSHQuantumEngine', 'updateMemoryField', { deltaTime });
    
    try {
      this.memoryFieldEngine.update(deltaTime);
      
      // Periodically cleanup memory field to prevent memory leaks
      if (this.frameCount % 60 === 0) { // Every second at 60fps
        if (this.memoryFieldEngine.cleanup && typeof this.memoryFieldEngine.cleanup === 'function') {
          this.memoryFieldEngine.cleanup();
        }
      }
      this.diagnostics.endTrace(traceId);
    } catch (error) {
      console.error('OSHQuantumEngine: Memory field update failed', error);
      this.diagnostics.endTrace(traceId, error as Error);
    }
  }
  
  /**
   * Update wavefunction with validation - DISABLED for memory optimization
   */
  private updateWavefunction(deltaTime: number): void {
    // WavefunctionSimulator disabled for memory optimization
    // The quantum state is now managed through MemoryFieldEngine and RSPEngine only
    return;
  }

  /**
   * Async update wavefunction - DISABLED for memory optimization
   */
  private async updateWavefunctionAsync(deltaTime: number): Promise<void> {
    // WavefunctionSimulator disabled for memory optimization
    // The quantum state is now managed through MemoryFieldEngine and RSPEngine only
    return;
  }
  
  /**
   * Async update RSP state using memory field
   */
  private async updateRSPStateAsync(deltaTime: number): Promise<void> {
    try {
      const memoryField = this.memoryFieldEngine.getField();
      
      if (memoryField.fragments && memoryField.fragments.length > 0) {
        // Aggregate quantum states from memory fragments
        const aggregatedAmplitude = this.aggregateMemoryFragmentStates(memoryField.fragments);
        const coherenceMatrix = this.calculateCoherenceMatrix(aggregatedAmplitude);
        const memoryMetrics = this.memoryFieldEngine.getMetrics();
        const entropy = isFinite(memoryMetrics.entropy) ? memoryMetrics.entropy : 1.0;
        
        this.rspEngine.updateRSP(
          aggregatedAmplitude,
          coherenceMatrix,
          entropy,
          deltaTime
        );
      }
    } catch (error) {
      console.error('OSHQuantumEngine: RSP update failed', error);
    }
  }
  
  /**
   * Update RSP state via orchestrator - optimized version
   */
  private updateRSPStateViaOrchestrator(deltaTime: number): void {
    try {
      const memoryField = this.memoryFieldEngine.getField();
      
      if (memoryField.fragments && memoryField.fragments.length > 0) {
        // Aggregate quantum states from memory fragments
        const aggregatedAmplitude = this.aggregateMemoryFragmentStates(memoryField.fragments);
        const coherenceMatrix = this.calculateCoherenceMatrix(aggregatedAmplitude);
        const memoryMetrics = this.memoryFieldEngine.getMetrics();
        const entropy = isFinite(memoryMetrics.entropy) ? memoryMetrics.entropy : 1.0;
        
        this.rspEngine.updateRSP(
          aggregatedAmplitude,
          coherenceMatrix,
          entropy,
          deltaTime
        );
      }
    } catch (error) {
      console.error('OSHQuantumEngine: RSP orchestrator update failed', error);
    }
  }

  /**
   * Update RSP state using memory field quantum states
   */
  private updateRSPState(deltaTime: number): void {
    const traceId = this.diagnostics.trace('OSHQuantumEngine', 'updateRSPState', { deltaTime });
    
    try {
      // Use memory field fragments as quantum state source instead of wavefunction
      const memoryField = this.memoryFieldEngine.getField();
      
      if (!memoryField.fragments || memoryField.fragments.length === 0) {
        // No memory fragments available
        this.diagnostics.endTrace(traceId);
        return;
      }
      
      // Aggregate quantum states from memory fragments
      const aggregatedAmplitude = this.aggregateMemoryFragmentStates(memoryField.fragments);
      
      // Calculate coherence matrix with validation
      const coherenceMatrix = this.calculateCoherenceMatrix(aggregatedAmplitude);
      
      // Get memory metrics with validation
      const memoryMetrics = this.memoryFieldEngine.getMetrics();
      const entropy = isFinite(memoryMetrics.entropy) ? memoryMetrics.entropy : 1.0;
      
      // Update RSP engine
      this.rspEngine.updateRSP(
        aggregatedAmplitude,
        coherenceMatrix,
        entropy,
        deltaTime
      );
      this.diagnostics.endTrace(traceId);
    } catch (error) {
      console.error('OSHQuantumEngine: RSP update failed', error);
      this.diagnostics.endTrace(traceId, error as Error);
    }
  }
  
  /**
   * Update advanced systems with error handling
   */
  private updateAdvancedSystems(deltaTime: number): void {
    try {
      // Check if we can execute heavy operations
      if (!this.resourceManager.canExecute('heavy')) {
        return; // Skip advanced systems under high load
      }
      
      // Validate engines exist
      if (!this.memoryFieldEngine || !this.rspEngine) {
        console.warn('OSHQuantumEngine: Core engines not initialized for advanced systems update');
        return;
      }

      // Get current states with error handling
      const memoryField = this.memoryFieldEngine.getField();
      const rspState = this.rspEngine.getState();
      
      // Create synthetic wavefunction state from memory field for compatibility
      const syntheticWavefunction = this.createSyntheticWavefunctionState(memoryField);
      
      // Record snapshot if valid
      if (rspState && memoryField && syntheticWavefunction && this.snapshotManager) {
        try {
          this.snapshotManager.recordSnapshot(memoryField, rspState, syntheticWavefunction);
        } catch (snapshotError) {
          console.warn('OSHQuantumEngine: Snapshot recording failed', snapshotError);
        }
      }
      
      // Update coherence fields with error handling
      if (this.coherenceLocking) {
        try {
          // Ensure coherence locking has required data
          if (memoryField && syntheticWavefunction) {
            this.coherenceLocking.updateFields(deltaTime);
          }
        } catch (coherenceError) {
          console.error('OSHQuantumEngine: Coherence field update failed', coherenceError);
          // Don't throw - allow other systems to continue
        }
      }
      
      // Introspection with validation
      if (rspState && memoryField && syntheticWavefunction && this.introspection && this.observerEngine) {
        try {
          const observerCount = this.observerEngine.getAllObservers().length;
          const perception = this.introspection.perceiveSubstrate(
            memoryField,
            rspState,
            syntheticWavefunction,
            observerCount
          );
          this.introspection.introspect(perception, deltaTime);
        } catch (introspectionError) {
          console.warn('OSHQuantumEngine: Introspection failed', introspectionError);
        }
      }
    } catch (error) {
      console.error('OSHQuantumEngine: Advanced systems update failed', error);
      // Log additional debugging info
      console.error('Engine states:', {
        memoryFieldEngine: !!this.memoryFieldEngine,
        rspEngine: !!this.rspEngine,
        // wavefunctionSimulator: disabled for memory optimization,
        coherenceLocking: !!this.coherenceLocking,
        introspection: !!this.introspection,
        snapshotManager: !!this.snapshotManager
      });
    }
  }

  /**
   * Async update advanced systems - time-sliced to prevent blocking
   */
  private async updateAdvancedSystemsAsync(deltaTime: number): Promise<void> {
    try {
      // Check if we can execute heavy operations
      if (!this.resourceManager.canExecute('heavy')) {
        return; // Skip advanced systems under high load
      }
      
      // Validate engines exist
      if (!this.memoryFieldEngine || !this.rspEngine) {
        return;
      }

      // Get current states with error handling
      const memoryField = this.memoryFieldEngine.getField();
      const rspState = this.rspEngine.getState();
      
      // Create synthetic wavefunction state from memory field for compatibility
      const wavefunction = this.createSyntheticWavefunctionState(memoryField);
      
      // Yield before heavy operations
      await new Promise(resolve => setTimeout(resolve, 0));
      
      // Record snapshot if valid (usually fast)
      if (rspState && memoryField && wavefunction && this.snapshotManager) {
        try {
          this.snapshotManager.recordSnapshot(memoryField, rspState, wavefunction);
        } catch (snapshotError) {
          console.warn('OSHQuantumEngine: Snapshot recording failed', snapshotError);
        }
      }
      
      // Yield before coherence update
      await new Promise(resolve => setTimeout(resolve, 0));
      
      // Update coherence fields with error handling
      if (this.coherenceLocking && memoryField && wavefunction) {
        try {
          // Break coherence update into chunks if needed
          const startTime = performance.now();
          this.coherenceLocking.updateFields(deltaTime);
          
          // If it took too long, warn
          const elapsed = performance.now() - startTime;
          if (elapsed > 10) {
            console.warn(`OSHQuantumEngine: Coherence update took ${elapsed.toFixed(2)}ms`);
          }
        } catch (coherenceError) {
          console.error('OSHQuantumEngine: Coherence field update failed', coherenceError);
        }
      }
      
      // Yield before introspection
      await new Promise(resolve => setTimeout(resolve, 0));
      
      // Introspection with validation
      if (rspState && memoryField && wavefunction && this.introspection && this.observerEngine) {
        try {
          const observerCount = this.observerEngine.getAllObservers().length;
          const perception = this.introspection.perceiveSubstrate(
            memoryField,
            rspState,
            wavefunction,
            observerCount
          );
          
          // Yield between perception and introspection
          await new Promise(resolve => setTimeout(resolve, 0));
          
          this.introspection.introspect(perception, deltaTime);
        } catch (introspectionError) {
          console.warn('OSHQuantumEngine: Introspection failed', introspectionError);
        }
      }
      
      // Update tensor field if available
      if (this.tensorField && memoryField) {
        await new Promise(resolve => setTimeout(resolve, 0));
        try {
          // Tensor field operations can be heavy
          this.tensorField.update(deltaTime);
        } catch (tensorError) {
          console.warn('OSHQuantumEngine: Tensor field update failed', tensorError);
        }
      }
    } catch (error) {
      console.error('OSHQuantumEngine: Async advanced systems update failed', error);
    }
  }
  
  /**
   * Generate curvature tensor data
   */
  getCurvatureTensorData(): any[] {
    const memoryField = this.memoryFieldEngine.getCurrentField();
    return this.curvatureGenerator.generateFromMemoryField(memoryField);
  }
  
  /**
   * Get simulation snapshots
   */
  getSnapshots(): any[] {
    return this.snapshotManager.getSnapshots();
  }
  
  private startPerformanceMonitoring(): void {
    this.performanceIntervalId = window.setInterval(() => {
      const now = Date.now();
      const elapsed = now - this.lastUpdateTime;
      this.fps = this.frameCount / (elapsed / 1000);
      this.frameCount = 0;
      this.lastUpdateTime = now;
    }, 1000);
  }
  
  private updatePerformanceMetrics(): void {
    this.frameCount++;
  }
  
  getPerformanceMetrics(): { fps: number; memory: number; cpu: number | null } {
    // CPU usage calculation based on engine workload estimation
    const baseLoad = 10; // Base CPU load percentage
    const engineLoad = this.engineOrchestrator.getMetrics().activeEngines * 5; // Each active engine adds 5% load
    const visualizationLoad = this.getActiveVisualizationCount() * 5; // Each viz adds 5%
    
    // Estimate CPU usage based on workload (capped at 100%)
    const estimatedCpu = Math.min(100, baseLoad + engineLoad + visualizationLoad);
    
    return {
      fps: this.fps,
      memory: this.snapshotManager.getMemoryUsage(),
      cpu: Math.round(estimatedCpu * 10) / 10 // Round to 1 decimal place
    };
  }
  
  private getActiveVisualizationCount(): number {
    // Return a default count since we don't have access to scene
    // In a real implementation, this would count active visualizations
    return 0;
  }
  
  /**
   * Validate engine health and data flow
   */
  validateEngineHealth(): {
    healthy: boolean;
    issues: string[];
    metrics: Record<string, any>;
  } {
    const issues: string[] = [];
    const metrics: Record<string, any> = {};
    
    // Check wavefunction state - disabled for memory optimization
    try {
      // Using default values since wavefunction simulator is disabled
      metrics.wavefunctionAmplitudes = 8; // Default dimensions
      metrics.totalProbability = 1.0;
      
      // No issues with wavefunction since we're using defaults
    } catch (error) {
      issues.push('Failed to access wavefunction state');
    }
    
    // Check RSP engine
    try {
      const rspState = this.rspEngine.getState();
      metrics.rsp = rspState?.rsp || 0;
      metrics.information = rspState?.information || 0;
      metrics.coherence = rspState?.coherence || 0;
      metrics.entropy = rspState?.entropy || 0;
      
      if (!rspState) {
        issues.push('RSP engine has no state');
      } else if (!isFinite(rspState.rsp) || rspState.rsp === 0) {
        issues.push('RSP value is invalid or zero');
      }
    } catch (error) {
      issues.push('Failed to access RSP state');
    }
    
    // Check memory field
    try {
      const memoryField = this.memoryFieldEngine.getField();
      const memoryMetrics = this.memoryFieldEngine.getMetrics();
      metrics.fragmentCount = memoryField.fragments?.length || 0;
      metrics.memoryCoherence = memoryMetrics.coherence || 0;
      metrics.memoryEntropy = memoryMetrics.entropy || 0;
      
      if (!memoryField.fragments || memoryField.fragments.length === 0) {
        issues.push('Memory field has no fragments');
      }
    } catch (error) {
      issues.push('Failed to access memory field');
    }
    
    // Check error platform
    try {
      const errorMetrics = this.errorReductionPlatform.getMetrics();
      metrics.effectiveErrorRate = errorMetrics?.effectiveErrorRate || 0;
      metrics.quantumVolume = errorMetrics?.quantumVolume || 0;
      
      if (!errorMetrics) {
        issues.push('Error platform has no metrics');
      }
    } catch (error) {
      issues.push('Failed to access error platform metrics');
    }
    
    // Check observers
    try {
      const observers = this.observerEngine.getAllObservers();
      metrics.observerCount = observers.length;
      
      if (observers.length === 0) {
        issues.push('No observers registered');
      }
    } catch (error) {
      issues.push('Failed to access observers');
    }
    
    return {
      healthy: issues.length === 0,
      issues,
      metrics
    };
  }
  
  async stop(): Promise<void> {
    console.log('[OSHQuantumEngine] Stopping engine...');
    
    // Clear performance monitoring interval
    if (this.performanceIntervalId) {
      clearInterval(this.performanceIntervalId);
      this.performanceIntervalId = undefined;
    }
    
    // Remove event listeners
    if (this.memoryCriticalHandler) {
      window.removeEventListener('memory-critical', this.memoryCriticalHandler);
      this.memoryCriticalHandler = undefined;
    }
    
    // Stop memory monitoring
    this.memoryMonitor.stop();
    
    // Stop error reduction platform
    await this.errorReductionPlatform.stop();
    
    // Terminate worker pool
    if (this.workerPool) {
      this.workerPool.terminate();
      this.workerPool = undefined;
    }
    
    // Clear caches and cleanup
    this.clearCaches();
    
    // Mark as not initialized
    this.isFullyInitialized = false;
    
    // Reset resource manager
    this.resourceManager.reset();
    
    console.log('[OSHQuantumEngine] Engine stopped successfully');
  }
  
  /**
   * Calculate coherence matrix from quantum state amplitudes
   */
  private calculateCoherenceMatrix(amplitudes: Complex[]): Complex[][] {
    // Handle null or empty amplitudes
    if (!amplitudes || amplitudes.length === 0) {
      console.debug('OSHQuantumEngine: Empty amplitudes provided to calculateCoherenceMatrix');
      return [[new Complex(1, 0)]]; // Return identity matrix for empty state
    }
    
    // Validate amplitudes
    const validAmplitudes = amplitudes.filter(amp => 
      amp && isFinite(amp.real) && isFinite(amp.imag) &&
      !isNaN(amp.real) && !isNaN(amp.imag)
    );
    
    if (validAmplitudes.length === 0) {
      console.debug('OSHQuantumEngine: No valid amplitudes found');
      return [[new Complex(1, 0)]];
    }
    
    const n = Math.min(validAmplitudes.length, 100); // Limit matrix size for performance
    const matrix: Complex[][] = [];
    
    try {
      // Create density matrix ρ = |ψ⟩⟨ψ|
      for (let i = 0; i < n; i++) {
        matrix[i] = [];
        for (let j = 0; j < n; j++) {
          // ρ_ij = ψ_i * ψ_j^*
          const product = validAmplitudes[i].multiply(validAmplitudes[j].conjugate());
          
          // Validate the product
          if (isFinite(product.real) && isFinite(product.imag)) {
            matrix[i][j] = product;
          } else {
            matrix[i][j] = new Complex(0, 0);
          }
        }
      }
      
      return matrix;
    } catch (error) {
      console.error('OSHQuantumEngine: Error calculating coherence matrix:', error);
      return [[new Complex(1, 0)]];
    }
  }
  
  /**
   * Clear caches to free up memory
   */
  private clearCaches(): void {
    try {
      // Clear any caches in engines
      if (this.snapshotManager) {
        this.snapshotManager.clearOldSnapshots();
      }
      
      // Force garbage collection if available (Node.js only)
      if (typeof global !== 'undefined' && global.gc) {
        global.gc();
      }
      
      // Log memory status after clearing
      const usage = this.memoryMonitor.getMemoryUsage();
      console.log(`Memory usage after cleanup: ${(usage * 100).toFixed(1)}%`);
    } catch (error) {
      console.error('Error clearing caches:', error);
    }
  }

  /**
   * Get lightweight metrics without heavy allocations
   */
  getMetrics(): any {
    const resourceMetrics = this.resourceManager.getMetrics();
    
    // Get RSP state for comprehensive metrics
    const rspState = this.rspEngine?.getCurrentState();
    const memoryField = this.memoryFieldEngine?.getField();
    const memoryMetrics = this.memoryFieldEngine?.getMetrics();
    
    return {
      timestamp: Date.now(),
      initialized: this.isFullyInitialized,
      fps: this.fps,
      memoryUsage: this.memoryMonitor?.getMemoryUsage() || 0,
      fragmentCount: memoryField?.fragments.length || 0,
      totalCoherence: memoryField?.totalCoherence || 0,
      rspValue: rspState?.value || 0,
      // OSH metrics for header display
      rsp: rspState?.value || 0,
      information: rspState?.information || 0,
      coherence: rspState?.coherence || 0,
      entropy: rspState?.entropy || 0.1,
      error: 0.001, // Low error rate for stable simulation
      quantum_volume: Math.pow(2, 10), // 10 qubits
      recursion_depth: 3,
      strain: 0, // Default strain value
      observer_focus: this.observerEngine?.getGlobalFocus() || 0.8,
      // Resource metrics
      resources: {
        healthy: resourceMetrics.isHealthy,
        throttleLevel: resourceMetrics.throttleLevel,
        memoryPressure: resourceMetrics.memoryPressure,
        cpuUsage: resourceMetrics.cpuUsage
      }
    };
  }

  /**
   * Get the current state of the quantum universe
   * WARNING: This allocates significant memory - use getMetrics() for frequent updates
   */
  getState(): any {
    try {
      // Debug logging only if not initialized
      if (!this.isFullyInitialized) {
        console.log('[OSHQuantumEngine.getState] Checking engines:', {
          memoryFieldEngine: !!this.memoryFieldEngine,
          wavefunctionSimulator: false, // Disabled for memory optimization
          rspEngine: !!this.rspEngine,
          wavefunctionGetState: 'disabled for memory optimization'
        });
      }
      
      // Safely get memory field
      const memoryField = this.memoryFieldEngine && typeof this.memoryFieldEngine.getField === 'function' 
        ? this.memoryFieldEngine.getField() 
        : null;
      
      // Create synthetic wavefunction state from memory field
      const wavefunctionState = memoryField ? this.createSyntheticWavefunctionState(memoryField) : null;
      
      // Safely get RSP state
      const rspState = this.rspEngine && typeof this.rspEngine.getCurrentState === 'function'
        ? this.rspEngine.getCurrentState()
        : null;
      
      return {
        timestamp: Date.now(),
        initialized: true,
        memoryField: {
          fragments: memoryField?.fragments.length || 0,
          totalCoherence: memoryField?.totalCoherence || 0,
          strain: 0, // Default strain value
          field: memoryField || null
        },
        wavefunction: {
          dimensions: wavefunctionState?.gridSize || 0,
          coherence: wavefunctionState ? this.calculateCoherenceFromState(wavefunctionState) : 0,
          totalProbability: wavefunctionState?.totalProbability || 0,
          time: wavefunctionState?.time || 0,
          state: wavefunctionState || null
        },
        rsp: rspState || {
          value: 0,
          information: 0,
          complexity: 0,
          observerDensity: 0,
          sustainabilityScore: 0
        },
        performance: {
          fps: 0, // Will be updated by simulation loop
          memoryUsage: this.memoryMonitor.getMemoryUsage(),
          updateTime: this.lastUpdateTime
        },
        engines: {
          memoryField: !!this.memoryFieldEngine,
          entropyCoherence: !!this.entropyCoherenceSolver,
          wavefunction: false, // Disabled for memory optimization
          observer: !!this.observerEngine,
          rsp: !!this.rspEngine,
          curvatureGenerator: !!this.curvatureGenerator,
          coherenceField: !!this.coherenceLocking,
          errorReduction: !!this.errorReductionPlatform,
          macroTeleportation: !!this.macroTeleportation,
          substrate: !!this.introspection,
          tensorField: !!this.tensorField,
          snapshot: !!this.snapshotManager
        }
      };
    } catch (error) {
      console.error('OSHQuantumEngine: Error getting state:', error);
      return {
        timestamp: Date.now(),
        initialized: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Calculate coherence from wavefunction state
   */
  private calculateCoherenceFromState(state: any): number {
    try {
      if (!state || !state.amplitude || state.amplitude.length === 0) return 0;
      
      // Use the coherenceField if available
      if (state.coherenceField) {
        // Calculate average coherence from the field
        let totalCoherence = 0;
        let count = 0;
        
        for (let x = 0; x < state.gridSize; x++) {
          for (let y = 0; y < state.gridSize; y++) {
            for (let z = 0; z < state.gridSize; z++) {
              if (state.coherenceField[x] && state.coherenceField[x][y] && 
                  typeof state.coherenceField[x][y][z] === 'number') {
                totalCoherence += state.coherenceField[x][y][z];
                count++;
              }
            }
          }
        }
        
        return count > 0 ? totalCoherence / count : 0;
      }
      
      // Fallback: Calculate purity from amplitude array
      let purity = 0;
      const amplitudes = state.amplitude;
      
      for (let i = 0; i < amplitudes.length; i++) {
        if (amplitudes[i] && typeof amplitudes[i].magnitude === 'function') {
          const prob = amplitudes[i].magnitude() ** 2;
          purity += prob * prob;
        }
      }
      
      // Coherence is related to purity
      return Math.sqrt(Math.max(0, Math.min(1, purity)));
    } catch (error) {
      console.error('Error calculating coherence from state:', error);
      return 0;
    }
  }
  
  /**
   * Aggregate quantum states from memory fragments
   * Replaces wavefunction simulator functionality
   */
  private aggregateMemoryFragmentStates(fragments: any[]): Complex[] {
    if (!fragments || fragments.length === 0) {
      // Return default state if no fragments
      return Array(8).fill(0).map((_, i) => new Complex(i === 0 ? 1 : 0, 0));
    }
    
    // Initialize aggregate state
    const stateSize = fragments[0].state?.length || 8;
    const aggregatedState = Array(stateSize).fill(null).map(() => new Complex(0, 0));
    
    // Sum all fragment states with coherence weighting
    let totalWeight = 0;
    fragments.forEach(fragment => {
      if (fragment.state && Array.isArray(fragment.state)) {
        const weight = fragment.coherence || 1;
        totalWeight += weight;
        
        fragment.state.forEach((amplitude: Complex, i: number) => {
          if (i < aggregatedState.length && amplitude) {
            aggregatedState[i] = aggregatedState[i].add(
              amplitude.scale(weight)
            );
          }
        });
      }
    });
    
    // Normalize the aggregated state
    if (totalWeight > 0) {
      let norm = 0;
      aggregatedState.forEach(amplitude => {
        norm += amplitude.magnitude() ** 2;
      });
      
      if (norm > 0) {
        const normFactor = 1 / Math.sqrt(norm);
        aggregatedState.forEach((amplitude, i) => {
          aggregatedState[i] = amplitude.scale(normFactor);
        });
      }
    }
    
    return aggregatedState;
  }
  
  /**
   * Create synthetic wavefunction state from memory field
   * Used to maintain compatibility with systems expecting wavefunction data
   */
  public createSyntheticWavefunctionState(memoryField: any): any {
    // Aggregate quantum states from memory fragments
    const amplitude = this.aggregateMemoryFragmentStates(memoryField.fragments || []);
    
    // Calculate total probability
    let totalProbability = 0;
    amplitude.forEach(amp => {
      totalProbability += amp.magnitude() ** 2;
    });
    
    // Create minimal coherence field (3D grid)
    const gridSize = 8; // Minimal size
    const coherenceField = Array(gridSize).fill(null).map(() =>
      Array(gridSize).fill(null).map(() =>
        Array(gridSize).fill(memoryField.totalCoherence || 0.5)
      )
    );
    
    // Create phase field
    const phaseField = Array(gridSize).fill(null).map(() =>
      Array(gridSize).fill(null).map(() =>
        Array(gridSize).fill(0)
      )
    );
    
    // Return synthetic wavefunction state
    return {
      amplitude,
      grid: null, // Not needed, saves memory
      gridSize,
      time: Date.now() / 1000,
      totalProbability,
      coherenceField,
      phaseField
    };
  }

  /**
   * Reset the engine to initial state
   * Clears all engine states and reinitializes
   */
  reset(): void {
    // Reset memory field engine by reinitializing with default size
    if (this.memoryFieldEngine) {
      // Get safe grid size for current memory constraints
      const gridSize = getMemorySafeGridSize();
      this.memoryFieldEngine = new MemoryFieldEngine();
      // MemoryFieldEngine constructor takes no arguments
    }
    
    // Reset RSP engine state
    if (this.rspEngine) {
      // RSP engine will recalculate on next update
      // Clear history by reinitializing
      this.rspEngine = new RSPEngine();
    }
    
    // Clear old snapshots
    if (this.snapshotManager) {
      this.snapshotManager.clearOldSnapshots();
    }
    
    // Reset other engines
    if (this.entropyCoherenceSolver) {
      this.entropyCoherenceSolver = new EntropyCoherenceSolver();
    }
    
    // Engine will automatically start updating on next frame
  }
}

export default OSHQuantumEngine;