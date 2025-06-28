/**
 * Engine Orchestrator
 * Enterprise-grade resource management and coordination for quantum simulation engines
 * Prevents resource exhaustion through intelligent scheduling and priority management
 */

import { EventEmitter } from '../utils/EventEmitter';
import { ResourceManager } from './ResourceManager';
import { DiagnosticsSystem } from '../utils/diagnostics';

export interface EngineConfig {
  id: string;
  priority: number; // 1-10, higher is more important
  resourceWeight: number; // Estimated resource consumption (0-1)
  required: boolean; // Core engines that must always run
  lazy: boolean; // Can be initialized on-demand
  updateFrequency: number; // How often to update (ms)
  dependencies?: string[]; // Other engines this depends on
}

export interface ManagedEngine {
  config: EngineConfig;
  instance: any;
  lastUpdate: number;
  updateCount: number;
  averageUpdateTime: number;
  isActive: boolean;
  isInitialized: boolean;
  errorCount?: number;
}

export class EngineOrchestrator extends EventEmitter {
  private static instance: EngineOrchestrator;
  private engines: Map<string, ManagedEngine> = new Map();
  private updateQueue: string[] = [];
  private isUpdating: boolean = false;
  private resourceManager: ResourceManager;
  private diagnostics: DiagnosticsSystem;
  private maxConcurrentUpdates: number = 3;
  private updateInterval?: number;
  private engineConfigs: Map<string, EngineConfig>;
  private frameMetrics = {
    updatesPerFrame: [] as number[],
    frameTime: [] as number[],
    lastFrameTime: 0
  };
  
  // Performance tracking
  private totalUpdateTime: number = 0;
  private updateCycles: number = 0;
  private lastResourceCheck: number = 0;
  private resourceCheckInterval: number = 1000; // Check resources every second
  
  private constructor() {
    super();
    this.resourceManager = ResourceManager.getInstance();
    this.diagnostics = DiagnosticsSystem.getInstance();
    this.engineConfigs = this.initializeEngineConfigs();
  }
  
  static getInstance(): EngineOrchestrator {
    if (!EngineOrchestrator.instance) {
      EngineOrchestrator.instance = new EngineOrchestrator();
    }
    return EngineOrchestrator.instance;
  }
  
  /**
   * Initialize engine configurations with optimal settings
   */
  private initializeEngineConfigs(): Map<string, EngineConfig> {
    const configs = new Map<string, EngineConfig>();
    
    // Core engines - always required
    configs.set('memoryField', {
      id: 'memoryField',
      priority: 10,
      resourceWeight: 0.15,
      required: true,
      lazy: false,
      updateFrequency: 16, // Every frame
      dependencies: []
    });
    
    configs.set('rsp', {
      id: 'rsp',
      priority: 9,
      resourceWeight: 0.1,
      required: true,
      lazy: false,
      updateFrequency: 33, // ~30fps
      dependencies: ['memoryField']
    });
    
    configs.set('observer', {
      id: 'observer',
      priority: 8,
      resourceWeight: 0.1,
      required: true,
      lazy: false,
      updateFrequency: 50,
      dependencies: ['memoryField']
    });
    
    configs.set('entropyCoherence', {
      id: 'entropyCoherence',
      priority: 7,
      resourceWeight: 0.05,
      required: true,
      lazy: false,
      updateFrequency: 100,
      dependencies: ['memoryField', 'rsp']
    });
    
    // Heavy computational engines - lazy loaded
    configs.set('errorReduction', {
      id: 'errorReduction',
      priority: 6,
      resourceWeight: 0.2,
      required: false,
      lazy: true,
      updateFrequency: 500,
      dependencies: ['memoryField', 'rsp']
    });
    
    configs.set('mlObserver', {
      id: 'mlObserver',
      priority: 5,
      resourceWeight: 0.25,
      required: false,
      lazy: true,
      updateFrequency: 1000,
      dependencies: ['observer']
    });
    
    configs.set('macroTeleportation', {
      id: 'macroTeleportation',
      priority: 4,
      resourceWeight: 0.15,
      required: false,
      lazy: true,
      updateFrequency: 2000,
      dependencies: ['memoryField', 'observer']
    });
    
    configs.set('curvatureTensor', {
      id: 'curvatureTensor',
      priority: 4,
      resourceWeight: 0.2,
      required: false,
      lazy: true,
      updateFrequency: 1000,
      dependencies: ['memoryField']
    });
    
    configs.set('coherenceLocking', {
      id: 'coherenceLocking',
      priority: 5,
      resourceWeight: 0.15,
      required: false,
      lazy: true,
      updateFrequency: 200,
      dependencies: ['memoryField', 'entropyCoherence']
    });
    
    configs.set('tensorField', {
      id: 'tensorField',
      priority: 3,
      resourceWeight: 0.3,
      required: false,
      lazy: true,
      updateFrequency: 2000,
      dependencies: ['memoryField', 'curvatureTensor']
    });
    
    configs.set('introspection', {
      id: 'introspection',
      priority: 3,
      resourceWeight: 0.2,
      required: false,
      lazy: true,
      updateFrequency: 3000,
      dependencies: ['memoryField', 'rsp', 'observer']
    });
    
    configs.set('snapshotManager', {
      id: 'snapshotManager',
      priority: 2,
      resourceWeight: 0.1,
      required: false,
      lazy: true,
      updateFrequency: 5000,
      dependencies: []
    });
    
    return configs;
  }
  
  /**
   * Register an engine with the orchestrator
   */
  registerEngine(id: string, engine: any): void {
    const config = this.engineConfigs.get(id);
    if (!config) {
      console.warn(`[EngineOrchestrator] Unknown engine ID: ${id}`);
      return;
    }
    
    const managedEngine: ManagedEngine = {
      config,
      instance: engine,
      lastUpdate: 0,
      updateCount: 0,
      averageUpdateTime: 0,
      isActive: config.required,
      isInitialized: !config.lazy
    };
    
    this.engines.set(id, managedEngine);
    console.log(`[EngineOrchestrator] Registered engine: ${id} (priority: ${config.priority}, weight: ${config.resourceWeight})`);
  }
  
  /**
   * Start orchestrated updates
   */
  start(): void {
    console.log('[EngineOrchestrator] Starting orchestrated engine updates');
    
    // Initialize required engines
    this.initializeRequiredEngines();
    
    // Start update loop
    this.updateInterval = window.setInterval(() => {
      this.orchestrateUpdates();
    }, 16); // 60fps base rate
    
    this.emit('started');
  }
  
  /**
   * Stop all engine updates
   */
  stop(): void {
    console.log('[EngineOrchestrator] Stopping orchestrated updates');
    
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = undefined;
    }
    
    // Deactivate all non-required engines
    this.engines.forEach((engine, id) => {
      if (!engine.config.required) {
        engine.isActive = false;
      }
    });
    
    this.emit('stopped');
  }
  
  /**
   * Initialize required engines
   */
  private initializeRequiredEngines(): void {
    this.engines.forEach((engine, id) => {
      if (engine.config.required && !engine.isInitialized) {
        console.log(`[EngineOrchestrator] Initializing required engine: ${id}`);
        engine.isInitialized = true;
        engine.isActive = true;
      }
    });
  }
  
  /**
   * Orchestrate engine updates based on priority and resources
   */
  private orchestrateUpdates(): void {
    if (this.isUpdating) return;
    
    this.isUpdating = true;
    const now = Date.now();
    
    // Check resources periodically
    if (now - this.lastResourceCheck > this.resourceCheckInterval) {
      this.checkAndAdjustResources();
      this.lastResourceCheck = now;
    }
    
    // Build update queue based on priority and timing
    this.buildUpdateQueue(now);
    
    // Process queue with concurrency limit
    this.processUpdateQueue();
    
    this.isUpdating = false;
  }
  
  /**
   * Check resources and adjust engine activity
   */
  private checkAndAdjustResources(): void {
    const metrics = this.resourceManager.getMetrics();
    
    if (metrics.memoryPressure > 0.8) {
      console.warn('[EngineOrchestrator] High memory pressure detected, deactivating non-essential engines');
      
      // Deactivate engines by priority (lowest first)
      const sortedEngines = Array.from(this.engines.entries())
        .sort((a, b) => a[1].config.priority - b[1].config.priority);
      
      for (const [id, engine] of sortedEngines) {
        if (!engine.config.required && engine.isActive) {
          console.log(`[EngineOrchestrator] Deactivating engine: ${id}`);
          engine.isActive = false;
          
          // Check if resources improved
          const newMetrics = this.resourceManager.getMetrics();
          if (newMetrics.memoryPressure < 0.7) break;
        }
      }
    } else if (metrics.memoryPressure < 0.5 && metrics.cpuUsage < 0.5) {
      // Resources are good, can activate more engines
      const inactiveEngines = Array.from(this.engines.entries())
        .filter(([_, engine]) => !engine.isActive && engine.isInitialized)
        .sort((a, b) => b[1].config.priority - a[1].config.priority);
      
      for (const [id, engine] of inactiveEngines) {
        // Check if activating this engine would exceed resource limits
        const projectedWeight = this.calculateTotalResourceWeight() + engine.config.resourceWeight;
        if (projectedWeight < 0.8) {
          console.log(`[EngineOrchestrator] Activating engine: ${id}`);
          engine.isActive = true;
        }
      }
    }
  }
  
  /**
   * Calculate total resource weight of active engines
   */
  private calculateTotalResourceWeight(): number {
    let total = 0;
    this.engines.forEach(engine => {
      if (engine.isActive) {
        total += engine.config.resourceWeight;
      }
    });
    return total;
  }
  
  /**
   * Build update queue based on timing and priority
   */
  private buildUpdateQueue(now: number): void {
    this.updateQueue = [];
    
    // Check each engine's update timing
    this.engines.forEach((engine, id) => {
      if (!engine.isActive || !engine.isInitialized) return;
      
      const timeSinceLastUpdate = now - engine.lastUpdate;
      if (timeSinceLastUpdate >= engine.config.updateFrequency) {
        // Check dependencies
        if (this.areDependenciesMet(engine.config.dependencies || [])) {
          this.updateQueue.push(id);
        }
      }
    });
    
    // Sort by priority (highest first)
    this.updateQueue.sort((a, b) => {
      const engineA = this.engines.get(a)!;
      const engineB = this.engines.get(b)!;
      return engineB.config.priority - engineA.config.priority;
    });
  }
  
  /**
   * Check if engine dependencies are met
   */
  private areDependenciesMet(dependencies: string[]): boolean {
    for (const dep of dependencies) {
      const depEngine = this.engines.get(dep);
      if (!depEngine || !depEngine.isActive || !depEngine.isInitialized) {
        return false;
      }
    }
    return true;
  }
  
  /**
   * Process update queue with concurrency control
   */
  private async processUpdateQueue(): Promise<void> {
    const concurrentUpdates: Promise<void>[] = [];
    const maxConcurrent = this.resourceManager.isHealthy() ? this.maxConcurrentUpdates : 1;
    
    for (let i = 0; i < Math.min(this.updateQueue.length, maxConcurrent); i++) {
      const engineId = this.updateQueue[i];
      concurrentUpdates.push(this.updateEngine(engineId));
    }
    
    // Wait for concurrent updates to complete
    await Promise.all(concurrentUpdates);
  }
  
  /**
   * Update a single engine with error handling
   */
  private async updateEngine(engineId: string): Promise<void> {
    const engine = this.engines.get(engineId);
    if (!engine) return;
    
    const startTime = performance.now();
    
    try {
      // Call engine's update method if it exists
      if (engine.instance.update && typeof engine.instance.update === 'function') {
        const deltaTime = Math.min((Date.now() - engine.lastUpdate) / 1000, 0.1);
        engine.instance.update(deltaTime);
      }
      
      // Update metrics
      const updateTime = performance.now() - startTime;
      engine.updateCount++;
      engine.averageUpdateTime = (engine.averageUpdateTime * (engine.updateCount - 1) + updateTime) / engine.updateCount;
      engine.lastUpdate = Date.now();
      
      // Warn if update took too long
      if (updateTime > engine.config.updateFrequency * 0.8) {
        console.warn(`[EngineOrchestrator] Engine ${engineId} update took ${updateTime.toFixed(2)}ms (target: ${engine.config.updateFrequency}ms)`);
      }
      
    } catch (error) {
      // Increment error count
      engine.errorCount = (engine.errorCount || 0) + 1;
      
      // Log error with more context
      console.error(`[EngineOrchestrator] Error updating engine ${engineId}:`, {
        error,
        errorCount: engine.errorCount,
        updateCount: engine.updateCount,
        engineType: engine.instance.constructor.name
      });
      
      // Deactivate engine if it fails repeatedly (but not on first few updates)
      if (engine.errorCount > 5 && engine.updateCount > 10 && !engine.config.required) {
        console.warn(`[EngineOrchestrator] Deactivating failing engine: ${engineId} after ${engine.errorCount} errors`);
        engine.isActive = false;
      }
    }
  }
  
  /**
   * Get engine metrics for monitoring
   */
  getMetrics(): any {
    // Get resource metrics from ResourceManager
    const resourceMetrics = this.resourceManager.getMetrics();
    
    const metrics: any = {
      activeEngines: 0,
      totalEngines: this.engines.size,
      queuedEngines: this.updateQueue.length,
      resourceWeight: 0,
      memoryPressure: resourceMetrics.memoryPressure,
      cpuUsage: resourceMetrics.cpuUsage / 100, // Convert to 0-1 range
      throttleLevel: resourceMetrics.throttleLevel,
      averageUpdatesPerFrame: this.frameMetrics.updatesPerFrame.length > 0
        ? this.frameMetrics.updatesPerFrame.reduce((a, b) => a + b, 0) / this.frameMetrics.updatesPerFrame.length
        : 0,
      engineMetrics: {}
    };
    
    this.engines.forEach((engine, id) => {
      if (engine.isActive) {
        metrics.activeEngines++;
        metrics.resourceWeight += engine.config.resourceWeight;
      }
      
      metrics.engineMetrics[id] = {
        active: engine.isActive,
        initialized: engine.isInitialized,
        priority: engine.config.priority,
        weight: engine.config.resourceWeight,
        updateFreq: engine.config.updateFrequency,
        avgUpdateTime: engine.averageUpdateTime.toFixed(2),
        updateCount: engine.updateCount
      };
    });
    
    return metrics;
  }
  
  /**
   * Called when a frame completes to track metrics
   */
  onFrameComplete(frameData: { frameTime: number; engineUpdatesCompleted: number }): void {
    // Track frame metrics
    this.frameMetrics.updatesPerFrame.push(frameData.engineUpdatesCompleted || 0);
    this.frameMetrics.frameTime.push(frameData.frameTime);
    this.frameMetrics.lastFrameTime = frameData.frameTime;
    
    // Keep only last 60 frames of data
    if (this.frameMetrics.updatesPerFrame.length > 60) {
      this.frameMetrics.updatesPerFrame.shift();
    }
    if (this.frameMetrics.frameTime.length > 60) {
      this.frameMetrics.frameTime.shift();
    }
  }
  
  /**
   * Activate a specific engine on-demand
   */
  activateEngine(engineId: string): boolean {
    const engine = this.engines.get(engineId);
    if (!engine) {
      console.warn(`[EngineOrchestrator] Cannot activate unknown engine: ${engineId}`);
      return false;
    }
    
    // Check resources before activation
    const projectedWeight = this.calculateTotalResourceWeight() + engine.config.resourceWeight;
    if (projectedWeight > 0.9) {
      console.warn(`[EngineOrchestrator] Cannot activate ${engineId} - would exceed resource limits`);
      return false;
    }
    
    // Initialize if needed
    if (!engine.isInitialized && engine.config.lazy) {
      console.log(`[EngineOrchestrator] Lazy-initializing engine: ${engineId}`);
      engine.isInitialized = true;
    }
    
    engine.isActive = true;
    console.log(`[EngineOrchestrator] Activated engine: ${engineId}`);
    return true;
  }
  
  /**
   * Deactivate a specific engine
   */
  deactivateEngine(engineId: string): boolean {
    const engine = this.engines.get(engineId);
    if (!engine) {
      console.warn(`[EngineOrchestrator] Cannot deactivate unknown engine: ${engineId}`);
      return false;
    }
    
    if (engine.config.required) {
      console.warn(`[EngineOrchestrator] Cannot deactivate required engine: ${engineId}`);
      return false;
    }
    
    engine.isActive = false;
    console.log(`[EngineOrchestrator] Deactivated engine: ${engineId}`);
    return true;
  }
}