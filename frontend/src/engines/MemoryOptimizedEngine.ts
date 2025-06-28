/**
 * Memory-Optimized Engine Factory
 * Provides memory-efficient initialization and lazy loading for quantum engines
 */

import { SimulationHarness, SimulationParameters } from './SimulationHarness';
import { OSHQuantumEngine } from './OSHQuantumEngine';

export interface OptimizedEngineConfig {
  gridSize?: number;
  enableWorkers?: boolean;
  lazyInit?: boolean;
  memoryLimit?: number; // MB
}

/**
 * Default configuration for memory-constrained environments
 */
export const MEMORY_SAFE_CONFIG: SimulationParameters = {
  gridSize: 8, // 8^3 = 512 cells (vs 64^3 = 262,144)
  timeStep: 0.01,
  memoryDecayRate: 0.05,
  coherenceDiffusion: 0.1,
  observerThreshold: 0.7,
  quantumCoupling: 0.5,
  entropyWeight: 1.0,
  informationFlow: 0.8
};

/**
 * Performance configuration for powerful systems
 */
export const PERFORMANCE_CONFIG: SimulationParameters = {
  gridSize: 32, // 32^3 = 32,768 cells
  timeStep: 0.005,
  memoryDecayRate: 0.03,
  coherenceDiffusion: 0.15,
  observerThreshold: 0.8,
  quantumCoupling: 0.6,
  entropyWeight: 1.2,
  informationFlow: 0.9
};

/**
 * Get optimal configuration based on available memory
 */
export function getOptimalConfig(): SimulationParameters {
  // Check available memory
  const memory = getAvailableMemory();
  
  if (memory < 1000) { // Less than 1GB available
    return MEMORY_SAFE_CONFIG;
  } else if (memory < 2000) { // Less than 2GB available
    return {
      ...MEMORY_SAFE_CONFIG,
      gridSize: 16 // 16^3 = 4,096 cells
    };
  } else {
    return PERFORMANCE_CONFIG;
  }
}

/**
 * Get available memory in MB
 */
function getAvailableMemory(): number {
  // Browser memory API
  if ('memory' in performance && (performance as any).memory) {
    const memory = (performance as any).memory;
    const availableBytes = memory.jsHeapSizeLimit - memory.usedJSHeapSize;
    return availableBytes / (1024 * 1024);
  }
  
  // Default to conservative estimate
  return 500; // 500MB
}

/**
 * Create memory-optimized OSH Quantum Engine
 */
export function createOptimizedEngine(config?: OptimizedEngineConfig): OSHQuantumEngine {
  const engineConfig = config || {};
  
  // Determine grid size based on memory
  const gridSize = engineConfig.gridSize || getOptimalConfig().gridSize;
  
  // Create engine - it will automatically use memory-safe configuration
  const engine = new OSHQuantumEngine();
  
  // Add memory monitoring
  if (engineConfig.memoryLimit) {
    monitorMemoryUsage(engine, engineConfig.memoryLimit);
  }
  
  return engine;
}

/**
 * Monitor memory usage and throttle if needed
 */
function monitorMemoryUsage(engine: OSHQuantumEngine, limitMB: number): void {
  let throttled = false;
  
  const checkMemory = () => {
    const memory = getAvailableMemory();
    
    if (memory < limitMB * 0.2 && !throttled) {
      // Less than 20% of limit available - throttle
      console.warn(`Memory low (${memory.toFixed(0)}MB), throttling engine`);
      // Throttle by reducing update frequency
      throttled = true;
    } else if (memory > limitMB * 0.5 && throttled) {
      // More than 50% available - resume normal
      console.log(`Memory recovered (${memory.toFixed(0)}MB), resuming normal operation`);
      throttled = false;
    }
  };
  
  // Check every 5 seconds
  setInterval(checkMemory, 5000);
}

/**
 * Lazy-loading engine wrapper
 */
export class LazyQuantumEngine {
  private engine?: OSHQuantumEngine;
  private config: OptimizedEngineConfig;
  private initialized = false;
  
  constructor(config?: OptimizedEngineConfig) {
    this.config = config || {};
  }
  
  /**
   * Initialize engine when first needed
   */
  private ensureInitialized(): OSHQuantumEngine {
    if (!this.engine || !this.initialized) {
      console.log('Lazy-initializing quantum engine...');
      this.engine = createOptimizedEngine(this.config);
      this.initialized = true;
    }
    return this.engine;
  }
  
  // Proxy methods
  start(): void {
    this.ensureInitialized().start();
  }
  
  stop(): void {
    if (this.engine) {
      this.engine.stop();
    }
  }
  
  async step(): Promise<void> {
    const engine = this.ensureInitialized();
    if (engine.simulationHarness) {
      await engine.simulationHarness.step();
    }
  }
  
  getState(): any {
    if (!this.engine) return null;
    return {
      metrics: this.engine.simulationHarness ? this.engine.simulationHarness.getMetrics() : {},
      simulationState: this.engine.simulationHarness ? this.engine.simulationHarness.getState() : null
    };
  }
  
  dispose(): void {
    if (this.engine) {
      this.engine.stop();
      this.engine = undefined;
      this.initialized = false;
    }
  }
}

/**
 * Batch processing for memory efficiency
 */
export class BatchProcessor {
  private batchSize: number;
  private queue: Array<() => void> = [];
  private processing = false;
  
  constructor(batchSize = 10) {
    this.batchSize = batchSize;
  }
  
  /**
   * Add task to batch queue
   */
  addTask(task: () => void): void {
    this.queue.push(task);
    if (!this.processing) {
      this.processNextBatch();
    }
  }
  
  /**
   * Process next batch of tasks
   */
  private async processNextBatch(): Promise<void> {
    if (this.queue.length === 0) {
      this.processing = false;
      return;
    }
    
    this.processing = true;
    const batch = this.queue.splice(0, this.batchSize);
    
    // Process batch with small delays to prevent blocking
    for (const task of batch) {
      task();
      await new Promise(resolve => setTimeout(resolve, 1));
    }
    
    // Schedule next batch
    requestAnimationFrame(() => this.processNextBatch());
  }
}