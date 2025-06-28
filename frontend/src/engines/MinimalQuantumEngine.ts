/**
 * Minimal Quantum Engine - Simplified simulation engine for testing
 * 
 * A lightweight implementation that maintains the core quantum simulation
 * functionality while avoiding heavy computations that block the browser.
 */

import { Complex } from '../utils/complex';

export interface MinimalEngineState {
  timestamp: number;
  coherence: number;
  entropy: number;
  rsp: number;
  fragments: number;
  fps: number;
}

export class MinimalQuantumEngine {
  private coherence: number = 1.0;
  private entropy: number = 0.0;
  private rsp: number = 1.0;
  private fragments: number = 1;
  private lastUpdateTime: number = Date.now();
  private frameCount: number = 0;
  private fps: number = 60;
  
  constructor() {
    console.log('[MinimalQuantumEngine] Initialized');
  }
  
  /**
   * Lightweight update that won't block
   */
  update(deltaTime: number): void {
    // Simple coherence decay
    this.coherence *= (1 - 0.001 * deltaTime);
    if (this.coherence < 0.1) this.coherence = 0.1;
    
    // Simple entropy increase
    this.entropy += 0.01 * deltaTime;
    if (this.entropy > 1) this.entropy = 1;
    
    // Simple RSP calculation
    this.rsp = this.coherence / (this.entropy + 0.1);
    
    // Fragment simulation
    if (Math.random() < 0.01) {
      this.fragments = Math.min(50, this.fragments + 1);
    }
    if (Math.random() < 0.005) {
      this.fragments = Math.max(1, this.fragments - 1);
    }
    
    // FPS calculation
    this.frameCount++;
    const now = Date.now();
    if (now - this.lastUpdateTime > 1000) {
      this.fps = this.frameCount;
      this.frameCount = 0;
      this.lastUpdateTime = now;
    }
  }
  
  /**
   * Get current state
   */
  getState(): MinimalEngineState {
    return {
      timestamp: Date.now(),
      coherence: this.coherence,
      entropy: this.entropy,
      rsp: this.rsp,
      fragments: this.fragments,
      fps: this.fps
    };
  }
  
  /**
   * Get lightweight metrics
   */
  getMetrics(): any {
    return {
      timestamp: Date.now(),
      initialized: true,
      fps: this.fps,
      memoryUsage: 0.1, // Minimal memory usage
      fragmentCount: this.fragments,
      totalCoherence: this.coherence,
      rspValue: this.rsp,
      resources: {
        healthy: true,
        throttleLevel: 0,
        memoryPressure: 0.1,
        cpuUsage: 5
      }
    };
  }
  
  /**
   * Stop the engine
   */
  async stop(): Promise<void> {
    console.log('[MinimalQuantumEngine] Stopped');
  }
  
  /**
   * Start the engine
   */
  start(): void {
    console.log('[MinimalQuantumEngine] Started');
  }
}

/**
 * Factory to create minimal engine with OSHQuantumEngine interface
 */
export function createMinimalEngine(): any {
  const engine = new MinimalQuantumEngine();
  
  // Add stub methods for compatibility
  const stubs = {
    memoryFieldEngine: { 
      getField: () => ({ fragments: [], totalCoherence: 1, strain: 0 }),
      update: () => {},
      cleanup: () => {}
    },
    // WavefunctionSimulator disabled for memory optimization
    wavefunctionSimulator: null as any, /*{
      getState: () => ({ amplitude: [], gridSize: 8, totalProbability: 1 }),
      evolve: () => {}
    },*/
    rspEngine: {
      getCurrentState: () => ({ value: 1, information: 1, complexity: 0.5 }),
      getState: () => ({ value: 1 }),
      updateRSP: () => {}
    },
    observerEngine: {
      getAllObservers: () => [],
      addObserver: () => {}
    }
  };
  
  return Object.assign(engine, stubs);
}