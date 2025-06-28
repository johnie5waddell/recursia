/**
 * Simulation Guard - Prevents infinite loops and excessive computation
 * 
 * Monitors simulation operations and enforces safety limits to prevent
 * browser hanging and out-of-memory crashes.
 */

export interface SimulationLimits {
  maxIterationsPerFrame: number;
  maxTimePerOperation: number;
  maxGridSize: number;
  maxObservers: number;
  maxFragments: number;
}

export class SimulationGuard {
  private static instance: SimulationGuard;
  
  private readonly DEFAULT_LIMITS: SimulationLimits = {
    maxIterationsPerFrame: 1000,    // Max iterations in any loop per frame
    maxTimePerOperation: 50,        // Max 50ms per operation
    maxGridSize: 8,                 // 8x8x8 = 512 cells max
    maxObservers: 10,               // Max 10 observers
    maxFragments: 50                // Max 50 memory fragments
  };
  
  private limits: SimulationLimits;
  private operationStartTimes: Map<string, number> = new Map();
  private iterationCounts: Map<string, number> = new Map();
  private violations: Array<{ operation: string; reason: string; timestamp: number }> = [];
  
  private constructor() {
    this.limits = { ...this.DEFAULT_LIMITS };
  }
  
  static getInstance(): SimulationGuard {
    if (!SimulationGuard.instance) {
      SimulationGuard.instance = new SimulationGuard();
    }
    return SimulationGuard.instance;
  }
  
  /**
   * Start monitoring an operation
   */
  startOperation(operationId: string): void {
    this.operationStartTimes.set(operationId, performance.now());
    this.iterationCounts.set(operationId, 0);
  }
  
  /**
   * End monitoring an operation
   */
  endOperation(operationId: string): void {
    this.operationStartTimes.delete(operationId);
    this.iterationCounts.delete(operationId);
  }
  
  /**
   * Check if operation should continue
   */
  checkOperation(operationId: string): boolean {
    const startTime = this.operationStartTimes.get(operationId);
    if (!startTime) return true;
    
    const elapsed = performance.now() - startTime;
    if (elapsed > this.limits.maxTimePerOperation) {
      this.recordViolation(operationId, `Operation exceeded time limit: ${elapsed.toFixed(2)}ms`);
      return false;
    }
    
    return true;
  }
  
  /**
   * Check iteration count
   */
  checkIteration(operationId: string): boolean {
    const count = (this.iterationCounts.get(operationId) || 0) + 1;
    this.iterationCounts.set(operationId, count);
    
    if (count > this.limits.maxIterationsPerFrame) {
      this.recordViolation(operationId, `Iteration limit exceeded: ${count}`);
      return false;
    }
    
    return true;
  }
  
  /**
   * Validate grid size
   */
  validateGridSize(size: number): number {
    if (size > this.limits.maxGridSize) {
      console.warn(`Grid size ${size} exceeds limit, capping to ${this.limits.maxGridSize}`);
      return this.limits.maxGridSize;
    }
    return size;
  }
  
  /**
   * Validate array size before operations
   */
  validateArrayOperation(arrayLength: number, operationName: string): boolean {
    const maxSize = this.limits.maxGridSize ** 3;
    if (arrayLength > maxSize) {
      this.recordViolation(operationName, `Array size ${arrayLength} exceeds limit ${maxSize}`);
      return false;
    }
    return true;
  }
  
  /**
   * Safe loop execution with automatic breaking
   */
  safeLoop<T>(
    items: T[],
    callback: (item: T, index: number) => void,
    operationId: string
  ): void {
    this.startOperation(operationId);
    
    try {
      for (let i = 0; i < items.length; i++) {
        if (!this.checkIteration(operationId) || !this.checkOperation(operationId)) {
          console.warn(`Loop ${operationId} terminated for safety`);
          break;
        }
        
        callback(items[i], i);
      }
    } finally {
      this.endOperation(operationId);
    }
  }
  
  /**
   * Safe nested loop execution
   */
  safeNestedLoop(
    sizeX: number,
    sizeY: number,
    sizeZ: number,
    callback: (x: number, y: number, z: number) => void,
    operationId: string
  ): void {
    this.startOperation(operationId);
    
    try {
      let iterations = 0;
      const maxIterations = Math.min(
        sizeX * sizeY * sizeZ,
        this.limits.maxIterationsPerFrame
      );
      
      outerLoop: for (let x = 0; x < sizeX; x++) {
        for (let y = 0; y < sizeY; y++) {
          for (let z = 0; z < sizeZ; z++) {
            if (iterations++ >= maxIterations) {
              console.warn(`Nested loop ${operationId} reached iteration limit`);
              break outerLoop;
            }
            
            if (!this.checkOperation(operationId)) {
              console.warn(`Nested loop ${operationId} exceeded time limit`);
              break outerLoop;
            }
            
            callback(x, y, z);
          }
        }
      }
    } finally {
      this.endOperation(operationId);
    }
  }
  
  /**
   * Record a safety violation
   */
  private recordViolation(operation: string, reason: string): void {
    this.violations.push({
      operation,
      reason,
      timestamp: Date.now()
    });
    
    // Keep only recent violations
    if (this.violations.length > 100) {
      this.violations = this.violations.slice(-50);
    }
    
    console.error(`[SimulationGuard] Safety violation in ${operation}: ${reason}`);
  }
  
  /**
   * Get recent violations
   */
  getViolations(): Array<{ operation: string; reason: string; timestamp: number }> {
    return [...this.violations];
  }
  
  /**
   * Update limits based on performance
   */
  adjustLimits(performanceMetrics: { fps: number; memoryUsage: number }): void {
    // Reduce limits if performance is poor
    if (performanceMetrics.fps < 10) {
      this.limits.maxIterationsPerFrame = Math.max(100, this.limits.maxIterationsPerFrame * 0.8);
      this.limits.maxGridSize = Math.max(4, this.limits.maxGridSize - 1);
      console.warn('[SimulationGuard] Reducing limits due to poor performance');
    }
    
    // Increase limits if performance is good
    if (performanceMetrics.fps > 50 && performanceMetrics.memoryUsage < 0.5) {
      this.limits.maxIterationsPerFrame = Math.min(2000, this.limits.maxIterationsPerFrame * 1.1);
      console.log('[SimulationGuard] Increasing limits due to good performance');
    }
  }
  
  /**
   * Get current limits
   */
  getLimits(): SimulationLimits {
    return { ...this.limits };
  }
  
  /**
   * Reset guard state
   */
  reset(): void {
    this.operationStartTimes.clear();
    this.iterationCounts.clear();
    this.violations = [];
    this.limits = { ...this.DEFAULT_LIMITS };
  }
}

export default SimulationGuard;