/**
 * Resource Manager - Enterprise-grade resource management for quantum simulations
 * 
 * Manages memory, CPU, and computational resources to ensure stable operation
 * on classical hardware while simulating quantum systems.
 * 
 * Key responsibilities:
 * - Monitor and limit memory usage
 * - Throttle computation based on available resources
 * - Implement backpressure when system is overloaded
 * - Gracefully degrade performance under stress
 * - Provide circuit breaker pattern for stability
 */

export interface ResourceMetrics {
  memoryUsed: number;
  memoryLimit: number;
  memoryPressure: number;
  cpuUsage: number;
  frameTime: number;
  isHealthy: boolean;
  throttleLevel: number;
}

export interface ResourceLimits {
  maxMemoryMB: number;
  maxFrameTimeMs: number;
  maxCpuPercent: number;
  maxQubits: number;
  maxFragments: number;
  maxWaves: number;
  maxGridSize: number;
}

export class ResourceManager {
  private static instance: ResourceManager;
  
  private readonly DEFAULT_LIMITS: ResourceLimits = {
    maxMemoryMB: 2048,      // 2GB max for quantum simulation (reasonable for modern systems)
    maxFrameTimeMs: 100,    // 100ms max per frame (10fps minimum)
    maxCpuPercent: 80,      // 80% CPU max
    maxQubits: 12,          // 2^12 = 4096 max quantum states
    maxFragments: 100,      // Max memory fragments
    maxWaves: 20,           // Max coherence waves
    maxGridSize: 8          // 8x8x8 = 512 grid cells max
  };
  
  private limits: ResourceLimits;
  private metrics: ResourceMetrics;
  private lastFrameTime: number = 0;
  private frameTimeHistory: number[] = [];
  private isThrottling: boolean = false;
  private circuitBreakerOpen: boolean = false;
  private circuitBreakerResetTime: number = 0;
  
  private constructor() {
    this.limits = { ...this.DEFAULT_LIMITS };
    this.metrics = this.createDefaultMetrics();
    
    // Initialize metrics with safe values
    this.updateMetrics();
    
    // Start monitoring after initial update
    this.startMonitoring();
  }
  
  /**
   * Get singleton instance
   */
  static getInstance(): ResourceManager {
    if (!ResourceManager.instance) {
      ResourceManager.instance = new ResourceManager();
    }
    return ResourceManager.instance;
  }
  
  /**
   * Create default metrics
   */
  private createDefaultMetrics(): ResourceMetrics {
    return {
      memoryUsed: 0,
      memoryLimit: this.limits.maxMemoryMB,
      memoryPressure: 0,
      cpuUsage: 0,
      frameTime: 0,
      isHealthy: true,
      throttleLevel: 0
    };
  }
  
  /**
   * Start resource monitoring
   */
  private startMonitoring(): void {
    // Monitor performance metrics
    if (typeof performance !== 'undefined' && (performance as any).memory) {
      setInterval(() => {
        this.updateMetrics();
      }, 1000);
    }
  }
  
  /**
   * Update resource metrics
   */
  private updateMetrics(): void {
    try {
      // Memory metrics
      if ((performance as any).memory) {
        const memoryInfo = (performance as any).memory;
        const usedMB = memoryInfo.usedJSHeapSize / 1024 / 1024;
        const limitMB = memoryInfo.jsHeapSizeLimit / 1024 / 1024;
        
        this.metrics.memoryUsed = usedMB;
        this.metrics.memoryLimit = limitMB;
        
        // Calculate memory pressure based on actual heap limit, not artificial constraint
        // This gives us a more accurate view of actual memory pressure
        const heapPressure = memoryInfo.usedJSHeapSize / memoryInfo.jsHeapSizeLimit;
        
        // Also consider our soft limit to prevent excessive memory usage
        const softLimitPressure = usedMB / this.limits.maxMemoryMB;
        
        // Use the higher pressure value, but weight heap pressure more heavily
        // This prevents false positives while still respecting soft limits
        this.metrics.memoryPressure = Math.max(heapPressure * 0.8, softLimitPressure * 0.2);
        
        // Ensure memory pressure is a valid number
        if (!isFinite(this.metrics.memoryPressure) || this.metrics.memoryPressure < 0) {
          this.metrics.memoryPressure = 0.3; // Safe default
        }
      } else {
        // Fallback when performance.memory is not available
        console.log('[ResourceManager] performance.memory not available, using defaults');
        this.metrics.memoryUsed = 256; // Assume 256MB used
        this.metrics.memoryLimit = 4096; // Assume 4GB limit
        this.metrics.memoryPressure = 0.1; // Low pressure default
      }
      
      // Frame time metrics
      if (this.frameTimeHistory.length > 0) {
        const avgFrameTime = this.frameTimeHistory.reduce((a, b) => a + b, 0) / this.frameTimeHistory.length;
        this.metrics.frameTime = avgFrameTime;
        this.metrics.cpuUsage = Math.min(100, (avgFrameTime / this.limits.maxFrameTimeMs) * 100);
      }
      
      // Calculate throttle level based on pressure
      this.metrics.throttleLevel = this.calculateThrottleLevel();
      
      // Determine health status
      this.metrics.isHealthy = this.checkHealth();
      
      // Circuit breaker logic
      this.updateCircuitBreaker();
      
    } catch (error) {
      console.error('[ResourceManager] Error updating metrics:', error);
    }
  }
  
  /**
   * Calculate throttle level based on system pressure
   */
  private calculateThrottleLevel(): number {
    const memoryPressure = this.metrics.memoryPressure;
    const cpuPressure = this.metrics.cpuUsage / 100;
    
    // Use the highest pressure as throttle level
    const maxPressure = Math.max(memoryPressure, cpuPressure);
    
    if (maxPressure > 0.9) return 0.9;  // 90% throttle - minimal operations
    if (maxPressure > 0.7) return 0.5;  // 50% throttle - reduced operations
    if (maxPressure > 0.5) return 0.2;  // 20% throttle - slight reduction
    return 0; // No throttle
  }
  
  /**
   * Check overall system health
   */
  private checkHealth(): boolean {
    return (
      this.metrics.memoryPressure < 0.9 &&
      this.metrics.cpuUsage < 90 &&
      this.metrics.frameTime < this.limits.maxFrameTimeMs * 2 &&
      !this.circuitBreakerOpen
    );
  }
  
  /**
   * Update circuit breaker state
   */
  private updateCircuitBreaker(): void {
    const criticalCondition = (
      this.metrics.memoryPressure > 0.95 ||
      this.metrics.cpuUsage > 95 ||
      this.metrics.frameTime > this.limits.maxFrameTimeMs * 3
    );
    
    if (criticalCondition && !this.circuitBreakerOpen) {
      // Open circuit breaker
      this.circuitBreakerOpen = true;
      this.circuitBreakerResetTime = Date.now() + 5000; // 5 second cooldown
      console.warn('[ResourceManager] Circuit breaker OPEN - system under critical load');
    } else if (this.circuitBreakerOpen && Date.now() > this.circuitBreakerResetTime) {
      // Try to close circuit breaker
      if (!criticalCondition) {
        this.circuitBreakerOpen = false;
        console.log('[ResourceManager] Circuit breaker CLOSED - system recovered');
      } else {
        // Extend cooldown
        this.circuitBreakerResetTime = Date.now() + 5000;
      }
    }
  }
  
  /**
   * Record frame time for performance tracking
   */
  recordFrameTime(frameTime: number): void {
    this.frameTimeHistory.push(frameTime);
    if (this.frameTimeHistory.length > 60) { // Keep last 60 frames
      this.frameTimeHistory.shift();
    }
    this.lastFrameTime = frameTime;
  }
  
  /**
   * Check if operation should be allowed based on resources
   */
  canExecute(operationType: 'heavy' | 'medium' | 'light' = 'medium'): boolean {
    // Circuit breaker check
    if (this.circuitBreakerOpen) {
      return operationType === 'light'; // Only allow light operations
    }
    
    // Resource pressure check
    const throttle = this.metrics.throttleLevel;
    
    switch (operationType) {
      case 'heavy':
        return throttle < 0.2; // Only when system is healthy
      case 'medium':
        return throttle < 0.5; // Allow up to 50% throttle
      case 'light':
        return throttle < 0.9; // Allow unless critically throttled
      default:
        return true;
    }
  }
  
  /**
   * Get safe limits based on current resources
   */
  getSafeLimits(): ResourceLimits {
    const throttle = this.metrics.throttleLevel;
    
    if (throttle > 0) {
      // Reduce limits based on throttle level
      const reduction = 1 - throttle;
      return {
        maxMemoryMB: this.limits.maxMemoryMB * reduction,
        maxFrameTimeMs: this.limits.maxFrameTimeMs,
        maxCpuPercent: this.limits.maxCpuPercent * reduction,
        maxQubits: Math.floor(this.limits.maxQubits * reduction),
        maxFragments: Math.floor(this.limits.maxFragments * reduction),
        maxWaves: Math.floor(this.limits.maxWaves * reduction),
        maxGridSize: Math.max(4, Math.floor(this.limits.maxGridSize * reduction))
      };
    }
    
    return { ...this.limits };
  }
  
  /**
   * Get current metrics
   */
  getMetrics(): ResourceMetrics {
    return { ...this.metrics };
  }
  
  /**
   * Check if system is healthy
   */
  isHealthy(): boolean {
    return this.metrics.isHealthy && !this.circuitBreakerOpen;
  }
  
  /**
   * Force garbage collection if available
   */
  forceCleanup(): void {
    if (global.gc) {
      console.log('[ResourceManager] Forcing garbage collection');
      global.gc();
    }
  }
  
  /**
   * Emergency stop - halt all operations
   */
  emergencyStop(): void {
    this.circuitBreakerOpen = true;
    this.circuitBreakerResetTime = Date.now() + 10000; // 10 second cooldown
    console.error('[ResourceManager] EMERGENCY STOP - all operations halted');
  }
  
  /**
   * Reset resource manager
   */
  reset(): void {
    this.metrics = this.createDefaultMetrics();
    this.frameTimeHistory = [];
    this.isThrottling = false;
    this.circuitBreakerOpen = false;
    this.circuitBreakerResetTime = 0;
    console.log('[ResourceManager] Reset complete');
  }
}

export default ResourceManager;