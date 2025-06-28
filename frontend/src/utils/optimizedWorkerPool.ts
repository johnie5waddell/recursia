/**
 * Optimized Worker Pool Manager
 * Manages Web Workers efficiently with automatic scaling and resource management
 */

import React from 'react';
import { getSystemResourceMonitor } from './systemResourceMonitor';

interface WorkerTask {
  id: string;
  type: string;
  data: any;
  priority: number;
  timestamp: number;
  timeout?: number;
}

interface WorkerInfo {
  worker: Worker;
  busy: boolean;
  taskCount: number;
  lastUsed: number;
  errors: number;
}

interface PoolConfig {
  minWorkers: number;
  maxWorkers: number;
  taskTimeout: number;
  idleTimeout: number;
  maxTasksPerWorker: number;
  enableAutoScaling: boolean;
}

export class OptimizedWorkerPool {
  private workers: Map<string, WorkerInfo> = new Map();
  private taskQueue: WorkerTask[] = [];
  private pendingTasks: Map<string, {
    resolve: (value: any) => void;
    reject: (error: any) => void;
    timeout?: NodeJS.Timeout;
  }> = new Map();
  
  private config: PoolConfig;
  private workerScript: string;
  private resourceMonitor = getSystemResourceMonitor();
  private scalingInterval?: NodeJS.Timeout;
  
  constructor(workerScript: string, config?: Partial<PoolConfig>) {
    this.workerScript = workerScript;
    this.config = {
      minWorkers: config?.minWorkers ?? 2,
      maxWorkers: config?.maxWorkers ?? (navigator.hardwareConcurrency || 4),
      taskTimeout: config?.taskTimeout ?? 30000,
      idleTimeout: config?.idleTimeout ?? 60000,
      maxTasksPerWorker: config?.maxTasksPerWorker ?? 100,
      enableAutoScaling: config?.enableAutoScaling ?? true
    };
    
    this.initialize();
  }
  
  /**
   * Initialize the worker pool
   */
  private initialize(): void {
    // Create initial workers
    for (let i = 0; i < this.config.minWorkers; i++) {
      this.createWorker();
    }
    
    // Start auto-scaling if enabled
    if (this.config.enableAutoScaling) {
      this.startAutoScaling();
    }
    
    // Start idle worker cleanup
    this.startIdleCleanup();
  }
  
  /**
   * Create a new worker
   */
  private createWorker(): string {
    const workerId = `worker-${Date.now()}-${Math.random()}`;
    
    try {
      // Skip worker creation for mock workers or empty scripts
      if (!this.workerScript || this.workerScript.startsWith('mock://')) {
        return workerId;
      }
      
      let worker: Worker;
      
      // Special handling for Vite workers
      if (this.workerScript === 'vite-worker') {
        // Use the imported QuantumWorker directly
        worker = new QuantumWorker();
      } else {
        // Create worker with proper type for other cases
        worker = new Worker(this.workerScript, { type: 'module' });
      }
      
      // Setup worker message handler
      worker.onmessage = (event) => {
        this.handleWorkerMessage(workerId, event);
      };
      
      // Setup error handler
      worker.onerror = (error) => {
        this.handleWorkerError(workerId, error);
      };
      
      // Store worker info
      this.workers.set(workerId, {
        worker,
        busy: false,
        taskCount: 0,
        lastUsed: Date.now(),
        errors: 0
      });
      
      return workerId;
    } catch (error) {
      console.error('Failed to create worker:', error);
      // Don't throw - just return the worker ID and it won't be used
      return workerId;
    }
  }
  
  /**
   * Handle message from worker
   */
  private handleWorkerMessage(workerId: string, event: MessageEvent): void {
    const { id: taskId, result, error } = event.data;
    
    const workerInfo = this.workers.get(workerId);
    if (workerInfo) {
      workerInfo.busy = false;
      workerInfo.lastUsed = Date.now();
      workerInfo.taskCount++;
    }
    
    // Resolve or reject the task
    const pending = this.pendingTasks.get(taskId);
    if (pending) {
      if (pending.timeout) {
        clearTimeout(pending.timeout);
      }
      
      if (error) {
        pending.reject(error);
      } else {
        pending.resolve(result);
      }
      
      this.pendingTasks.delete(taskId);
    }
    
    // Process next task in queue
    this.processQueue();
  }
  
  /**
   * Handle worker error
   */
  private handleWorkerError(workerId: string, error: ErrorEvent): void {
    console.error(`Worker ${workerId} error:`, error);
    
    const workerInfo = this.workers.get(workerId);
    if (workerInfo) {
      workerInfo.errors++;
      workerInfo.busy = false;
      
      // Terminate worker if too many errors
      if (workerInfo.errors > 3) {
        this.terminateWorker(workerId);
      }
    }
    
    // Process queue with remaining workers
    this.processQueue();
  }
  
  /**
   * Add task to the pool
   */
  async execute<T = any>(
    type: string,
    data: any,
    options?: {
      priority?: number;
      timeout?: number;
    }
  ): Promise<T> {
    const task: WorkerTask = {
      id: `task-${Date.now()}-${Math.random()}`,
      type,
      data,
      priority: options?.priority ?? 0,
      timestamp: Date.now(),
      timeout: options?.timeout ?? this.config.taskTimeout
    };
    
    return new Promise((resolve, reject) => {
      // Store promise callbacks
      this.pendingTasks.set(task.id, {
        resolve,
        reject
      });
      
      // Add to queue
      this.enqueueTask(task);
      
      // Process queue
      this.processQueue();
    });
  }
  
  /**
   * Enqueue task with priority
   */
  private enqueueTask(task: WorkerTask): void {
    // Insert task in priority order
    let inserted = false;
    for (let i = 0; i < this.taskQueue.length; i++) {
      if (task.priority > this.taskQueue[i].priority) {
        this.taskQueue.splice(i, 0, task);
        inserted = true;
        break;
      }
    }
    
    if (!inserted) {
      this.taskQueue.push(task);
    }
  }
  
  /**
   * Process task queue
   */
  private processQueue(): void {
    while (this.taskQueue.length > 0) {
      const availableWorker = this.getAvailableWorker();
      if (!availableWorker) {
        // No available workers, check if we can scale up
        if (this.shouldScaleUp()) {
          this.scaleUp();
        }
        break;
      }
      
      const task = this.taskQueue.shift()!;
      this.assignTask(availableWorker, task);
    }
  }
  
  /**
   * Get available worker
   */
  private getAvailableWorker(): string | null {
    for (const [workerId, info] of this.workers) {
      if (!info.busy && info.errors < 3) {
        return workerId;
      }
    }
    return null;
  }
  
  /**
   * Assign task to worker
   */
  private assignTask(workerId: string, task: WorkerTask): void {
    const workerInfo = this.workers.get(workerId);
    
    // Handle mock workers by executing synchronously
    if (this.workerScript.startsWith('mock://')) {
      const pending = this.pendingTasks.get(task.id);
      if (pending) {
        // Execute task synchronously for mock workers
        setTimeout(() => {
          try {
            // Mock execution - just resolve with the data
            pending.resolve({ type: task.type, data: task.data });
          } catch (error) {
            pending.reject(error);
          }
          this.pendingTasks.delete(task.id);
        }, 0);
      }
      return;
    }
    
    if (!workerInfo) return;
    
    workerInfo.busy = true;
    workerInfo.lastUsed = Date.now();
    
    // Setup timeout
    const pending = this.pendingTasks.get(task.id);
    if (pending && task.timeout) {
      pending.timeout = setTimeout(() => {
        pending.reject(new Error(`Task ${task.id} timed out`));
        this.pendingTasks.delete(task.id);
        workerInfo.busy = false;
      }, task.timeout);
    }
    
    // Send task to worker in the format expected by quantum.worker.ts
    workerInfo.worker.postMessage({
      id: task.id,
      type: task.type,
      data: task.data
    });
  }
  
  /**
   * Start auto-scaling monitoring
   */
  private startAutoScaling(): void {
    this.scalingInterval = setInterval(() => {
      const metrics = this.resourceMonitor.getMetrics();
      
      // Scale based on queue size and resource usage
      if (this.shouldScaleUp() && metrics.cpu.usage < 80) {
        this.scaleUp();
      } else if (this.shouldScaleDown() && metrics.memory.percentage < 70) {
        this.scaleDown();
      }
    }, 5000); // Check every 5 seconds
  }
  
  /**
   * Check if should scale up
   */
  private shouldScaleUp(): boolean {
    const workerCount = this.workers.size;
    const busyWorkers = Array.from(this.workers.values()).filter(w => w.busy).length;
    
    return (
      workerCount < this.config.maxWorkers &&
      this.taskQueue.length > 0 &&
      busyWorkers === workerCount
    );
  }
  
  /**
   * Check if should scale down
   */
  private shouldScaleDown(): boolean {
    const workerCount = this.workers.size;
    const idleWorkers = Array.from(this.workers.values()).filter(w => !w.busy).length;
    
    return (
      workerCount > this.config.minWorkers &&
      idleWorkers > workerCount / 2
    );
  }
  
  /**
   * Scale up by adding workers
   */
  private scaleUp(): void {
    const currentCount = this.workers.size;
    if (currentCount < this.config.maxWorkers) {
      this.createWorker();
      console.log(`Scaled up to ${this.workers.size} workers`);
    }
  }
  
  /**
   * Scale down by removing idle workers
   */
  private scaleDown(): void {
    const currentCount = this.workers.size;
    if (currentCount > this.config.minWorkers) {
      // Find most idle worker
      let mostIdleWorker: string | null = null;
      let maxIdleTime = 0;
      
      for (const [workerId, info] of this.workers) {
        if (!info.busy) {
          const idleTime = Date.now() - info.lastUsed;
          if (idleTime > maxIdleTime) {
            maxIdleTime = idleTime;
            mostIdleWorker = workerId;
          }
        }
      }
      
      if (mostIdleWorker) {
        this.terminateWorker(mostIdleWorker);
        console.log(`Scaled down to ${this.workers.size} workers`);
      }
    }
  }
  
  /**
   * Start idle worker cleanup
   */
  private startIdleCleanup(): void {
    setInterval(() => {
      const now = Date.now();
      
      for (const [workerId, info] of this.workers) {
        if (!info.busy && 
            now - info.lastUsed > this.config.idleTimeout &&
            this.workers.size > this.config.minWorkers) {
          this.terminateWorker(workerId);
        }
      }
    }, 30000); // Check every 30 seconds
  }
  
  /**
   * Terminate a worker
   */
  private terminateWorker(workerId: string): void {
    const workerInfo = this.workers.get(workerId);
    if (workerInfo) {
      workerInfo.worker.terminate();
      this.workers.delete(workerId);
      console.log(`Terminated worker ${workerId}`);
    }
  }
  
  /**
   * Get pool statistics
   */
  getStats(): {
    totalWorkers: number;
    busyWorkers: number;
    idleWorkers: number;
    queueLength: number;
    totalTasksProcessed: number;
    averageTasksPerWorker: number;
  } {
    const workers = Array.from(this.workers.values());
    const busyWorkers = workers.filter(w => w.busy).length;
    const totalTasksProcessed = workers.reduce((sum, w) => sum + w.taskCount, 0);
    
    return {
      totalWorkers: this.workers.size,
      busyWorkers,
      idleWorkers: this.workers.size - busyWorkers,
      queueLength: this.taskQueue.length,
      totalTasksProcessed,
      averageTasksPerWorker: totalTasksProcessed / this.workers.size || 0
    };
  }
  
  /**
   * Terminate all workers and cleanup
   */
  terminate(): void {
    // Clear scaling interval
    if (this.scalingInterval) {
      clearInterval(this.scalingInterval);
    }
    
    // Reject all pending tasks
    for (const [taskId, pending] of this.pendingTasks) {
      if (pending.timeout) {
        clearTimeout(pending.timeout);
      }
      pending.reject(new Error('Worker pool terminated'));
    }
    this.pendingTasks.clear();
    
    // Terminate all workers
    for (const [workerId] of this.workers) {
      this.terminateWorker(workerId);
    }
    
    // Clear queue
    this.taskQueue = [];
  }
}

/**
 * Create a singleton worker pool for quantum computations
 */
let quantumWorkerPool: OptimizedWorkerPool | null = null;

// Import worker with Vite's special syntax
import QuantumWorker from '../workers/quantum.worker.ts?worker';

export function getQuantumWorkerPool(): OptimizedWorkerPool {
  if (!quantumWorkerPool) {
    try {
      // Create a placeholder worker script URL
      // The actual worker is handled by Vite's worker import
      quantumWorkerPool = new OptimizedWorkerPool('vite-worker', {
        minWorkers: 2,
        maxWorkers: navigator.hardwareConcurrency || 4,
        enableAutoScaling: true,
        maxTasksPerWorker: 50,
        taskTimeout: 30000,
        idleTimeout: 60000
      });
    } catch (error) {
      console.error('Failed to create quantum worker pool:', error);
      // Fallback to minimal configuration
      quantumWorkerPool = new OptimizedWorkerPool('', {
        minWorkers: 0,
        maxWorkers: 0,
        enableAutoScaling: false
      });
    }
  }
  return quantumWorkerPool;
}

/**
 * React hook for using worker pool
 */
export function useWorkerPool(workerScript: string, config?: Partial<PoolConfig>) {
  const poolRef = React.useRef<OptimizedWorkerPool | null>(null);
  
  React.useEffect(() => {
    poolRef.current = new OptimizedWorkerPool(workerScript, config);
    
    return () => {
      poolRef.current?.terminate();
    };
  }, [workerScript]);
  
  const execute = React.useCallback(async <T = any>(
    type: string,
    data: any,
    options?: { priority?: number; timeout?: number }
  ): Promise<T> => {
    if (!poolRef.current) {
      throw new Error('Worker pool not initialized');
    }
    return poolRef.current.execute<T>(type, data, options);
  }, []);
  
  return { execute, getStats: () => poolRef.current?.getStats() };
}