/**
 * Vite-compatible Worker Pool Manager
 * Uses Vite's worker import syntax
 */

import { createQuantumWorker } from './quantumWorker';

export interface WorkerTask<T = any> {
  id: string;
  type: string;
  data: T;
  priority?: number;
}

export interface WorkerResult<T = any> {
  id: string;
  result: T;
  error?: string;
}

export class ViteWorkerPool {
  private workers: Worker[] = [];
  private availableWorkers: Worker[] = [];
  private taskQueue: Array<{
    task: WorkerTask;
    resolve: (result: any) => void;
    reject: (error: any) => void;
  }> = [];
  private activeJobs: Map<string, {
    worker: Worker;
    resolve: (result: any) => void;
    reject: (error: any) => void;
  }> = new Map();
  
  constructor(
    private poolSize: number = navigator.hardwareConcurrency || 4
  ) {
    this.initializeWorkers();
  }
  
  /**
   * Initialize worker pool
   */
  private initializeWorkers(): void {
    for (let i = 0; i < this.poolSize; i++) {
      try {
        const worker = createQuantumWorker();
        
        worker.addEventListener('message', (event) => {
          this.handleWorkerMessage(worker, event);
        });
        
        worker.addEventListener('error', (event) => {
          console.error('Worker error:', event);
        });
        
        this.workers.push(worker);
        this.availableWorkers.push(worker);
      } catch (error) {
        console.error('Failed to create worker:', error);
      }
    }
    
    console.log(`Worker pool initialized with ${this.workers.length} workers`);
  }
  
  /**
   * Handle message from worker
   */
  private handleWorkerMessage(worker: Worker, event: MessageEvent): void {
    const { id, result, error } = event.data;
    const job = this.activeJobs.get(id);
    
    if (job) {
      this.activeJobs.delete(id);
      this.availableWorkers.push(worker);
      
      if (error) {
        job.reject(new Error(error));
      } else {
        job.resolve(result);
      }
      
      // Process next task in queue
      this.processQueue();
    }
  }
  
  /**
   * Submit task to worker pool
   */
  async submitTask<T = any>(task: WorkerTask): Promise<T> {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ task, resolve, reject });
      this.processQueue();
    });
  }
  
  /**
   * Process task queue
   */
  private processQueue(): void {
    while (this.availableWorkers.length > 0 && this.taskQueue.length > 0) {
      const worker = this.availableWorkers.pop()!;
      const { task, resolve, reject } = this.taskQueue.shift()!;
      
      this.activeJobs.set(task.id, { worker, resolve, reject });
      worker.postMessage(task);
    }
  }
  
  /**
   * Terminate all workers
   */
  terminate(): void {
    this.workers.forEach(worker => worker.terminate());
    this.workers = [];
    this.availableWorkers = [];
    this.taskQueue = [];
    this.activeJobs.clear();
  }
  
  /**
   * Get pool statistics
   */
  getStats(): {
    totalWorkers: number;
    availableWorkers: number;
    activeJobs: number;
    queuedTasks: number;
  } {
    return {
      totalWorkers: this.workers.length,
      availableWorkers: this.availableWorkers.length,
      activeJobs: this.activeJobs.size,
      queuedTasks: this.taskQueue.length
    };
  }
}