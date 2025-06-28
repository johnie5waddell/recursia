/**
 * Web Worker Pool Manager
 * Manages a pool of workers for parallel computation
 */

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

export class WorkerPool {
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
    private workerScript: string,
    private poolSize: number = navigator.hardwareConcurrency || 4
  ) {
    this.initializeWorkers();
  }
  
  /**
   * Initialize worker pool
   */
  private initializeWorkers(): void {
    for (let i = 0; i < this.poolSize; i++) {
      const worker = new Worker(this.workerScript, { type: 'module' });
      
      worker.addEventListener('message', (event) => {
        this.handleWorkerMessage(worker, event);
      });
      
      worker.addEventListener('error', (error) => {
        console.error('Worker error:', error);
        this.handleWorkerError(worker, error);
      });
      
      this.workers.push(worker);
      this.availableWorkers.push(worker);
    }
  }
  
  /**
   * Execute task on worker
   */
  async execute<T>(task: WorkerTask): Promise<T> {
    return new Promise((resolve, reject) => {
      // Add to queue
      this.taskQueue.push({
        task,
        resolve,
        reject
      });
      
      // Sort by priority if provided
      if (task.priority !== undefined) {
        this.taskQueue.sort((a, b) => 
          (b.task.priority || 0) - (a.task.priority || 0)
        );
      }
      
      // Try to process immediately
      this.processQueue();
    });
  }
  
  /**
   * Execute multiple tasks in parallel
   */
  async executeBatch<T>(tasks: WorkerTask[]): Promise<T[]> {
    const promises = tasks.map(task => this.execute<T>(task));
    return Promise.all(promises);
  }
  
  /**
   * Process task queue
   */
  private processQueue(): void {
    while (this.availableWorkers.length > 0 && this.taskQueue.length > 0) {
      const worker = this.availableWorkers.shift()!;
      const { task, resolve, reject } = this.taskQueue.shift()!;
      
      // Track active job
      this.activeJobs.set(task.id, {
        worker,
        resolve,
        reject
      });
      
      // Send task to worker
      worker.postMessage(task);
    }
  }
  
  /**
   * Handle worker message
   */
  private handleWorkerMessage(worker: Worker, event: MessageEvent): void {
    const { id, result, error } = event.data;
    
    const job = this.activeJobs.get(id);
    if (!job) {
      console.error('Received result for unknown job:', id);
      return;
    }
    
    // Remove from active jobs
    this.activeJobs.delete(id);
    
    // Return worker to available pool
    this.availableWorkers.push(worker);
    
    // Resolve or reject promise
    if (error) {
      job.reject(new Error(error));
    } else {
      job.resolve(result);
    }
    
    // Process next task
    this.processQueue();
  }
  
  /**
   * Handle worker error
   */
  private handleWorkerError(worker: Worker, error: ErrorEvent): void {
    // Find and reject all jobs for this worker
    for (const [id, job] of this.activeJobs.entries()) {
      if (job.worker === worker) {
        job.reject(error);
        this.activeJobs.delete(id);
      }
    }
    
    // Remove worker from pool
    const index = this.workers.indexOf(worker);
    if (index !== -1) {
      this.workers.splice(index, 1);
    }
    
    // Remove from available workers
    const availableIndex = this.availableWorkers.indexOf(worker);
    if (availableIndex !== -1) {
      this.availableWorkers.splice(availableIndex, 1);
    }
    
    // Terminate the failed worker
    worker.terminate();
    
    // Create replacement worker
    if (this.workers.length < this.poolSize) {
      const newWorker = new Worker(this.workerScript, { type: 'module' });
      
      newWorker.addEventListener('message', (event) => {
        this.handleWorkerMessage(newWorker, event);
      });
      
      newWorker.addEventListener('error', (error) => {
        console.error('Worker error:', error);
        this.handleWorkerError(newWorker, error);
      });
      
      this.workers.push(newWorker);
      this.availableWorkers.push(newWorker);
      
      // Process queue with new worker
      this.processQueue();
    }
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
  
  /**
   * Terminate all workers
   */
  terminate(): void {
    // Cancel all pending tasks
    for (const { reject } of this.taskQueue) {
      reject(new Error('Worker pool terminated'));
    }
    this.taskQueue = [];
    
    // Cancel all active jobs
    for (const [, job] of this.activeJobs) {
      job.reject(new Error('Worker pool terminated'));
    }
    this.activeJobs.clear();
    
    // Terminate all workers
    for (const worker of this.workers) {
      worker.terminate();
    }
    
    this.workers = [];
    this.availableWorkers = [];
  }
}