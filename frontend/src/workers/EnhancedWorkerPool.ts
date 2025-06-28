/**
 * Enhanced Worker Pool with Memory Management
 * Enterprise-grade worker pool with automatic resource cleanup and monitoring
 */

import { getMemoryManager, ResourceType } from '../utils/memoryManager';

export interface WorkerTask<T = any> {
  id: string;
  type: string;
  data: T;
  priority?: number;
  timeout?: number;
}

export interface WorkerResult<T = any> {
  id: string;
  result?: T;
  error?: string;
  duration?: number;
}

interface QueuedTask<T = any> {
  task: WorkerTask<T>;
  resolve: (result: T) => void;
  reject: (error: Error) => void;
  timestamp: number;
  timeoutId?: number;
}

interface ActiveJob<T = any> {
  worker: Worker;
  task: WorkerTask<T>;
  resolve: (result: T) => void;
  reject: (error: Error) => void;
  startTime: number;
  timeoutId?: number;
}

interface WorkerMetrics {
  id: string;
  tasksCompleted: number;
  tasksFailed: number;
  totalTime: number;
  averageTime: number;
  lastTaskTime?: number;
  errors: number;
}

/**
 * Enhanced worker pool with comprehensive memory management
 */
export class EnhancedWorkerPool {
  private workers: Map<string, Worker> = new Map();
  private availableWorkers: Set<string> = new Set();
  private taskQueue: QueuedTask[] = [];
  private activeJobs: Map<string, ActiveJob> = new Map();
  private workerMetrics: Map<string, WorkerMetrics> = new Map();
  private memoryManager = getMemoryManager();
  private componentName: string;
  private terminated = false;
  private defaultTimeout = 30000; // 30 seconds
  private maxQueueSize = 1000;
  private maxRetries = 3;
  private retryDelay = 1000; // 1 second
  
  // Performance monitoring
  private performanceObserver?: PerformanceObserver;
  private taskPerformance: Map<string, number> = new Map();

  constructor(
    private workerScript: string,
    private poolSize: number = navigator.hardwareConcurrency || 4,
    componentName: string = 'WorkerPool'
  ) {
    this.componentName = componentName;
    this.initializeWorkers();
    this.setupPerformanceMonitoring();
  }

  /**
   * Initialize worker pool with memory tracking
   */
  private initializeWorkers(): void {
    for (let i = 0; i < this.poolSize; i++) {
      this.createWorker(i);
    }
  }

  /**
   * Create a single worker with full instrumentation
   */
  private createWorker(index: number): void {
    const workerId = `${this.componentName}-worker-${index}`;
    const worker = new Worker(this.workerScript, { type: 'module' });
    
    // Initialize metrics
    this.workerMetrics.set(workerId, {
      id: workerId,
      tasksCompleted: 0,
      tasksFailed: 0,
      totalTime: 0,
      averageTime: 0,
      errors: 0
    });

    // Setup event handlers
    worker.addEventListener('message', (event) => {
      this.handleWorkerMessage(workerId, event);
    });

    worker.addEventListener('error', (error) => {
      this.handleWorkerError(workerId, error);
    });

    worker.addEventListener('messageerror', (event) => {
      console.error(`Message error in worker ${workerId}:`, event);
      this.handleWorkerError(workerId, new ErrorEvent('messageerror', { message: 'Failed to deserialize message' }));
    });

    // Track in memory manager
    this.memoryManager.track(
      workerId,
      ResourceType.WORKER,
      () => {
        worker.terminate();
        this.workers.delete(workerId);
        this.availableWorkers.delete(workerId);
        this.workerMetrics.delete(workerId);
      },
      {
        component: this.componentName,
        description: `Worker ${index} for ${this.workerScript}`,
        size: 1048576, // 1MB estimate
        priority: 'high'
      }
    );

    this.workers.set(workerId, worker);
    this.availableWorkers.add(workerId);
  }

  /**
   * Setup performance monitoring
   */
  private setupPerformanceMonitoring(): void {
    if ('PerformanceObserver' in window) {
      try {
        this.performanceObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.entryType === 'measure' && entry.name.startsWith('worker-task-')) {
              const taskId = entry.name.replace('worker-task-', '');
              this.taskPerformance.set(taskId, entry.duration);
            }
          }
        });
        this.performanceObserver.observe({ entryTypes: ['measure'] });
      } catch (e) {
        console.warn('Performance monitoring not available:', e);
      }
    }
  }

  /**
   * Execute task on worker with retry logic
   */
  async execute<T>(task: WorkerTask<T>, retryCount = 0): Promise<T> {
    if (this.terminated) {
      throw new Error('Worker pool is terminated');
    }

    if (this.taskQueue.length >= this.maxQueueSize) {
      throw new Error(`Task queue is full (${this.maxQueueSize} tasks)`);
    }

    return new Promise((resolve, reject) => {
      const queuedTask: QueuedTask<T> = {
        task,
        resolve: (result: T) => {
          // Clean up timeout tracking
          if (queuedTask.timeoutId) {
            this.memoryManager.untrack(`timeout-${task.id}`);
          }
          resolve(result);
        },
        reject: (error: Error) => {
          // Clean up timeout tracking
          if (queuedTask.timeoutId) {
            this.memoryManager.untrack(`timeout-${task.id}`);
          }
          
          // Retry logic
          if (retryCount < this.maxRetries) {
            console.warn(`Retrying task ${task.id} (attempt ${retryCount + 1}/${this.maxRetries})`);
            setTimeout(() => {
              this.execute(task, retryCount + 1).then(resolve).catch(reject);
            }, this.retryDelay * (retryCount + 1));
          } else {
            reject(error);
          }
        },
        timestamp: Date.now()
      };

      // Add to queue with priority sorting
      this.taskQueue.push(queuedTask);
      if (task.priority !== undefined) {
        this.taskQueue.sort((a, b) => 
          (b.task.priority || 0) - (a.task.priority || 0)
        );
      }

      // Process queue
      this.processQueue();
    });
  }

  /**
   * Execute multiple tasks in parallel
   */
  async executeBatch<T>(tasks: WorkerTask<T>[]): Promise<T[]> {
    const promises = tasks.map(task => this.execute<T>(task));
    return Promise.all(promises);
  }

  /**
   * Process task queue
   */
  private processQueue(): void {
    while (this.availableWorkers.size > 0 && this.taskQueue.length > 0) {
      const workerId = this.availableWorkers.values().next().value;
      const worker = this.workers.get(workerId);
      
      if (!worker) continue;
      
      this.availableWorkers.delete(workerId);
      const queuedTask = this.taskQueue.shift()!;
      
      // Mark performance start
      performance.mark(`worker-task-start-${queuedTask.task.id}`);
      
      // Create active job
      const activeJob: ActiveJob = {
        worker,
        task: queuedTask.task,
        resolve: queuedTask.resolve,
        reject: queuedTask.reject,
        startTime: Date.now()
      };

      // Set timeout
      const timeout = queuedTask.task.timeout || this.defaultTimeout;
      const timeoutId = window.setTimeout(() => {
        this.handleTaskTimeout(queuedTask.task.id);
      }, timeout);

      activeJob.timeoutId = timeoutId;
      queuedTask.timeoutId = timeoutId;

      // Track timeout in memory manager
      this.memoryManager.track(
        `timeout-${queuedTask.task.id}`,
        ResourceType.TIMER,
        () => {
          window.clearTimeout(timeoutId);
        },
        {
          component: this.componentName,
          description: `Task timeout for ${queuedTask.task.id}`,
          priority: 'low'
        }
      );

      // Store active job
      this.activeJobs.set(queuedTask.task.id, activeJob);

      // Send task to worker
      try {
        worker.postMessage(queuedTask.task);
      } catch (error) {
        console.error(`Failed to post message to worker:`, error);
        this.handleTaskError(queuedTask.task.id, error as Error);
      }
    }
  }

  /**
   * Handle worker message
   */
  private handleWorkerMessage(workerId: string, event: MessageEvent<WorkerResult>): void {
    const { id, result, error, duration } = event.data;
    const job = this.activeJobs.get(id);
    
    if (!job) {
      console.error('Received result for unknown job:', id);
      return;
    }

    // Mark performance end
    performance.mark(`worker-task-end-${id}`);
    performance.measure(
      `worker-task-${id}`,
      `worker-task-start-${id}`,
      `worker-task-end-${id}`
    );

    // Clear timeout
    if (job.timeoutId) {
      window.clearTimeout(job.timeoutId);
      this.memoryManager.untrack(`timeout-${id}`);
    }

    // Update metrics
    const metrics = this.workerMetrics.get(workerId)!;
    const taskDuration = duration || (Date.now() - job.startTime);
    
    if (error) {
      metrics.tasksFailed++;
      metrics.errors++;
    } else {
      metrics.tasksCompleted++;
    }
    
    metrics.totalTime += taskDuration;
    metrics.averageTime = metrics.totalTime / (metrics.tasksCompleted + metrics.tasksFailed);
    metrics.lastTaskTime = taskDuration;

    // Clean up job
    this.activeJobs.delete(id);
    this.availableWorkers.add(workerId);

    // Resolve or reject
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
  private handleWorkerError(workerId: string, error: ErrorEvent): void {
    console.error(`Worker ${workerId} error:`, error);
    
    const metrics = this.workerMetrics.get(workerId);
    if (metrics) {
      metrics.errors++;
    }

    // Find and reject all jobs for this worker
    const jobsToReject: string[] = [];
    for (const [taskId, job] of this.activeJobs.entries()) {
      if (this.getWorkerIdForJob(job) === workerId) {
        jobsToReject.push(taskId);
      }
    }

    for (const taskId of jobsToReject) {
      this.handleTaskError(taskId, new Error(`Worker error: ${error.message}`));
    }

    // Remove failed worker
    this.workers.delete(workerId);
    this.availableWorkers.delete(workerId);
    this.memoryManager.untrack(workerId);

    // Create replacement worker if pool not terminated
    if (!this.terminated && this.workers.size < this.poolSize) {
      this.createWorker(this.workers.size);
      this.processQueue();
    }
  }

  /**
   * Handle task timeout
   */
  private handleTaskTimeout(taskId: string): void {
    const job = this.activeJobs.get(taskId);
    if (!job) return;

    console.warn(`Task ${taskId} timed out`);
    this.handleTaskError(taskId, new Error('Task timeout'));
  }

  /**
   * Handle task error
   */
  private handleTaskError(taskId: string, error: Error): void {
    const job = this.activeJobs.get(taskId);
    if (!job) return;

    // Clear timeout
    if (job.timeoutId) {
      window.clearTimeout(job.timeoutId);
      this.memoryManager.untrack(`timeout-${taskId}`);
    }

    // Update metrics
    const workerId = this.getWorkerIdForJob(job);
    if (workerId) {
      const metrics = this.workerMetrics.get(workerId);
      if (metrics) {
        metrics.tasksFailed++;
      }
      this.availableWorkers.add(workerId);
    }

    // Clean up and reject
    this.activeJobs.delete(taskId);
    job.reject(error);

    // Process next task
    this.processQueue();
  }

  /**
   * Get worker ID for a job
   */
  private getWorkerIdForJob(job: ActiveJob): string | undefined {
    for (const [workerId, worker] of this.workers.entries()) {
      if (worker === job.worker) {
        return workerId;
      }
    }
    return undefined;
  }

  /**
   * Get pool statistics
   */
  getStats(): {
    totalWorkers: number;
    availableWorkers: number;
    activeJobs: number;
    queuedTasks: number;
    metrics: WorkerMetrics[];
    averageTaskTime: number;
    totalTasksCompleted: number;
    totalTasksFailed: number;
    errorRate: number;
  } {
    const metrics = Array.from(this.workerMetrics.values());
    const totalCompleted = metrics.reduce((sum, m) => sum + m.tasksCompleted, 0);
    const totalFailed = metrics.reduce((sum, m) => sum + m.tasksFailed, 0);
    const totalTasks = totalCompleted + totalFailed;
    
    return {
      totalWorkers: this.workers.size,
      availableWorkers: this.availableWorkers.size,
      activeJobs: this.activeJobs.size,
      queuedTasks: this.taskQueue.length,
      metrics,
      averageTaskTime: metrics.reduce((sum, m) => sum + m.averageTime, 0) / metrics.length || 0,
      totalTasksCompleted: totalCompleted,
      totalTasksFailed: totalFailed,
      errorRate: totalTasks > 0 ? totalFailed / totalTasks : 0
    };
  }

  /**
   * Terminate all workers and clean up resources
   */
  terminate(): void {
    if (this.terminated) return;
    this.terminated = true;

    // Clear performance observer
    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }

    // Cancel all pending tasks
    for (const { reject, timeoutId } of this.taskQueue) {
      if (timeoutId) {
        window.clearTimeout(timeoutId);
      }
      reject(new Error('Worker pool terminated'));
    }
    this.taskQueue = [];

    // Cancel all active jobs
    for (const [taskId, job] of this.activeJobs) {
      if (job.timeoutId) {
        window.clearTimeout(job.timeoutId);
        this.memoryManager.untrack(`timeout-${taskId}`);
      }
      job.reject(new Error('Worker pool terminated'));
    }
    this.activeJobs.clear();

    // Terminate all workers
    for (const [workerId, worker] of this.workers) {
      worker.terminate();
      this.memoryManager.untrack(workerId);
    }

    this.workers.clear();
    this.availableWorkers.clear();
    this.workerMetrics.clear();
    this.taskPerformance.clear();
  }

  /**
   * Graceful shutdown with timeout
   */
  async shutdown(timeout = 5000): Promise<void> {
    // Stop accepting new tasks
    this.terminated = true;

    // Wait for active jobs to complete or timeout
    const shutdownPromise = new Promise<void>((resolve) => {
      const checkInterval = setInterval(() => {
        if (this.activeJobs.size === 0) {
          clearInterval(checkInterval);
          resolve();
        }
      }, 100);
    });

    const timeoutPromise = new Promise<void>((resolve) => {
      setTimeout(resolve, timeout);
    });

    await Promise.race([shutdownPromise, timeoutPromise]);

    // Force terminate
    this.terminate();
  }
}