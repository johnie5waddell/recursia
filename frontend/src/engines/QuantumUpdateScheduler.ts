/**
 * Quantum Update Scheduler - Breaks up heavy computations into manageable chunks
 * 
 * Prevents browser blocking by scheduling quantum simulation updates
 * across multiple animation frames using time-slicing.
 */

export interface UpdateTask {
  id: string;
  priority: number;
  execute: () => Promise<void> | void;
  maxTime?: number;
}

export class QuantumUpdateScheduler {
  private static instance: QuantumUpdateScheduler;
  private taskQueue: UpdateTask[] = [];
  private isRunning: boolean = false;
  private currentTask: UpdateTask | null = null;
  private frameStartTime: number = 0;
  private readonly FRAME_BUDGET_MS = 16; // Target 60fps
  private readonly MAX_TASK_TIME_MS = 5; // Max 5ms per task
  
  private constructor() {}
  
  static getInstance(): QuantumUpdateScheduler {
    if (!QuantumUpdateScheduler.instance) {
      QuantumUpdateScheduler.instance = new QuantumUpdateScheduler();
    }
    return QuantumUpdateScheduler.instance;
  }
  
  /**
   * Schedule a task for execution
   */
  scheduleTask(task: UpdateTask): void {
    // Insert task based on priority (higher priority first)
    const insertIndex = this.taskQueue.findIndex(t => t.priority < task.priority);
    if (insertIndex === -1) {
      this.taskQueue.push(task);
    } else {
      this.taskQueue.splice(insertIndex, 0, task);
    }
    
    // Start processing if not already running
    if (!this.isRunning) {
      this.startProcessing();
    }
  }
  
  /**
   * Schedule multiple tasks
   */
  scheduleTasks(tasks: UpdateTask[]): void {
    tasks.forEach(task => this.scheduleTask(task));
  }
  
  /**
   * Clear all pending tasks
   */
  clearTasks(): void {
    this.taskQueue = [];
    this.currentTask = null;
  }
  
  /**
   * Get pending task count
   */
  getPendingCount(): number {
    return this.taskQueue.length + (this.currentTask ? 1 : 0);
  }
  
  /**
   * Start processing tasks
   */
  private startProcessing(): void {
    if (this.isRunning) return;
    
    this.isRunning = true;
    this.frameStartTime = performance.now();
    this.processNextTask();
  }
  
  /**
   * Process the next task in the queue
   */
  private async processNextTask(): Promise<void> {
    // Check if we should stop
    if (this.taskQueue.length === 0) {
      this.isRunning = false;
      return;
    }
    
    // Check frame budget
    const elapsed = performance.now() - this.frameStartTime;
    if (elapsed > this.FRAME_BUDGET_MS) {
      // Yield to browser and continue next frame
      requestAnimationFrame(() => {
        this.frameStartTime = performance.now();
        this.processNextTask();
      });
      return;
    }
    
    // Get next task
    this.currentTask = this.taskQueue.shift() || null;
    if (!this.currentTask) {
      this.isRunning = false;
      return;
    }
    
    const taskStartTime = performance.now();
    const maxTime = Math.min(
      this.currentTask.maxTime || this.MAX_TASK_TIME_MS,
      this.FRAME_BUDGET_MS - elapsed
    );
    
    try {
      // Create a timeout promise
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Task timeout')), maxTime);
      });
      
      // Execute task with timeout
      const result = this.currentTask.execute();
      if (result instanceof Promise) {
        await Promise.race([result, timeoutPromise]);
      }
      
      const taskTime = performance.now() - taskStartTime;
      if (taskTime > maxTime) {
        console.warn(`Task ${this.currentTask.id} exceeded time budget: ${taskTime.toFixed(2)}ms`);
      }
      
    } catch (error) {
      if (error instanceof Error && error.message === 'Task timeout') {
        console.warn(`Task ${this.currentTask.id} timed out after ${maxTime}ms`);
        // Re-queue the task with lower priority
        if (this.currentTask) {
          this.scheduleTask({
            ...this.currentTask,
            priority: this.currentTask.priority - 1
          });
        }
      } else {
        console.error(`Task ${this.currentTask?.id} failed:`, error);
      }
    }
    
    this.currentTask = null;
    
    // Continue processing
    if (performance.now() - this.frameStartTime < this.FRAME_BUDGET_MS) {
      // Process immediately if we have time
      await this.processNextTask();
    } else {
      // Yield and continue next frame
      requestAnimationFrame(() => {
        this.frameStartTime = performance.now();
        this.processNextTask();
      });
    }
  }
  
  /**
   * Wait for all tasks to complete
   */
  async waitForCompletion(timeoutMs: number = 5000): Promise<void> {
    const startTime = Date.now();
    
    while (this.getPendingCount() > 0) {
      if (Date.now() - startTime > timeoutMs) {
        throw new Error('Task completion timeout');
      }
      await new Promise(resolve => setTimeout(resolve, 10));
    }
  }
  
  /**
   * Emergency stop
   */
  emergencyStop(): void {
    console.error('[QuantumUpdateScheduler] Emergency stop - clearing all tasks');
    this.clearTasks();
    this.isRunning = false;
  }
}

/**
 * Helper to create chunked tasks from an array
 */
export function createChunkedTasks<T>(
  items: T[],
  chunkSize: number,
  taskIdPrefix: string,
  processor: (chunk: T[]) => void,
  priority: number = 0
): UpdateTask[] {
  const tasks: UpdateTask[] = [];
  
  for (let i = 0; i < items.length; i += chunkSize) {
    const chunk = items.slice(i, i + chunkSize);
    tasks.push({
      id: `${taskIdPrefix}-${i / chunkSize}`,
      priority,
      execute: () => processor(chunk)
    });
  }
  
  return tasks;
}

/**
 * Helper to break up nested loops
 */
export function createNestedLoopTasks(
  sizeX: number,
  sizeY: number,
  sizeZ: number,
  taskIdPrefix: string,
  processor: (x: number, y: number, z: number) => void,
  chunksPerDimension: number = 4
): UpdateTask[] {
  const tasks: UpdateTask[] = [];
  const chunkSizeX = Math.ceil(sizeX / chunksPerDimension);
  const chunkSizeY = Math.ceil(sizeY / chunksPerDimension);
  const chunkSizeZ = Math.ceil(sizeZ / chunksPerDimension);
  
  let taskIndex = 0;
  
  for (let cx = 0; cx < sizeX; cx += chunkSizeX) {
    for (let cy = 0; cy < sizeY; cy += chunkSizeY) {
      for (let cz = 0; cz < sizeZ; cz += chunkSizeZ) {
        const startX = cx;
        const endX = Math.min(cx + chunkSizeX, sizeX);
        const startY = cy;
        const endY = Math.min(cy + chunkSizeY, sizeY);
        const startZ = cz;
        const endZ = Math.min(cz + chunkSizeZ, sizeZ);
        
        tasks.push({
          id: `${taskIdPrefix}-${taskIndex++}`,
          priority: 0,
          execute: () => {
            for (let x = startX; x < endX; x++) {
              for (let y = startY; y < endY; y++) {
                for (let z = startZ; z < endZ; z++) {
                  processor(x, y, z);
                }
              }
            }
          }
        });
      }
    }
  }
  
  return tasks;
}

export default QuantumUpdateScheduler;