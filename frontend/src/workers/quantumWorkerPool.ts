/**
 * Quantum Worker Pool Singleton
 * Provides a ready-to-use worker pool for quantum computations
 */

import { EnhancedWorkerPool } from './EnhancedWorkerPool';

// Create singleton instance with Vite worker import
let quantumWorkerPool: EnhancedWorkerPool | null = null;

export function getQuantumWorkerPool(): EnhancedWorkerPool {
  if (!quantumWorkerPool) {
    // For Vite, we need to use the worker URL directly
    const workerUrl = new URL('./quantum.worker.ts', import.meta.url).href;
    quantumWorkerPool = new EnhancedWorkerPool(
      workerUrl,
      navigator.hardwareConcurrency || 4,
      'QuantumWorkerPool'
    );
  }
  return quantumWorkerPool;
}

// Cleanup function
export async function terminateQuantumWorkerPool(): Promise<void> {
  if (quantumWorkerPool) {
    await quantumWorkerPool.shutdown();
    quantumWorkerPool = null;
  }
}

// Convenience function for single quantum computation
export async function computeQuantum<T>(
  type: string,
  data: any,
  priority?: number
): Promise<T> {
  const pool = getQuantumWorkerPool();
  return pool.execute<T>({
    id: `quantum-${Date.now()}-${Math.random()}`,
    type,
    data,
    priority
  });
}

// Convenience function for batch quantum computations
export async function computeQuantumBatch<T>(
  tasks: Array<{ type: string; data: any; priority?: number }>
): Promise<T[]> {
  const pool = getQuantumWorkerPool();
  const workerTasks = tasks.map((task, index) => ({
    id: `quantum-batch-${Date.now()}-${index}`,
    type: task.type,
    data: task.data,
    priority: task.priority
  }));
  return pool.executeBatch<T>(workerTasks);
}