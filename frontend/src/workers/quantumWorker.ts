// Import worker with Vite syntax
import QuantumWorker from './quantum.worker?worker';

export function createQuantumWorker(): Worker {
  return new QuantumWorker();
}