/**
 * Memory Stability Test Suite
 * Automated tests to ensure memory usage remains stable
 */

import { getMemoryManager } from '../utils/memoryManager';
import { getQuantumWorkerPool } from '../workers/quantumWorkerPool';
import { ThreeMemoryManager } from '../utils/ThreeMemoryManager';

interface TestResult {
  name: string;
  passed: boolean;
  duration: number;
  memoryBefore: number;
  memoryAfter: number;
  memoryDelta: number;
  resourcesBefore: number;
  resourcesAfter: number;
  details?: string;
}

/**
 * Get current memory usage
 */
function getMemoryUsage(): number {
  if ('memory' in performance && (performance as any).memory) {
    return (performance as any).memory.usedJSHeapSize / 1048576; // MB
  }
  return 0;
}

/**
 * Wait for garbage collection
 */
async function waitForGC(): Promise<void> {
  // Try to trigger GC if available
  if (typeof (globalThis as any).gc === 'function') {
    (globalThis as any).gc();
  }
  
  // Wait a bit for GC to complete
  await new Promise(resolve => setTimeout(resolve, 100));
}

/**
 * Run a test with memory tracking
 */
async function runMemoryTest(
  name: string,
  testFn: () => Promise<void>
): Promise<TestResult> {
  const memoryManager = getMemoryManager();
  
  // Initial GC
  await waitForGC();
  
  const memoryBefore = getMemoryUsage();
  const resourcesBefore = memoryManager.getStats().resourceCount;
  const startTime = performance.now();
  
  try {
    // Run the test
    await testFn();
    
    // Force cleanup
    memoryManager.forceGC();
    await waitForGC();
    
    const endTime = performance.now();
    const memoryAfter = getMemoryUsage();
    const resourcesAfter = memoryManager.getStats().resourceCount;
    
    const result: TestResult = {
      name,
      passed: true,
      duration: endTime - startTime,
      memoryBefore,
      memoryAfter,
      memoryDelta: memoryAfter - memoryBefore,
      resourcesBefore,
      resourcesAfter
    };
    
    // Check for memory leaks
    if (result.memoryDelta > 10) { // 10MB threshold
      result.passed = false;
      result.details = `Memory increased by ${result.memoryDelta.toFixed(2)}MB`;
    }
    
    if (resourcesAfter > resourcesBefore) {
      result.passed = false;
      result.details = `${resourcesAfter - resourcesBefore} resources leaked`;
    }
    
    return result;
  } catch (error) {
    return {
      name,
      passed: false,
      duration: performance.now() - startTime,
      memoryBefore,
      memoryAfter: getMemoryUsage(),
      memoryDelta: 0,
      resourcesBefore,
      resourcesAfter: memoryManager.getStats().resourceCount,
      details: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

/**
 * Test: Worker Pool Creation and Destruction
 */
async function testWorkerPoolMemory(): Promise<TestResult> {
  return runMemoryTest('Worker Pool Memory', async () => {
    const iterations = 100;
    
    for (let i = 0; i < iterations; i++) {
      const pool = getQuantumWorkerPool();
      
      // Execute some tasks
      const tasks = Array.from({ length: 10 }, (_, j) => ({
        id: `task-${i}-${j}`,
        type: 'test',
        data: { value: Math.random() }
      }));
      
      await pool.executeBatch(tasks);
    }
    
    // Cleanup
    await getQuantumWorkerPool().shutdown();
  });
}

/**
 * Test: React Component Mounting/Unmounting
 */
async function testComponentLifecycle(): Promise<TestResult> {
  return runMemoryTest('Component Lifecycle', async () => {
    const { render, unmountComponentAtNode } = await import('react-dom');
    const { MemoryMonitor } = await import('../components/MemoryMonitor');
    const React = await import('react');
    
    const container = document.createElement('div');
    document.body.appendChild(container);
    
    try {
      const iterations = 50;
      
      for (let i = 0; i < iterations; i++) {
        // Mount component
        render(React.createElement(MemoryMonitor, {
          position: 'top-left',
          collapsed: true
        }), container);
        
        // Wait a bit
        await new Promise(resolve => setTimeout(resolve, 10));
        
        // Unmount component
        unmountComponentAtNode(container);
        
        // Wait for cleanup
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    } finally {
      document.body.removeChild(container);
    }
  });
}

/**
 * Test: Three.js Scene Creation and Disposal
 */
async function testThreeJSMemory(): Promise<TestResult> {
  return runMemoryTest('Three.js Memory', async () => {
    const THREE = await import('three');
    const iterations = 20;
    
    for (let i = 0; i < iterations; i++) {
      const threeManager = new ThreeMemoryManager(`three-test-${i}`);
      const scene = new THREE.Scene();
      
      // Add objects to scene
      for (let j = 0; j < 100; j++) {
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(
          Math.random() * 10 - 5,
          Math.random() * 10 - 5,
          Math.random() * 10 - 5
        );
        scene.add(mesh);
      }
      
      // Track scene
      threeManager.trackScene(scene, `scene-${i}`);
      
      // Simulate some work
      await new Promise(resolve => setTimeout(resolve, 50));
      
      // Dispose everything
      threeManager.disposeScene(scene);
      threeManager.disposeAll();
    }
  });
}

/**
 * Test: Timer and Event Listener Cleanup
 */
async function testTimersAndEvents(): Promise<TestResult> {
  return runMemoryTest('Timers and Events', async () => {
    const memoryManager = getMemoryManager();
    const iterations = 100;
    
    for (let i = 0; i < iterations; i++) {
      const timers: number[] = [];
      const listeners: Array<{ target: EventTarget; type: string; handler: () => void }> = [];
      
      // Create timers
      for (let j = 0; j < 10; j++) {
        const timerId = window.setTimeout(() => {}, 1000000);
        timers.push(timerId);
        
        memoryManager.track(`timer-${i}-${j}`, 'timer' as any, () => {
          window.clearTimeout(timerId);
        });
      }
      
      // Create event listeners
      for (let j = 0; j < 10; j++) {
        const handler = () => {};
        const target = document.createElement('div');
        target.addEventListener('click', handler);
        listeners.push({ target, type: 'click', handler });
        
        memoryManager.track(`listener-${i}-${j}`, 'event_listener' as any, () => {
          target.removeEventListener('click', handler);
        });
      }
      
      // Cleanup
      for (const timerId of timers) {
        window.clearTimeout(timerId);
      }
      
      for (const { target, type, handler } of listeners) {
        target.removeEventListener(type, handler);
      }
      
      // Cleanup tracked resources
      for (let j = 0; j < 10; j++) {
        memoryManager.cleanup(`timer-${i}-${j}`);
        memoryManager.cleanup(`listener-${i}-${j}`);
      }
    }
  });
}

/**
 * Test: Large Data Structure Management
 */
async function testLargeDataStructures(): Promise<TestResult> {
  return runMemoryTest('Large Data Structures', async () => {
    const memoryManager = getMemoryManager();
    const iterations = 10;
    
    for (let i = 0; i < iterations; i++) {
      // Create large array
      const largeArray = new Float32Array(1000000); // ~4MB
      
      memoryManager.track(`array-${i}`, 'component' as any, () => {
        // Clear reference
        largeArray.fill(0);
      }, {
        size: largeArray.byteLength
      });
      
      // Simulate work
      for (let j = 0; j < largeArray.length; j += 1000) {
        largeArray[j] = Math.random();
      }
      
      // Cleanup
      memoryManager.cleanup(`array-${i}`);
    }
  });
}

/**
 * Run all memory stability tests
 */
export async function runMemoryStabilityTests(): Promise<TestResult[]> {
  console.log('Starting memory stability tests...');
  
  const tests = [
    testWorkerPoolMemory,
    testComponentLifecycle,
    testThreeJSMemory,
    testTimersAndEvents,
    testLargeDataStructures
  ];
  
  const results: TestResult[] = [];
  
  for (const test of tests) {
    console.log(`Running ${test.name}...`);
    const result = await test();
    results.push(result);
    console.log(`${result.passed ? '✅' : '❌'} ${result.name}:`, {
      duration: `${result.duration.toFixed(2)}ms`,
      memoryDelta: `${result.memoryDelta.toFixed(2)}MB`,
      resourceDelta: result.resourcesAfter - result.resourcesBefore,
      details: result.details
    });
    
    // Wait between tests
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  // Summary
  const passed = results.filter(r => r.passed).length;
  const failed = results.filter(r => !r.passed).length;
  const totalMemoryDelta = results.reduce((sum, r) => sum + r.memoryDelta, 0);
  
  console.log('\n=== Memory Stability Test Summary ===');
  console.log(`Total tests: ${results.length}`);
  console.log(`Passed: ${passed}`);
  console.log(`Failed: ${failed}`);
  console.log(`Total memory delta: ${totalMemoryDelta.toFixed(2)}MB`);
  console.log('===================================\n');
  
  return results;
}

/**
 * Continuous memory monitoring
 */
export function startContinuousMonitoring(intervalMs = 5000): () => void {
  const memoryManager = getMemoryManager();
  let lastMemory = getMemoryUsage();
  let lastResources = memoryManager.getStats().resourceCount;
  
  const interval = setInterval(() => {
    const currentMemory = getMemoryUsage();
    const currentResources = memoryManager.getStats().resourceCount;
    const stats = memoryManager.getStats();
    
    const memoryDelta = currentMemory - lastMemory;
    const resourceDelta = currentResources - lastResources;
    
    if (Math.abs(memoryDelta) > 1 || resourceDelta !== 0) {
      console.log('[Memory Monitor]', {
        memory: `${currentMemory.toFixed(2)}MB (${memoryDelta > 0 ? '+' : ''}${memoryDelta.toFixed(2)}MB)`,
        resources: `${currentResources} (${resourceDelta > 0 ? '+' : ''}${resourceDelta})`,
        byType: stats.resourcesByType
      });
    }
    
    lastMemory = currentMemory;
    lastResources = currentResources;
  }, intervalMs);
  
  return () => clearInterval(interval);
}

// Export for use in browser console
if (typeof window !== 'undefined') {
  (window as any).memoryTests = {
    runAll: runMemoryStabilityTests,
    startMonitoring: startContinuousMonitoring,
    stopMonitoring: null as (() => void) | null
  };
}