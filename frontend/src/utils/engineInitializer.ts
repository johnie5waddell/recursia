/**
 * Engine Initialization Utilities
 * Ensures proper initialization sequence and error handling
 */

import { OSHQuantumEngine } from '../engines/OSHQuantumEngine';
import { Complex } from './complex';

/**
 * Initialize the OSH Quantum Engine with proper error handling and validation
 */
export async function initializeEngine(): Promise<OSHQuantumEngine> {
  const startTime = performance.now();
  
  try {
    console.log('[EngineInit] Starting engine initialization...');
    console.log('[EngineInit] Available memory:', 
      'memory' in performance ? 
      `${((performance as any).memory.jsHeapSizeLimit / 1024 / 1024).toFixed(0)}MB` : 
      'unknown');
    
    // Create engine asynchronously to avoid blocking
    console.log('[EngineInit] Creating OSHQuantumEngine instance...');
    const engineStartTime = performance.now();
    
    // Create engine directly without setTimeout wrapper
    let engine: OSHQuantumEngine;
    try {
      console.log('[EngineInit] Constructing OSHQuantumEngine...');
      engine = new OSHQuantumEngine();
      console.log('[EngineInit] OSHQuantumEngine constructed in', 
        (performance.now() - engineStartTime).toFixed(1), 'ms');
    } catch (error) {
      console.error('[EngineInit] Failed to construct engine:', error);
      throw error;
    }
    
    console.log('[EngineInit] Engine instance created');
    
    // Start the engine asynchronously
    console.log('[EngineInit] Starting engine subsystems...');
    const startEngineTime = performance.now();
    
    try {
      engine.start();
      console.log('[EngineInit] Engine started in', 
        (performance.now() - startEngineTime).toFixed(1), 'ms');
    } catch (error) {
      console.error('[EngineInit] Failed to start engine:', error);
      throw error;
    }
    
    // Initialize simulation harness if needed
    if (engine.simulationHarness) {
      const harnessStartTime = performance.now();
      console.log('[EngineInit] Checking simulation harness initialization...');
      
      if (!engine.simulationHarness['isInitialized']) {
        console.log('[EngineInit] Initializing simulation harness...');
        await engine.simulationHarness.initialize();
        console.log('[EngineInit] Simulation harness initialized in',
          (performance.now() - harnessStartTime).toFixed(1), 'ms');
      } else {
        console.log('[EngineInit] Simulation harness already initialized');
      }
    }
    
    // Run initial update cycles to populate engine state
    console.log('[EngineInit] Running initial update cycles...');
    const updateStartTime = performance.now();
    try {
      // Run a few update cycles to ensure all systems are initialized
      for (let i = 0; i < 3; i++) {
        engine.update(0.016); // 16ms per frame at 60fps
        await new Promise(resolve => setTimeout(resolve, 10)); // Small delay between updates
      }
      console.log('[EngineInit] Initial updates completed in',
        (performance.now() - updateStartTime).toFixed(1), 'ms');
    } catch (error) {
      console.warn('[EngineInit] Initial update warning:', error);
      // Non-fatal - continue with initialization
    }
    
    // Wait for initial state to be ready
    console.log('[EngineInit] Waiting for engine to be ready...');
    const readyStartTime = performance.now();
    await waitForEngineReady(engine);
    console.log('[EngineInit] Engine ready in',
      (performance.now() - readyStartTime).toFixed(1), 'ms');
    
    // Initialize universe with quantum states asynchronously
    console.log('[EngineInit] Initializing quantum universe...');
    const universeStartTime = performance.now();
    
    await new Promise<void>((resolve) => {
      setTimeout(() => {
        try {
          initializeQuantumUniverse(engine);
          console.log('[EngineInit] Quantum universe initialized in',
            (performance.now() - universeStartTime).toFixed(1), 'ms');
          resolve();
        } catch (error) {
          console.error('[EngineInit] Failed to initialize quantum universe:', error);
          throw error;
        }
      }, 10);
    });
    
    const totalTime = performance.now() - startTime;
    console.log('[EngineInit] Total initialization time:', totalTime.toFixed(1), 'ms');
    
    if (totalTime > 5000) {
      console.warn('[EngineInit] Initialization took longer than expected!');
    }
    
    return engine;
  } catch (error) {
    const totalTime = performance.now() - startTime;
    console.error('[EngineInit] Engine initialization failed after', 
      totalTime.toFixed(1), 'ms:', error);
    throw error;
  }
}

/**
 * Wait for engine to be ready with timeout
 */
async function waitForEngineReady(engine: OSHQuantumEngine, timeout: number = 5000): Promise<void> {
  const startTime = Date.now();
  
  while (Date.now() - startTime < timeout) {
    try {
      // Check if all subsystems are initialized
      // Since WavefunctionSimulator is disabled, only check essential components
      const memoryField = engine.memoryFieldEngine?.getField();
      const rspState = engine.rspEngine?.getState();
      const observers = engine.observerEngine?.getAllObservers();
      
      // Verify core engines are initialized and have valid state
      if (memoryField && memoryField.fragments && memoryField.fragments.length > 0 &&
          rspState && typeof rspState.rsp === 'number' && isFinite(rspState.rsp) &&
          observers && Array.isArray(observers)) {
        // Engine is ready - all core systems operational
        console.log('[EngineInit] Engine ready - Memory fragments:', memoryField.fragments.length, 
                    'RSP:', rspState.rsp, 'Observers:', observers.length);
        return;
      }
    } catch (error) {
      // Not ready yet - this is expected during initialization
      // Only log if we're getting close to timeout
      if (Date.now() - startTime > timeout * 0.8) {
        console.warn('[EngineInit] Still waiting for engine readiness:', error);
      }
    }
    
    // Wait a bit before checking again
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  // Provide detailed error message for debugging
  let statusMessage = 'Engine initialization timeout. Status:';
  try {
    statusMessage += `\n- MemoryFieldEngine: ${engine.memoryFieldEngine ? 'exists' : 'missing'}`;
    statusMessage += `\n- Fragments: ${engine.memoryFieldEngine?.getField()?.fragments?.length || 0}`;
    statusMessage += `\n- RSPEngine: ${engine.rspEngine ? 'exists' : 'missing'}`;
    statusMessage += `\n- RSP value: ${engine.rspEngine?.getState()?.rsp || 'N/A'}`;
    statusMessage += `\n- ObserverEngine: ${engine.observerEngine ? 'exists' : 'missing'}`;
  } catch (e) {
    statusMessage += `\n- Error collecting status: ${e}`;
  }
  
  throw new Error(statusMessage);
}

/**
 * Initialize the quantum universe with baseline states
 */
function initializeQuantumUniverse(engine: OSHQuantumEngine): void {
  // Initialize with multiple quantum states for robust initialization
  const positions: [number, number, number][] = [
    [0, 0, 0],
    [2, 1, -1], 
    [-2, -1, 1],
    [1, -2, 2]
  ];
  
  positions.forEach((position, index) => {
    // Create different quantum states for variety
    const stateVector = Array(8).fill(0).map((_, i) => {
      if (index === 0 && i === 0) {
        // Ground state |000⟩
        return new Complex(1, 0);
      } else if (index === 1 && i === 1) {
        // First excited state |001⟩
        return new Complex(0.707, 0);
      } else if (index === 2 && i === 2) {
        // Second excited state |010⟩
        return new Complex(0.5, 0.5);
      } else if (index === 3 && i === 3) {
        // Third excited state |011⟩
        return new Complex(0.6, 0);
      }
      return new Complex(0, 0);
    });
    
    // Normalize the state vector
    let norm = 0;
    stateVector.forEach(c => norm += c.magnitude() ** 2);
    if (norm > 0) {
      const normFactor = 1 / Math.sqrt(norm);
      stateVector.forEach(c => {
        c.real *= normFactor;
        c.imag *= normFactor;
      });
    }
    
    // Add all fragments to ensure proper initialization
    engine.memoryFieldEngine.addFragment(stateVector, position);
  });
  
  // Add initial observers
  const observerPositions: [number, number, number][] = [
    [3, 0, 0],
    [-3, 0, 0],
    [0, 3, 0],
    [0, -3, 0]
  ];
  
  observerPositions.forEach((pos, i) => {
    engine.observerEngine.addObserver({
      name: `Observer-${i + 1}`,
      focus: pos,
      phase: i * Math.PI / 2,
      collapseThreshold: 0.7,
      memoryParticipation: 0.5,
      entangledObservers: [],
      observationHistory: []
    });
  });
}