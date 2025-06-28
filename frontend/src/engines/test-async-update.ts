/**
 * Test the async update implementation
 */

import { OSHQuantumEngine } from './OSHQuantumEngine';

async function testAsyncUpdate() {
  console.log('=== Testing OSH Quantum Engine Async Updates ===');
  
  try {
    // Create engine instance
    console.log('Creating OSH Quantum Engine...');
    const engine = new OSHQuantumEngine();
    
    // Start engine
    console.log('Starting engine...');
    engine.start();
    
    // Wait for initialization
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // Test update with timing
    console.log('\nTesting update performance...');
    
    const testIterations = 5;
    const updateTimes: number[] = [];
    
    for (let i = 0; i < testIterations; i++) {
      const startTime = performance.now();
      
      // Call update method
      engine.update(0.016); // 16ms = 60fps
      
      const endTime = performance.now();
      const updateTime = endTime - startTime;
      updateTimes.push(updateTime);
      
      console.log(`Update ${i + 1}: ${updateTime.toFixed(2)}ms`);
      
      // Small delay between updates
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    
    // Calculate average
    const avgTime = updateTimes.reduce((a, b) => a + b, 0) / updateTimes.length;
    console.log(`\nAverage update time: ${avgTime.toFixed(2)}ms`);
    
    // Check if any update exceeded 5 seconds
    const maxTime = Math.max(...updateTimes);
    console.log(`Max update time: ${maxTime.toFixed(2)}ms`);
    
    if (maxTime > 5000) {
      console.error('❌ Update exceeded 5 second timeout!');
    } else if (maxTime > 100) {
      console.warn('⚠️  Update time is high but within limits');
    } else {
      console.log('✅ Update times are good');
    }
    
    // Test metrics
    console.log('\nEngine metrics:', engine.getMetrics());
    
    // Stop engine
    console.log('\nStopping engine...');
    await engine.stop();
    
    console.log('Test complete!');
    
  } catch (error) {
    console.error('Test failed:', error);
  }
}

// Run test if this file is executed directly
if (import.meta.url === `file://${__filename}`) {
  testAsyncUpdate().catch(console.error);
}

export { testAsyncUpdate };