/**
 * API Debugger Utility
 * Intercepts and logs all API requests and responses for debugging metrics flow
 */

// Store original fetch for restoration
const originalFetch = window.fetch;

/**
 * Enable API debugging with request/response logging
 */
export function enableAPIDebugging() {
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    const method = init?.method || 'GET';
    
    console.group(`[API Debug] ${method} ${url}`);
    
    // Log request details
    if (init?.body) {
      try {
        const body = typeof init.body === 'string' ? JSON.parse(init.body) : init.body;
        console.log('Request Body:', body);
      } catch {
        console.log('Request Body (raw):', init.body);
      }
    }
    
    try {
      // Make the actual request
      const response = await originalFetch(input, init);
      
      // Clone response so we can read it without consuming
      const responseClone = response.clone();
      
      // Try to parse JSON response
      try {
        const data = await responseClone.json();
        console.log('Response Status:', response.status);
        console.log('Response Data:', data);
        
        // Special logging for metrics endpoints
        if (url.includes('/metrics')) {
          // For /api/metrics endpoint, the response IS the metrics
          const nonZeroMetrics = Object.entries(data)
            .filter(([key, value]) => typeof value === 'number' && value !== 0 && value !== 0.0)
            .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});
          
          console.log('%cMetrics in Response:', 'color: #4CAF50; font-weight: bold');
          console.table(nonZeroMetrics);
        } else if (url.includes('/execute')) {
          // For execute endpoint, metrics are nested
          if (data.metrics) {
            const nonZeroMetrics = Object.entries(data.metrics)
              .filter(([_, value]) => value !== 0 && value !== 0.0)
              .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});
            
            console.log('%cMetrics in Execute Response:', 'color: #4CAF50; font-weight: bold');
            console.table(nonZeroMetrics);
          } else {
            console.warn('%cNo metrics in execute response!', 'color: #FF5722; font-weight: bold');
          }
        }
      } catch {
        // Not JSON, log as text
        const text = await responseClone.text();
        console.log('Response (text):', text);
      }
      
      console.groupEnd();
      return response;
    } catch (error) {
      console.error('Request failed:', error);
      console.groupEnd();
      throw error;
    }
  };
  
  console.log('%c[API Debugger] Enabled - All API calls will be logged', 'color: #2196F3; font-weight: bold');
}

/**
 * Disable API debugging and restore original fetch
 */
export function disableAPIDebugging() {
  window.fetch = originalFetch;
  console.log('%c[API Debugger] Disabled', 'color: #9E9E9E');
}

/**
 * Test API endpoints directly
 */
export async function testAPIEndpoints(baseUrl: string = 'http://localhost:8080') {
  console.group('[API Test] Testing all endpoints');
  
  // Test metrics endpoint
  try {
    console.log('\n1. Testing /api/metrics...');
    const metricsRes = await fetch(`${baseUrl}/api/metrics`);
    const metrics = await metricsRes.json();
    console.log('Metrics response:', metrics);
    
    const nonZero = Object.entries(metrics)
      .filter(([_, value]) => value !== 0 && value !== 0.0);
    console.log('Non-zero metrics:', Object.fromEntries(nonZero));
  } catch (error) {
    console.error('Metrics endpoint failed:', error);
  }
  
  // Test states endpoint
  try {
    console.log('\n2. Testing /api/states...');
    const statesRes = await fetch(`${baseUrl}/api/states`);
    const states = await statesRes.json();
    console.log('States response:', states);
  } catch (error) {
    console.error('States endpoint failed:', error);
  }
  
  // Test execution with simple program
  try {
    console.log('\n3. Testing /api/execute...');
    const code = `
state TestState : quantum_type {
  state_qubits: 2,
  state_coherence: 0.95
};
measure TestState by coherence;
print "Test complete";
`;
    
    const execRes = await fetch(`${baseUrl}/api/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, iterations: 1 })
    });
    
    const result = await execRes.json();
    console.log('Execution result:', result);
    
    if (result.metrics) {
      const nonZero = Object.entries(result.metrics)
        .filter(([_, value]) => value !== 0 && value !== 0.0);
      console.log('Metrics from execution:', Object.fromEntries(nonZero));
    }
  } catch (error) {
    console.error('Execute endpoint failed:', error);
  }
  
  console.groupEnd();
}

// Auto-enable debugging in development - DISABLED to avoid confusion
// The API debugger intercepts REST calls which show default values
// WebSocket metrics are the real-time values that should be displayed
if (process.env.NODE_ENV === 'development') {
  // Don't auto-enable - it's confusing to see REST API defaults
  // enableAPIDebugging();
  
  // Add global debugging functions
  (window as any).apiDebug = {
    enable: enableAPIDebugging,
    disable: disableAPIDebugging,
    test: testAPIEndpoints
  };
  
  console.log('%cAPI Debugger available but NOT auto-enabled. Use window.apiDebug.enable() to enable.', 'color: #9E9E9E');
}