import React, { useEffect, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Provider } from 'react-redux';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';

import { store } from './store';
import { QuantumOSHStudio } from './components/QuantumOSHStudio';
import { ErrorBoundary } from './utils/ErrorBoundary';
import { getMemoryManager } from './utils/memoryManager';
import { terminateQuantumWorkerPool } from './workers/quantumWorkerPool';
import { getMemorySafeGridSize, getOptimalWorkerCount } from './config/memoryConfig';
import { EngineAPIProvider } from './contexts/EngineAPIContext';

// Styles
import './styles/globals.css';
import './styles/theme-variables.css';
import './styles/quantum-osh-studio.css';
import './styles/rsp-dashboard-v2.css';

// Create a query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  
  
  // Initialize memory management
  useEffect(() => {
    const memoryManager = getMemoryManager();
    
    // Cleanup on unmount
    return () => {
      // Terminate worker pools
      terminateQuantumWorkerPool();
      
      // Dispose memory manager
      memoryManager.dispose();
    };
  }, []);
  
  
  try {
    
    return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <Provider store={store}>
          <EngineAPIProvider>
            <Router>
              <Routes>
                <Route path="/" element={<QuantumOSHStudio />} />
                <Route path="/studio" element={<QuantumOSHStudio />} />
              </Routes>
            </Router>
            <Toaster 
              position="bottom-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#1a1a1a',
                  color: '#fff',
                  border: '1px solid #2a2a2a',
                },
              }}
/>
          </EngineAPIProvider>
        </Provider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
  } catch (error: any) {
    console.error('[Recursia] App render error:', error);
    return (
      <div style={{ padding: '20px', background: '#ff0000', color: 'white' }}>
        <h1>App Error</h1>
        <pre>{error.message}</pre>
        <pre>{error.stack}</pre>
      </div>
    );
  }
}

export default App;