import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import './styles/tooltip.css';
import './styles/theory-of-everything.css';
import './styles/common-components.css';
import App from './App';

// Enable API debugging in development
if (process.env.NODE_ENV === 'development' || !process.env.NODE_ENV) {
  import('./utils/apiDebugger').then(module => {
    console.log('[Recursia] API Debugger loaded');
  }).catch(err => {
    console.warn('[Recursia] Failed to load API debugger:', err);
  });
}

// Enhanced debugging
console.log('[Recursia] Index.tsx loading at', new Date().toISOString());

// Verify imports loaded
console.log('[Recursia] React loaded:', !!React);
console.log('[Recursia] React version:', React.version);
console.log('[Recursia] App component loaded:', !!App);

// Check DOM
const container = document.getElementById('root');
console.log('[Recursia] Root container found:', !!container);

if (!container) {
  document.body.innerHTML = '<h1 style="color: red;">Root container not found!</h1>';
  throw new Error('Root container not found');
}

try {
  console.log('[Recursia] Creating React root...');
  const root = createRoot(container);
  
  console.log('[Recursia] Starting React render...');
  // Temporarily disable StrictMode to debug WebSocket issues
  // StrictMode causes double mounting which can interfere with WebSocket connections
  root.render(
    <App />
  );
  
  console.log('[Recursia] React render initiated successfully');
  
  // Check if render actually happened after a delay
  setTimeout(() => {
    if (container.children.length === 0) {
      console.error('[Recursia] WARNING: Root container is still empty after render!');
    } else {
      console.log('[Recursia] SUCCESS: React rendered', container.children.length, 'children');
    }
  }, 1000);
  
} catch (error) {
  console.error('[Recursia] React render error:', error);
  document.body.innerHTML = `
    <div style="padding: 20px; background: #ff0000; color: white;">
      <h1>Application Error</h1>
      <pre>${error}\n${error.stack}</pre>
    </div>
  `;
}