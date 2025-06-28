import React, { useEffect, useState } from 'react';

export const WebSocketTest: React.FC = () => {
  const [logs, setLogs] = useState<string[]>([]);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [status, setStatus] = useState('Initializing...');

  const addLog = (message: string) => {
    console.log(`[WebSocketTest] ${message}`);
    setLogs(prev => [...prev, `${new Date().toISOString()} - ${message}`]);
  };

  useEffect(() => {
    addLog('Component mounted');
    
    // Test multiple WebSocket URLs
    const testConnections = async () => {
      // Test 1: Relative path (should use Vite proxy)
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const proxyUrl = `${wsProtocol}//${window.location.host}/ws`;
      addLog(`Testing proxied URL: ${proxyUrl}`);
      
      try {
        const ws1 = new WebSocket(proxyUrl);
        
        ws1.onopen = () => {
          addLog('✓ Proxied WebSocket connected!');
          setStatus('Connected via proxy');
          setWs(ws1);
        };
        
        ws1.onerror = (event) => {
          addLog(`✗ Proxied WebSocket error`);
          console.error('Proxy WebSocket error:', event);
        };
        
        ws1.onclose = (event) => {
          addLog(`Proxied WebSocket closed: ${event.code} ${event.reason}`);
        };
        
        ws1.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'metrics_update') {
              addLog(`Metrics: running=${data.data.universe_running}, iter=${data.data.iteration_count}`);
            } else {
              addLog(`Message: ${data.type}`);
            }
          } catch (e) {
            addLog(`Raw message: ${event.data.substring(0, 100)}`);
          }
        };
        
      } catch (e) {
        addLog(`Failed to create proxied WebSocket: ${e}`);
      }
      
      // Test 2: Direct connection (after 2 seconds)
      setTimeout(() => {
        addLog('Testing direct connection...');
        const directUrl = 'ws://localhost:8080/ws';
        
        try {
          const ws2 = new WebSocket(directUrl);
          
          ws2.onopen = () => {
            addLog('✓ Direct WebSocket also works!');
            ws2.close();
          };
          
          ws2.onerror = () => {
            addLog('✗ Direct connection failed');
          };
          
        } catch (e) {
          addLog(`Direct connection error: ${e}`);
        }
      }, 2000);
    };
    
    testConnections();
    
    return () => {
      if (ws) {
        addLog('Closing WebSocket on unmount');
        ws.close();
      }
    };
  }, []);

  return (
    <div style={{
      position: 'fixed',
      top: 20,
      right: 20,
      width: 500,
      maxHeight: 400,
      background: 'rgba(0, 0, 0, 0.95)',
      border: '2px solid #0f0',
      borderRadius: 8,
      padding: 20,
      fontFamily: 'monospace',
      fontSize: 12,
      color: '#0f0',
      overflow: 'auto',
      zIndex: 10000
    }}>
      <h2 style={{ margin: '0 0 10px 0' }}>WebSocket Connection Test</h2>
      <div style={{ marginBottom: 10 }}>
        Status: <span style={{ color: ws ? '#0f0' : '#f00' }}>{status}</span>
      </div>
      <div style={{ 
        maxHeight: 300, 
        overflow: 'auto',
        background: '#000',
        padding: 10,
        borderRadius: 4
      }}>
        {logs.map((log, i) => (
          <div key={i} style={{ 
            fontSize: 10, 
            marginBottom: 2,
            color: log.includes('✓') ? '#0f0' : log.includes('✗') ? '#f00' : '#fff'
          }}>
            {log}
          </div>
        ))}
      </div>
    </div>
  );
};