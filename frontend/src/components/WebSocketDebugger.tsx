import React, { useEffect, useState } from 'react';
import { useEngineAPI } from '../hooks/useEngineAPI';

export const WebSocketDebugger: React.FC = () => {
  const engineAPI = useEngineAPI();
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    const addLog = (message: string) => {
      const timestamp = new Date().toISOString();
      setLogs(prev => [...prev, `[${timestamp}] ${message}`]);
    };

    // Log initial state
    addLog(`WebSocket Status: ${engineAPI.wsStatus}`);
    addLog(`Is Connected: ${engineAPI.isConnected}`);
    addLog(`Error: ${engineAPI.error || 'None'}`);

    // Log metrics updates
    const metricsStr = JSON.stringify({
      rsp: engineAPI.metrics.rsp,
      coherence: engineAPI.metrics.coherence,
      universe_running: engineAPI.metrics.universe_running,
      iteration_count: engineAPI.metrics.iteration_count
    });
    addLog(`Current Metrics: ${metricsStr}`);

    // Set up interval to log status
    const interval = setInterval(() => {
      addLog(`[Status Check] wsStatus: ${engineAPI.wsStatus}, isConnected: ${engineAPI.isConnected}`);
    }, 5000);

    return () => clearInterval(interval);
  }, [engineAPI.wsStatus, engineAPI.isConnected, engineAPI.error]);

  return (
    <div style={{
      position: 'fixed',
      bottom: 20,
      right: 20,
      width: 400,
      maxHeight: 300,
      background: 'rgba(0, 0, 0, 0.9)',
      border: '1px solid #0f0',
      borderRadius: 4,
      padding: 10,
      fontFamily: 'monospace',
      fontSize: 12,
      color: '#0f0',
      overflow: 'auto',
      zIndex: 9999
    }}>
      <h3 style={{ margin: '0 0 10px 0' }}>WebSocket Debugger</h3>
      <div>
        <div>Status: <span style={{ color: engineAPI.isConnected ? '#0f0' : '#f00' }}>
          {engineAPI.wsStatus}
        </span></div>
        <div>Connected: {engineAPI.isConnected ? 'Yes' : 'No'}</div>
        <div>Error: {engineAPI.error || 'None'}</div>
        <hr style={{ border: '1px solid #0f0', margin: '10px 0' }} />
        <div style={{ maxHeight: 150, overflow: 'auto' }}>
          {logs.map((log, i) => (
            <div key={i} style={{ fontSize: 10, opacity: 0.8 }}>{log}</div>
          ))}
        </div>
      </div>
    </div>
  );
};