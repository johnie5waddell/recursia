/**
 * Diagnostic Panel - Real-time performance and execution monitoring
 * 
 * Displays diagnostics information to help identify performance bottlenecks
 * and hanging operations in the quantum simulation.
 */

import React, { useEffect, useState, useRef } from 'react';
import { DiagnosticsSystem, DiagnosticEntry, PerformanceMetric } from '../utils/diagnostics';
import { AlertTriangle, Activity, Clock, Zap, X, Wifi, WifiOff } from 'lucide-react';
import { useEngineAPIContext } from '../contexts/EngineAPIContext';

interface DiagnosticPanelProps {
  onClose?: () => void;
}

export const DiagnosticPanel: React.FC<DiagnosticPanelProps> = ({ onClose }) => {
  const [activeTraces, setActiveTraces] = useState<Array<{ id: string; duration: number }>>([]);
  const [slowOps, setSlowOps] = useState<DiagnosticEntry[]>([]);
  const [errors, setErrors] = useState<DiagnosticEntry[]>([]);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetric[]>([]);
  const [updateCounter, setUpdateCounter] = useState(0);
  
  // Get real-time metrics from websocket
  const { metrics, isConnected } = useEngineAPIContext();
  
  const diagnostics = DiagnosticsSystem.getInstance();
  const intervalRef = useRef<NodeJS.Timeout>();
  
  useEffect(() => {
    // Update diagnostics data every 500ms
    const updateData = () => {
      setActiveTraces(diagnostics.getActiveTraces());
      setSlowOps(diagnostics.getSlowOperations(50));
      setErrors(diagnostics.getErrors());
      setPerformanceMetrics(diagnostics.getPerformanceReport().slice(0, 10));
      setUpdateCounter(prev => prev + 1);
    };
    
    updateData();
    intervalRef.current = setInterval(updateData, 500);
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);
  
  const formatDuration = (ms: number): string => {
    if (ms < 1) return `${ms.toFixed(2)}ms`;
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };
  
  const exportDiagnostics = () => {
    const data = diagnostics.export();
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `diagnostics-${new Date().toISOString()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  return (
    <div className="diagnostic-panel">
      <div className="panel-header">
        <h3>
          <Activity className="icon" />
          Diagnostics Monitor
          {isConnected ? (
            <Wifi size={14} color="#4ade80" style={{ marginLeft: '8px' }} />
          ) : (
            <WifiOff size={14} color="#ef4444" style={{ marginLeft: '8px' }} />
          )}
        </h3>
        <button onClick={onClose} className="close-btn">
          <X size={16} />
        </button>
      </div>
      
      <div className="panel-content">
        {/* Real-time WebSocket Metrics */}
        {metrics && (
          <div className="section" style={{ marginBottom: '16px' }}>
            <h4 style={{ color: '#888', fontSize: '12px', marginBottom: '8px' }}>Real-time Metrics</h4>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(2, 1fr)', 
              gap: '8px',
              padding: '8px',
              backgroundColor: 'rgba(255, 255, 255, 0.02)',
              borderRadius: '4px'
            }}>
              <div style={{ fontSize: '11px' }}>
                <span style={{ color: '#666' }}>RSP:</span>
                <span style={{ color: '#4fc3f7', marginLeft: '8px', fontWeight: 'bold' }}>
                  {metrics.rsp.toFixed(2)}
                </span>
              </div>
              <div style={{ fontSize: '11px' }}>
                <span style={{ color: '#666' }}>Coherence:</span>
                <span style={{ color: '#4fc3f7', marginLeft: '8px', fontWeight: 'bold' }}>
                  {(metrics.coherence * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{ fontSize: '11px' }}>
                <span style={{ color: '#666' }}>FPS:</span>
                <span style={{ 
                  color: metrics.fps < 30 ? '#ef4444' : '#4ade80', 
                  marginLeft: '8px',
                  fontWeight: 'bold'
                }}>
                  {metrics.fps.toFixed(0)}
                </span>
              </div>
              <div style={{ fontSize: '11px' }}>
                <span style={{ color: '#666' }}>Error:</span>
                <span style={{ 
                  color: metrics.error > 0.1 ? '#ef4444' : '#4ade80', 
                  marginLeft: '8px',
                  fontWeight: 'bold'
                }}>
                  {(metrics.error * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{ fontSize: '11px' }}>
                <span style={{ color: '#666' }}>Quantum Vol:</span>
                <span style={{ color: '#a78bfa', marginLeft: '8px', fontWeight: 'bold' }}>
                  {metrics.quantum_volume.toFixed(1)}
                </span>
              </div>
              <div style={{ fontSize: '11px' }}>
                <span style={{ color: '#666' }}>Observers:</span>
                <span style={{ color: '#60a5fa', marginLeft: '8px', fontWeight: 'bold' }}>
                  {metrics.observer_count}
                </span>
              </div>
            </div>
          </div>
        )}
        
        {/* Active/Hanging Operations */}
        <div className="section">
          <h4>
            <Clock className="icon" />
            Active Operations ({activeTraces.length})
          </h4>
          {activeTraces.length === 0 ? (
            <p className="empty-state">No active operations</p>
          ) : (
            <div className="trace-list">
              {activeTraces.map(trace => (
                <div 
                  key={trace.id} 
                  className={`trace-item ${trace.duration > 1000 ? 'hanging' : ''}`}
                >
                  <span className="trace-id">{trace.id.split('-')[0]}</span>
                  <span className={`duration ${trace.duration > 1000 ? 'critical' : ''}`}>
                    {formatDuration(trace.duration)}
                  </span>
                  {trace.duration > 1000 && <AlertTriangle className="warning-icon" size={14} />}
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Performance Metrics */}
        <div className="section">
          <h4>
            <Zap className="icon" />
            Performance Metrics
          </h4>
          {performanceMetrics.length === 0 ? (
            <p className="empty-state">No metrics collected</p>
          ) : (
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Operation</th>
                  <th>Count</th>
                  <th>Avg</th>
                  <th>Max</th>
                </tr>
              </thead>
              <tbody>
                {performanceMetrics.map(metric => (
                  <tr key={metric.name}>
                    <td className="operation-name">{metric.name}</td>
                    <td>{metric.count}</td>
                    <td className={metric.avgTime > 50 ? 'slow' : ''}>
                      {formatDuration(metric.avgTime)}
                    </td>
                    <td className={metric.maxTime > 100 ? 'critical' : ''}>
                      {formatDuration(metric.maxTime)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        
        {/* Recent Errors */}
        {errors.length > 0 && (
          <div className="section errors">
            <h4>
              <AlertTriangle className="icon" />
              Recent Errors ({errors.length})
            </h4>
            <div className="error-list">
              {errors.slice(-5).map(error => (
                <div key={error.id} className="error-item">
                  <div className="error-header">
                    <span className="error-location">{error.component}.{error.method}</span>
                    <span className="error-time">
                      {new Date(error.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  {error.metadata?.error && (
                    <div className="error-message">{error.metadata.error}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Slow Operations */}
        {slowOps.length > 0 && (
          <div className="section">
            <h4>Slow Operations ({slowOps.length})</h4>
            <div className="slow-ops-list">
              {slowOps.slice(0, 5).map(op => (
                <div key={op.id} className="slow-op-item">
                  <span className="op-name">{op.component}.{op.method}</span>
                  <span className="op-duration">{formatDuration(op.duration || 0)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      <div className="panel-footer">
        <button onClick={() => diagnostics.clear()} className="btn-secondary">
          Clear Data
        </button>
        <button onClick={exportDiagnostics} className="btn-primary">
          Export Diagnostics
        </button>
      </div>
      
      <style>{`
        .diagnostic-panel {
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 600px;
          max-height: 80vh;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
          z-index: 10000;
          display: flex;
          flex-direction: column;
        }
        
        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px;
          border-bottom: 1px solid #333;
        }
        
        .panel-header h3 {
          display: flex;
          align-items: center;
          gap: 8px;
          margin: 0;
          color: #fff;
        }
        
        .close-btn {
          background: transparent;
          border: none;
          color: #999;
          cursor: pointer;
          padding: 4px;
        }
        
        .close-btn:hover {
          color: #fff;
        }
        
        .panel-content {
          flex: 1;
          overflow-y: auto;
          padding: 16px;
        }
        
        .section {
          margin-bottom: 24px;
        }
        
        .section h4 {
          display: flex;
          align-items: center;
          gap: 8px;
          margin: 0 0 12px 0;
          color: #fff;
          font-size: 14px;
        }
        
        .icon {
          width: 16px;
          height: 16px;
        }
        
        .empty-state {
          color: #666;
          font-style: italic;
          margin: 8px 0;
        }
        
        .trace-list {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        
        .trace-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 12px;
          background: #222;
          border-radius: 4px;
          font-family: monospace;
          font-size: 12px;
        }
        
        .trace-item.hanging {
          background: #442222;
          border: 1px solid #663333;
        }
        
        .trace-id {
          color: #4ecdc4;
        }
        
        .duration {
          color: #999;
        }
        
        .duration.critical {
          color: #ff6b6b;
          font-weight: bold;
        }
        
        .warning-icon {
          color: #ffa500;
        }
        
        .metrics-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 12px;
        }
        
        .metrics-table th {
          text-align: left;
          padding: 8px;
          border-bottom: 1px solid #333;
          color: #999;
          font-weight: normal;
        }
        
        .metrics-table td {
          padding: 8px;
          border-bottom: 1px solid #222;
        }
        
        .operation-name {
          color: #4ecdc4;
          font-family: monospace;
        }
        
        .slow {
          color: #ffa500;
        }
        
        .critical {
          color: #ff6b6b;
        }
        
        .error-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .error-item {
          background: #332222;
          border: 1px solid #553333;
          border-radius: 4px;
          padding: 8px 12px;
          font-size: 12px;
        }
        
        .error-header {
          display: flex;
          justify-content: space-between;
          margin-bottom: 4px;
        }
        
        .error-location {
          color: #ff6b6b;
          font-family: monospace;
        }
        
        .error-time {
          color: #666;
          font-size: 11px;
        }
        
        .error-message {
          color: #ccc;
          margin-top: 4px;
          font-family: monospace;
          font-size: 11px;
          word-break: break-word;
        }
        
        .slow-ops-list {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        
        .slow-op-item {
          display: flex;
          justify-content: space-between;
          padding: 6px 12px;
          background: #222;
          border-radius: 4px;
          font-size: 12px;
        }
        
        .op-name {
          color: #4ecdc4;
          font-family: monospace;
        }
        
        .op-duration {
          color: #ffa500;
        }
        
        .panel-footer {
          display: flex;
          justify-content: flex-end;
          gap: 8px;
          padding: 16px;
          border-top: 1px solid #333;
        }
        
        .btn-primary, .btn-secondary {
          padding: 8px 16px;
          border: none;
          border-radius: 4px;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .btn-primary {
          background: #4ecdc4;
          color: #000;
        }
        
        .btn-primary:hover {
          background: #3db5ad;
        }
        
        .btn-secondary {
          background: #333;
          color: #fff;
        }
        
        .btn-secondary:hover {
          background: #444;
        }
      `}</style>
    </div>
  );
};

export default DiagnosticPanel;