import React, { useEffect, useState } from 'react';
import { OSHQuantumEngine } from '../engines/OSHQuantumEngine';
import { Settings2, X, CheckCircle, AlertCircle, Cpu, Database, Brain, Zap } from 'lucide-react';
import { Tooltip } from './ui/Tooltip';
import { useEngineAPIContext } from '../contexts/EngineAPIContext';
import { ErrorBoundary } from './common/ErrorBoundary';
import { Skeleton } from './common/LoadingStates';

interface EngineStatusProps {
  engine: OSHQuantumEngine | null;
  primaryColor?: string;
}

const EngineStatusComponent = React.forwardRef<any, EngineStatusProps>(({ engine, primaryColor = '#ffd700' }, ref) => {
  const [status, setStatus] = useState<Record<string, boolean>>({});
  const [isOpen, setIsOpen] = useState(false);
  const [showMetrics, setShowMetrics] = useState(true);
  const { metrics, isConnected } = useEngineAPIContext();
  
  useEffect(() => {
    if (!engine) {
      setStatus({ engine: false });
      return;
    }
    
    // Check each subsystem
    const checkStatus = {
      engine: !!engine,
      memoryFieldEngine: !!engine.memoryFieldEngine,
      entropyCoherenceSolver: !!engine.entropyCoherenceSolver,
      rspEngine: !!engine.rspEngine,
      observerEngine: !!engine.observerEngine,
      simulationHarness: !!engine.simulationHarness,
      errorReductionPlatform: !!engine.errorReductionPlatform,
      mlObserver: !!engine.mlObserver,
      macroTeleportation: !!engine.macroTeleportation,
      curvatureGenerator: !!engine.curvatureGenerator,
      snapshotManager: !!engine.snapshotManager,
      coherenceLocking: !!engine.coherenceLocking,
      tensorField: !!engine.tensorField,
      introspection: !!engine.introspection
    };
    
    setStatus(checkStatus);
  }, [engine]);
  
  const allReady = Object.values(status).every(v => v);
  
  return (
    <>
      {/* Toggle Button */}
      <Tooltip content="Toggle Engine Status">
        <button
          onClick={() => setIsOpen(!isOpen)}
          style={{
            position: 'fixed',
            bottom: '50px',
            right: '20px',
            width: '36px',
            height: '36px',
            background: allReady ? primaryColor : '#ff6b6b',
            border: 'none',
            borderRadius: '50%',
            color: '#000',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
            zIndex: 10001,
            transition: 'all 0.2s ease'
          }}
        >
          <Settings2 size={20} />
        </button>
      </Tooltip>
      
      {/* Status Panel */}
      {isOpen && (
        <div style={{
          position: 'fixed',
          bottom: '50px',
          right: '70px',
          background: 'rgba(0, 0, 0, 0.95)',
          border: '1px solid #333',
          borderRadius: '8px',
          padding: '16px',
          color: 'white',
          fontSize: '12px',
          fontFamily: 'monospace',
          maxHeight: '300px',
          minWidth: '250px',
          overflow: 'auto',
          zIndex: 10000,
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)'
        }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '12px'
          }}>
            <h4 style={{ margin: 0, color: allReady ? primaryColor : '#ff6b6b' }}>
              Engine Status
            </h4>
            <button
              onClick={() => setIsOpen(false)}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#999',
                cursor: 'pointer',
                padding: '4px'
              }}
            >
              <X size={16} />
            </button>
          </div>
          
          <div style={{ fontSize: '11px', color: '#888', marginBottom: '8px' }}>
            {allReady ? 'All systems operational' : 'Initializing subsystems...'}
          </div>
          
          {/* Engine Subsystems */}
          <div style={{ marginBottom: '12px' }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '4px',
              marginBottom: '8px',
              color: '#ccc',
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              <Cpu size={12} />
              Engine Subsystems
            </div>
            {Object.entries(status).map(([key, value]) => (
              <div key={key} style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                marginBottom: '3px',
                padding: '2px 0',
                fontSize: '11px'
              }}>
                <span style={{ color: '#999' }}>
                  {key.replace(/([A-Z])/g, ' $1').trim()}:
                </span>
                <span style={{ color: value ? primaryColor : '#ff6b6b' }}>
                  {value ? '✓' : '✗'}
                </span>
              </div>
            ))}
          </div>
          
          {/* Backend Connection */}
          <div style={{ marginBottom: '12px' }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '4px',
              marginBottom: '8px',
              color: '#ccc',
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              <Database size={12} />
              Backend Connection
            </div>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              marginBottom: '3px',
              padding: '2px 0',
              fontSize: '11px'
            }}>
              <span style={{ color: '#999' }}>API Status:</span>
              <span style={{ 
                color: isConnected ? primaryColor : '#ff6b6b',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}>
                {isConnected ? (
                  <><CheckCircle size={10} /> Connected</>
                ) : (
                  <><AlertCircle size={10} /> Disconnected</>
                )}
              </span>
            </div>
          </div>
          
          {/* Live Metrics */}
          {metrics && (
            <div>
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '4px',
                marginBottom: '8px',
                color: '#ccc',
                fontSize: '11px',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                cursor: 'pointer'
              }} onClick={() => setShowMetrics(!showMetrics)}>
                <Brain size={12} />
                Live Metrics
                <span style={{ marginLeft: 'auto', fontSize: '10px' }}>
                  {showMetrics ? '−' : '+'}
                </span>
              </div>
              
              {showMetrics && (
                <div style={{ fontSize: '10px' }}>
                  <MetricRow label="RSP" value={metrics.rsp} format="number" />
                  <MetricRow label="Coherence" value={metrics.coherence} format="percentage" />
                  <MetricRow label="Entropy" value={metrics.entropy} format="decimal" />
                  <MetricRow label="Information" value={metrics.information} format="decimal" />
                  <MetricRow label="Φ (Phi)" value={metrics.phi} format="decimal" />
                  <MetricRow label="Error Rate" value={metrics.error} format="percentage" />
                  <MetricRow label="Observer Count" value={metrics.observer_count} format="integer" />
                  <MetricRow label="State Count" value={metrics.state_count} format="integer" />
                  <MetricRow label="FPS" value={metrics.fps} format="integer" />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </>
  );
});

// Helper component for metric rows
const MetricRow: React.FC<{ label: string; value: any; format: string }> = ({ label, value, format }) => {
  const formatValue = (val: any, fmt: string) => {
    if (val === undefined || val === null) return '—';
    
    switch (fmt) {
      case 'percentage':
        return `${(val * 100).toFixed(1)}%`;
      case 'decimal':
        return val.toFixed(3);
      case 'number':
        return val.toFixed(2);
      case 'integer':
        return Math.round(val).toString();
      default:
        return val.toString();
    }
  };
  
  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'space-between',
      marginBottom: '2px',
      padding: '1px 0'
    }}>
      <span style={{ color: '#888' }}>{label}:</span>
      <span style={{ color: '#ccc', fontFamily: 'monospace' }}>
        {formatValue(value, format)}
      </span>
    </div>
  );
};

// Export with error boundary
export const EngineStatus = React.forwardRef<any, EngineStatusProps>((props, ref) => (
  <ErrorBoundary componentName="EngineStatus">
    <EngineStatusComponent ref={ref} {...props} />
  </ErrorBoundary>
));