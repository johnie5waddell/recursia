/**
 * System Health Dashboard
 * Comprehensive monitoring of all frontend components
 */

import React, { useEffect, useState } from 'react';
import { 
  Activity, 
  CheckCircle, 
  AlertTriangle, 
  XCircle,
  RefreshCw,
  Download,
  ChevronDown,
  ChevronUp,
  Cpu,
  Database,
  Globe,
  Zap
} from 'lucide-react';
import { useEngineAPIContext } from '../contexts/EngineAPIContext';
import { ComponentHealth, runFullSystemHealthCheck } from '../utils/componentHealthCheck';
import { ErrorBoundary } from './common/ErrorBoundary';
import { motion, AnimatePresence } from 'framer-motion';

interface SystemHealthDashboardProps {
  primaryColor?: string;
  onClose?: () => void;
}

const SystemHealthDashboardComponent: React.FC<SystemHealthDashboardProps> = ({ 
  primaryColor = '#4fc3f7',
  onClose 
}) => {
  const { metrics, isConnected } = useEngineAPIContext();
  const [componentHealth, setComponentHealth] = useState<ComponentHealth[]>([]);
  const [expandedComponents, setExpandedComponents] = useState<Set<string>>(new Set());
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Run health checks
  useEffect(() => {
    const runHealthCheck = () => {
      const components = [
        {
          name: 'QuantumOSHStudio',
          metrics,
          isConnected,
          hasErrorBoundary: false, // Main component doesn't have one yet
          hasLoadingState: true,
          hasEmptyState: false,
          displayedFields: ['rsp', 'coherence', 'entropy', 'information', 'error', 'qv'],
          availableFields: metrics ? Object.keys(metrics) : []
        },
        {
          name: 'RSPDashboard',
          metrics,
          isConnected,
          hasErrorBoundary: false,
          hasLoadingState: true,
          hasEmptyState: true,
          displayedFields: ['rsp', 'information', 'coherence', 'entropy', 'drsp_dt'],
          availableFields: ['rsp', 'information', 'coherence', 'entropy', 'drsp_dt', 'di_dt', 'dc_dt', 'de_dt']
        },
        {
          name: 'OSHCalculationsPanel',
          metrics,
          isConnected,
          hasErrorBoundary: false,
          hasLoadingState: true,
          hasEmptyState: false
        },
        // MemoryFieldVisualizer removed
        // {
        //   name: 'MemoryFieldVisualizer',
        //   metrics,
        //   isConnected,
        //   hasErrorBoundary: false,
        //   hasLoadingState: true,
        //   hasEmptyState: false,
        //   displayedFields: ['memory_fragments', 'strain', 'coherence'],
        //   availableFields: ['memory_fragments', 'strain', 'coherence', 'field_energy']
        // },
        {
          name: 'OSHUniverse3D',
          metrics,
          isConnected,
          hasErrorBoundary: true, // This one has error boundary
          hasLoadingState: true,
          hasEmptyState: false
        },
        {
          name: 'GravitationalWaveEchoVisualizer',
          metrics,
          isConnected,
          hasErrorBoundary: false,
          hasLoadingState: true,
          hasEmptyState: true,
          displayedFields: ['information_curvature', 'strain'],
          availableFields: ['information_curvature', 'strain', 'field_energy']
        },
        {
          name: 'InformationalCurvatureMap',
          metrics,
          isConnected,
          hasErrorBoundary: false,
          hasLoadingState: false,
          hasEmptyState: false
        },
        {
          name: 'QuantumCircuitDesigner',
          metrics,
          isConnected,
          hasErrorBoundary: false,
          hasLoadingState: false,
          hasEmptyState: false
        },
        {
          name: 'DataFlowMonitor',
          metrics,
          isConnected,
          hasErrorBoundary: false,
          hasLoadingState: false,
          hasEmptyState: false
        },
        {
          name: 'DiagnosticPanel',
          metrics,
          isConnected,
          hasErrorBoundary: false,
          hasLoadingState: false,
          hasEmptyState: false
        },
        {
          name: 'EngineStatus',
          metrics,
          isConnected,
          hasErrorBoundary: true, // Updated with error boundary
          hasLoadingState: false,
          hasEmptyState: false
        }
      ];

      const health = runFullSystemHealthCheck(components);
      setComponentHealth(health);
      setLastUpdate(new Date());
    };

    runHealthCheck();

    if (autoRefresh) {
      const interval = setInterval(runHealthCheck, 5000);
      return () => clearInterval(interval);
    }
  }, [metrics, isConnected, autoRefresh]);

  const toggleComponent = (name: string) => {
    setExpandedComponents(prev => {
      const newSet = new Set(prev);
      if (newSet.has(name)) {
        newSet.delete(name);
      } else {
        newSet.add(name);
      }
      return newSet;
    });
  };

  const getStatusIcon = (status: ComponentHealth['status']) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle size={16} color="#4ade80" />;
      case 'warning':
        return <AlertTriangle size={16} color="#fbbf24" />;
      case 'error':
        return <XCircle size={16} color="#ef4444" />;
    }
  };

  const getStatusColor = (status: ComponentHealth['status']) => {
    switch (status) {
      case 'healthy':
        return '#4ade80';
      case 'warning':
        return '#fbbf24';
      case 'error':
        return '#ef4444';
    }
  };

  const exportHealthReport = () => {
    const report = {
      timestamp: new Date().toISOString(),
      systemStatus: {
        apiConnected: isConnected,
        metricsAvailable: !!metrics,
        totalComponents: componentHealth.length,
        healthyComponents: componentHealth.filter(c => c.status === 'healthy').length,
        warningComponents: componentHealth.filter(c => c.status === 'warning').length,
        errorComponents: componentHealth.filter(c => c.status === 'error').length
      },
      componentDetails: componentHealth
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `system-health-${new Date().toISOString()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const overallStatus = componentHealth.some(c => c.status === 'error') ? 'error' :
                       componentHealth.some(c => c.status === 'warning') ? 'warning' : 'healthy';

  const healthyCount = componentHealth.filter(c => c.status === 'healthy').length;
  const warningCount = componentHealth.filter(c => c.status === 'warning').length;
  const errorCount = componentHealth.filter(c => c.status === 'error').length;

  return (
    <div className="system-health-dashboard">
      <div className="dashboard-header">
        <div className="header-title">
          <Activity size={20} />
          <h2>System Health Monitor</h2>
          <div className={`status-indicator ${overallStatus}`} />
        </div>
        <div className="header-actions">
          <button 
            className="action-button"
            onClick={() => setAutoRefresh(!autoRefresh)}
            title={autoRefresh ? "Disable auto-refresh" : "Enable auto-refresh"}
          >
            <RefreshCw size={16} className={autoRefresh ? 'spinning' : ''} />
          </button>
          <button 
            className="action-button"
            onClick={exportHealthReport}
            title="Export health report"
          >
            <Download size={16} />
          </button>
          {onClose && (
            <button className="close-button" onClick={onClose}>×</button>
          )}
        </div>
      </div>

      <div className="dashboard-summary">
        <div className="summary-card">
          <Cpu size={24} color={primaryColor} />
          <div className="summary-content">
            <span className="summary-label">Components</span>
            <span className="summary-value">{componentHealth.length}</span>
          </div>
        </div>
        <div className="summary-card healthy">
          <CheckCircle size={24} />
          <div className="summary-content">
            <span className="summary-label">Healthy</span>
            <span className="summary-value">{healthyCount}</span>
          </div>
        </div>
        <div className="summary-card warning">
          <AlertTriangle size={24} />
          <div className="summary-content">
            <span className="summary-label">Warnings</span>
            <span className="summary-value">{warningCount}</span>
          </div>
        </div>
        <div className="summary-card error">
          <XCircle size={24} />
          <div className="summary-content">
            <span className="summary-label">Errors</span>
            <span className="summary-value">{errorCount}</span>
          </div>
        </div>
      </div>

      <div className="dashboard-connection">
        <div className="connection-status">
          <Database size={16} />
          <span>Backend API:</span>
          <span className={`status ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <div className="connection-status">
          <Globe size={16} />
          <span>WebSocket:</span>
          <span className={`status ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? 'Active' : 'Inactive'}
          </span>
        </div>
        <div className="connection-status">
          <Zap size={16} />
          <span>Last Update:</span>
          <span className="timestamp">{lastUpdate.toLocaleTimeString()}</span>
        </div>
      </div>

      <div className="dashboard-components">
        {componentHealth.map(component => (
          <motion.div 
            key={component.name}
            className={`component-card ${component.status}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div 
              className="component-header"
              onClick={() => toggleComponent(component.name)}
              style={{ cursor: 'pointer' }}
            >
              <div className="component-info">
                {getStatusIcon(component.status)}
                <span className="component-name">{component.name}</span>
                <span className="component-message">{component.message}</span>
              </div>
              <div className="component-toggle">
                {expandedComponents.has(component.name) ? 
                  <ChevronUp size={16} /> : 
                  <ChevronDown size={16} />
                }
              </div>
            </div>

            <AnimatePresence>
              {expandedComponents.has(component.name) && (
                <motion.div 
                  className="component-details"
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  {component.checks.map((check, index) => (
                    <div key={index} className="check-item">
                      <span className={`check-status ${check.passed ? 'passed' : 'failed'}`}>
                        {check.passed ? '✓' : '✗'}
                      </span>
                      <span className="check-name">{check.name}:</span>
                      <span className="check-message">{check.message}</span>
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>

      <style>{`
        .system-health-dashboard {
          background: rgba(0, 0, 0, 0.95);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 24px;
          max-width: 800px;
          margin: 0 auto;
          color: #fff;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
        }

        .header-title {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .header-title h2 {
          margin: 0;
          font-size: 20px;
          font-weight: 600;
        }

        .status-indicator {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          animation: pulse 2s ease-in-out infinite;
        }

        .status-indicator.healthy { background: #4ade80; }
        .status-indicator.warning { background: #fbbf24; }
        .status-indicator.error { background: #ef4444; }

        .header-actions {
          display: flex;
          gap: 8px;
        }

        .action-button {
          background: rgba(255, 255, 255, 0.1);
          border: none;
          border-radius: 6px;
          padding: 8px;
          color: #fff;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .action-button:hover {
          background: rgba(255, 255, 255, 0.2);
        }

        .spinning {
          animation: spin 2s linear infinite;
        }

        .close-button {
          background: transparent;
          border: none;
          color: #999;
          font-size: 24px;
          cursor: pointer;
          padding: 0 8px;
        }

        .dashboard-summary {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 16px;
          margin-bottom: 24px;
        }

        .summary-card {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          padding: 16px;
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .summary-card.healthy { border: 1px solid rgba(74, 222, 128, 0.3); }
        .summary-card.warning { border: 1px solid rgba(251, 191, 36, 0.3); }
        .summary-card.error { border: 1px solid rgba(239, 68, 68, 0.3); }

        .summary-content {
          display: flex;
          flex-direction: column;
        }

        .summary-label {
          font-size: 12px;
          color: #999;
        }

        .summary-value {
          font-size: 24px;
          font-weight: 600;
        }

        .dashboard-connection {
          display: flex;
          gap: 24px;
          margin-bottom: 24px;
          padding: 16px;
          background: rgba(255, 255, 255, 0.02);
          border-radius: 8px;
        }

        .connection-status {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
        }

        .status.connected { color: #4ade80; }
        .status.disconnected { color: #ef4444; }
        .timestamp { color: #999; }

        .dashboard-components {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .component-card {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          overflow: hidden;
          border: 1px solid transparent;
          transition: all 0.2s ease;
        }

        .component-card.healthy { border-color: rgba(74, 222, 128, 0.2); }
        .component-card.warning { border-color: rgba(251, 191, 36, 0.2); }
        .component-card.error { border-color: rgba(239, 68, 68, 0.2); }

        .component-header {
          padding: 16px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .component-info {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .component-name {
          font-weight: 500;
        }

        .component-message {
          color: #999;
          font-size: 14px;
        }

        .component-details {
          padding: 0 16px 16px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          overflow: hidden;
        }

        .check-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 0;
          font-size: 13px;
        }

        .check-status {
          font-weight: 600;
          width: 20px;
        }

        .check-status.passed { color: #4ade80; }
        .check-status.failed { color: #ef4444; }

        .check-name {
          color: #999;
          min-width: 150px;
        }

        .check-message {
          color: #ccc;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

// Export with error boundary
export const SystemHealthDashboard: React.FC<SystemHealthDashboardProps> = (props) => (
  <ErrorBoundary componentName="SystemHealthDashboard">
    <SystemHealthDashboardComponent {...props} />
  </ErrorBoundary>
);