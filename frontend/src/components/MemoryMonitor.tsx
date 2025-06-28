/**
 * Memory Monitor Component
 * Real-time memory usage visualization and management interface
 */

import React, { useState, useEffect, useCallback } from 'react';
import { 
  AlertTriangle, 
  Activity, 
  Trash2, 
  RefreshCw, 
  BarChart3,
  Database,
  Cpu,
  Zap,
  X
} from 'lucide-react';
import { getMemoryManager, MemoryStats, ResourceType } from '../utils/memoryManager';
import { Tooltip } from './ui/Tooltip';

interface MemoryMonitorProps {
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  collapsed?: boolean;
  primaryColor?: string;
  onClose?: () => void;
}

/**
 * Format bytes to human readable format
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format duration to human readable format
 */
function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

/**
 * Memory usage bar component
 */
const MemoryBar: React.FC<{
  used: number;
  total: number;
  warning: number;
  critical: number;
  label: string;
}> = ({ used, total, warning, critical, label }) => {
  const percentage = (used / total) * 100;
  const warningPercentage = (warning / total) * 100;
  const criticalPercentage = (critical / total) * 100;
  
  let barColor = '#4ade80'; // green
  if (used > critical) {
    barColor = '#ef4444'; // red
  } else if (used > warning) {
    barColor = '#f59e0b'; // yellow
  }

  return (
    <div className="memory-bar-container">
      <div className="memory-bar-header">
        <span className="memory-bar-label">{label}</span>
        <span className="memory-bar-value">
          {formatBytes(used * 1048576)} / {formatBytes(total * 1048576)}
        </span>
      </div>
      <div className="memory-bar-track">
        <div 
          className="memory-bar-fill"
          style={{ 
            width: `${percentage}%`,
            backgroundColor: barColor,
            transition: 'width 0.3s ease-out'
          }}
        />
        <Tooltip content="Warning threshold">
          <div 
            className="memory-bar-warning"
            style={{ left: `${warningPercentage}%` }}
          />
        </Tooltip>
        <Tooltip content="Critical threshold">
          <div 
            className="memory-bar-critical"
            style={{ left: `${criticalPercentage}%` }}
          />
        </Tooltip>
      </div>
      <div className="memory-bar-percentage">{percentage.toFixed(1)}%</div>
    </div>
  );
};

/**
 * Resource type icon
 */
const ResourceIcon: React.FC<{ type: ResourceType }> = ({ type }) => {
  const icons: Record<ResourceType, React.ReactNode> = {
    [ResourceType.COMPONENT]: <Activity size={14} />,
    [ResourceType.WORKER]: <Cpu size={14} />,
    [ResourceType.CANVAS]: <BarChart3 size={14} />,
    [ResourceType.WEBGL]: <Zap size={14} />,
    [ResourceType.TIMER]: <RefreshCw size={14} />,
    [ResourceType.SUBSCRIPTION]: <Database size={14} />,
    [ResourceType.EVENT_LISTENER]: <Activity size={14} />,
    [ResourceType.DOM_NODE]: <Database size={14} />,
    [ResourceType.OBSERVABLE]: <Activity size={14} />,
    [ResourceType.ANIMATION_FRAME]: <RefreshCw size={14} />,
    [ResourceType.WEBSOCKET]: <Database size={14} />,
    [ResourceType.MEDIA_STREAM]: <Activity size={14} />
  };
  
  return icons[type] || <Database size={14} />;
};

/**
 * Memory Monitor Component
 */
export const MemoryMonitor: React.FC<MemoryMonitorProps> = ({
  position = 'bottom-right',
  collapsed: initialCollapsed = false,
  primaryColor = '#3b82f6',
  onClose
}) => {
  const [isCollapsed, setIsCollapsed] = useState(initialCollapsed);
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [resources, setResources] = useState<any[]>([]);
  const [showDetails, setShowDetails] = useState(false);
  const [autoGC, setAutoGC] = useState(true);
  const memoryManager = getMemoryManager();

  // Update stats periodically
  useEffect(() => {
    const updateStats = () => {
      setStats(memoryManager.getStats());
      setResources(memoryManager.getResourceDetails());
    };

    updateStats();
    const interval = setInterval(updateStats, 1000);

    // Listen to memory events
    const handleWarning = (stats: MemoryStats) => {
      console.warn('Memory warning:', stats);
    };

    const handleCritical = (stats: MemoryStats) => {
      console.error('Memory critical:', stats);
      if (autoGC) {
        memoryManager.forceGC();
      }
    };

    const handleGC = (event: any) => {
      // GC completed - event contains cleanup details
    };

    memoryManager.on('memory:warning', handleWarning);
    memoryManager.on('memory:critical', handleCritical);
    memoryManager.on('gc:completed', handleGC);

    return () => {
      clearInterval(interval);
      memoryManager.off('memory:warning', handleWarning);
      memoryManager.off('memory:critical', handleCritical);
      memoryManager.off('gc:completed', handleGC);
    };
  }, [memoryManager, autoGC]);

  const handleForceGC = useCallback(() => {
    memoryManager.forceGC();
  }, [memoryManager]);

  const handleClearResource = useCallback((id: string) => {
    memoryManager.cleanup(id);
  }, [memoryManager]);

  if (!stats) return null;

  const positionStyles: Record<string, React.CSSProperties> = {
    'top-left': { top: 20, left: 20 },
    'top-right': { top: 20, right: 20 },
    'bottom-left': { bottom: 20, left: 20 },
    'bottom-right': { bottom: 20, right: 20 }
  };

  return (
    <div 
      className={`memory-monitor ${isCollapsed ? 'collapsed' : ''}`}
      style={{
        position: 'fixed',
        ...positionStyles[position],
        zIndex: 9999,
        background: 'rgba(0, 0, 0, 0.95)',
        border: `1px solid ${primaryColor}`,
        borderRadius: '8px',
        color: 'white',
        fontSize: '12px',
        fontFamily: 'monospace',
        transition: 'all 0.3s ease',
        backdropFilter: 'blur(10px)',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
      }}
    >
      {/* Header */}
      <div 
        className="memory-monitor-header"
        style={{
          padding: '8px 12px',
          borderBottom: isCollapsed ? 'none' : `1px solid ${primaryColor}30`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer'
        }}
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={16} style={{ color: primaryColor }} />
          <span style={{ fontWeight: 'bold' }}>Memory Monitor</span>
          {stats.percentUsed > 80 && (
            <AlertTriangle size={14} style={{ color: '#ef4444' }} />
          )}
        </div>
        <div style={{ display: 'flex', gap: '4px' }}>
          {onClose && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onClose();
              }}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#666',
                cursor: 'pointer',
                padding: '2px'
              }}
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      {!isCollapsed && (
        <div className="memory-monitor-content" style={{ padding: '12px' }}>
          {/* Memory Usage */}
          <MemoryBar
            used={stats.used}
            total={stats.total}
            warning={512}
            critical={1024}
            label="JS Heap"
          />

          {/* Resource Summary */}
          <div style={{ marginTop: '12px' }}>
            <div style={{ marginBottom: '8px', fontWeight: 'bold' }}>
              Resources ({stats.resourceCount})
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
              {Object.entries(stats.resourcesByType).map(([type, count]) => (
                <div 
                  key={type}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '2px 4px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '4px'
                  }}
                >
                  <ResourceIcon type={type as ResourceType} />
                  <span style={{ fontSize: '10px', opacity: 0.8 }}>
                    {type.replace(/_/g, ' ')}
                  </span>
                  <span style={{ marginLeft: 'auto', fontWeight: 'bold' }}>
                    {count}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div style={{ marginTop: '12px', display: 'flex', gap: '8px' }}>
            <Tooltip content="Force garbage collection">
              <button
                onClick={handleForceGC}
                className="memory-action-btn"
                style={{
                  flex: 1,
                  padding: '6px',
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '4px',
                  color: 'white',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '4px',
                  fontSize: '11px',
                  transition: 'all 0.2s'
                }}
              >
                <Trash2 size={12} />
                GC
              </button>
            </Tooltip>
            
            <Tooltip content="View resource details">
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="memory-action-btn"
                style={{
                  flex: 1,
                  padding: '6px',
                  background: showDetails ? primaryColor : 'rgba(255, 255, 255, 0.1)',
                  border: `1px solid ${showDetails ? primaryColor : 'rgba(255, 255, 255, 0.2)'}`,
                  borderRadius: '4px',
                  color: 'white',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '4px',
                  fontSize: '11px',
                  transition: 'all 0.2s'
                }}
              >
                <BarChart3 size={12} />
                Details
              </button>
            </Tooltip>

            <Tooltip content={`Auto GC: ${autoGC ? 'ON' : 'OFF'}`}>
              <button
                onClick={() => setAutoGC(!autoGC)}
                className="memory-action-btn"
                style={{
                  flex: 1,
                  padding: '6px',
                  background: autoGC ? primaryColor : 'rgba(255, 255, 255, 0.1)',
                  border: `1px solid ${autoGC ? primaryColor : 'rgba(255, 255, 255, 0.2)'}`,
                  borderRadius: '4px',
                  color: 'white',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '4px',
                  fontSize: '11px',
                  transition: 'all 0.2s'
                }}
              >
                <RefreshCw size={12} />
                Auto
              </button>
            </Tooltip>
          </div>

          {/* GC Stats */}
          <div style={{ 
            marginTop: '8px', 
            fontSize: '10px', 
            opacity: 0.6,
            textAlign: 'center'
          }}>
            Last GC: {formatDuration(Date.now() - stats.lastGC)} ago • 
            Total GCs: {stats.gcCount}
          </div>

          {/* Resource Details */}
          {showDetails && (
            <div style={{
              marginTop: '12px',
              maxHeight: '200px',
              overflowY: 'auto',
              background: 'rgba(0, 0, 0, 0.5)',
              borderRadius: '4px',
              padding: '8px'
            }}>
              <div style={{ marginBottom: '8px', fontWeight: 'bold' }}>
                Top Resources
              </div>
              {resources.slice(0, 10).map((resource) => (
                <div
                  key={resource.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '4px',
                    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                    fontSize: '10px'
                  }}
                >
                  <ResourceIcon type={resource.type} />
                  <div style={{ flex: 1 }}>
                    <div style={{ opacity: 0.8 }}>{resource.id}</div>
                    {resource.component && (
                      <div style={{ opacity: 0.6 }}>{resource.component}</div>
                    )}
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div>{resource.size}</div>
                    <div style={{ opacity: 0.6 }}>{resource.age}</div>
                  </div>
                  <button
                    onClick={() => handleClearResource(resource.id)}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#ef4444',
                      cursor: 'pointer',
                      padding: '2px'
                    }}
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

/**
 * Global styles for memory monitor
 */
const styles = `
  .memory-monitor {
    min-width: 300px;
    max-width: 400px;
  }
  
  .memory-monitor.collapsed {
    min-width: auto;
  }
  
  .memory-bar-container {
    margin-bottom: 8px;
  }
  
  .memory-bar-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
    font-size: 11px;
  }
  
  .memory-bar-track {
    position: relative;
    height: 20px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
  }
  
  .memory-bar-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    border-radius: 4px;
  }
  
  .memory-bar-warning,
  .memory-bar-critical {
    position: absolute;
    top: 0;
    width: 2px;
    height: 100%;
    background: rgba(255, 255, 255, 0.3);
  }
  
  .memory-bar-critical {
    background: rgba(239, 68, 68, 0.5);
  }
  
  .memory-bar-percentage {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-weight: bold;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
  }
  
  .memory-action-btn:hover {
    background: rgba(255, 255, 255, 0.2) !important;
  }
`;

// Inject styles
if (typeof document !== 'undefined') {
  const styleElement = document.createElement('style');
  styleElement.textContent = styles;
  document.head.appendChild(styleElement);
}