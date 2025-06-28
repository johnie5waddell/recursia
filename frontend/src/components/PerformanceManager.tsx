/**
 * Performance Manager Component
 * Monitors and manages application performance in real-time
 */

import React, { useEffect, useState, useCallback } from 'react';
import { Activity, AlertTriangle, Cpu, HardDrive, Zap, TrendingUp, Wifi, WifiOff } from 'lucide-react';
import { getSystemResourceMonitor, useSystemMetrics } from '../utils/systemResourceMonitor';
import { MemoryLeakDetector, useMemoryLeakDetector } from '../utils/memoryLeakDetector';
import { getPerformanceOptimizer } from '../utils/performanceOptimizer';
import { getResourceThrottler } from '../utils/resourceThrottler';
import { getQuantumWorkerPool } from '../utils/optimizedWorkerPool';
import { Tooltip } from './ui/Tooltip';
import { useEngineAPIContext } from '../contexts/EngineAPIContext';

interface PerformanceManagerProps {
  primaryColor?: string;
}

export const PerformanceManager: React.FC<PerformanceManagerProps> = ({
  primaryColor = '#ffd700'
}) => {
  const { metrics, health } = useSystemMetrics();
  const [isOpen, setIsOpen] = useState(false);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [autoOptimize, setAutoOptimize] = useState(true);
  
  // Get real-time metrics from websocket
  const { metrics: wsMetrics, isConnected } = useEngineAPIContext();
  
  // Use memory leak detector
  useMemoryLeakDetector('PerformanceManager');
  
  // Get instances
  const resourceMonitor = getSystemResourceMonitor();
  const leakDetector = MemoryLeakDetector.getInstance();
  const performanceOptimizer = getPerformanceOptimizer();
  const resourceThrottler = getResourceThrottler();
  const workerPool = getQuantumWorkerPool();
  
  // Worker stats state - must be declared before any conditional returns
  const [workerStats, setWorkerStats] = useState(() => workerPool.getStats());
  
  // Update worker stats frequently
  useEffect(() => {
    const updateStats = () => {
      setWorkerStats(workerPool.getStats());
    };
    
    const interval = setInterval(updateStats, 500); // Update every 500ms
    
    return () => clearInterval(interval);
  }, [workerPool]);
  
  // Update recommendations periodically
  useEffect(() => {
    const updateRecommendations = () => {
      const newRecommendations = [
        ...resourceMonitor.getRecommendations(),
        ...performanceOptimizer.getRecommendations()
      ];
      setRecommendations(newRecommendations);
    };
    
    updateRecommendations();
    const interval = setInterval(updateRecommendations, 5000);
    
    return () => clearInterval(interval);
  }, [resourceMonitor, performanceOptimizer]);
  
  // Worker pool will show activity when actual quantum calculations are performed
  // No need for test tasks - the pool will be used by the quantum engines
  
  // Auto-optimize when enabled
  useEffect(() => {
    if (autoOptimize && metrics) {
      // Apply optimizations based on health status
      if (health === 'critical') {
        performanceOptimizer.applyOptimizations();
        
        // Clear memory leaks
        const leakSummary = leakDetector.getSummary();
        if (leakSummary.totalLeaks > 5) {
          leakDetector.cleanup();
        }
      }
    }
  }, [autoOptimize, health, metrics, performanceOptimizer, leakDetector]);
  
  // Handle optimization actions
  const handleClearCache = useCallback(() => {
    if ('caches' in window) {
      caches.keys().then(names => {
        names.forEach(name => caches.delete(name));
      });
    }
    
    // Force garbage collection if available
    if (typeof (global as any).gc === 'function') {
      (global as any).gc();
    }
    
    // Cache cleared and garbage collection triggered
  }, []);
  
  const handleClearLeaks = useCallback(() => {
    leakDetector.cleanup();
    // Memory leaks cleaned up
  }, [leakDetector]);
  
  const handleOptimizeNow = useCallback(() => {
    performanceOptimizer.applyOptimizations();
    // Optimizations applied
  }, [performanceOptimizer]);
  
  const healthColor = health === 'healthy' ? '#4ade80' : 
                     health === 'warning' ? '#fbbf24' : '#ef4444';
  
  const throttlerStats = resourceThrottler.getStats();
  const leakSummary = leakDetector.getSummary();
  
  // Always render the button, even if metrics aren't loaded yet
  return (
    <>
      {/* Toggle Button - positioned on left side opposite to EngineStatus */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          position: 'fixed',
          bottom: '50px',
          left: '20px',  // Left side instead of right
          width: '36px',
          height: '36px',
          background: health === 'healthy' ? (primaryColor || '#ffd700') : health === 'warning' ? '#fbbf24' : '#ef4444',
          border: 'none',
          borderRadius: '50%',
          color: '#000',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
          zIndex: 10001,
          transition: 'all 0.2s ease',
          animation: 'fadeInScale 0.3s ease-out'
        }}
        title="Toggle Performance Monitor"
      >
        <Activity size={20} />
      </button>
      
      {/* Performance Panel */}
      {isOpen && (
        <div style={{
          position: 'fixed',
          bottom: '50px',
          left: '70px',  // Left side instead of right
          background: 'rgba(0, 0, 0, 0.95)',
          border: '1px solid #333',
          borderRadius: '8px',
          padding: '16px',
          color: 'white',
          fontSize: '12px',
          fontFamily: 'monospace',
          maxHeight: '400px',
          minWidth: '320px',
          maxWidth: '400px',
          overflow: 'auto',
          zIndex: 10000,
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)'
        }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '12px',
            borderBottom: '1px solid #333',
            paddingBottom: '12px'
          }}>
            <h4 style={{ margin: 0, color: healthColor, fontSize: '14px' }}>
              Performance Monitor
            </h4>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <Tooltip content={autoOptimize ? "Disable auto-optimization" : "Enable auto-optimization"}>
                <button 
                  style={{
                    background: autoOptimize ? primaryColor : 'transparent',
                    border: `1px solid ${autoOptimize ? primaryColor : '#666'}`,
                    borderRadius: '4px',
                    color: autoOptimize ? 'white' : '#999',
                    padding: '4px 8px',
                    cursor: 'pointer',
                    fontSize: '11px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    transition: 'all 0.2s ease'
                  }}
                  onClick={() => setAutoOptimize(!autoOptimize)}
                >
                  <Zap size={12} />
                </button>
              </Tooltip>
              <button
                onClick={() => setIsOpen(false)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#999',
                  cursor: 'pointer',
                  padding: '2px',
                  fontSize: '16px',
                  lineHeight: 1
                }}
              >
                ×
              </button>
            </div>
          </div>
      
          {/* Content */}
          <div style={{ marginTop: '12px' }}>
            {!metrics ? (
              <div style={{ 
                textAlign: 'center', 
                padding: '20px',
                color: '#666'
              }}>
                <Activity size={24} style={{ marginBottom: '8px', opacity: 0.5 }} />
                <p style={{ margin: 0, fontSize: '12px' }}>Loading metrics...</p>
              </div>
            ) : (
              <>
                {/* System Metrics */}
                <div style={{ marginBottom: '16px' }}>
                  <h5 style={{ margin: '0 0 8px 0', color: '#999', fontSize: '12px', fontWeight: '600' }}>
                    System Resources
                  </h5>
                  <div style={{ display: 'grid', gap: '12px' }}>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      <Cpu size={16} style={{ color: primaryColor }} />
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>CPU</span>
                          <span style={{ fontSize: '14px', fontWeight: '600' }}>{metrics.cpu.usage.toFixed(1)}%</span>
                    </div>
                    <div style={{ 
                      height: '4px', 
                      background: 'rgba(255, 255, 255, 0.1)', 
                      borderRadius: '2px',
                      marginTop: '4px',
                      overflow: 'hidden'
                    }}>
                      <div style={{ 
                        height: '100%',
                        width: `${metrics.cpu.usage}%`,
                        background: metrics.cpu.usage > 80 ? '#ef4444' : primaryColor,
                        transition: 'all 0.3s ease'
                      }} />
                    </div>
                  </div>
                </div>
            
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <HardDrive size={16} style={{ color: primaryColor }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>Memory</span>
                      <span style={{ fontSize: '14px', fontWeight: '600' }}>{metrics.memory.percentage.toFixed(1)}%</span>
                    </div>
                    <div style={{ 
                      height: '4px', 
                      background: 'rgba(255, 255, 255, 0.1)', 
                      borderRadius: '2px',
                      marginTop: '4px',
                      overflow: 'hidden'
                    }}>
                      <div style={{ 
                        height: '100%',
                        width: `${metrics.memory.percentage}%`,
                        background: metrics.memory.percentage > 80 ? '#ef4444' : primaryColor,
                        transition: 'all 0.3s ease'
                      }} />
                    </div>
                  </div>
                </div>
                
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <Activity size={16} style={{ color: primaryColor }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>FPS</span>
                      <span style={{ fontSize: '14px', fontWeight: '600' }}>
                        {wsMetrics ? wsMetrics.fps.toFixed(0) : metrics.performance.fps.toFixed(0)}
                      </span>
                    </div>
                    <div style={{ fontSize: '10px', color: '#999', marginTop: '2px' }}>
                      Frame: {metrics.performance.frameTime.toFixed(1)}ms
                    </div>
                  </div>
                </div>
          </div>
        </div>
        
            {/* WebSocket Real-time Metrics */}
            {wsMetrics && (
              <div style={{ marginBottom: '16px', paddingTop: '16px', borderTop: '1px solid #333' }}>
                <h5 style={{ 
                  margin: '0 0 8px 0', 
                  color: '#999', 
                  fontSize: '12px', 
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  Real-time Engine Metrics
                  <Wifi size={12} style={{ color: '#4ade80' }} />
                </h5>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
                  <div style={{ fontSize: '11px' }}>
                    <span style={{ color: '#666' }}>RSP:</span>
                    <span style={{ color: primaryColor, marginLeft: '6px', fontWeight: 'bold' }}>
                      {wsMetrics.rsp.toFixed(2)}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px' }}>
                    <span style={{ color: '#666' }}>Coherence:</span>
                    <span style={{ color: primaryColor, marginLeft: '6px', fontWeight: 'bold' }}>
                      {(wsMetrics.coherence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div style={{ fontSize: '11px' }}>
                    <span style={{ color: '#666' }}>Quantum Vol:</span>
                    <span style={{ color: '#a78bfa', marginLeft: '6px', fontWeight: 'bold' }}>
                      {wsMetrics.quantum_volume.toFixed(1)}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px' }}>
                    <span style={{ color: '#666' }}>Stability:</span>
                    <span style={{ 
                      color: wsMetrics.temporal_stability < 0.5 ? '#ef4444' : '#4ade80', 
                      marginLeft: '6px', 
                      fontWeight: 'bold' 
                    }}>
                      {(wsMetrics.temporal_stability * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
        
            {/* Worker Pool Stats */}
            <div style={{ marginBottom: '16px', paddingTop: '16px', borderTop: '1px solid #333' }}>
              <h5 style={{ margin: '0 0 8px 0', color: '#999', fontSize: '12px', fontWeight: '600' }}>
                Worker Pool
              </h5>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', fontSize: '11px', color: '#ccc' }}>
                <span>Workers: {workerStats.busyWorkers}/{workerStats.totalWorkers}</span>
                <span>Queue: {workerStats.queueLength}</span>
                <span>Processed: {workerStats.totalTasksProcessed}</span>
              </div>
            </div>
        
            {/* Memory Leaks */}
            {leakSummary.totalLeaks > 0 && (
              <div style={{ 
                marginBottom: '16px', 
                padding: '12px',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '4px'
              }}>
                <h5 style={{ 
                  margin: '0 0 8px 0', 
                  color: '#ef4444', 
                  fontSize: '12px', 
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  <AlertTriangle size={14} />
                  Memory Leaks Detected
                </h5>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '11px', color: '#fca5a5' }}>
                    {leakSummary.totalLeaks} leaks ({(leakSummary.totalSize / 1024).toFixed(1)} KB)
                  </span>
                  <button 
                    style={{
                      padding: '4px 8px',
                      background: '#ef4444',
                      border: 'none',
                      borderRadius: '4px',
                      color: 'white',
                      fontSize: '11px',
                      cursor: 'pointer'
                    }}
                    onClick={handleClearLeaks}
                  >
                    Clear Leaks
                  </button>
                </div>
              </div>
            )}
        
            {/* Actions */}
            <div style={{ 
              display: 'flex', 
              gap: '8px', 
              marginTop: '16px',
              paddingTop: '16px',
              borderTop: '1px solid #333'
            }}>
              <button 
                style={{
                  flex: 1,
                  padding: '6px 12px',
                  background: primaryColor,
                  border: 'none',
                  borderRadius: '4px',
                  color: '#000',
                  fontSize: '12px',
                  cursor: 'pointer',
                  transition: 'opacity 0.2s ease'
                }}
                onClick={handleOptimizeNow}
                onMouseEnter={(e) => e.currentTarget.style.opacity = '0.8'}
                onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
              >
                Optimize Now
              </button>
              <button 
                style={{
                  flex: 1,
                  padding: '6px 12px',
                  background: 'transparent',
                  border: `1px solid ${primaryColor}`,
                  borderRadius: '4px',
                  color: primaryColor,
                  fontSize: '12px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
                onClick={handleClearCache}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = primaryColor;
                  e.currentTarget.style.color = 'white';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.color = primaryColor;
                }}
              >
                Clear Cache
              </button>
            </div>
            </>
            )}
          </div>
        </div>
      )}
    </>
  );
};

