/**
 * Data Flow Monitor Component
 * Provides real-time monitoring and validation of data flow throughout the OSH Quantum Engine
 * Ensures all subsystems are properly connected and transmitting valid data
 */

import React, { useEffect, useState } from 'react';
import { OSHQuantumEngine } from '../engines/OSHQuantumEngine';
import { CheckCircle, AlertCircle, XCircle } from 'lucide-react';
import { useEngineAPIContext } from '../contexts/EngineAPIContext';

interface DataFlowStatus {
  subsystem: string;
  status: 'healthy' | 'warning' | 'error';
  message: string;
  lastUpdate: number;
  dataQuality: number; // 0-1 score
}

interface DataFlowMonitorProps {
  engine: OSHQuantumEngine | null;
  compact?: boolean;
}

export const DataFlowMonitor: React.FC<DataFlowMonitorProps> = ({ engine, compact = false }) => {
  const [flowStatuses, setFlowStatuses] = useState<DataFlowStatus[]>([]);
  const [overallHealth, setOverallHealth] = useState<'healthy' | 'warning' | 'error'>('healthy');
  
  // Get real-time metrics from websocket
  const { metrics, states, isConnected } = useEngineAPIContext();

  useEffect(() => {
    if (!engine) return;

    const checkDataFlows = () => {
      const statuses: DataFlowStatus[] = [];
      
      // Check WebSocket connection status
      statuses.push({
        subsystem: 'WebSocket Connection',
        status: isConnected ? 'healthy' : 'error',
        message: isConnected ? 'Real-time data flowing' : 'Disconnected from backend',
        lastUpdate: Date.now(),
        dataQuality: isConnected ? 1.0 : 0
      });

      // Check Memory Field Engine with real-time metrics
      try {
        const memoryField = engine.memoryFieldEngine.getField();
        const memoryMetrics = engine.memoryFieldEngine.getMetrics();
        
        const memoryStatus: DataFlowStatus = {
          subsystem: 'Memory Field',
          status: 'healthy',
          message: 'Operating normally',
          lastUpdate: Date.now(),
          dataQuality: metrics ? metrics.memory_field_coupling : 1.0
        };

        if (!memoryField || !memoryField.fragments) {
          memoryStatus.status = 'error';
          memoryStatus.message = 'No memory field data';
          memoryStatus.dataQuality = 0;
        } else if (metrics && metrics.memory_field_coupling < 0.5) {
          memoryStatus.status = 'warning';
          memoryStatus.message = `Low coupling strength: ${(metrics.memory_field_coupling * 100).toFixed(1)}%`;
        } else if (metrics) {
          memoryStatus.message = `Coupling: ${(metrics.memory_field_coupling * 100).toFixed(1)}%`;
        } else if (memoryField.fragments.length === 0) {
          memoryStatus.status = 'warning';
          memoryStatus.message = 'No memory fragments';
          memoryStatus.dataQuality = 0.5;
        } else if (!isFinite(memoryMetrics.coherence) || !isFinite(memoryMetrics.entropy)) {
          memoryStatus.status = 'warning';
          memoryStatus.message = 'Invalid metrics';
          memoryStatus.dataQuality = 0.7;
        } else {
          memoryStatus.message = `${memoryField.fragments.length} fragments, C=${memoryMetrics.coherence.toFixed(2)}`;
        }

        statuses.push(memoryStatus);
      } catch (error) {
        statuses.push({
          subsystem: 'Memory Field',
          status: 'error',
          message: 'Failed to access',
          lastUpdate: Date.now(),
          dataQuality: 0
        });
      }

      // Check RSP Engine with real-time metrics
      try {
        const rspState = engine.rspEngine.getState();
        
        const rspStatus: DataFlowStatus = {
          subsystem: 'RSP Engine',
          status: 'healthy',
          message: 'Operating normally',
          lastUpdate: Date.now(),
          dataQuality: 1.0
        };

        // Use real-time metrics from websocket if available
        if (metrics) {
          const { rsp, information, drsp_dt } = metrics;
          
          if (!isFinite(rsp) || rsp === 0) {
            rspStatus.status = 'error';
            rspStatus.message = 'Invalid RSP value';
            rspStatus.dataQuality = 0;
          } else if (Math.abs(drsp_dt) > 10) {
            rspStatus.status = 'warning';
            rspStatus.message = `RSP rapidly changing: ${drsp_dt.toFixed(2)}/s`;
            rspStatus.dataQuality = 0.6;
          } else {
            rspStatus.message = `RSP=${rsp.toFixed(2)}, I=${information.toFixed(2)}, dRSP/dt=${drsp_dt.toFixed(3)}`;
            rspStatus.dataQuality = Math.min(1.0, rsp / 100); // Quality based on RSP value
          }
        } else if (!rspState) {
          rspStatus.status = 'error';
          rspStatus.message = 'No RSP state';
          rspStatus.dataQuality = 0;
        } else if (!isFinite(rspState.rsp) || rspState.rsp === 0) {
          rspStatus.status = 'warning';
          rspStatus.message = 'Invalid RSP value';
          rspStatus.dataQuality = 0.5;
        } else if (rspState.isDiverging) {
          rspStatus.status = 'warning';
          rspStatus.message = 'RSP diverging!';
          rspStatus.dataQuality = 0.6;
        } else {
          rspStatus.message = `RSP=${rspState.rsp.toFixed(2)}, I=${rspState.information.toFixed(2)}`;
        }

        statuses.push(rspStatus);
      } catch (error) {
        statuses.push({
          subsystem: 'RSP Engine',
          status: 'error',
          message: 'Failed to access',
          lastUpdate: Date.now(),
          dataQuality: 0
        });
      }

      // Check Wavefunction Simulator with real-time metrics
      try {
        // Create synthetic wavefunction from memory field
        const memoryField = engine.memoryFieldEngine.getField();
        const wavefunctionState = memoryField ? {
          totalProbability: 1.0,
          coherence: metrics ? metrics.coherence : (memoryField.totalCoherence || 0),
          amplitude: memoryField.fragments || []
        } : null;
        
        const wfStatus: DataFlowStatus = {
          subsystem: 'Wavefunction',
          status: 'healthy',
          message: 'Operating normally',
          lastUpdate: Date.now(),
          dataQuality: metrics ? metrics.coherence : 1.0
        };

        if (!wavefunctionState || !wavefunctionState.amplitude) {
          wfStatus.status = 'error';
          wfStatus.message = 'No wavefunction data';
          wfStatus.dataQuality = 0;
        } else if (wavefunctionState.amplitude.length === 0) {
          wfStatus.status = 'error';
          wfStatus.message = 'Empty amplitude array';
          wfStatus.dataQuality = 0;
        } else if (wavefunctionState.totalProbability < 0.9 || wavefunctionState.totalProbability > 1.1) {
          wfStatus.status = 'warning';
          wfStatus.message = `Norm=${wavefunctionState.totalProbability.toFixed(3)}`;
          wfStatus.dataQuality = 0.7;
        } else if (metrics) {
          // Use real-time coherence from websocket
          wfStatus.message = `Coherence=${(metrics.coherence * 100).toFixed(1)}%, Entropy=${metrics.entropy.toFixed(3)}`;
          wfStatus.dataQuality = metrics.coherence;
          
          if (metrics.coherence < 0.5) {
            wfStatus.status = 'warning';
            wfStatus.message += ' - Low coherence!';
          }
        } else {
          const validAmps = wavefunctionState.amplitude.filter((a: any) => 
            a && isFinite(a.real || a.coherence) && isFinite(a.imag || a.phase)
          ).length;
          wfStatus.message = `${validAmps}/${wavefunctionState.amplitude.length} valid amplitudes`;
          wfStatus.dataQuality = validAmps / wavefunctionState.amplitude.length;
        }

        statuses.push(wfStatus);
      } catch (error) {
        statuses.push({
          subsystem: 'Wavefunction',
          status: 'error',
          message: 'Failed to access',
          lastUpdate: Date.now(),
          dataQuality: 0
        });
      }

      // Check Observer Engine with real-time metrics
      try {
        const observers = engine.observerEngine.getAllObservers();
        const observerCount = metrics ? metrics.observer_count : observers.length;
        const observerInfluence = metrics ? metrics.observer_influence : 1.0;
        
        const obsStatus: DataFlowStatus = {
          subsystem: 'Observers',
          status: 'healthy',
          message: `${observerCount} active`,
          lastUpdate: Date.now(),
          dataQuality: observerInfluence
        };

        if (observerCount === 0) {
          obsStatus.status = 'warning';
          obsStatus.message = 'No observers';
        } else if (metrics) {
          obsStatus.message = `${observerCount} observers, Influence=${(observerInfluence * 100).toFixed(1)}%`;
          if (observerInfluence < 0.3) {
            obsStatus.status = 'warning';
            obsStatus.message += ' - Low influence!';
          }
        }

        statuses.push(obsStatus);
      } catch (error) {
        statuses.push({
          subsystem: 'Observers',
          status: 'error',
          message: 'Failed to access',
          lastUpdate: Date.now(),
          dataQuality: 0
        });
      }

      // Check Error Reduction Platform with real-time metrics
      try {
        const errorMetrics = engine.errorReductionPlatform.getMetrics();
        const errorRate = metrics ? metrics.error : (errorMetrics ? errorMetrics.effectiveErrorRate : 0);
        
        const errorStatus: DataFlowStatus = {
          subsystem: 'Error Platform',
          status: 'healthy',
          message: 'Operating normally',
          lastUpdate: Date.now(),
          dataQuality: 1.0 - Math.min(1.0, errorRate)
        };

        if (!errorMetrics) {
          errorStatus.status = 'warning';
          errorStatus.message = 'No metrics available';
          errorStatus.dataQuality = 0.5;
        } else if (errorMetrics.effectiveErrorRate > 0.1) {
          errorStatus.status = 'warning';
          errorStatus.message = `High error rate: ${(errorMetrics.effectiveErrorRate * 100).toFixed(1)}%`;
          errorStatus.dataQuality = 1 - errorMetrics.effectiveErrorRate;
        } else {
          errorStatus.message = `Error rate: ${(errorMetrics.effectiveErrorRate * 100).toFixed(2)}%`;
        }

        statuses.push(errorStatus);
      } catch (error) {
        statuses.push({
          subsystem: 'Error Platform',
          status: 'warning',
          message: 'Not initialized',
          lastUpdate: Date.now(),
          dataQuality: 0.5
        });
      }

      // Update overall health
      const errors = statuses.filter(s => s.status === 'error').length;
      const warnings = statuses.filter(s => s.status === 'warning').length;
      
      if (errors > 0) {
        setOverallHealth('error');
      } else if (warnings > 2) {
        setOverallHealth('warning');
      } else {
        setOverallHealth('healthy');
      }

      setFlowStatuses(statuses);
    };

    // Initial check
    checkDataFlows();

    // Set up interval
    const interval = setInterval(checkDataFlows, 1000);

    return () => clearInterval(interval);
  }, [engine, metrics, states, isConnected]); // Re-run when websocket data updates

  const getStatusIcon = (status: 'healthy' | 'warning' | 'error') => {
    switch (status) {
      case 'healthy':
        return <CheckCircle size={16} className="text-green-500" />;
      case 'warning':
        return <AlertCircle size={16} className="text-yellow-500" />;
      case 'error':
        return <XCircle size={16} className="text-red-500" />;
    }
  };

  const getHealthColor = () => {
    switch (overallHealth) {
      case 'healthy': return 'bg-green-500';
      case 'warning': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
    }
  };

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${getHealthColor()} animate-pulse`} />
        <span className="text-xs text-gray-400">
          Data Flow: {overallHealth}
        </span>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Data Flow Monitor</h3>
        <div className={`px-2 py-1 rounded text-xs font-medium ${
          overallHealth === 'healthy' ? 'bg-green-900 text-green-300' :
          overallHealth === 'warning' ? 'bg-yellow-900 text-yellow-300' :
          'bg-red-900 text-red-300'
        }`}>
          {overallHealth.toUpperCase()}
        </div>
      </div>
      
      <div className="space-y-2">
        {flowStatuses.map((status, index) => (
          <div key={index} className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              {getStatusIcon(status.status)}
              <span className="text-gray-400">{status.subsystem}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-gray-500">{status.message}</span>
              <div className="w-12 h-1 bg-gray-800 rounded-full overflow-hidden">
                <div 
                  className={`h-full transition-all duration-300 ${
                    status.dataQuality > 0.8 ? 'bg-green-500' :
                    status.dataQuality > 0.5 ? 'bg-yellow-500' :
                    'bg-red-500'
                  }`}
                  style={{ width: `${status.dataQuality * 100}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {overallHealth !== 'healthy' && (
        <div className="mt-3 pt-3 border-t border-gray-800">
          <p className="text-xs text-gray-500">
            {overallHealth === 'error' 
              ? 'Critical data flow issues detected. Check engine initialization.'
              : 'Some subsystems are experiencing issues. Monitor for stability.'}
          </p>
        </div>
      )}
      
      {/* Real-time metrics summary */}
      {metrics && (
        <div className="mt-3 pt-3 border-t border-gray-800">
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="text-center">
              <div className="text-gray-500">FPS</div>
              <div className={`font-mono ${metrics.fps < 30 ? 'text-yellow-400' : 'text-green-400'}`}>
                {metrics.fps.toFixed(0)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-500">Quantum Vol</div>
              <div className="font-mono text-blue-400">
                {metrics.quantum_volume.toFixed(1)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-500">Stability</div>
              <div className={`font-mono ${metrics.temporal_stability < 0.5 ? 'text-red-400' : 'text-green-400'}`}>
                {(metrics.temporal_stability * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataFlowMonitor;