/**
 * RSP Dashboard - Production-Ready Implementation
 * 
 * Real-time monitoring and analysis of Recursive Simulation Potential (RSP)
 * Tracks RSP(t) = I(t)·C(t)/E(t) with comprehensive metrics and visualizations
 * 
 * Features:
 * - Live time series charts for RSP, Information, Coherence, and Entropy
 * - Differential analysis with derivatives and rate of change
 * - Divergence condition monitoring with alerts
 * - Memory attractor visualization in phase space
 * - Entropy plateau detection and analysis
 * - Critical threshold monitoring
 * - Export functionality for data analysis
 */

import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { RSPEngine, RSPState, MemoryAttractor, DivergenceCondition, EntropyPlateau } from '../../engines/RSPEngine';
import { Complex } from '../../utils/complex';
import { useEngineAPIContext } from '../../contexts/EngineAPIContext';
import { useUniverse } from '../../contexts/UniverseContext';
import { Line, Bar, Scatter, Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler,
  ChartOptions
} from 'chart.js';
import { Tooltip } from '../ui/Tooltip';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  AlertCircle, 
  Download, 
  RefreshCw,
  Info,
  Zap,
  Brain,
  Waves,
  GitBranch,
  BarChart3,
  Clock,
  AlertTriangle,
  CheckCircle
} from 'lucide-react';
import {
  RSP_PROTO_CONSCIOUSNESS,
  RSP_ACTIVE_CONSCIOUSNESS,
  RSP_ADVANCED_CONSCIOUSNESS
} from '../../config/physicsConstants';
import { generateColorPalette, adjustLightness, mixColors, hexToRgb } from '../../utils/colorUtils';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  Legend,
  Filler
);

interface RSPDashboardProps {
  rspEngine?: RSPEngine;
  primaryColor?: string;
  isActive?: boolean;
  isSimulating?: boolean;
}

// Constants for thresholds and limits
const CRITICAL_RSP_THRESHOLD = RSP_ACTIVE_CONSCIOUSNESS; // 1e10 from MATHEMATICAL_SUPPLEMENT.md
const WARNING_RSP_THRESHOLD = RSP_PROTO_CONSCIOUSNESS; // 1e3 from MATHEMATICAL_SUPPLEMENT.md
const ENTROPY_PLATEAU_THRESHOLD = 0.01;
const MAX_HISTORY_POINTS = 1000; // Increased to store more history
const UPDATE_INTERVAL_MS = 100;

/**
 * Main RSP Dashboard Component
 */
export const RSPDashboard: React.FC<RSPDashboardProps> = ({ 
  rspEngine: externalEngine, 
  primaryColor = '#4fc3f7', 
  isActive,
  isSimulating: propIsSimulating
}) => {
  // Generate color palette from primary color
  const palette = useMemo(() => generateColorPalette(primaryColor), [primaryColor]);
  
  // Get universe context - handle case where context might not be available
  let universeContext;
  let contextIsSimulating = false;
  try {
    universeContext = useUniverse();
    contextIsSimulating = universeContext?.isSimulating || false;
  } catch (error) {
    // Context not available, use default values
    // Running without UniverseContext
  }
  
  // Use prop isSimulating if available, otherwise fall back to context
  const isSimulating = propIsSimulating ?? contextIsSimulating ?? false;
  const isActiveState = isActive ?? isSimulating;
  
  // Create or use RSP engine
  const [internalEngine] = useState(() => new RSPEngine());
  const rspEngine = externalEngine || internalEngine;
  
  // State management
  const [rspHistory, setRspHistory] = useState<RSPState[]>([]);
  const [divergenceConditions, setDivergenceConditions] = useState<DivergenceCondition[]>([]);
  const [entropyPlateaus, setEntropyPlateaus] = useState<EntropyPlateau[]>([]);
  const [memoryAttractors, setMemoryAttractors] = useState<MemoryAttractor[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1m' | '5m' | '15m' | 'all'>('5m'); // Default to 5m for more data
  const [showDerivatives, setShowDerivatives] = useState(true); // Show all lines by default
  const [autoScale, setAutoScale] = useState(true);
  const [isInitialized, setIsInitialized] = useState(false);
  const [criticalAlert, setCriticalAlert] = useState(false);
  
  // Refs for performance
  const updateIntervalRef = useRef<NodeJS.Timeout>();
  const frameCountRef = useRef(0);
  const lastUpdateTimeRef = useRef(Date.now());
  
  // API connection
  const { metrics, isConnected } = useEngineAPIContext();
  
  
  /**
   * Initialize RSP engine when simulation starts
   */
  useEffect(() => {
    if (isSimulating && !isInitialized && rspEngine) {
      // RSP engine ready for real-time data
      setIsInitialized(true);
    } else if (!isSimulating && isInitialized) {
      setIsInitialized(false);
      // Don't clear history - preserve data across mode changes
      // Only clear if explicitly requested or on component unmount
    }
  }, [isSimulating, isInitialized, rspEngine]);
  
  /**
   * Update dashboard data from engine
   */
  const updateDashboard = useCallback(() => {
    if (!rspEngine || !isActiveState) return;
    
    const currentState = rspEngine.getCurrentState();
    
    // Always update if we have valid data (including when value is 0)
    if (currentState) {
      // Use metrics RSP if available to ensure we have the latest data
      if (metrics?.rsp !== undefined) {
        currentState.value = metrics.rsp;
        currentState.rsp = metrics.rsp;
        // Use nullish coalescing to preserve zero values
        currentState.information = metrics.information ?? 1;
        currentState.coherence = metrics.coherence ?? 0.5;
        currentState.entropy = metrics.entropy ?? 0.5;
      }
      
      // Update history without filtering in the state setter
      setRspHistory(prev => {
        const newHistory = [...prev, currentState];
        // Simply maintain the maximum history length
        return newHistory.slice(-MAX_HISTORY_POINTS);
      });
      
      // Check for critical conditions
      if (currentState.value > CRITICAL_RSP_THRESHOLD) {
        setCriticalAlert(true);
      } else if (currentState.value < WARNING_RSP_THRESHOLD) {
        setCriticalAlert(false);
      }
      
      // Update other analysis data
      const divergence = rspEngine.getDivergenceConditions();
      const plateaus = rspEngine.getEntropyPlateaus();
      const attractors = rspEngine.getMemoryAttractors();
      
      setDivergenceConditions(divergence.slice(-10));
      setEntropyPlateaus(plateaus.slice(-10));
      setMemoryAttractors(attractors.slice(-20));
    }
    
    frameCountRef.current++;
  }, [rspEngine, isActiveState, metrics]);
  
  /**
   * Main update loop
   */
  useEffect(() => {
    if (!isActiveState || !isSimulating) {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
      return;
    }
    
    // Initial update
    updateDashboard();
    
    // Set up interval
    updateIntervalRef.current = setInterval(updateDashboard, UPDATE_INTERVAL_MS);
    
    return () => {
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
    };
  }, [isActiveState, isSimulating, updateDashboard]);
  
  /**
   * Sync with backend metrics if available
   */
  useEffect(() => {
    if (metrics && rspEngine && isConnected) {
      // Transform backend metrics to RSPEngine format
      const rspMetrics = {
        rsp_value: metrics.rsp ?? 0,
        information: metrics.information ?? 0,  // Changed default to 0 to preserve zero values
        coherence: metrics.coherence ?? 0.5,
        entropy: metrics.entropy ?? 0.5,
        is_diverging: metrics.rsp > RSP_ACTIVE_CONSCIOUSNESS,
        // Use derivatives from backend
        drsp_dt: metrics.drsp_dt ?? 0,
        di_dt: metrics.di_dt ?? 0,
        dc_dt: metrics.dc_dt ?? 0,
        de_dt: metrics.de_dt ?? 0,
        acceleration: metrics.acceleration ?? 0
      };
      
      // Update engine with backend metrics
      rspEngine.updateFromBackend(rspMetrics);
      
      // Updated with backend metrics including derivatives
    }
  }, [metrics, rspEngine, isConnected]);
  
  /**
   * Filter data by time range
   */
  const filterByTimeRange = (data: RSPState[], range: string): RSPState[] => {
    if (range === 'all') return data;
    
    const now = Date.now();
    const ranges: Record<string, number> = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000
    };
    
    const cutoff = now - (ranges[range] || 0);
    return data.filter(d => d.timestamp >= cutoff);
  };
  
  // Filter history once for all uses
  const filteredHistory = useMemo(() => 
    filterByTimeRange(rspHistory, selectedTimeRange),
    [rspHistory, selectedTimeRange]
  );

  /**
   * Calculate statistics for display
   */
  const statistics = useMemo(() => {
    // Use metrics directly if history is empty but we have current data
    if (filteredHistory.length === 0 && metrics && metrics.rsp !== undefined) {
      const current = metrics.rsp || 0;
      return {
        current,
        average: current,
        max: current,
        min: current,
        trend: 'stable' as const,
        volatility: 0,
        divergenceRisk: Math.min(1, Math.max(0, 
          (current / WARNING_RSP_THRESHOLD - 1) / 9
        ))
      };
    }
    
    if (filteredHistory.length === 0) {
      return {
        current: 0,
        average: 0,
        max: 0,
        min: 0,
        trend: 'stable' as const,
        volatility: 0,
        divergenceRisk: 0
      };
    }
    
    const values = filteredHistory.map(h => h.value);
    const current = values[values.length - 1];
    const average = values.reduce((a, b) => a + b, 0) / values.length;
    const max = Math.max(...values);
    const min = Math.min(...values);
    
    // Calculate trend
    const recentValues = values.slice(-10);
    const trend: 'increasing' | 'decreasing' | 'stable' = recentValues.length > 1 && 
      recentValues[recentValues.length - 1] > recentValues[0] * 1.1 ? 'increasing' :
      recentValues[recentValues.length - 1] < recentValues[0] * 0.9 ? 'decreasing' : 
      'stable';
    
    // Calculate volatility (protect against division by zero and extreme values)
    const variance = values.reduce((acc, val) => 
      acc + Math.pow(val - average, 2), 0) / values.length;
    // Coefficient of variation capped at reasonable values
    const rawVolatility = average > 0.001 ? Math.sqrt(variance) / average : 0;
    const volatility = Math.min(rawVolatility, 10); // Cap at 1000%
    
    // Calculate divergence risk
    const divergenceRisk = Math.min(1, Math.max(0, 
      (current / WARNING_RSP_THRESHOLD - 1) / 9
    ));
    
    return {
      current,
      average,
      max,
      min,
      trend,
      volatility,
      divergenceRisk
    };
  }, [filteredHistory, metrics]);
  
  /**
   * Export data
   */
  const exportData = useCallback(() => {
    const data = {
      timestamp: Date.now(),
      statistics,
      history: rspHistory,
      divergenceConditions,
      entropyPlateaus,
      memoryAttractors,
      settings: {
        timeRange: selectedTimeRange,
        showDerivatives,
        autoScale
      }
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], 
      { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rsp_analysis_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [statistics, rspHistory, divergenceConditions, entropyPlateaus, 
      memoryAttractors, selectedTimeRange, showDerivatives, autoScale]);
  
  // If not active, show minimal state
  if (!isActiveState) {
    return (
      <div style={{
        background: palette.bgHover,
        border: `1px solid ${palette.borderPrimary}`,
        borderRadius: '8px',
        padding: '20px',
        textAlign: 'center',
        opacity: 0.5
      }}>
        <Activity size={32} style={{ marginBottom: '10px', opacity: 0.3 }} />
        <p style={{ margin: 0, fontSize: '14px', color: palette.textSecondary }}>RSP Dashboard Inactive</p>
        <p style={{ margin: '5px 0 0 0', fontSize: '12px', color: palette.textTertiary }}>
          Start the simulation to begin analysis
        </p>
      </div>
    );
  }
  
  return (
    <div style={{ 
      width: '100%',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden'
    }}>
      {/* Combined header with badges, stats, and controls */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '8px',
        gap: '16px',
        flexWrap: 'wrap',
        flexShrink: 0
      }}>
        {/* Left: Status badges */}
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '8px',
          flexShrink: 0
        }}>
          {isConnected && (
            <span style={{
              background: `${primaryColor}30`,
              color: primaryColor,
              padding: '2px 6px',
              borderRadius: '3px',
              fontSize: '10px',
              fontWeight: '500'
            }}>
              LIVE
            </span>
          )}
          {criticalAlert && (
            <span style={{
              background: palette.error + '30',
              color: palette.error,
              padding: '2px 6px',
              borderRadius: '3px',
              fontSize: '10px',
              fontWeight: '500',
              animation: 'pulse 1s infinite'
            }}>
              CRITICAL
            </span>
          )}
        </div>
        
        {/* Middle: Compact stats */}
        <div style={{ 
          display: 'flex', 
          gap: '12px',
          alignItems: 'center',
          flex: '1 1 auto',
          justifyContent: 'center',
          flexWrap: 'wrap',
          minWidth: 0
        }}>
          <CompactStat
            label="RSP"
            value={formatRSPValue(statistics.current)}
            color={primaryColor}
            trend={statistics.trend}
          />
          <CompactStat
            label="Risk"
            value={`${Math.min(100, (statistics.divergenceRisk || 0) * 100).toFixed(0)}%`}
            color={statistics.divergenceRisk > 0.7 ? '#ff4444' : '#ffa726'}
          />
          <CompactStat
            label="Observers"
            value={`${metrics.observer_count || 0}`}
            color="#00bcd4"
          />
          <CompactStat
            label="Effect"
            value={`${((metrics.observer_influence || 0) * (metrics.coherence || 0) * 100).toFixed(1)}%`}
            color="#9c27b0"
          />
        </div>
        
        {/* Right: Controls */}
        <div style={{ 
          display: 'flex', 
          gap: '4px', 
          alignItems: 'center',
          flexShrink: 0
        }}>
          {/* Time range selector - compact dropdown */}
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value as any)}
            style={{
              height: '24px',
              minWidth: '50px',
              padding: '0 18px 0 6px',
              background: 'rgba(0, 0, 0, 0.5)',
              border: `1px solid ${primaryColor}40`,
              borderRadius: '3px',
              color: primaryColor,
              cursor: 'pointer',
              fontSize: '11px',
              outline: 'none',
              appearance: 'none',
              WebkitAppearance: 'none',
              MozAppearance: 'none',
              backgroundImage: `url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='${encodeURIComponent(primaryColor)}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e")`,
              backgroundRepeat: 'no-repeat',
              backgroundPosition: 'right 4px center',
              backgroundSize: '12px'
            }}
          >
            <option value="1m" style={{ background: 'rgba(0, 0, 0, 0.9)', color: primaryColor }}>1m</option>
            <option value="5m" style={{ background: 'rgba(0, 0, 0, 0.9)', color: primaryColor }}>5m</option>
            <option value="15m" style={{ background: 'rgba(0, 0, 0, 0.9)', color: primaryColor }}>15m</option>
            <option value="all" style={{ background: 'rgba(0, 0, 0, 0.9)', color: primaryColor }}>All</option>
          </select>
          
          {/* Show all lines toggle */}
          <Tooltip content="Show all metrics">
            <button
              onClick={() => setShowDerivatives(!showDerivatives)}
              style={{
                height: '24px',
                minWidth: '40px',
                padding: '0 8px',
                background: showDerivatives ? `${primaryColor}20` : 'transparent',
                border: `1px solid ${primaryColor}40`,
                borderRadius: '3px',
                color: showDerivatives ? primaryColor : '#666',
                cursor: 'pointer',
                fontSize: '11px',
                transition: 'all 0.2s',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              All
            </button>
          </Tooltip>
          
          {/* Auto scale toggle */}
          <Tooltip content="Auto scale">
            <button
              onClick={() => setAutoScale(!autoScale)}
              style={{
                height: '24px',
                minWidth: '40px',
                padding: '0 8px',
                background: autoScale ? `${primaryColor}20` : 'transparent',
                border: `1px solid ${primaryColor}40`,
                borderRadius: '3px',
                color: autoScale ? primaryColor : '#666',
                cursor: 'pointer',
                fontSize: '11px',
                transition: 'all 0.2s',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              Auto
            </button>
          </Tooltip>
          
          {/* Export button */}
          <Tooltip content="Export data">
            <button
              onClick={exportData}
              style={{
                height: '24px',
                width: '24px',
                padding: '0',
                background: 'transparent',
                border: `1px solid ${primaryColor}40`,
                borderRadius: '3px',
                color: primaryColor,
                cursor: 'pointer',
                fontSize: '11px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <Download size={12} />
            </button>
          </Tooltip>
        </div>
      </div>
      
      {/* New layout with 70/30 split */}
      <div style={{
        display: 'flex',
        gap: '8px',
        flex: '1 1 auto',
        minHeight: 0,
        overflow: 'hidden'
      }}>
        {/* Left side - 70% width */}
        <div style={{
          flex: '0 0 70%',
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gridTemplateRows: '35% 32.5% 32.5%',
          gap: '6px',
          minWidth: 0,
          overflow: 'hidden'
        }}>
          {/* Time Series Analysis - spans both columns */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: `1px solid ${primaryColor}20`,
            borderRadius: '6px',
            padding: '8px',
            gridColumn: 'span 2',
            height: '100%'
          }}>
            <h3 style={{ 
              margin: '0 0 8px 0', 
              fontSize: '12px', 
              fontWeight: '600',
              color: '#ffffff'
            }}>
              Time Series Analysis
            </h3>
            <div style={{ height: 'calc(100% - 28px)' }}>
              <TimeSeriesChart
                data={{
                  timestamps: filteredHistory.map(h => h.timestamp),
                  rspValues: filteredHistory.map(h => h.value),
                  information: filteredHistory.map(h => h.information),
                  coherence: filteredHistory.map(h => h.coherence),
                  entropy: filteredHistory.map(h => h.entropy)
                }}
                primaryColor={primaryColor}
                showDerivatives={showDerivatives}
                autoScale={autoScale}
              />
            </div>
          </div>
          
          {/* Differential Analysis */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: `1px solid ${primaryColor}20`,
            borderRadius: '6px',
            padding: '8px',
            overflow: 'hidden'
          }}>
            <DifferentialAnalysis
              currentState={rspHistory[rspHistory.length - 1] || null}
              primaryColor={primaryColor}
            />
          </div>
          
          {/* Divergence Conditions */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: `1px solid ${primaryColor}20`,
            borderRadius: '6px',
            padding: '8px',
            overflow: 'hidden'
          }}>
            <DivergenceIndicator
              conditions={divergenceConditions}
              primaryColor={primaryColor}
            />
          </div>
          
          {/* Entropy Plateaus - spans both columns */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: `1px solid ${primaryColor}20`,
            borderRadius: '6px',
            padding: '8px',
            gridColumn: 'span 2',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <EntropyPlateauAnalysis
              plateaus={entropyPlateaus}
              primaryColor={primaryColor}
            />
          </div>
        </div>
        
        {/* Right side - 30% width for Memory Attractors */}
        <div style={{
          flex: '0 0 30%',
          background: 'rgba(255, 255, 255, 0.02)',
          border: `1px solid ${primaryColor}20`,
          borderRadius: '6px',
          padding: '8px',
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0
        }}>
          <h3 style={{ 
            margin: '0 0 8px 0', 
            fontSize: '12px', 
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            flexShrink: 0
          }}>
            <Brain size={12} />
            Memory Attractors
          </h3>
          <div style={{ 
            flex: 1, 
            minHeight: 0,
            position: 'relative'
          }}>
            <AttractorMap
              attractors={memoryAttractors}
              primaryColor={primaryColor}
            />
          </div>
        </div>
      </div>
      
      {/* CSS for animations and hover effects */}
      <style>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
        
        .stat-card {
          transition: all 0.2s ease;
        }
        
        .stat-card:hover {
          background: rgba(255, 255, 255, 0.05) !important;
          border-color: rgba(255, 255, 255, 0.2) !important;
        }
        
        .diff-row {
          transition: background 0.2s ease;
        }
        
        .diff-row:hover {
          background: rgba(255, 255, 255, 0.05) !important;
        }
      `}</style>
    </div>
  );
};

/**
 * Compact stat for header display
 */
const CompactStat: React.FC<{
  label: string;
  value: string;
  color: string;
  trend?: 'increasing' | 'decreasing' | 'stable';
}> = ({ label, value, color, trend }) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '11px'
  }}>
    <span style={{ opacity: 0.6 }}>{label}:</span>
    <span style={{ 
      fontWeight: '600', 
      color,
      display: 'flex',
      alignItems: 'center',
      gap: '2px'
    }}>
      {value}
      {trend && (
        <span style={{ fontSize: '9px', opacity: 0.8 }}>
          {trend === 'increasing' ? <TrendingUp size={10} /> : 
           trend === 'decreasing' ? <TrendingDown size={10} /> : 
           null}
        </span>
      )}
    </span>
  </div>
);

/**
 * Sleek metric card component with horizontal layout
 */
const StatCard: React.FC<{
  title: string;
  value: string;
  icon: React.ReactNode;
  color: string;
  trend?: 'increasing' | 'decreasing' | 'stable';
  tooltip: string;
}> = ({ title, value, icon, color, trend, tooltip }) => (
  <Tooltip content={tooltip}>
    <div style={{
      background: 'rgba(255, 255, 255, 0.02)',
      border: `1px solid ${color}20`,
      borderRadius: '4px',
      padding: '8px 12px',
      cursor: 'pointer',
      transition: 'all 0.2s',
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      minWidth: '140px'
    }}
    className="stat-card"
    data-hover-color={color}>
      <span style={{ color, opacity: 0.8, flexShrink: 0 }}>{icon}</span>
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column',
        gap: '2px',
        minWidth: 0,
        flex: 1
      }}>
        <div style={{ 
          fontSize: '10px', 
          opacity: 0.6,
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis'
        }}>
          {title}
        </div>
        <div style={{ 
          fontSize: '14px', 
          fontWeight: '600', 
          color,
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}>
          {value}
          {trend && (
            <span style={{ fontSize: '10px', opacity: 0.8 }}>
              {trend === 'increasing' ? <TrendingUp size={12} /> : 
               trend === 'decreasing' ? <TrendingDown size={12} /> : 
               null}
            </span>
          )}
        </div>
      </div>
    </div>
  </Tooltip>
);

/**
 * Time series chart component with enhanced features
 */
const TimeSeriesChart: React.FC<{
  data: {
    timestamps: number[];
    rspValues: number[];
    information: number[];
    coherence: number[];
    entropy: number[];
  };
  primaryColor: string;
  showDerivatives: boolean;
  autoScale: boolean;
}> = ({ data, primaryColor, showDerivatives, autoScale }) => {
  const chartData = {
    labels: data.timestamps.map(t => {
      const date = new Date(t);
      return `${date.getHours().toString().padStart(2, '0')}:${
        date.getMinutes().toString().padStart(2, '0')}:${
        date.getSeconds().toString().padStart(2, '0')}`;
    }),
    datasets: [
      {
        label: 'RSP(t)',
        data: data.rspValues,
        borderColor: primaryColor,
        backgroundColor: primaryColor + '20',
        borderWidth: 2,
        tension: 0.4,
        yAxisID: 'y',
        pointRadius: 0,
        pointHoverRadius: 4
      },
      {
        label: 'I(t)',
        data: data.information,
        borderColor: '#ff6b9d',
        backgroundColor: '#ff6b9d20',
        borderWidth: 1.5,
        tension: 0.4,
        yAxisID: 'y1',
        pointRadius: 0,
        pointHoverRadius: 4
      },
      {
        label: 'C(t)',
        data: data.coherence,
        borderColor: '#45b7d1',
        backgroundColor: '#45b7d120',
        borderWidth: 1.5,
        tension: 0.4,
        yAxisID: 'y2',
        pointRadius: 0,
        pointHoverRadius: 4
      },
      {
        label: 'E(t)',
        data: data.entropy,
        borderColor: '#96ceb4',
        backgroundColor: '#96ceb420',
        borderWidth: 1.5,
        tension: 0.4,
        yAxisID: 'y2',
        pointRadius: 0,
        pointHoverRadius: 4
      }
    ]
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#fff',
          usePointStyle: true,
          padding: 15,
          font: {
            size: 10
          }
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        borderColor: primaryColor,
        borderWidth: 1,
        titleFont: {
          size: 11
        },
        bodyFont: {
          size: 10
        },
        callbacks: {
          label: (context) => {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            if (label === 'RSP(t)') {
              return `${label}: ${formatRSPValue(value)}`;
            }
            return `${label}: ${(value || 0).toFixed(4)}`;
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: false
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.05)'
        },
        ticks: {
          color: '#666',
          maxRotation: 45,
          minRotation: 45,
          font: {
            size: 9
          }
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'RSP',
          color: primaryColor,
          font: {
            size: 10
          }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.05)'
        },
        ticks: {
          color: primaryColor,
          font: {
            size: 9
          },
          callback: (value) => formatRSPValue(value as number)
        },
        ...(autoScale ? {} : {
          min: 0,
          max: Math.max(...data.rspValues) * 1.2
        })
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Information',
          color: '#ff6b9d',
          font: {
            size: 10
          }
        },
        grid: {
          drawOnChartArea: false
        },
        ticks: {
          color: '#ff6b9d',
          font: {
            size: 9
          }
        },
        min: 0
      },
      y2: {
        type: 'linear',
        display: true,
        position: 'right',
        offset: true,
        title: {
          display: true,
          text: 'C(t) / E(t)',
          color: '#888',
          font: {
            size: 10
          }
        },
        grid: {
          drawOnChartArea: false
        },
        ticks: {
          color: '#888',
          font: {
            size: 9
          }
        },
        min: 0,
        max: 1
      }
    }
  };

  return (
    <div style={{ height: '100%' }}>
      <Line data={chartData} options={options} />
    </div>
  );
};

/**
 * Differential analysis component
 */
const DifferentialAnalysis: React.FC<{
  currentState: RSPState | null;
  primaryColor: string;
}> = ({ currentState, primaryColor }) => {
  if (!currentState) {
    return (
      <div style={{ textAlign: 'center', padding: '20px', opacity: 0.5 }}>
        <p style={{ fontSize: '12px' }}>No data available</p>
      </div>
    );
  }

  const derivatives = currentState.derivatives || {
    dRSP_dt: 0,
    dI_dt: 0,
    dC_dt: 0,
    dE_dt: 0,
    acceleration: 0
  };

  return (
    <div>
      <h3 style={{ 
        margin: '0 0 8px 0', 
        fontSize: '12px', 
        fontWeight: '600',
        display: 'flex',
        alignItems: 'center',
        gap: '4px'
      }}>
        <TrendingUp size={12} />
        Differential Analysis
      </h3>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <DiffRow 
          label="dRSP/dt"
          value={derivatives.dRSP_dt}
          color={primaryColor}
          tooltip="Rate of change of RSP"
        />
        <DiffRow 
          label="dI/dt"
          value={derivatives.dI_dt}
          color="#ff6b9d"
          tooltip="Information flow rate"
        />
        <DiffRow 
          label="dC/dt"
          value={derivatives.dC_dt}
          color="#45b7d1"
          tooltip="Coherence change rate"
        />
        <DiffRow 
          label="dE/dt"
          value={derivatives.dE_dt}
          color="#96ceb4"
          tooltip="Entropy production rate"
        />
        <div style={{
          marginTop: '8px',
          padding: '8px',
          background: 'rgba(255, 255, 255, 0.02)',
          borderRadius: '4px',
          border: `1px solid ${primaryColor}20`
        }}>
          <div style={{ fontSize: '10px', opacity: 0.6, marginBottom: '4px' }}>
            Acceleration
          </div>
          <div style={{ 
            fontSize: '14px', 
            fontWeight: '600', 
            color: derivatives.acceleration > 0 ? '#4fc3f7' : '#ff6b9d' 
          }}>
            {(derivatives.acceleration || 0).toFixed(6)} /s²
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Differential row component
 */
const DiffRow: React.FC<{
  label: string;
  value: number;
  color: string;
  tooltip: string;
}> = ({ label, value, color, tooltip }) => (
  <Tooltip content={tooltip}>
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '6px 8px',
      background: 'rgba(255, 255, 255, 0.02)',
      borderRadius: '3px',
      cursor: 'pointer',
      transition: 'all 0.2s'
    }}
    className="diff-row">
      <span style={{ fontSize: '11px', opacity: 0.7 }}>{label}</span>
      <span style={{ 
        fontSize: '12px', 
        fontWeight: '600', 
        color,
        display: 'flex',
        alignItems: 'center',
        gap: '4px'
      }}>
        {value > 0 && '+'}
        {(value || 0).toFixed(6)}
        {value !== 0 && (
          <span style={{ fontSize: '10px' }}>
            {value > 0 ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
          </span>
        )}
      </span>
    </div>
  </Tooltip>
);

/**
 * Divergence indicator component
 */
const DivergenceIndicator: React.FC<{
  conditions: DivergenceCondition[];
  primaryColor: string;
}> = ({ conditions, primaryColor }) => {
  const latestCondition = conditions[conditions.length - 1];
  // Use the risk value from the condition if available, otherwise calculate
  const risk = latestCondition ? 
    (latestCondition.risk !== undefined ? latestCondition.risk : Math.min(1, latestCondition.strength / 2)) : 0;
  
  return (
    <div>
      <h3 style={{ 
        margin: '0 0 8px 0', 
        fontSize: '12px', 
        fontWeight: '600',
        display: 'flex',
        alignItems: 'center',
        gap: '4px'
      }}>
        <AlertCircle size={12} />
        Divergence Monitor
      </h3>
      
      <div style={{
        padding: '8px',
        background: `linear-gradient(90deg, 
          rgba(76, 195, 247, 0.1) 0%, 
          rgba(255, 167, 38, 0.1) 50%, 
          rgba(255, 68, 68, 0.1) 100%)`,
        borderRadius: '4px',
        marginBottom: '8px'
      }}>
        <div style={{ fontSize: '10px', opacity: 0.6, marginBottom: '4px' }}>
          Current Risk Level
        </div>
        <div style={{ 
          fontSize: '24px', 
          fontWeight: 'bold',
          color: risk > 0.7 ? '#ff4444' : risk > 0.3 ? '#ffa726' : '#4fc3f7'
        }}>
          {((risk || 0) * 100).toFixed(1)}%
        </div>
        <div style={{
          marginTop: '8px',
          height: '4px',
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '2px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${risk * 100}%`,
            height: '100%',
            background: risk > 0.7 ? '#ff4444' : risk > 0.3 ? '#ffa726' : '#4fc3f7',
            transition: 'width 0.3s'
          }} />
        </div>
      </div>
      
      {/* Recent conditions */}
      <div style={{ fontSize: '11px', opacity: 0.7, marginBottom: '8px' }}>
        Recent Conditions
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {conditions.slice(-3).reverse().map((condition, i) => (
          <div key={i} style={{
            padding: '6px 8px',
            background: 'rgba(255, 255, 255, 0.02)',
            borderRadius: '3px',
            fontSize: '10px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span style={{ opacity: 0.6 }}>
              {new Date(condition.timestamp).toLocaleTimeString()}
            </span>
            <span style={{
              color: condition.strength > 0.7 ? '#ff4444' : 
                     condition.strength > 0.3 ? '#ffa726' : '#4fc3f7',
              fontWeight: '500'
            }}>
              {((Math.min(1, condition.strength / 2) || 0) * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Memory attractor map
 */
const AttractorMap: React.FC<{
  attractors: MemoryAttractor[];
  primaryColor: string;
}> = ({ attractors, primaryColor }) => {
  // Ensure we have valid attractor data
  const validAttractors = attractors.filter(a => 
    a && typeof a.position === 'object' && a.position.length >= 2
  );

  // Create multiple datasets for better visualization
  const datasets = [];
  
  // Main attractor points - use different colors based on strength
  datasets.push({
    label: 'Attractors',
    type: 'bubble' as const,
    data: validAttractors.map(a => ({
      x: a.position[0] || 0,
      y: a.position[1] || 0,
      r: Math.sqrt(a.strength || 1) * 8 + 5  // Minimum size of 5
    })),
    backgroundColor: primaryColor + '80', // Primary color with alpha
    borderColor: primaryColor,
    borderWidth: 2
  });
  
  // Influence fields (showing radius of effect)
  validAttractors.forEach((attractor, i) => {
    if (attractor.radius > 0) {
      const points = [];
      const steps = 32;
      for (let j = 0; j <= steps; j++) {
        const angle = (j / steps) * 2 * Math.PI;
        points.push({
          x: attractor.position[0] + Math.cos(angle) * attractor.radius,
          y: attractor.position[1] + Math.sin(angle) * attractor.radius
        });
      }
      
      // Offset color - complementary to primary
      const offsetColor = adjustLightness(primaryColor, -20);
      
      datasets.push({
        label: `Field ${i}`,
        type: 'line' as const,
        data: points,
        borderColor: offsetColor + '60', // Offset color with 38% opacity
        backgroundColor: 'transparent',
        borderWidth: 1,
        borderDash: [2, 2],
        pointRadius: 0,
        showLine: true,
        tension: 0
      });
    }
  });
  
  // Add fragments as white dots if available
  validAttractors.forEach((attractor, i) => {
    if (attractor.capturedFragments && attractor.capturedFragments.length > 0) {
      const fragmentData = attractor.capturedFragments.slice(0, 5).map(fragment => ({
        x: attractor.position[0] + (Math.random() - 0.5) * attractor.radius * 0.8,
        y: attractor.position[1] + (Math.random() - 0.5) * attractor.radius * 0.8,
        r: 3
      }));
      
      datasets.push({
        label: `Fragments ${i}`,
        type: 'bubble' as const,
        data: fragmentData,
        backgroundColor: '#ffffff80', // White with 50% opacity
        borderColor: '#ffffff',
        borderWidth: 1,
        pointRadius: 3
      });
    }
  });

  const chartData = { datasets };

  const options: ChartOptions<'scatter'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            if (context.dataset.type === 'bubble') {
              const point = context.raw as any;
              const attractorIndex = context.dataIndex;
              const attractor = validAttractors[attractorIndex];
              return [
                `ID: ${attractor?.id || 'Unknown'}`,
                `Position: (${(point.x || 0).toFixed(2)}, ${(point.y || 0).toFixed(2)})`,
                `Strength: ${(attractor?.strength || 0).toFixed(3)}`,
                `RSP Density: ${(attractor?.rspDensity || 0).toFixed(2)}`,
                `Fragments: ${attractor?.capturedFragments?.length || 0}`
              ];
            }
            return null;
          }
        },
        filter: (tooltipItem) => tooltipItem.dataset.type === 'bubble'
      }
    },
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        title: {
          display: true,
          text: 'Information (Phase Space)',
          color: '#666',
          font: { size: 10 }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.05)'
        },
        ticks: {
          color: '#666',
          font: { size: 9 }
        }
      },
      y: {
        type: 'linear',
        title: {
          display: true,
          text: 'Coherence (Phase Space)',
          color: '#666',
          font: { size: 10 }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.05)'
        },
        ticks: {
          color: '#666',
          font: { size: 9 }
        }
      }
    }
  };

  // Show placeholder if no valid attractors
  if (validAttractors.length === 0) {
    return (
      <div style={{ 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        opacity: 0.5,
        fontSize: '12px'
      }}>
        No memory attractors detected yet
      </div>
    );
  }

  return (
    <div style={{ height: '100%' }}>
      <Scatter data={chartData} options={options} />
    </div>
  );
};

/**
 * Entropy plateau analysis
 */
const EntropyPlateauAnalysis: React.FC<{
  plateaus: EntropyPlateau[];
  primaryColor: string;
}> = ({ plateaus, primaryColor }) => {
  return (
    <div style={{ 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      overflow: 'hidden'
    }}>
      <h3 style={{ 
        margin: '0 0 6px 0', 
        fontSize: '12px', 
        fontWeight: '600',
        display: 'flex',
        alignItems: 'center',
        gap: '4px',
        flexShrink: 0
      }}>
        <Zap size={12} />
        Entropy Plateaus
        <Tooltip content="Periods of stable entropy indicating phase transitions">
          <Info size={10} style={{ opacity: 0.5 }} />
        </Tooltip>
      </h3>
      
      {plateaus.length === 0 ? (
        <div style={{ 
          opacity: 0.5, 
          textAlign: 'center', 
          padding: '12px',
          background: 'rgba(255, 255, 255, 0.02)',
          borderRadius: '4px',
          fontSize: '11px'
        }}>
          No entropy plateaus detected
        </div>
      ) : (
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          gap: '4px',
          overflow: 'auto',
          flex: '1 1 auto'
        }}>
          {plateaus.slice(-4).map((plateau, i) => {
            const duration = (plateau.endTime - plateau.startTime) / 1000;
            const stability = plateau.stability;
            
            return (
              <div
                key={i}
                style={{
                  padding: '6px',
                  background: 'rgba(255, 255, 255, 0.02)',
                  border: `1px solid ${primaryColor}20`,
                  borderRadius: '4px',
                  fontSize: '10px'
                }}
              >
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '4px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                    <span style={{ fontSize: '10px', opacity: 0.6 }}>
                      Plateau #{plateaus.length - (plateaus.length - i - 1)}
                    </span>
                    <span style={{ 
                      fontWeight: '600',
                      color: primaryColor
                    }}>
                      E = {(plateau.averageEntropy || 0).toFixed(4)}
                    </span>
                  </div>
                  <span style={{ fontWeight: '500' }}>
                    {duration.toFixed(1)}s
                  </span>
                </div>
                
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: '10px',
                  opacity: 0.6
                }}>
                  <span>
                    {new Date(plateau.startTime).toLocaleTimeString()}
                  </span>
                  <span>
                    Stability: {((plateau.stability || 0) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

/**
 * Utility functions
 */
function formatRSPValue(value: number | undefined | null): string {
  if (!value || value === 0) return '0';
  if (value < 0.001) return value.toExponential(2);
  if (value < 1) return value.toFixed(4);
  if (value < 1000) return value.toFixed(2);
  if (value < 1e6) return (value / 1000).toFixed(1) + 'K';
  if (value < 1e9) return (value / 1e6).toFixed(1) + 'M';
  return value.toExponential(2);
}

// Memoize component to prevent unnecessary re-renders
export default RSPDashboard;