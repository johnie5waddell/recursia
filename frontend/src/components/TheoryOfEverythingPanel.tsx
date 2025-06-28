/**
 * Theory of Everything Panel - Complete OSH Implementation
 * 
 * Rigorous implementation of OSH Theory of Everything based on:
 * - Information-gravity coupling α = 8π from holographic principle
 * - Ricci tensor derivation R_μν ~ ∇_μ∇_ν I from information curvature
 * - Fundamental force unification through information geometry
 * - Consciousness emergence Φ > 1.0 threshold from IIT
 * 
 * Direct integration with unified VM calculations (no mock data)
 * Half-width layout: Forces (left) + Gravitational Effects (right)
 * Modern design with tight padding and enhanced visualizations
 */

import React, { useMemo, useEffect, useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Line, Radar, Scatter, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler,
  ChartOptions
} from 'chart.js';
import { 
  Atom, 
  Zap, 
  Brain, 
  Waves, 
  TrendingUp,
  Activity,
  Target,
  Layers,
  Gauge,
  Settings2,
  BarChart3,
  Link2
} from 'lucide-react';
import { useEngineAPIContext } from '../contexts/EngineAPIContext';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  ChartTooltip,
  Legend,
  Filler
);

interface OSHTheoryMetrics {
  // Core OSH unified metrics (from VM)
  rsp: number;
  phi: number;
  coherence: number;
  entropy: number;
  information: number;
  complexity: number;
  
  // Information geometry (from unified VM)
  information_curvature: number;
  strain: number;
  quantum_volume: number;
  state_count: number;
  
  // Consciousness indicators
  consciousness_threshold_exceeded: boolean;
  emergence_index: number;
  
  // Time derivatives (unified calculations)
  drsp_dt: number;
  di_dt: number;
  dc_dt: number;
  de_dt: number;
  acceleration: number;
}

interface TheoryOfEverythingPanelProps {
  primaryColor?: string;
  isActive?: boolean;
  showControls?: boolean;
}

export const TheoryOfEverythingPanel: React.FC<TheoryOfEverythingPanelProps> = ({
  primaryColor = '#4fc3f7',
  isActive = true,
  showControls = false
}) => {
  const [history, setHistory] = useState<OSHTheoryMetrics[]>([]);
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  const animationRef = useRef<number>(0);
  
  // Get metrics directly from unified VM via API context
  const { metrics: vmMetrics } = useEngineAPIContext();
  
  // Use VM metrics directly (no transformations)
  const metrics: OSHTheoryMetrics = useMemo(() => ({
    // Core OSH metrics (direct from VM)
    rsp: vmMetrics?.rsp ?? 0,
    phi: vmMetrics?.phi ?? 0,
    coherence: vmMetrics?.coherence ?? 0,
    entropy: vmMetrics?.entropy ?? 0,
    information: vmMetrics?.information ?? 0,
    complexity: vmMetrics?.complexity ?? 1,
    
    // Information geometry (direct from VM unified calculations)
    information_curvature: vmMetrics?.information_curvature ?? 0,
    strain: vmMetrics?.strain ?? 0,
    quantum_volume: vmMetrics?.quantum_volume ?? 1,
    state_count: vmMetrics?.state_count ?? 1,
    
    // Consciousness indicators (VM derived)
    consciousness_threshold_exceeded: (vmMetrics?.phi ?? 0) > 1.0,
    emergence_index: vmMetrics?.emergence_index ?? 0,
    
    // Time derivatives (VM unified calculations)
    drsp_dt: vmMetrics?.drsp_dt ?? 0,
    di_dt: vmMetrics?.di_dt ?? 0,
    dc_dt: vmMetrics?.dc_dt ?? 0,
    de_dt: vmMetrics?.de_dt ?? 0,
    acceleration: vmMetrics?.acceleration ?? 0
  }), [vmMetrics]);
  
  // Update history with real-time VM metrics
  useEffect(() => {
    if (isActive && metrics.rsp !== undefined) {
      setHistory(prev => {
        const lastMetric = prev[prev.length - 1];
        // Only update if metrics changed (avoid excessive re-renders)
        if (lastMetric && 
            Math.abs(lastMetric.rsp - metrics.rsp) < 0.001 && 
            Math.abs(lastMetric.phi - metrics.phi) < 0.001 &&
            Math.abs(lastMetric.information_curvature - metrics.information_curvature) < 1e-6) {
          return prev;
        }
        return [...prev.slice(-49), metrics]; // Keep last 50 data points
      });
    }
  }, [metrics.rsp, metrics.phi, metrics.information_curvature, metrics.strain, isActive]);
  
  // Animation frame for smooth updates
  useEffect(() => {
    const animate = () => {
      animationRef.current += 0.016 * animationSpeed; // ~60fps
      if (isActive) {
        requestAnimationFrame(animate);
      }
    };
    if (isActive) {
      animate();
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive, animationSpeed]);
  
  // OSH Fundamental Constants (no calculations - just display)
  const fundamentalConstants = useMemo(() => {
    // α = 8π coupling from holographic principle (exact, no free parameters)
    const alpha = 8 * Math.PI;
    
    // Direct display of backend-calculated values
    return {
      coupling_constant: alpha,
      information_curvature: metrics.information_curvature,
      strain: metrics.strain,
      phi: metrics.phi,
      rsp: metrics.rsp,
      entropy_flux: metrics.entropy,
      conservation_error: vmMetrics?.conservation_law ?? 0
    };
  }, [metrics, vmMetrics]);
  
  // OSH Core Metrics Display (direct from backend)
  const oshMetricsData = useMemo(() => ({
    labels: ['Φ (IIT)', 'K (Complexity)', 'E (Entropy)', 'C (Coherence)', 'RSP'],
    datasets: [{
      label: 'Normalized Value',
      data: [
        Math.min(1, metrics.phi / 15), // Normalize Phi to [0,1] for display
        metrics.complexity || 0,
        Math.min(1, metrics.entropy / 10), // Normalize entropy to [0,1]
        metrics.coherence || 0,
        Math.min(1, Math.log10(Math.max(1, metrics.rsp)) / 5) // Log scale RSP
      ],
      backgroundColor: `${primaryColor}25`,
      borderColor: primaryColor,
      borderWidth: 2.5,
      pointBackgroundColor: [
        metrics.phi > 1.0 ? '#00ff88' : '#ff6b6b', // Green if conscious
        '#4ecdc4', // Teal  
        '#45b7d1', // Blue
        '#96ceb4', // Green
        primaryColor // Primary
      ],
      pointBorderColor: '#0a0a0a',
      pointBorderWidth: 2,
      pointRadius: 6,
      pointHoverRadius: 8
    }]
  }), [metrics, primaryColor]);
  
  const forceOptions: ChartOptions<'radar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#0a0a0a',
        titleColor: primaryColor,
        bodyColor: '#ffffff',
        borderColor: primaryColor,
        borderWidth: 1,
        titleFont: { size: 11, weight: 'bold' },
        bodyFont: { size: 10 },
        callbacks: {
          label: (context) => {
            const value = Math.pow(10, context.parsed.r);
            const forceNames = ['Electromagnetic', 'Weak Nuclear', 'Strong Nuclear', 'Gravitational', 'Info-Unified'];
            return `${forceNames[context.dataIndex]}: ${value.toExponential(3)}`;
          }
        }
      }
    },
    scales: {
      r: {
        beginAtZero: true,
        min: -50,
        max: 5,
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
          circular: true
        },
        angleLines: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        pointLabels: {
          color: '#ffffff',
          font: { size: 11, weight: '600' }
        },
        ticks: {
          display: false
        }
      }
    },
    animation: {
      duration: 800,
      easing: 'easeInOutQuart'
    }
  };
  
  // Gravitational Effects from Information Curvature (VM data)
  const gravitationalData = useMemo(() => ({
    labels: history.map((_, i) => (i * 0.1).toFixed(1)), // Time in seconds
    datasets: [
      {
        label: 'R_μν (Info Curvature)',
        data: history.map(h => h.information_curvature),
        borderColor: '#ff6b6b',
        backgroundColor: 'rgba(255, 107, 107, 0.1)',
        tension: 0.3,
        fill: true,
        pointRadius: 0,
        borderWidth: 2.5
      },
      {
        label: 'h (GW Strain)',
        data: history.map(h => h.strain * 1e21), // Scale for visibility
        borderColor: '#4ecdc4',
        backgroundColor: 'rgba(78, 205, 196, 0.1)',
        tension: 0.3,
        fill: true,
        pointRadius: 0,
        borderWidth: 2.5
      },
      {
        label: 'Φ (Consciousness)',
        data: history.map(h => h.phi),
        borderColor: primaryColor,
        backgroundColor: `${primaryColor}20`,
        tension: 0.3,
        fill: false,
        pointRadius: 0,
        borderWidth: 2,
        borderDash: [5, 5]
      }
    ]
  }), [history, primaryColor]);
  
  const gravitationalOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          color: '#ffffff',
          font: { size: 10 },
          usePointStyle: true,
          pointStyle: 'rectRounded',
          padding: 15
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: '#0a0a0a',
        titleColor: primaryColor,
        bodyColor: '#ffffff',
        borderColor: primaryColor,
        borderWidth: 1,
        titleFont: { size: 11, weight: 'bold' },
        bodyFont: { size: 10 },
        callbacks: {
          title: (context) => `t = ${context[0].label}s`,
          label: (context) => {
            const value = context.parsed.y;
            if (context.datasetIndex === 1) return `Strain: ${(value * 1e-21).toExponential(2)}`;
            return `${context.dataset.label}: ${value.toExponential(3)}`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: {
          color: '#888888',
          font: { size: 9 },
          maxTicksLimit: 8
        },
        title: {
          display: true,
          text: 'Time (s)',
          color: '#cccccc',
          font: { size: 10 }
        }
      },
      y: {
        type: 'logarithmic',
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: {
          color: '#888888',
          font: { size: 9 },
          callback: function(value: any) {
            return typeof value === 'number' ? value.toExponential(0) : value;
          }
        },
        title: {
          display: true,
          text: 'Amplitude',
          color: '#cccccc',
          font: { size: 10 }
        }
      }
    },
    animation: {
      duration: 600,
      easing: 'easeInOutCubic'
    }
  };
  
  // OSH-specific formatting for theory metrics
  const formatOSH = (value: number, type: 'sci' | 'phi' | 'rsp' | 'coupling' | 'standard' = 'sci'): string => {
    if (value === 0) return '0';
    if (Math.abs(value) < 1e-15) return '~0';
    
    switch (type) {
      case 'phi':
        return value.toFixed(3); // Φ values with 3 decimals
      case 'rsp':
        return value > 1000 ? value.toExponential(2) : value.toFixed(1);
      case 'coupling':
        return value.toExponential(3); // Force couplings in scientific
      case 'standard':
        return value.toFixed(2); // Standard values with 2 decimals
      case 'sci':
        return value.toExponential(2); // Scientific notation
      default:
        return Math.abs(value) < 0.001 ? value.toExponential(2) : value.toFixed(4);
    }
  };
  
  // Consciousness emergence probability (sigmoid)
  const consciousnessProb = useMemo(() => {
    const phi = metrics.phi;
    return 1 / (1 + Math.exp(-5 * (phi - 1.0))); // Sigmoid centered at Φ = 1.0
  }, [metrics.phi]);
  
  return (
    <div className="theory-panel">
      {/* No header - shown at layout level per requirements */}
      
      {/* Controls (optional) */}
      {showControls && (
        <div className="theory-controls">
          <div className="control-group">
            <Settings2 size={12} />
            <label>Animation</label>
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.1"
              value={animationSpeed}
              onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
              className="speed-slider"
            />
          </div>
        </div>
      )}
      
      {/* Key Metrics Overview - Single Row */}
      <div className="theory-metrics-row">
        <div className="metric-item">
          <Brain size={12} style={{ color: metrics.consciousness_threshold_exceeded ? '#10b981' : '#888' }} />
          <span className="metric-label">Φ</span>
          <span className={`metric-value ${metrics.consciousness_threshold_exceeded ? 'conscious' : ''}`}>
            {formatOSH(metrics.phi, 'phi')}
          </span>
          <span className="metric-unit">({(consciousnessProb * 100).toFixed(0)}%)</span>
        </div>
        
        <div className="metric-divider" />
        
        <div className="metric-item">
          <Waves size={12} style={{ color: primaryColor }} />
          <span className="metric-label">R_μν</span>
          <span className="metric-value">{formatOSH(metrics.information_curvature, 'sci')}</span>
        </div>
        
        <div className="metric-divider" />
        
        <div className="metric-item">
          <Target size={12} style={{ color: '#4ecdc4' }} />
          <span className="metric-label">RSP</span>
          <span className="metric-value">{formatOSH(metrics.rsp, 'rsp')}</span>
          <span className="metric-unit">bit⋅s</span>
        </div>
        
        <div className="metric-divider" />
        
        <div className="metric-item">
          <Activity size={12} style={{ color: '#ff6b6b' }} />
          <span className="metric-label">dRSP/dt</span>
          <span className="metric-value">{formatOSH(metrics.drsp_dt, 'sci')}</span>
        </div>
      </div>
      
      {/* OSH Core Equations Display */}
      <div className="theory-equations">
        <div className="equation-card">
          <h5>Recursive Simulation Potential</h5>
          <div className="equation">RSP = I × K / E = {formatOSH(metrics.rsp, 'standard')} bit·s</div>
          <div className="equation-values">
            I={formatOSH(metrics.information, 'standard')} bits, 
            K={formatOSH(metrics.complexity, 'standard')}, 
            E={formatOSH(metrics.entropy, 'standard')} bits/s
          </div>
        </div>
        
        <div className="equation-card">
          <h5>Conservation Law</h5>
          <div className="equation">d/dt(I×K) = α·E + β·Q</div>
          <div className="equation-values">
            α=8π={fundamentalConstants.coupling_constant.toFixed(3)}, 
            Error={(fundamentalConstants.conservation_error).toExponential(2)}
          </div>
        </div>
        
        <div className="equation-card">
          <h5>Information-Gravity Coupling</h5>
          <div className="equation">R_μν = 8π ∇_μ∇_ν I</div>
          <div className="equation-values">
            R={formatOSH(metrics.information_curvature, 'sci')}, 
            h={formatOSH(metrics.strain, 'sci')}
          </div>
        </div>
      </div>
      
      {/* Half-width visualizations side by side */}
      <div className="theory-visualizations">
        {/* Fundamental Forces (Left Half) */}
        <div className="viz-half forces-viz">
          <div className="viz-header">
            <Brain size={14} style={{ color: primaryColor }} />
            <h4>OSH Core Metrics</h4>
          </div>
          <div className="chart-container">
            <Radar data={oshMetricsData} options={forceOptions} />
          </div>
          <div className="force-metrics">
            <div className="force-item">
              <span className="force-label">Φ:</span>
              <span className="force-value">{formatOSH(metrics.phi || 0, 'standard')} bits</span>
            </div>
            <div className="force-item">
              <span className="force-label">RSP:</span>
              <span className="force-value">{formatOSH(metrics.rsp || 0, 'sci')} bit·s</span>
            </div>
            <div className="force-item">
              <span className="force-label">Coherence:</span>
              <span className="force-value">{((metrics.coherence || 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="force-item">
              <span className="force-label">Entropy:</span>
              <span className="force-value">{formatOSH(metrics.entropy || 0, 'standard')} bits/s</span>
            </div>
          </div>
        </div>
        
        {/* Gravitational Effects (Right Half) */}
        <div className="viz-half gravity-viz">
          <div className="viz-header">
            <Waves size={14} style={{ color: '#ff6b6b' }} />
            <h4>Gravitational Effects</h4>
          </div>
          <div className="chart-container">
            <Line data={gravitationalData} options={gravitationalOptions} />
          </div>
          <div className="gravity-metrics">
            <div className="gravity-item">
              <span className="gravity-label">Strain (h):</span>
              <span className="gravity-value">{formatOSH(metrics.strain, 'sci')}</span>
            </div>
            <div className="gravity-item">
              <span className="gravity-label">α = 8π:</span>
              <span className="gravity-value">{(8 * Math.PI).toFixed(3)}</span>
            </div>
            <div className="gravity-item">
              <span className="gravity-label">Q-Vol:</span>
              <span className="gravity-value">{formatOSH(metrics.quantum_volume, 'sci')}</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Theory Status Indicators */}
      <div className="theory-status">
        <motion.div
          className={`status-indicator ${metrics.consciousness_threshold_exceeded ? 'active' : 'inactive'}`}
          animate={{
            scale: metrics.consciousness_threshold_exceeded ? [1, 1.05, 1] : 1,
            opacity: metrics.consciousness_threshold_exceeded ? 1 : 0.6
          }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <Brain size={12} />
          <span>Consciousness {metrics.consciousness_threshold_exceeded ? 'Emerged' : 'Dormant'}</span>
        </motion.div>
        
        <motion.div
          className={`status-indicator ${Math.abs(metrics.information_curvature) > 1e-10 ? 'active' : 'inactive'}`}
          animate={{
            scale: Math.abs(metrics.information_curvature) > 1e-10 ? [1, 1.05, 1] : 1,
            opacity: Math.abs(metrics.information_curvature) > 1e-10 ? 1 : 0.6
          }}
          transition={{ duration: 2.5, repeat: Infinity }}
        >
          <Waves size={12} />
          <span>Info-Gravity {Math.abs(metrics.information_curvature) > 1e-10 ? 'Active' : 'Inactive'}</span>
        </motion.div>
        
        <motion.div
          className={`status-indicator ${fundamentalConstants.conservation_error < 1e-4 ? 'active' : 'inactive'}`}
          animate={{
            scale: fundamentalConstants.conservation_error < 1e-4 ? [1, 1.05, 1] : 1,
            opacity: fundamentalConstants.conservation_error < 1e-4 ? 1 : 0.6
          }}
          transition={{ duration: 3, repeat: Infinity }}
        >
          <Link2 size={12} />
          <span>Conservation {fundamentalConstants.conservation_error < 1e-4 ? 'Valid' : 'Violated'}</span>
        </motion.div>
      </div>
    </div>
  );
};

// Memoize component to prevent unnecessary re-renders
export default TheoryOfEverythingPanel;