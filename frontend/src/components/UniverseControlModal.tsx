/**
 * Universe Control Modal
 * Enterprise-grade simulation control panel for universe mode
 * Provides real-time parameter adjustment and monitoring
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useEngineAPIContext } from '../contexts/EngineAPIContext';
import {
  Play,
  Pause,
  RotateCcw,
  Sliders,
  Users,
  Cpu,
  GitBranch,
  Activity,
  Zap,
  X,
  Plus,
  Minus,
  Info,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Waves,
  Brain,
  Database,
  Network,
  AlertTriangle,
  HardDrive
} from 'lucide-react';
import { Tooltip } from './ui/Tooltip';
import { 
  PHI_SCALING_FACTOR_BETA,
  CONSCIOUSNESS_SIGMOID_K,
  CONSCIOUSNESS_SIGMOID_PHI_C,
  DEFAULT_COHERENCE,
  DEFAULT_ENTROPY
} from '../config/physicsConstants';
import { getSystemResourceMonitor, useSystemMetrics } from '../utils/systemResourceMonitor';
import { getMemorySafeGridSize, MEMORY_CONFIG } from '../config/memoryConfig';

interface UniverseControlModalProps {
  isOpen: boolean;
  onClose: () => void;
  primaryColor: string;
  isSimulating: boolean;
  onToggleSimulation: () => void;
  onUpdateParameters: (params: UniverseParameters) => void;
  currentParameters?: UniverseParameters;
  metrics?: any;
}

export interface UniverseParameters {
  // Quantum field parameters
  universeQubits: number;
  memoryQubits: number;
  consciousnessQubits: number;
  
  // Coherence settings
  universeCoherence: number;
  memoryCoherence: number;
  consciousnessCoherence: number;
  
  // Entropy settings
  universeEntropy: number;
  memoryEntropy: number;
  consciousnessEntropy: number;
  
  // Observer configuration
  observerCount: number;
  observerCollapseThreshold: number;
  observerSelfAwareness: number;
  
  // Gate configuration
  gateTypes: string[];
  gateComplexity: number;
  
  // Advanced parameters
  scalingFactor: number;
  recursionDepth: number;
  consciousnessThreshold: number;
  decoherenceRate: number;
  
  // Evolution settings
  evolutionSteps: number;
  simulationSteps: number;
  timeScale: number;
  
  // Dynamic universe settings
  universeMode: string;
  evolutionRate: number;
  chaoseFactor: number;
  interactionStrength: number;
}

const DEFAULT_PARAMETERS: UniverseParameters = {
  universeQubits: 16,
  memoryQubits: 12,
  consciousnessQubits: 10,
  
  universeCoherence: DEFAULT_COHERENCE,
  memoryCoherence: 0.92,
  consciousnessCoherence: 0.88,
  
  universeEntropy: DEFAULT_ENTROPY,
  memoryEntropy: 0.12,
  consciousnessEntropy: 0.15,
  
  observerCount: 3,
  observerCollapseThreshold: 0.85,
  observerSelfAwareness: 0.9,
  
  gateTypes: ['H', 'CNOT', 'RX', 'RY', 'RZ', 'CZ', 'TOFFOLI'],
  gateComplexity: 3,
  
  scalingFactor: PHI_SCALING_FACTOR_BETA,
  recursionDepth: 22,
  consciousnessThreshold: CONSCIOUSNESS_SIGMOID_PHI_C,
  decoherenceRate: 333.3,
  
  evolutionSteps: 3,
  simulationSteps: 2,
  timeScale: 1.0,
  
  universeMode: 'standard',
  evolutionRate: 1.0,
  chaoseFactor: 0.1,
  interactionStrength: 0.5
};

const GATE_OPTIONS = [
  { value: 'H', label: 'Hadamard', description: 'Creates superposition' },
  { value: 'X', label: 'Pauli-X', description: 'Bit flip gate' },
  { value: 'Y', label: 'Pauli-Y', description: 'Bit and phase flip' },
  { value: 'Z', label: 'Pauli-Z', description: 'Phase flip gate' },
  { value: 'CNOT', label: 'CNOT', description: 'Controlled NOT' },
  { value: 'CZ', label: 'CZ', description: 'Controlled Z' },
  { value: 'TOFFOLI', label: 'Toffoli', description: 'Controlled-controlled NOT' },
  { value: 'RX', label: 'RX', description: 'X-axis rotation' },
  { value: 'RY', label: 'RY', description: 'Y-axis rotation' },
  { value: 'RZ', label: 'RZ', description: 'Z-axis rotation' },
  { value: 'SWAP', label: 'SWAP', description: 'Swaps two qubits' }
];

// Universe mode options
const UNIVERSE_MODES = [
  { value: 'standard', label: 'Standard Universe', description: 'Balanced evolution with moderate interaction' },
  { value: 'high_entropy', label: 'High Entropy', description: 'Rapid decoherence and high entropy production' },
  { value: 'coherent', label: 'Coherent Universe', description: 'Highly coherent states with minimal decoherence' },
  { value: 'chaotic', label: 'Chaotic Universe', description: 'Unpredictable evolution with strong interactions' },
  { value: 'quantum_critical', label: 'Quantum Critical', description: 'Universe at the edge of phase transition' }
];

export const UniverseControlModal: React.FC<UniverseControlModalProps> = ({
  isOpen,
  onClose,
  primaryColor,
  isSimulating,
  onToggleSimulation,
  onUpdateParameters,
  currentParameters = DEFAULT_PARAMETERS,
  metrics
}) => {
  const [parameters, setParameters] = useState<UniverseParameters>(currentParameters);
  const { setUniverseMode, updateUniverseParameters, startUniverseSimulation, stopUniverseSimulation } = useEngineAPIContext();
  const { metrics: systemMetrics, health: systemHealth } = useSystemMetrics();
  
  // Calculate memory-based limits
  const memoryLimits = useMemo(() => {
    if (!systemMetrics?.memory) {
      return {
        maxUniverseQubits: 16,
        maxMemoryQubits: 12,
        maxConsciousnessQubits: 10,
        maxObservers: 5,
        memoryPercentage: 50,
        availableMemoryMB: 1000,
        recommendedUniverseQubits: 12,
        warningThreshold: false,
        criticalThreshold: false
      };
    }
    
    const availableMemoryMB = systemMetrics.memory.free / (1024 * 1024);
    const memoryPercentage = systemMetrics.memory.percentage;
    
    // Calculate safe qubit limits based on available memory
    // Each qubit exponentially increases memory usage (2^n states)
    // Base calculation: ~100MB per qubit at n=16
    const safeMemoryForQuantum = availableMemoryMB * 0.6; // Use 60% of available memory
    
    let maxUniverseQubits = 8; // Minimum
    let maxMemoryQubits = 6;
    let maxConsciousnessQubits = 4;
    let maxObservers = 2;
    
    if (safeMemoryForQuantum > 3000) {
      maxUniverseQubits = 20;
      maxMemoryQubits = 16;
      maxConsciousnessQubits = 14;
      maxObservers = 8;
    } else if (safeMemoryForQuantum > 2000) {
      maxUniverseQubits = 18;
      maxMemoryQubits = 14;
      maxConsciousnessQubits = 12;
      maxObservers = 6;
    } else if (safeMemoryForQuantum > 1000) {
      maxUniverseQubits = 16;
      maxMemoryQubits = 12;
      maxConsciousnessQubits = 10;
      maxObservers = 5;
    } else if (safeMemoryForQuantum > 500) {
      maxUniverseQubits = 14;
      maxMemoryQubits = 10;
      maxConsciousnessQubits = 8;
      maxObservers = 4;
    } else if (safeMemoryForQuantum > 250) {
      maxUniverseQubits = 12;
      maxMemoryQubits = 8;
      maxConsciousnessQubits = 6;
      maxObservers = 3;
    }
    
    // Calculate recommended values (75% of max)
    const recommendedUniverseQubits = Math.floor(maxUniverseQubits * 0.75);
    
    return {
      maxUniverseQubits,
      maxMemoryQubits,
      maxConsciousnessQubits,
      maxObservers,
      memoryPercentage,
      availableMemoryMB,
      recommendedUniverseQubits,
      warningThreshold: memoryPercentage > MEMORY_CONFIG.WARNING_MEMORY_THRESHOLD * 100,
      criticalThreshold: memoryPercentage > MEMORY_CONFIG.CRITICAL_MEMORY_THRESHOLD * 100
    };
  }, [systemMetrics]);
  
  const [expandedSections, setExpandedSections] = useState({
    quantum: true,
    coherence: true,
    entropy: false,
    observers: true,
    gates: false,
    advanced: false
  });

  // Update local parameters when props change
  useEffect(() => {
    setParameters(currentParameters);
  }, [currentParameters]);

  // Handle parameter changes
  const updateParameter = useCallback((key: keyof UniverseParameters, value: any) => {
    setParameters(prev => {
      const updated = { ...prev, [key]: value };
      onUpdateParameters(updated);
      
      // Send specific updates to backend
      if (key === 'universeMode') {
        setUniverseMode(value);
      } else {
        updateUniverseParameters({ [key]: value });
      }
      
      return updated;
    });
  }, [onUpdateParameters, setUniverseMode, updateUniverseParameters]);

  // Toggle section expansion
  const toggleSection = useCallback((section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  }, []);

  // Calculate estimated consciousness probability
  const calculateConsciousnessProbability = useCallback(() => {
    const phi = parameters.scalingFactor * parameters.consciousnessQubits * Math.pow(parameters.consciousnessCoherence, 2);
    return 1 / (1 + Math.exp(-CONSCIOUSNESS_SIGMOID_K * (phi - parameters.consciousnessThreshold)));
  }, [parameters]);

  // Listen for universe start confirmation
  useEffect(() => {
    const handleUniverseStartConfirmed = (event: CustomEvent) => {
      // Force update local state if needed
      if (!isSimulating && event.detail.running) {
        onToggleSimulation();
      }
    };
    
    window.addEventListener('universeStartConfirmed', handleUniverseStartConfirmed as EventListener);
    
    return () => {
      window.removeEventListener('universeStartConfirmed', handleUniverseStartConfirmed as EventListener);
    };
  }, [isSimulating, onToggleSimulation]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="universe-control-backdrop"
        onClick={onClose}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.85)',
          backdropFilter: 'blur(8px)',
          zIndex: 10000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          className="universe-control-content"
          onClick={(e) => e.stopPropagation()}
          style={{
            background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
            border: `1px solid ${primaryColor}30`,
            borderRadius: '12px',
            padding: '0',
            width: '90%',
            maxWidth: '800px',
            maxHeight: '90vh',
            overflow: 'hidden',
            boxShadow: `0 25px 50px rgba(0, 0, 0, 0.8), 0 0 30px ${primaryColor}15`,
            backdropFilter: 'blur(20px)'
          }}
        >
          {/* Header */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0.75rem 1rem',
            borderBottom: `1px solid ${primaryColor}20`,
            background: `linear-gradient(90deg, ${primaryColor}08 0%, transparent 100%)`
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Sparkles size={20} color={primaryColor} />
              <h2 style={{ margin: 0, color: primaryColor, fontSize: '1.25rem', fontWeight: 600 }}>
                Universe Simulation Control
              </h2>
            </div>
            
            {/* Memory Status Indicator */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.25rem 0.75rem',
                borderRadius: '6px',
                background: memoryLimits.criticalThreshold ? 'rgba(239, 68, 68, 0.2)' : 
                           memoryLimits.warningThreshold ? 'rgba(251, 191, 36, 0.2)' : 
                           'rgba(34, 197, 94, 0.2)',
                border: `1px solid ${
                  memoryLimits.criticalThreshold ? '#ef4444' : 
                  memoryLimits.warningThreshold ? '#fbbf24' : 
                  '#22c55e'
                }40`
              }}>
                <HardDrive size={16} color={
                  memoryLimits.criticalThreshold ? '#ef4444' : 
                  memoryLimits.warningThreshold ? '#fbbf24' : 
                  '#22c55e'
                } />
                <span style={{ 
                  fontSize: '0.75rem', 
                  color: memoryLimits.criticalThreshold ? '#ef4444' : 
                         memoryLimits.warningThreshold ? '#fbbf24' : 
                         '#22c55e',
                  fontWeight: 600
                }}>
                  {memoryLimits.memoryPercentage.toFixed(1)}% RAM
                </span>
                <Tooltip primaryColor={primaryColor} content={`Available: ${memoryLimits.availableMemoryMB.toFixed(0)}MB`}>
                  <Info size={12} style={{ opacity: 0.7 }} />
                </Tooltip>
              </div>
              
              <button
                onClick={onClose}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#999',
                  cursor: 'pointer',
                  padding: '0.25rem',
                  borderRadius: '6px',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                  e.currentTarget.style.color = '#fff';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'none';
                  e.currentTarget.style.color = '#999';
                }}
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {/* Control Buttons */}
          <div style={{
            display: 'flex',
            gap: '0.5rem',
            padding: '0.75rem 1rem',
            borderBottom: `1px solid ${primaryColor}10`
          }}>
            <button
              onClick={() => {
                if (isSimulating) {
                  stopUniverseSimulation();
                } else {
                  startUniverseSimulation();
                }
                onToggleSimulation();
              }}
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                border: 'none',
                background: isSimulating ? '#ff4444' : primaryColor,
                color: '#fff',
                fontSize: '1rem',
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              {isSimulating ? <Pause size={20} /> : <Play size={20} />}
              {isSimulating ? 'Stop Simulation' : 'Start Simulation'}
            </button>
            
            <button
              onClick={() => setParameters(DEFAULT_PARAMETERS)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                border: `1px solid ${primaryColor}40`,
                background: 'transparent',
                color: primaryColor,
                fontSize: '1rem',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = `${primaryColor}20`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
              }}
            >
              <RotateCcw size={18} />
              Reset
            </button>
          </div>

          {/* Universe Mode Selection */}
          <div style={{
            padding: '0.75rem 1rem',
            borderBottom: `1px solid ${primaryColor}10`
          }}>
            <div style={{ marginBottom: '1rem' }}>
              <label style={{ fontSize: '0.875rem', color: '#ccc', display: 'block', marginBottom: '0.5rem' }}>
                Universe Evolution Mode
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.5rem' }}>
                {UNIVERSE_MODES.map(mode => (
                  <button
                    key={mode.value}
                    onClick={() => updateParameter('universeMode', mode.value)}
                    style={{
                      padding: '0.5rem',
                      borderRadius: '6px',
                      border: `2px solid ${parameters.universeMode === mode.value ? primaryColor : '#333'}`,
                      background: parameters.universeMode === mode.value ? `${primaryColor}20` : 'transparent',
                      color: parameters.universeMode === mode.value ? primaryColor : '#ccc',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: '0.125rem'
                    }}
                  >
                    <span style={{ fontWeight: 500, fontSize: '0.875rem' }}>{mode.label}</span>
                    <span style={{ fontSize: '0.625rem', opacity: 0.7, lineHeight: 1.2 }}>{mode.description}</span>
                  </button>
                ))}
              </div>
            </div>
            
            {/* Evolution Rate Control */}
            <ParameterSlider
              label="Evolution Rate"
              value={parameters.evolutionRate}
              onChange={(v) => updateParameter('evolutionRate', v)}
              min={0.1}
              max={5.0}
              step={0.1}
              primaryColor={primaryColor}
              icon={<Zap size={16} />}
              tooltip="Speed of universe evolution (1.0 = normal)"
            />
          </div>

          {/* Live Metrics */}
          {metrics && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))',
              gap: '0.75rem',
              padding: '0.75rem 1.5rem',
              background: `linear-gradient(135deg, ${primaryColor}05 0%, rgba(255, 255, 255, 0.02) 100%)`,
              borderBottom: `1px solid ${primaryColor}10`,
              borderRadius: '8px 8px 0 0'
            }}>
              <div style={{ 
                textAlign: 'center',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: `1px solid ${primaryColor}15`
              }}>
                <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Universe Time</div>
                <div style={{ fontSize: '1.1rem', color: primaryColor, fontWeight: 700, fontFamily: 'monospace' }}>
                  {(typeof metrics?.universe_time === 'number' ? metrics.universe_time : 0).toFixed(2)}s
                </div>
              </div>
              <div style={{ 
                textAlign: 'center',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: `1px solid ${primaryColor}15`
              }}>
                <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Iterations</div>
                <div style={{ fontSize: '1.1rem', color: primaryColor, fontWeight: 700, fontFamily: 'monospace' }}>
                  {(typeof metrics?.iteration_count === 'number' ? metrics.iteration_count : 0).toLocaleString()}
                </div>
              </div>
              <div style={{ 
                textAlign: 'center',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: `1px solid ${primaryColor}15`
              }}>
                <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>RSP</div>
                <div style={{ fontSize: '1.1rem', color: primaryColor, fontWeight: 700, fontFamily: 'monospace' }}>
                  {(metrics?.rsp || 0).toFixed(1)}
                </div>
              </div>
              <div style={{ 
                textAlign: 'center',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: `1px solid ${primaryColor}15`
              }}>
                <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Coherence</div>
                <div style={{ fontSize: '1.1rem', color: primaryColor, fontWeight: 700, fontFamily: 'monospace' }}>
                  {((metrics?.coherence || 0) * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ 
                textAlign: 'center',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: `1px solid ${primaryColor}15`
              }}>
                <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Entropy</div>
                <div style={{ fontSize: '1.1rem', color: primaryColor, fontWeight: 700, fontFamily: 'monospace' }}>
                  {(metrics?.entropy || 0).toFixed(3)}
                </div>
              </div>
              <div style={{ 
                textAlign: 'center',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: `1px solid ${primaryColor}15`
              }}>
                <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Entanglements</div>
                <div style={{ fontSize: '1.1rem', color: primaryColor, fontWeight: 700, fontFamily: 'monospace' }}>
                  {(metrics?.num_entanglements || 0).toLocaleString()}
                </div>
              </div>
              <div style={{ 
                textAlign: 'center',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: `1px solid ${primaryColor}15`
              }}>
                <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Observers</div>
                <div style={{ fontSize: '1.1rem', color: primaryColor, fontWeight: 700, fontFamily: 'monospace' }}>
                  {(metrics?.observer_count || metrics?.observers || 0).toLocaleString()}
                </div>
              </div>
              <div style={{ 
                textAlign: 'center',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: `1px solid ${primaryColor}15`
              }}>
                <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>States</div>
                <div style={{ fontSize: '1.1rem', color: primaryColor, fontWeight: 700, fontFamily: 'monospace' }}>
                  {(metrics?.state_count || metrics?.states || 0).toLocaleString()}
                </div>
              </div>
              <div style={{ 
                textAlign: 'center',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: `1px solid ${primaryColor}15`
              }}>
                <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Φ Value</div>
                <div style={{ fontSize: '1.1rem', color: primaryColor, fontWeight: 700, fontFamily: 'monospace' }}>
                  {(metrics?.phi || 0).toFixed(2)}
                </div>
              </div>
              <div style={{ 
                textAlign: 'center',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: `1px solid ${primaryColor}15`
              }}>
                <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Mode</div>
                <div style={{ fontSize: '0.85rem', color: primaryColor, fontWeight: 600, textAlign: 'center' }}>
                  {(metrics?.universe_mode || 'Standard').replace(' Universe', '')}
                </div>
              </div>
              <div style={{ 
                textAlign: 'center',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.03)',
                border: `1px solid ${primaryColor}15`
              }}>
                <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Status</div>
                <div style={{ 
                  fontSize: '0.9rem', 
                  color: metrics?.universe_running ? '#22c55e' : '#ef4444', 
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.3rem'
                }}>
                  <div style={{
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    backgroundColor: metrics?.universe_running ? '#22c55e' : '#ef4444',
                    boxShadow: metrics?.universe_running ? '0 0 8px #22c55e40' : '0 0 8px #ef444440'
                  }} />
                  {metrics?.universe_running ? 'ACTIVE' : 'IDLE'}
                </div>
              </div>
            </div>
          )}

          {/* Parameter Controls - Removed all slider sections */}
          <div style={{
            padding: '1rem',
            overflowY: 'auto',
            maxHeight: 'calc(95vh - 280px)',
            display: 'none'  // Hide the entire parameter controls section
          }}>
            {/* Quantum Fields Section */}
            <div style={{ marginBottom: '1.5rem' }}>
              <button
                onClick={() => toggleSection('quantum')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  width: '100%',
                  padding: '0.75rem',
                  background: `${primaryColor}10`,
                  border: `1px solid ${primaryColor}20`,
                  borderRadius: '8px',
                  color: primaryColor,
                  fontSize: '1rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                  marginBottom: expandedSections.quantum ? '1rem' : 0,
                  transition: 'all 0.2s ease'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Cpu size={18} />
                  Quantum Fields
                </div>
                {expandedSections.quantum ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>
              
              {expandedSections.quantum && (
                <div style={{ display: 'grid', gap: '0.75rem' }}>
                  {memoryLimits.criticalThreshold && (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      padding: '0.5rem 0.75rem',
                      background: 'rgba(239, 68, 68, 0.1)',
                      border: '1px solid rgba(239, 68, 68, 0.3)',
                      borderRadius: '6px',
                      fontSize: '0.75rem',
                      color: '#ef4444'
                    }}>
                      <AlertTriangle size={14} />
                      Memory critical! Reduced qubit limits to prevent system freeze.
                    </div>
                  )}
                  
                  <ParameterSlider
                    label="Universe Qubits"
                    value={parameters.universeQubits}
                    onChange={(v) => updateParameter('universeQubits', v)}
                    min={4}
                    max={memoryLimits.maxUniverseQubits}
                    step={1}
                    primaryColor={primaryColor}
                    icon={<Network size={14} />}
                    tooltip={`Number of qubits in the universe field (max: ${memoryLimits.maxUniverseQubits} based on available RAM)`}
                    recommended={memoryLimits.recommendedUniverseQubits}
                    memoryLimited={parameters.universeQubits === memoryLimits.maxUniverseQubits}
                  />
                  <ParameterSlider
                    label="Memory Qubits"
                    value={parameters.memoryQubits}
                    onChange={(v) => updateParameter('memoryQubits', v)}
                    min={2}
                    max={memoryLimits.maxMemoryQubits}
                    step={1}
                    primaryColor={primaryColor}
                    icon={<Database size={14} />}
                    tooltip={`Number of qubits in the memory field (max: ${memoryLimits.maxMemoryQubits} based on available RAM)`}
                    memoryLimited={parameters.memoryQubits === memoryLimits.maxMemoryQubits}
                  />
                  <ParameterSlider
                    label="Consciousness Qubits"
                    value={parameters.consciousnessQubits}
                    onChange={(v) => updateParameter('consciousnessQubits', v)}
                    min={2}
                    max={memoryLimits.maxConsciousnessQubits}
                    step={1}
                    primaryColor={primaryColor}
                    icon={<Brain size={14} />}
                    tooltip={`Number of qubits in the consciousness field (max: ${memoryLimits.maxConsciousnessQubits} based on available RAM)`}
                    memoryLimited={parameters.consciousnessQubits === memoryLimits.maxConsciousnessQubits}
                  />
                </div>
              )}
            </div>

            {/* Coherence Section */}
            <div style={{ marginBottom: '1.5rem' }}>
              <button
                onClick={() => toggleSection('coherence')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  width: '100%',
                  padding: '0.75rem',
                  background: `${primaryColor}10`,
                  border: `1px solid ${primaryColor}20`,
                  borderRadius: '8px',
                  color: primaryColor,
                  fontSize: '1rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                  marginBottom: expandedSections.coherence ? '1rem' : 0,
                  transition: 'all 0.2s ease'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Waves size={18} />
                  Coherence Settings
                </div>
                {expandedSections.coherence ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>
              
              {expandedSections.coherence && (
                <div style={{ display: 'grid', gap: '1rem' }}>
                  <ParameterSlider
                    label="Universe Coherence"
                    value={parameters.universeCoherence}
                    onChange={(v) => updateParameter('universeCoherence', v)}
                    min={0}
                    max={1}
                    step={0.01}
                    primaryColor={primaryColor}
                    percentage
                    tooltip="Quantum coherence of the universe field"
                  />
                  <ParameterSlider
                    label="Memory Coherence"
                    value={parameters.memoryCoherence}
                    onChange={(v) => updateParameter('memoryCoherence', v)}
                    min={0}
                    max={1}
                    step={0.01}
                    primaryColor={primaryColor}
                    percentage
                    tooltip="Quantum coherence of the memory field"
                  />
                  <ParameterSlider
                    label="Consciousness Coherence"
                    value={parameters.consciousnessCoherence}
                    onChange={(v) => updateParameter('consciousnessCoherence', v)}
                    min={0}
                    max={1}
                    step={0.01}
                    primaryColor={primaryColor}
                    percentage
                    tooltip="Quantum coherence of the consciousness field"
                  />
                </div>
              )}
            </div>

            {/* Observer Section */}
            <div style={{ marginBottom: '1.5rem' }}>
              <button
                onClick={() => toggleSection('observers')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  width: '100%',
                  padding: '0.75rem',
                  background: `${primaryColor}10`,
                  border: `1px solid ${primaryColor}20`,
                  borderRadius: '8px',
                  color: primaryColor,
                  fontSize: '1rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                  marginBottom: expandedSections.observers ? '1rem' : 0,
                  transition: 'all 0.2s ease'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Users size={18} />
                  Observer Configuration
                </div>
                {expandedSections.observers ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>
              
              {expandedSections.observers && (
                <div style={{ display: 'grid', gap: '1rem' }}>
                  <ParameterSlider
                    label="Observer Count"
                    value={parameters.observerCount}
                    onChange={(v) => updateParameter('observerCount', v)}
                    min={0}
                    max={memoryLimits.maxObservers}
                    step={1}
                    primaryColor={primaryColor}
                    icon={<Users size={14} />}
                    tooltip={`Number of active observers in the system (max: ${memoryLimits.maxObservers} based on available RAM)`}
                    memoryLimited={parameters.observerCount === memoryLimits.maxObservers}
                  />
                  <ParameterSlider
                    label="Collapse Threshold"
                    value={parameters.observerCollapseThreshold}
                    onChange={(v) => updateParameter('observerCollapseThreshold', v)}
                    min={0.5}
                    max={1}
                    step={0.01}
                    primaryColor={primaryColor}
                    percentage
                    tooltip="Threshold for observer-induced state collapse"
                  />
                  <ParameterSlider
                    label="Self Awareness"
                    value={parameters.observerSelfAwareness}
                    onChange={(v) => updateParameter('observerSelfAwareness', v)}
                    min={0}
                    max={1}
                    step={0.01}
                    primaryColor={primaryColor}
                    percentage
                    tooltip="Level of observer self-awareness"
                  />
                </div>
              )}
            </div>

            {/* Dynamic Evolution Section */}
            <div style={{ marginBottom: '1.5rem' }}>
              <button
                onClick={() => toggleSection('advanced')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  width: '100%',
                  padding: '0.75rem',
                  background: `${primaryColor}10`,
                  border: `1px solid ${primaryColor}20`,
                  borderRadius: '8px',
                  color: primaryColor,
                  fontSize: '1rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                  marginBottom: expandedSections.advanced ? '1rem' : 0,
                  transition: 'all 0.2s ease'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Activity size={18} />
                  Dynamic Evolution
                </div>
                {expandedSections.advanced ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>
              
              {expandedSections.advanced && (
                <div style={{ display: 'grid', gap: '1rem' }}>
                  <ParameterSlider
                    label="Chaos Factor"
                    value={parameters.chaoseFactor}
                    onChange={(v) => updateParameter('chaoseFactor', v)}
                    min={0}
                    max={1}
                    step={0.01}
                    primaryColor={primaryColor}
                    percentage
                    icon={<Sparkles size={16} />}
                    tooltip="Randomness in universe evolution"
                  />
                  <ParameterSlider
                    label="Interaction Strength"
                    value={parameters.interactionStrength}
                    onChange={(v) => updateParameter('interactionStrength', v)}
                    min={0}
                    max={1}
                    step={0.01}
                    primaryColor={primaryColor}
                    percentage
                    icon={<GitBranch size={16} />}
                    tooltip="How strongly quantum states interact"
                  />
                  <ParameterSlider
                    label="Decoherence Rate"
                    value={parameters.decoherenceRate}
                    onChange={(v) => updateParameter('decoherenceRate', v)}
                    min={0}
                    max={1000}
                    step={10}
                    primaryColor={primaryColor}
                    icon={<Activity size={16} />}
                    tooltip="Rate of quantum decoherence (Hz)"
                  />
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

// Parameter Slider Component
interface ParameterSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  primaryColor: string;
  percentage?: boolean;
  icon?: React.ReactNode;
  tooltip?: string;
  recommended?: number;
  memoryLimited?: boolean;
}

const ParameterSlider: React.FC<ParameterSliderProps> = ({
  label,
  value,
  onChange,
  min,
  max,
  step,
  primaryColor,
  percentage = false,
  icon,
  tooltip,
  recommended,
  memoryLimited = false
}) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {icon && <span style={{ color: primaryColor }}>{icon}</span>}
          <label style={{ fontSize: '0.875rem', color: '#ccc' }}>{label}</label>
          {tooltip && (
            <Tooltip primaryColor={primaryColor} content={tooltip}>
              <Info size={14} style={{ color: '#666', cursor: 'help' }} />
            </Tooltip>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.875rem', color: primaryColor, fontWeight: 500 }}>
            {percentage ? `${(value * 100).toFixed(0)}%` : value}
          </span>
          {memoryLimited && (
            <span style={{ 
              fontSize: '0.625rem', 
              color: '#fbbf24',
              padding: '0.125rem 0.375rem',
              background: 'rgba(251, 191, 36, 0.15)',
              borderRadius: '4px',
              border: '1px solid rgba(251, 191, 36, 0.3)',
              fontWeight: 600
            }}>
              RAM LIMIT
            </span>
          )}
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <button
          onClick={() => onChange(Math.max(min, value - step))}
          style={{
            background: 'rgba(255, 255, 255, 0.1)',
            border: 'none',
            borderRadius: '4px',
            color: '#999',
            cursor: 'pointer',
            padding: '0.25rem',
            transition: 'all 0.2s ease'
          }}
        >
          <Minus size={14} />
        </button>
        <div style={{ flex: 1, position: 'relative' }}>
          {recommended !== undefined && (
            <div style={{
              position: 'absolute',
              left: `${((recommended - min) / (max - min)) * 100}%`,
              top: '50%',
              transform: 'translate(-50%, -50%)',
              width: '2px',
              height: '12px',
              background: primaryColor + '60',
              pointerEvents: 'none',
              zIndex: 1
            }} />
          )}
          <input
            type="range"
            value={value}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            min={min}
            max={max}
            step={step}
            style={{
              width: '100%',
              height: '4px',
              background: `linear-gradient(to right, ${primaryColor} 0%, ${primaryColor} ${((value - min) / (max - min)) * 100}%, rgba(255, 255, 255, 0.1) ${((value - min) / (max - min)) * 100}%, rgba(255, 255, 255, 0.1) 100%)`,
              borderRadius: '2px',
              outline: 'none',
              WebkitAppearance: 'none',
              cursor: 'pointer',
              position: 'relative'
            }}
          />
        </div>
        <button
          onClick={() => onChange(Math.min(max, value + step))}
          style={{
            background: 'rgba(255, 255, 255, 0.1)',
            border: 'none',
            borderRadius: '4px',
            color: '#999',
            cursor: 'pointer',
            padding: '0.25rem',
            transition: 'all 0.2s ease'
          }}
        >
          <Plus size={14} />
        </button>
      </div>
    </div>
  );
};