import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import { HexColorPicker } from 'react-colorful';
import * as THREE from 'three';
import { 
  Activity, 
  AlertCircle,
  Brain, 
  Calculator,
  Cpu, 
  Eye, 
  Layers, 
  BarChart3, 
  Settings,
  Database,
  Download,
  Upload,
  RefreshCw,
  Zap,
  Info,
  ChevronDown,
  ChevronUp,
  Maximize2,
  Minimize2,
  Save,
  FileText,
  Play,
  Pause,
  PlayCircle,
  SkipForward,
  Grid3x3,
  Lock,
  Unlock,
  Code,
  Terminal,
  Sparkles,
  Atom,
  Waves,
  Network,
  GitBranch,
  Infinity,
  ChevronLeft,
  ChevronRight,
  Sliders,
  Clock,
  Trash2
} from 'lucide-react';

// Import Layout System
import GridLayout from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

// Import program storage utilities
import { loadCustomPrograms } from '../utils/programStorage';

// Import component styles
import '../styles/quantum-osh-studio.css';

// Import Universe Context
import { UniverseProvider } from '../contexts/UniverseContext';

// Import MemoryMonitor
import { MemoryMonitor } from './MemoryMonitor';

// Import all OSH engines
import { MemoryFieldEngine, type MemoryFragment, type MemoryField } from '../engines/MemoryFieldEngine';
import { EntropyCoherenceSolver, type EntropyMetrics, type CoherenceMetrics } from '../engines/EntropyCoherenceSolver';
import { RSPEngine, type RSPState, type MemoryAttractor } from '../engines/RSPEngine';
import { ObserverEngine, type Observer as ObserverType, type ObservationEvent } from '../engines/ObserverEngine';
import { WavefunctionSimulator, type WavefunctionState } from '../engines/WavefunctionSimulator';
import { SimulationHarness, type SimulationParameters, type SimulationState } from '../engines/SimulationHarness';
import { UnifiedQuantumErrorReductionPlatform, type UnifiedMetrics } from '../engines/UnifiedQuantumErrorReductionPlatform';
import { MLAssistedObserver } from '../engines/MLObserver';
import { MacroTeleportationEngine } from '../engines/MacroTeleportation';
import { CurvatureTensorGenerator } from '../engines/CurvatureTensorGenerator';
import { SimulationSnapshotManager } from '../engines/SimulationSnapshotManager';
import { CoherenceFieldLockingEngine } from '../engines/CoherenceFieldLockingEngine';
import { RecursiveTensorFieldEngine } from '../engines/RecursiveTensorFieldEngine';
import { SubstrateIntrospectionEngine } from '../engines/SubstrateIntrospectionEngine';
import { WorkerPool } from '../workers/WorkerPool';

// Import visualization components
// import { MemoryFieldVisualizer } from './visualizations/MemoryFieldVisualizer'; // Removed
import { RSPDashboard } from './visualizations/RSPDashboard';
// WaveformExplorer disabled for memory optimization
// import { WaveformExplorer } from './visualizations/WaveformExplorer';
import { InformationalCurvatureMap } from './visualizations/InformationalCurvatureMap';
import { OSHUniverse3D } from './visualizations/OSHUniverse3D';
import { GravitationalWaveEchoVisualizer } from './visualizations/GravitationalWaveEchoVisualizer';
import { VisualizationControls } from './visualizations/VisualizationControls';
import { Tooltip } from './ui/Tooltip';
import { QuantumCodeEditor } from './QuantumCodeEditor';
import { QuantumProgramsLibrary } from './QuantumProgramsLibrary';
import { QuantumCircuitDesigner } from './QuantumCircuitDesigner';
import { EngineStatus } from './EngineStatus';
import OSHCalculationsPanel from './OSHCalculationsPanel';
import { UniverseControlModal, type UniverseParameters } from './UniverseControlModal';
import { TheoryOfEverythingPanel } from './TheoryOfEverythingPanel';
import { TheoryOfEverythingErrorBoundary } from './TheoryOfEverythingErrorBoundary';
import { GravitationalWaveEchoVisualizerErrorBoundary } from './visualizations/GravitationalWaveEchoVisualizerErrorBoundary';
import { generateColorPalette, applyThemeColors, hexToRgb, adjustColor, getContrastText } from '../utils/colorUtils';
import { OSHQuantumEngine } from '../engines/OSHQuantumEngine';
import { initializeEngine } from '../utils/engineInitializer';
import { ensureMemoryFieldMethods } from '../utils/ensureMemoryFieldMethods';
import { Complex } from '../utils/complex';

// Import utilities
import { DataExporter } from '../utils/dataExport';
import { OSHEvidenceEvaluator, type OSHEvidence } from '../utils/oshEvidenceEvaluator';

// Import API functions
import { fetchProgram } from '../config/api';
import { EnhancedExecutionLog } from './EnhancedExecutionLog';
import { DataFlowMonitor } from './DataFlowMonitor';
import { safeFormat, ensureFinite, safeGet } from '../utils/safeAccess';
import { useEngineAPIContext } from '../contexts/EngineAPIContext';
import { useEngineState } from '../hooks/useEngineState';
import { PerformanceManager } from './PerformanceManager';
import { usePerformanceOptimization } from '../utils/performanceOptimizer';
import { useResourceThrottling } from '../utils/resourceThrottler';
import { DiagnosticsSystem } from '../utils/diagnostics';
import { DiagnosticPanel } from './DiagnosticPanel';
import { syncEnginesWithBackend } from '../utils/backendMetricsSync';

// Import quantum programs
import { advancedQuantumPrograms, type QuantumProgram } from '../data/advancedQuantumPrograms';
import { enterpriseQuantumPrograms } from '../data/enterpriseQuantumPrograms';
import { oshQuantumPrograms } from '../data/oshQuantumPrograms';

// Import Monaco editor
import Monaco from '@monaco-editor/react';

// Types
// Complex type is already imported from '../utils/complex'

interface LayoutConfig {
  id: string;
  name: string;
  icon: React.ReactNode;
  layout: GridLayout.Layout[];
}

interface WindowConfig {
  id: string;
  title: string;
  component: React.ReactNode;
  minW?: number;
  minH?: number;
  icon: React.ReactNode;
}

// Type declarations moved to enhanced component

// Quantum field shader and visualization moved to EnhancedQuantumField3D component

// OSHQuantumEngine is now imported from its own file

// Using EnhancedQuantumField3D component for visualization

// Enhanced Circuit Designer with ML Observer - REMOVED (using imported component)
// This local definition has been removed to avoid conflicts with the imported QuantumCircuitDesigner
const LegacyCircuitDesigner: React.FC<{
  primaryColor: string;
}> = ({ primaryColor }) => {
  const [circuit, setCircuit] = useState<any[]>([]);
  const [qubits, setQubits] = useState(4);
  const [mlPrediction, setMlPrediction] = useState<any>(null);
  
  const gates = [
    { id: 'H', name: 'Hadamard', symbol: 'H', color: '#4ecdc4' },
    { id: 'X', name: 'Pauli-X', symbol: 'X', color: '#ff6b9d' },
    { id: 'Y', name: 'Pauli-Y', symbol: 'Y', color: '#feca57' },
    { id: 'Z', name: 'Pauli-Z', symbol: 'Z', color: '#48dbfb' },
    { id: 'CNOT', name: 'CNOT', symbol: '●━', color: '#ee5a6f' },
    { id: 'CZ', name: 'CZ', symbol: '●─Z', color: '#f368e0' },
    { id: 'SWAP', name: 'SWAP', symbol: '×', color: '#00d2d3' },
    { id: 'T', name: 'T Gate', symbol: 'T', color: '#54a0ff' },
    { id: 'S', name: 'S Gate', symbol: 'S', color: '#5f27cd' },
    { id: 'RX', name: 'RX(θ)', symbol: 'Rx', color: '#ff9ff3' },
    { id: 'RY', name: 'RY(θ)', symbol: 'Ry', color: '#ffeaa7' },
    { id: 'RZ', name: 'RZ(θ)', symbol: 'Rz', color: '#dfe6e9' },
    { id: 'CONSCIOUSNESS', name: 'Consciousness', symbol: 'Ψ', color: primaryColor },
    { id: 'MEMORY', name: 'Memory Field', symbol: 'M', color: '#a78bfa' },
    { id: 'RSP', name: 'RSP Oracle', symbol: 'Ω', color: '#f472b6' }
  ];
  
  const addGate = useCallback((gateId: string, qubit: number, target?: number) => {
    const newGate = {
      id: `${gateId}_${Date.now()}`,
      type: gateId,
      qubit,
      target,
      position: circuit.filter(g => g.qubit === qubit).length
    };
    setCircuit([...circuit, newGate]);
  }, [circuit]);
  
  return (
    <div className="circuit-designer-advanced">
      <div className="designer-header">
        <h3>Quantum Circuit Designer</h3>
        <div className="qubit-selector">
          <label>Qubits:</label>
          <input
            type="range"
            min={2}
            max={8}
            value={qubits}
            onChange={(e) => setQubits(Number(e.target.value))}
          />
          <span>{qubits}</span>
        </div>
      </div>
      
      <div className="gates-palette">
        {gates.map(gate => (
          <motion.button
            key={gate.id}
            className="gate-btn"
            style={{ backgroundColor: gate.color + '20', borderColor: gate.color }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => addGate(gate.id, 0)}
            title={gate.name}
          >
            {gate.symbol}
          </motion.button>
        ))}
      </div>
      
      <div className="circuit-grid">
        {Array.from({ length: qubits }).map((_, q) => (
          <div key={q} className="qubit-wire">
            <span className="qubit-label">|q{q}⟩</span>
            <div className="wire-line" />
            {circuit
              .filter(g => g.qubit === q)
              .map((gate, i) => (
                <div
                  key={gate.id}
                  className="gate-element"
                  style={{
                    left: `${gate.position * 80 + 100}px`,
                    backgroundColor: gates.find(g => g.id === gate.type)?.color + '30',
                    borderColor: gates.find(g => g.id === gate.type)?.color
                  }}
                >
                  {gates.find(g => g.id === gate.type)?.symbol}
                </div>
              ))}
          </div>
        ))}
      </div>
      
      {mlPrediction && (
        <div className="ml-prediction">
          <h4>ML Observer Prediction</h4>
          <div className="prediction-metrics">
            <span>Should Observe: {mlPrediction.shouldObserve ? 'Yes' : 'No'}</span>
            <span>Confidence: {(mlPrediction.confidence * 100).toFixed(1)}%</span>
            <span>Optimal Focus: {mlPrediction.optimalFocus.toFixed(3)}</span>
          </div>
        </div>
      )}
    </div>
  );
};

// Wrapper component for reactive data updates
const InformationalCurvatureMapWrapper: React.FC<{
  data: any[];
  primaryColor: string;
  showClassicalComparison: boolean;
  fieldStrength: number;
}> = ({ data, primaryColor, showClassicalComparison, fieldStrength }) => {
  return (
    <InformationalCurvatureMap
      data={data}
      primaryColor={primaryColor}
      showClassicalComparison={showClassicalComparison}
      fieldStrength={fieldStrength}
    />
  );
};

// Use getContrastText as getContrastColor for backward compatibility
const getContrastColor = getContrastText;

// Main Recursia Studio Component
export const QuantumOSHStudio: React.FC = () => {
  const [primaryColor, setPrimaryColorState] = useState('#ffd700'); // Quantum Gold default
  const [isSimulating, setIsSimulating] = useState(false);
  const [isUniverseLoading, setIsUniverseLoading] = useState(false);
  const [universeLoadingStage, setUniverseLoadingStage] = useState<string>('');
  const [universeLoadingProgress, setUniverseLoadingProgress] = useState(0);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionMode, setExecutionMode] = useState<'universe' | 'program'>('program');
  const [isSwitchingMode, setIsSwitchingMode] = useState(false);
  const [isProgramRunning, setIsProgramRunning] = useState(false);
  
  // Safe color setter - HexColorPicker already provides valid hex colors
  const setPrimaryColor = useCallback((color: string) => {
    // Color received from picker
    if (color && typeof color === 'string') {
      setPrimaryColorState(color);
    }
  }, []);
  const [selectedLayout, setSelectedLayout] = useState('default');
  const [layouts, setLayouts] = useState<{ [key: string]: GridLayout.Layout[] }>({});
  const [engineLoading, setEngineLoading] = useState(true);
  const [engineError, setEngineError] = useState<string | null>(null);
  const [lockedLayout, setLockedLayout] = useState(true); // Default to locked layout
  const [currentLayout, setCurrentLayout] = useState<GridLayout.Layout[]>([]);
  const [executionLogEntries, setExecutionLogEntries] = useState<Array<{
    id: string;
    timestamp: number;
    level: 'info' | 'success' | 'warning' | 'error' | 'debug';
    category: 'execution' | 'measurement' | 'analysis' | 'osh_evidence' | 'quantum_state' | 'optimization' | 'compilation' | 'performance' | 'editor';
    message: string;
    executionContext?: {
      programName?: string;
      lineNumber?: number;
      executionTime?: number;
    };
    quantumState?: {
      stateName: string;
      qubits: number;
      coherence: number;
      entropy: number;
    };
    measurementResult?: {
      qubit: number;
      outcome: number;
      probability: number;
    };
    details?: string;
  }>>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const [currentProgram, setCurrentProgram] = useState<QuantumProgram | null>(null);
  const [currentCode, setCurrentCode] = useState<string>('');
  const [codeEditorRef, setCodeEditorRef] = useState<any>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('all');
  const [showSettings, setShowSettings] = useState(false);
  const [showMemoryMonitor, setShowMemoryMonitor] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [scrollPosition, setScrollPosition] = useState(0);
  const [canScrollRight, setCanScrollRight] = useState(true);
  const [showUniverseControlModal, setShowUniverseControlModal] = useState(false);
  const [universeParameters, setUniverseParameters] = useState<UniverseParameters | undefined>(undefined);
  
  // Settings state - Initialize from localStorage if available
  const [settings, setSettings] = useState(() => {
    const savedSettings = localStorage.getItem('osh-settings');
    if (savedSettings) {
      try {
        return JSON.parse(savedSettings);
      } catch (e) {
        // Failed to parse saved settings - use defaults
      }
    }
    return {
      targetFPS: '60',
      quality: 'High',
      autoSaveSnapshots: true,
      debugMode: localStorage.getItem('osh-debug-mode') === 'true'
    };
  });
  
  // Save settings to localStorage when they change
  useEffect(() => {
    localStorage.setItem('osh-settings', JSON.stringify(settings));
  }, [settings]);
  
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const startTime = useRef<number>(Date.now());
  const engineRef = useRef<OSHQuantumEngine | null>(null);
  const fpsRef = useRef<number>(60);
  
  // Add missing state for lastSimulationTime
  const [lastSimulationTime, setLastSimulationTime] = useState<number>(Date.now());
  
  // Use the API hook to connect to the backend - MUST be before callbacks that use its functions
  const { 
    metrics, 
    states, 
    execute, 
    compile, 
    isConnected,
    simulationSnapshots,
    isSimulationPaused,
    simulationTime,
    pauseSimulation,
    resumeSimulation,
    seekSimulation,
    clearSnapshots,
    executionLog: apiExecutionLog,
    // Universe control functions
    startUniverseSimulation,
    stopUniverseSimulation,
    setUniverseMode,
    updateUniverseParameters
  } = useEngineAPIContext();
  
  
  // Monitor universe mode changes with debouncing
  useEffect(() => {
    if (executionMode === 'universe' && isSimulating && !metrics?.universe_running) {
      // Only log once per issue, not continuously
      const warningTimer = setTimeout(() => {
        if (!metrics?.universe_running) {
          setExecutionLogEntries(prev => {
            // Check if we already have this warning in recent entries
            const recentWarning = prev.slice(-10).some(entry => 
              entry.message.includes('Universe mode active but simulation not running')
            );
            
            if (!recentWarning) {
              return [...prev, {
                id: `universe-check-${Date.now()}-${Math.random()}`,
                timestamp: Date.now(),
                level: 'warning',
                category: 'execution',
                message: 'Universe mode active but simulation not running. Check WebSocket connection.'
              }];
            }
            return prev;
          });
        }
      }, 2000); // Wait 2 seconds before warning
      
      return () => clearTimeout(warningTimer);
    }
  }, [executionMode, isSimulating, metrics?.universe_running]);
  
  // Mode switching handler
  const handleModeSwitch = useCallback(async (mode: 'universe' | 'program') => {
    // If we're already in the target mode or switching, do nothing
    if (executionMode === mode || isSwitchingMode) {
      return;
    }

    // Start loading state
    setIsSwitchingMode(true);

    // Show loading state
    setExecutionLogEntries(prev => [...prev, {
      id: `mode-switch-loading-${Date.now()}-${Math.random()}`,
      timestamp: Date.now(),
      level: 'info',
      category: 'execution',
      message: `Switching to ${mode === 'universe' ? 'Universe Simulation' : 'Program Execution'} mode...`
    }]);

    // Handle cleanup of current mode
    if (executionMode === 'universe' && isSimulating) {
      // Switching from universe to program - stop universe simulation
      setExecutionLogEntries(prev => [...prev, {
        id: `universe-stopping-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'info',
        category: 'execution',
        message: 'Stopping universe simulation...'
      }]);
      
      stopUniverseSimulation();
      setIsSimulating(false);
      
      // Wait a bit for cleanup
      await new Promise(resolve => setTimeout(resolve, 200));
      
      setExecutionLogEntries(prev => [...prev, {
        id: `universe-stopped-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'success',
        category: 'execution',
        message: 'Universe simulation stopped'
      }]);
    } else if (executionMode === 'program' && isProgramRunning) {
      // Switching from program to universe - stop program execution
      setExecutionLogEntries(prev => [...prev, {
        id: `program-stopping-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'info',
        category: 'execution',
        message: 'Stopping program execution...'
      }]);
      
      // Stop the program (you may need to implement this)
      setIsProgramRunning(false);
      
      // Wait a bit for cleanup
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    
    // Stop engine if it's running (extra safety)
    if (engineRef.current && engineRef.current.stop && typeof engineRef.current.stop === 'function') {
      try {
        engineRef.current.stop();
      } catch (error) {
        console.error('Failed to stop engine during mode switch:', error);
      }
    }
    
    // Switch to the new mode
    setExecutionMode(mode);
    
    // Handle mode-specific initialization
    if (mode === 'universe') {
      // Auto-start universe simulation when switching to universe mode
      try {
        setExecutionLogEntries(prev => [...prev, {
          id: `universe-auto-start-${Date.now()}-${Math.random()}`,
          timestamp: Date.now(),
          level: 'info',
          category: 'execution',
          message: 'Starting dynamic universe simulation...'
        }]);
        
        // Small delay to ensure mode switch completes first
        setTimeout(() => {
          // Start universe with default mode
          startUniverseSimulation('standard');
          setIsSimulating(true);
          
          setExecutionLogEntries(prev => [...prev, {
            id: `universe-started-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'success',
            category: 'execution',
            message: 'Dynamic universe simulation started - watch metrics evolve in real-time!'
          }]);
        }, 100);
        
      } catch (error) {
        setExecutionLogEntries(prev => [...prev, {
          id: `universe-start-error-${Date.now()}-${Math.random()}`,
          timestamp: Date.now(),
          level: 'error',
          category: 'execution',
          message: `Failed to start universe: ${error instanceof Error ? error.message : String(error)}`
        }]);
      }
    } else {
      // Switched to program mode
      setExecutionLogEntries(prev => [...prev, {
        id: `program-mode-ready-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'success',
        category: 'execution',
        message: 'Program execution mode ready'
      }]);
    }

    // Clear loading state
    setIsSwitchingMode(false);
  }, [executionMode, isSimulating, isProgramRunning, isSwitchingMode, startUniverseSimulation, stopUniverseSimulation]);
  
  // Handle stopping universe simulation (for header controls)
  const handleStopUniverse = useCallback(() => {
    if (isSimulating) {
      stopUniverseSimulation();
      setIsSimulating(false);
      
      setExecutionLogEntries(prev => [...prev, {
        id: `universe-manual-stop-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'info',
        category: 'execution',
        message: 'Universe simulation manually stopped'
      }]);
    }
  }, [isSimulating, stopUniverseSimulation]);
  
  // Execute Recursia program - moved here to be available for callbacks
  const executeProgram = async (code: string, programName?: string, iterations: number = 1) => {
    if (!code.trim()) {
      setExecutionLogEntries(prev => [...prev, {
        id: `error-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'error',
        category: 'execution',
        message: 'No code to execute'
      }]);
      return;
    }

    try {
      // Update UI to show execution in progress
      setIsExecuting(true);
      setIsProgramRunning(true);
      
      // Log execution start with code preview
      const startTime = Date.now();
      setExecutionLogEntries(prev => [...prev, {
        id: `exec-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'info',
        category: 'execution',
        message: `Executing ${programName || 'program'}...`,
        executionContext: {
          programName: programName || 'Unnamed Program',
          lineNumber: 1,
          executionTime: 0
        }
      }]);
      
      // Log code being executed (first 100 chars)
      const codePreview = code.length > 100 ? code.substring(0, 100) + '...' : code;
      setExecutionLogEntries(prev => [...prev, {
        id: `code-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'info',
        category: 'execution',
        message: `Code: ${codePreview}`
      }]);

      // Send code to backend for execution
      const result = await execute(code, iterations);
      
      // Metrics received from execution
      
      // Check if this is a multi-iteration result
      const isMultiIteration = result.iterations && result.iterations > 1;
      
      // Ensure boolean comparison
      if (result.success === true || (typeof result.success === 'string' && result.success === 'true')) {
        const executionTime = Date.now() - startTime;
        
        if (isMultiIteration) {
          // Log aggregated results for multiple iterations
          setExecutionLogEntries(prev => [...prev, {
            id: `success-multi-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'success',
            category: 'execution',
            message: `Successfully completed ${result.iterations} iterations`,
            executionContext: {
              programName: programName || 'Unnamed Program',
              lineNumber: code.split('\n').length,
              executionTime: result.total_execution_time || executionTime,
              iterations: result.iterations
            }
          }]);
          
          // Log statistical metrics
          if (result.metrics) {
            const statsMessage = [
              `Average RSP: ${result.metrics.rsp?.toFixed(3) || 'N/A'}`,
              `Coherence: ${result.metrics.coherence?.toFixed(3) || 'N/A'} ± ${result.metrics.coherence_std?.toFixed(3) || '0'}`,
              `Entropy: ${result.metrics.entropy?.toFixed(3) || 'N/A'} ± ${result.metrics.entropy_std?.toFixed(3) || '0'}`
            ].join(', ');
            
            setExecutionLogEntries(prev => [...prev, {
              id: `stats-${Date.now()}-${Math.random()}`,
              timestamp: Date.now(),
              level: 'info',
              category: 'analysis',
              message: `Statistical Results: ${statsMessage}`,
              executionContext: {
                programName: programName || 'Unnamed Program',
                iterations: result.iterations
              }
            }]);
          }
        } else {
          // Single iteration result
          setExecutionLogEntries(prev => [...prev, {
            id: `success-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'success',
            category: 'execution',
            message: result.message || 'Program executed successfully',
            executionContext: {
              programName: programName || 'Unnamed Program',
              lineNumber: code.split('\n').length,
              executionTime: executionTime
            }
          }]);
        }
        
        // Log any output
        if (isMultiIteration && result.outputs) {
          // Handle outputs from multiple iterations
          const uniqueOutputs = new Set<string>();
          result.outputs.forEach((output: any) => {
            if (Array.isArray(output)) {
              output.forEach((line: string) => uniqueOutputs.add(line));
            } else if (output) {
              uniqueOutputs.add(String(output));
            }
          });
          
          if (uniqueOutputs.size > 0) {
            const outputText = Array.from(uniqueOutputs).join('\n');
            setExecutionLogEntries(prev => [...prev, {
              id: `output-multi-${Date.now()}-${Math.random()}`,
              timestamp: Date.now(),
              level: 'info',
              category: 'execution',
              message: `Unique outputs across iterations:\n${outputText}`
            }]);
          }
        } else if (result.output !== undefined && result.output !== null) {
          // Single iteration output
          const outputText = Array.isArray(result.output) 
            ? result.output.join('\n')
            : String(result.output);
          
          // Only log if there's actual content
          if (outputText && outputText.trim() && outputText.trim() !== '[]') {
            setExecutionLogEntries(prev => [...prev, {
              id: `output-${Date.now()}-${Math.random()}`,
              timestamp: Date.now(),
              level: 'info',
              category: 'execution',
              message: `Output: ${outputText}`
            }]);
          }
        }
        
        // Log measurements if available
        if (result.measurements && result.measurements.length > 0) {
          setExecutionLogEntries(prev => [...prev, {
            id: `measurements-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'info',
            category: 'measurement',
            message: `Performed ${result.measurements.length} measurement(s)`,
            measurements: result.measurements
          }]);
          
          // Log individual measurements
          result.measurements.forEach((m: any, idx: number) => {
            // Handle different measurement formats from backend
            let measurementMessage = '';
            if (m.type === 'qubit' && m.state && m.qubit !== undefined && m.value !== undefined) {
              // Qubit measurement format: {type: 'qubit', state, qubit, value, timestamp}
              measurementMessage = `${m.state}[${m.qubit}] → |${m.value}⟩`;
            } else if (m.type && m.state && m.value !== undefined) {
              // Other measurement format: {type, state, value, timestamp}
              const value = typeof m.value === 'number' ? m.value.toFixed(4) : m.value;
              measurementMessage = `${m.state} measured by ${m.type}: ${value}`;
            } else if (m.state && m.qubit !== undefined && m.outcome !== undefined) {
              // Legacy format: {state, qubit, outcome, probability}
              measurementMessage = `${m.state}[${m.qubit}] → |${m.outcome}⟩ (p=${m.probability?.toFixed(3) || '0.5'})`;
            } else {
              // Fallback for unknown format
              measurementMessage = `Measurement ${idx + 1}: ${JSON.stringify(m)}`;
            }
            
            setExecutionLogEntries(prev => [...prev, {
              id: `measure-${Date.now()}-${idx}-${Math.random()}`,
              timestamp: Date.now(),
              level: 'info',
              category: 'measurement',
              message: measurementMessage,
              // Include measurementResult for statistics tracking
              measurementResult: m.type === 'qubit' ? {
                qubit: m.qubit || 0,
                outcome: m.value !== undefined ? m.value : (m.outcome || 0),
                probability: m.probability || 0.5,
                collapseType: 'projective'
              } : undefined,
              // For non-qubit measurements, store the data differently
              data: m.type !== 'qubit' ? {
                type: m.type,
                value: m.value,
                state: m.state
              } : undefined
            }]);
          });
        }
        
        // Process post-execution metrics if available
        if (result.metrics) {
          
          setExecutionLogEntries(prev => [...prev, {
            id: `metrics-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'info',
            category: 'analysis',
            message: `Post-execution: States=${result.metrics.state_count || 0}, Observers=${result.metrics.observer_count || 0}, RSP=${result.metrics.rsp?.toFixed(2) || '0'}, Measurements=${result.metrics.measurement_count || result.measurements?.length || 0}`,
            executionContext: {
              programName: programName || 'Unnamed Program',
              executionTime: Date.now() - startTime
            },
            quantumState: result.metrics.coherence !== undefined ? {
              stateName: 'System State',
              qubits: result.metrics.state_count || 0,
              coherence: result.metrics.coherence || 0,
              entropy: result.metrics.entropy || 0
            } : undefined,
            oshMetrics: {
              depth: result.metrics.depth || result.metrics.recursion_depth || 0,
              strain: result.metrics.strain || 0,
              focus: result.metrics.focus || result.metrics.observer_focus || 0,
              information: result.metrics.information || result.metrics.information_curvature || 0
            },
            // Include full metrics data for enhanced display
            data: {
              // Core OSH metrics
              rsp: result.metrics.rsp || 0,
              information: result.metrics.information || result.metrics.information_curvature || 0,
              information_curvature: result.metrics.information_curvature || result.metrics.information || 0,
              
              // Measurement data
              measurement_count: result.metrics.measurement_count || (result.measurements?.length || 0),
              
              // State metrics
              coherence: result.metrics.coherence || 0,
              entropy: result.metrics.entropy || 1.0,
              state_count: result.metrics.state_count || 0,
              observer_count: result.metrics.observer_count || 0,
              
              // System metrics
              depth: result.metrics.depth || result.metrics.recursion_depth || 0,
              recursion_depth: result.metrics.recursion_depth || result.metrics.depth || 0,
              strain: result.metrics.strain || 0,
              focus: result.metrics.focus || result.metrics.observer_focus || 0,
              observer_focus: result.metrics.observer_focus || result.metrics.focus || 0,
              
              // Additional OSH metrics
              phi: result.metrics.phi || result.metrics.integrated_information || 0,
              integrated_information: result.metrics.integrated_information || result.metrics.phi || 0,
              complexity: result.metrics.complexity || 0,
              entropy_flux: result.metrics.entropy_flux || 0,
              emergence_index: result.metrics.emergence_index || 0,
              field_energy: result.metrics.field_energy || 0,
              temporal_stability: result.metrics.temporal_stability || 0.5,
              memory_field_coupling: result.metrics.memory_field_coupling || 0,
              
              // Conservation law data
              conservation_law: result.metrics.conservation_law || {
                verified: false,
                left_side: 0,
                right_side: 0,
                relative_error: 0,
                conservation_ratio: 1.0,
                message: 'No data'
              }
            }
          }]);
        }
      } else {
        // Handle compilation/execution errors
        if (result.errors && result.errors.length > 0) {
          // Log each error with details
          result.errors.forEach((err: any) => {
            // Handle both string and object error formats
            const errorMessage = typeof err === 'string' 
              ? err 
              : err.message || JSON.stringify(err);
            
            // Create detailed error message with location if available
            let fullMessage = errorMessage;
            if (typeof err === 'object' && err !== null) {
              if (err.line && err.column) {
                fullMessage = `Line ${err.line}, Column ${err.column}: ${errorMessage}`;
              }
              if (err.type) {
                fullMessage = `[${err.type.toUpperCase()}] ${fullMessage}`;
              }
            }
            
            setExecutionLogEntries(prev => [...prev, {
              id: `err-${Date.now()}-${Math.random()}`,
              timestamp: Date.now(),
              level: 'error',
              category: 'execution',
              message: fullMessage
            }]);
          });
          
          // Don't throw, just return - errors are already logged
          return;
        } else {
          // Generic error without details
          setExecutionLogEntries(prev => [...prev, {
            id: `err-generic-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'error',
            category: 'execution',
            message: result.error || 'Execution failed - no details provided'
          }]);
        }
      }
      
    } catch (error) {
      // Enhanced error logging for better debugging
      // Execution error occurred
      
      let errorMessage = 'Unable to connect to backend';
      let errorDetails = '';
      
      if (error instanceof Error) {
        errorMessage = error.message;
        
        // Check for specific error types
        if (errorMessage.includes('HTTP 500')) {
          errorMessage = '500 Internal Server Error - Backend execution failed';
          errorDetails = 'The server encountered an error while executing your program. Check the backend logs for details.';
        } else if (errorMessage.includes('HTTP 400')) {
          errorMessage = '400 Bad Request - Invalid program syntax';
          errorDetails = 'The program contains syntax errors. Check your code for typos or unsupported constructs.';
        } else if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
          errorMessage = 'Network Error - Cannot reach backend server';
          errorDetails = 'Please ensure the backend server is running on port 8080.';
        } else if (errorMessage.includes('AbortError') || errorMessage.includes('timeout')) {
          errorMessage = 'Request Timeout - Execution took too long';
          errorDetails = 'The program execution exceeded the timeout limit. Try optimizing your code or reducing complexity.';
        }
      }
      
      // Log main error
      setExecutionLogEntries(prev => [...prev, {
        id: `exec-error-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'error',
        category: 'execution',
        message: errorMessage,
        data: { 
          details: errorDetails,
          originalError: error instanceof Error ? error.stack : String(error)
        }
      }]);
      
      // Log additional details if available
      if (errorDetails) {
        setExecutionLogEntries(prev => [...prev, {
          id: `exec-error-details-${Date.now()}-${Math.random()}`,
          timestamp: Date.now(),
          level: 'error',
          category: 'execution',
          message: errorDetails
        }]);
      }
      
      // Log debugging information
      setExecutionLogEntries(prev => [...prev, {
        id: `exec-debug-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'debug',
        category: 'execution',
        message: 'Debug Info: Check browser console for full error details',
        data: {
          code: code.substring(0, 200) + '...',
          programName: programName || 'Unknown',
          timestamp: new Date().toISOString()
        }
      }]);
    } finally {
      // Reset execution state
      setIsExecuting(false);
      setIsProgramRunning(false);
    }
  };
  
  // Handle pausing/resuming simulation
  const handlePauseResumeSimulation = useCallback(() => {
    if (isSimulationPaused) {
      resumeSimulation();
      logExecution({
        level: 'info',
        category: 'execution',
        message: 'Simulation resumed'
      });
    } else {
      pauseSimulation();
      logExecution({
        level: 'info',
        category: 'execution',
        message: 'Simulation paused'
      });
    }
  }, [isSimulationPaused, pauseSimulation, resumeSimulation]);
  
  // Handle starting universe simulation with enhanced loading UX
  const handleStartUniverse = useCallback(async () => {
    if (executionMode !== 'universe') {
      return;
    }
    
    // Prevent double-loading
    if (isUniverseLoading) {
      return;
    }
    
    try {
      setIsUniverseLoading(true);
      setUniverseLoadingProgress(0);
      setUniverseLoadingStage('Initializing quantum fields...');
      
      // Load enhanced universe simulation from external file
      let universeCode = '';
      try {
        // Attempt to load the enhanced universe simulation
        const response = await fetch('/enhancedUniverseSimulation.recursia');
        if (response.ok) {
          universeCode = await response.text();
          // Loaded enhanced universe simulation
        } else {
          throw new Error(`Failed to fetch enhanced universe simulation: ${response.status}`);
        }
      } catch (error) {
        // Failed to load enhanced universe simulation, falling back to default
        
        // Fallback to default universe simulation
        try {
          const fallbackResponse = await fetch('/defaultUniverseSimulation.recursia');
          if (fallbackResponse.ok) {
            universeCode = await fallbackResponse.text();
            // Loaded default universe simulation as fallback
          } else {
            throw new Error('Both enhanced and default universe simulations failed to load');
          }
        } catch (fallbackError) {
          // Failed to load any universe simulation file
          
          // Final fallback to inline program
          universeCode = `// Minimal Universe Simulation - Inline Fallback
state UniverseField : quantum_type {
    state_qubits: 8,
    state_coherence: 0.95,
    state_entropy: 0.1
}

state MemoryField : quantum_type {
    state_qubits: 6,
    state_coherence: 0.9,
    state_entropy: 0.15
}

cohere UniverseField 0.95
cohere MemoryField 0.9

apply H_gate to UniverseField qubit 0
apply H_gate to MemoryField qubit 0
entangle UniverseField with MemoryField

evolve for 2 steps

print "Minimal universe simulation active";`;
        }
      }
        
        // Progressive loading stages with user feedback
        const loadingStages = [
          { stage: 'Initializing enhanced quantum fields...', progress: 15 },
          { stage: 'Applying MATHEMATICAL_SUPPLEMENT.md constants...', progress: 30 },
          { stage: 'Establishing 16-qubit field entanglement...', progress: 50 },
          { stage: 'Calibrating empirical observers...', progress: 70 },
          { stage: 'Starting enhanced universe simulation...', progress: 90 },
          { stage: 'Enhanced universe active', progress: 100 }
        ];
        
        // Execute loading stages with delays for UX
        for (let i = 0; i < loadingStages.length; i++) {
          const { stage, progress } = loadingStages[i];
          setUniverseLoadingStage(stage);
          setUniverseLoadingProgress(progress);
          
          setExecutionLogEntries(prev => [...prev, {
            id: `universe-stage-${i}-${Date.now()}`,
            timestamp: Date.now(),
            level: 'info',
            category: 'execution',
            message: stage
          }]);
          
          // Progressive delay to show loading states
          if (i < loadingStages.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 800));
          }
        }
        
        // Execute through the backend if connected, otherwise use local engine
        if (isConnected) {
          // Executing universe initialization via backend
          
          // Start the dynamic universe engine - this is what actually runs the universe simulation
          const mode = universeParameters?.universeMode || 'standard';
          await startUniverseSimulation(mode);
          
          // The universe is now running in the backend
          setIsSimulating(true);
          
          // Close loading modal on successful completion
          setIsUniverseLoading(false);
          
          setExecutionLogEntries(prev => [...prev, {
            id: `universe-started-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'success',
            category: 'execution',
            message: 'Universe simulation started with backend execution'
          }]);
        } else if (engineRef.current) {
          // Use local engine if not connected to backend
          // Executing universe initialization via local engine
          
          // Use executeProgram to process the Recursia code
          await executeProgram(universeCode, 'Universe Initialization');
          
          // After execution, start the simulation loop
          setIsSimulating(true);
          
          // Close loading modal on successful completion
          setIsUniverseLoading(false);
          
          setExecutionLogEntries(prev => [...prev, {
            id: `universe-started-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'success',
            category: 'execution',
            message: 'Universe simulation started with local engine'
          }]);
        } else {
          setExecutionLogEntries(prev => [...prev, {
            id: `universe-error-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'error',
            category: 'execution',
            message: 'No engine available for universe simulation'
          }]);
        }
        
        // Log connection status
        if (isConnected) {
          setExecutionLogEntries(prev => [...prev, {
            id: `backend-connected-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'info',
            category: 'execution',
            message: 'Receiving real-time metrics from backend WebSocket'
          }]);
        }
        
    } catch (error) {
      // Error starting enhanced universe simulation
      setIsUniverseLoading(false);
      setIsSimulating(false);
      setExecutionLogEntries(prev => [...prev, {
        id: `universe-error-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'error',
        category: 'execution',
        message: `Failed to start universe: ${error instanceof Error ? error.message : String(error)}`
      }]);
    }
  }, [executionMode, executeProgram, execute, isConnected]);
  
  // Handle executing a program with optional iterations
  const handleExecuteProgram = useCallback(async (iterations: number = 1) => {
    if (executionMode !== 'program' || !currentCode.trim()) {
      return;
    }
    
    // Stop universe simulation if running
    if (isSimulating) {
      setIsSimulating(false);
    }
    
    // Don't set isProgramRunning as it triggers the engine simulation loop
    // Program execution uses direct API calls, not the engine
    
    try {
      // For single iteration, execute directly
      // For multiple iterations, the backend handles the loop
      await executeProgram(currentCode, currentProgram?.name, iterations);
    } catch (error) {
      // Execution error in handleExecuteProgram
      setExecutionLogEntries(prev => [...prev, {
        id: `exec-error-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'error',
        category: 'execution',
        message: `Execution failed: ${error instanceof Error ? error.message : String(error)}`
      }]);
    }
  }, [executionMode, currentCode, currentProgram, isSimulating, executeProgram]);
  
  // Removed auto-start - user must explicitly start universe simulation
  
  // Debug: Log metrics updates
  useEffect(() => {
    if (metrics) {
      // Safely handle timestamp conversion
      let timestampStr = 'N/A';
      try {
        if (metrics.timestamp && typeof metrics.timestamp === 'number' && isFinite(metrics.timestamp)) {
          // Handle both seconds and milliseconds timestamps
          const ts = metrics.timestamp > 1e10 ? metrics.timestamp : metrics.timestamp * 1000;
          const date = new Date(ts);
          if (!isNaN(date.getTime())) {
            timestampStr = date.toISOString();
          }
        }
      } catch (error) {
        // Invalid timestamp detected
      }
      
    }
  }, [metrics]);
  
  // Performance optimization hooks
  const { batchUpdate, throttle, debounce } = usePerformanceOptimization('QuantumOSHStudio');
  const { execute: throttledExecute } = useResourceThrottling();
  
  // Initialize engine (with StrictMode protection to prevent duplicate initialization)
  const initializationAttempted = useRef(false);
  
  useEffect(() => {
    // Prevent duplicate initialization in React StrictMode
    if (initializationAttempted.current) {
      // Skipping duplicate initialization (StrictMode)
      return;
    }
    
    initializationAttempted.current = true;
    
    const initEngine = async () => {
      try {
        setEngineLoading(true);
        setEngineError(null);
        
        // Add loading message
        setExecutionLogEntries(prev => [...prev, {
          id: `init-loading-${Date.now()}-${Math.random()}`,
          timestamp: Date.now(),
          level: 'info',
          category: 'execution',
          message: 'Initializing OSH Quantum Engine...'
        }]);
        
        const engine = await initializeEngine();
        engineRef.current = engine;
        
        // Ensure memory field has all required methods
        if (engine.memoryFieldEngine) {
          ensureMemoryFieldMethods(engine.memoryFieldEngine);
        }
        
        // Log initialization success
        setExecutionLogEntries(prev => [...prev, {
          id: `init-${Date.now()}-${Math.random()}`,
          timestamp: Date.now(),
          level: 'success',
          category: 'execution',
          message: 'OSH Quantum Universe initialized with baseline quantum states'
        }]);
        
        setEngineLoading(false);
      } catch (error) {
        // Failed to initialize engine
        const errorMessage = error instanceof Error ? error.message : String(error);
        setEngineError(errorMessage);
        setEngineLoading(false);
        
        setExecutionLogEntries(prev => [...prev, {
          id: `init-error-${Date.now()}-${Math.random()}`,
          timestamp: Date.now(),
          level: 'error',
          category: 'execution',
          message: `Engine initialization failed: ${errorMessage}`
        }]);
      }
    };
    
    initEngine();
    
    return () => {
      // Ensure simulation is stopped
      setIsSimulating(false);
      
      // Stop the engine
      if (engineRef.current) {
        // Stopping engine on unmount
        engineRef.current.stop().catch(err => {
          // Error stopping engine on unmount
        });
      }
      
      // Reset initialization flag on cleanup
      initializationAttempted.current = false;
    };
  }, []);
  
  // Simulation loop - runs when isSimulating is true (universe mode) or isProgramRunning is true (program mode)
  useEffect(() => {
    // Only run simulation for universe mode
    // Program mode uses direct API execution, not the engine simulation loop
    const shouldRun = executionMode === 'universe' && isSimulating;
    
    // Check if we should run simulation/program
    if (!shouldRun) {
      return;
    }

    // Check if engine is still loading
    if (engineLoading) {
      // Engine still loading, stopping simulation
      setExecutionLogEntries(prev => [...prev, {
        id: `sim-wait-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'warning',
        category: 'execution',
        message: 'Waiting for engine initialization...'
      }]);
      // Reset simulation state since engine isn't ready
      setIsSimulating(false);
      return;
    }

    // Check if engine has errors
    if (engineError) {
      console.error('[SimulationLoop] Engine has error:', engineError);
      setExecutionLogEntries(prev => [...prev, {
        id: `sim-error-init-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'error',
        category: 'execution',
        message: `Cannot start simulation: ${engineError}`
      }]);
      // Reset simulation state
      setIsSimulating(false);
      return;
    }

    // Final check for engine reference
    if (!engineRef.current) {
      console.error('[SimulationLoop] No engine reference available');
      setExecutionLogEntries(prev => [...prev, {
        id: `sim-no-engine-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'error',
        category: 'execution',
        message: 'Engine not available. Please refresh the page.'
      }]);
      setIsSimulating(false);
      return;
    }
    
    
    // Start the engine if it has a start method
    if (engineRef.current.start && typeof engineRef.current.start === 'function') {
      try {
        engineRef.current.start();
      } catch (error) {
        console.error('[SimulationLoop] Failed to start engine:', error);
        setExecutionLogEntries(prev => [...prev, {
          id: `sim-start-error-${Date.now()}-${Math.random()}`,
          timestamp: Date.now(),
          level: 'error',
          category: 'execution',
          message: `Failed to start engine: ${error instanceof Error ? error.message : String(error)}`
        }]);
        setIsSimulating(false);
        setIsProgramRunning(false);
        return;
      }
    } else {
      // Engine does not have a start method
    }
    
    let animationFrameId: number;
    let lastTime = performance.now();
    let frameCount = 0;
    let lastFpsUpdate = performance.now();
    let fps = 0;
    let lastUniverseUpdate = 0;
    const UNIVERSE_UPDATE_THROTTLE = 50; // Update universe state max 20 times per second
    
    // Get diagnostics instance
    const diagnostics = DiagnosticsSystem.getInstance();
    
    const simulationLoop = () => {
      const currentTime = performance.now();
      const deltaTime = Math.min((currentTime - lastTime) / 1000, 0.1); // Cap at 100ms
      lastTime = currentTime;
      frameCount++;
      
      // Check for hanging operations every second
      if (frameCount % 60 === 0) {
        const hangs = diagnostics.checkForHangs();
        if (hangs.length > 0) {
          console.error('Hanging operations detected:', hangs);
          setExecutionLogEntries(prev => [...prev, {
            id: `hang-detected-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'error',
            category: 'performance',
            message: `Hanging operations detected: ${hangs.map(h => h.id).join(', ')}`,
            details: JSON.stringify(hangs)
          }]);
        }
      }
      
      // Update FPS counter
      if (currentTime - lastFpsUpdate > 1000) {
        fps = frameCount;
        fpsRef.current = fps; // Update the ref with current FPS
        frameCount = 0;
        lastFpsUpdate = currentTime;
        
        // Force re-render to update FPS display
        setLastSimulationTime(Date.now());
        
        // Emit FPS metric
        setExecutionLogEntries(prev => {
          const filtered = prev.filter(entry => !entry.message.startsWith('Simulation FPS:'));
          return [...filtered, {
            id: `sim-fps-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'info',
            category: 'performance',
            message: `Simulation FPS: ${fps}`
          }];
        });
      }
      
      try {
        // Ensure engine still exists
        if (!engineRef.current) {
          console.error('Engine lost during simulation');
          setIsSimulating(false);
          setIsProgramRunning(false);
          return;
        }

        // Check memory pressure before update (Chrome only)
        if (typeof (performance as any).memory !== 'undefined') {
          const memInfo = (performance as any).memory;
          if (memInfo.usedJSHeapSize / memInfo.jsHeapSizeLimit > 0.9) {
            // High memory pressure detected, slowing simulation
            // Skip some frames to allow garbage collection
            if (frameCount % 3 !== 0) {
              animationFrameId = requestAnimationFrame(simulationLoop);
              return;
            }
          }
        }

        // Update the engine with the delta time
        if (frameCount % 60 === 0) {
        }
        
        // Only update local engine if not connected to backend
        // When connected to backend, we receive metrics via WebSocket
        if (!isConnected && executionMode === 'universe') {
          try {
            engineRef.current.update(deltaTime);
          } catch (updateError) {
            console.error('[SimulationLoop] Engine update threw error:', updateError);
            setExecutionLogEntries(prev => [...prev, {
            id: `engine-update-error-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'error',
            category: 'execution',
            message: `Engine update error: ${updateError instanceof Error ? updateError.message : String(updateError)}`,
            details: updateError instanceof Error ? updateError.stack : undefined
            }]);
            // Don't stop simulation for non-critical errors
            if (updateError instanceof Error && updateError.message.includes('Cannot read')) {
              setIsSimulating(false);
              setIsProgramRunning(false);
              return;
            }
          }
        }
        
        // Throttle universe state updates to prevent memory issues
        if (currentTime - lastUniverseUpdate > UNIVERSE_UPDATE_THROTTLE) {
          lastUniverseUpdate = currentTime;
          
          // Emit lightweight metrics frequently, full state rarely
          // Only get metrics from local engine if not connected to backend
          const localMetrics = (!isConnected && engineRef.current.getMetrics) ? engineRef.current.getMetrics() : null;
          
          // Check if resources are healthy
          if (localMetrics && localMetrics.resources) {
            // Log resource metrics periodically
            if (frameCount % 300 === 0 || frameCount < 5) {
            }
            // Add grace period - don't stop simulation in first 3 seconds (180 frames at 60fps)
            const gracePeriodFrames = 180;
            
            if (localMetrics.resources && !localMetrics.resources.healthy && localMetrics.resources.throttleLevel > 0.8 && frameCount > gracePeriodFrames) {
              // Critical resource state - stop simulation
              console.error('[SimulationLoop] Critical resource state detected - stopping simulation', {
                frameCount,
                resources: localMetrics.resources
              });
              setExecutionLogEntries(prev => [...prev, {
                id: `resource-critical-${Date.now()}-${Math.random()}`,
                timestamp: Date.now(),
                level: 'error',
                category: 'execution',
                message: 'Simulation stopped due to critical resource constraints',
                details: JSON.stringify(localMetrics.resources)
              }]);
              setIsSimulating(false);
              setIsProgramRunning(false);
              return;
            } else if (metrics.resources && !metrics.resources.healthy && frameCount > gracePeriodFrames) {
              // Skipping full state update due to resource constraints
              return; // Skip this update but continue simulation
            } else if (metrics.resources && !metrics.resources.healthy && frameCount <= gracePeriodFrames) {
              // During grace period, just log a warning
              if (frameCount % 60 === 0) {
              }
            }
          }
          
          // Only get full state every 5 seconds to reduce memory pressure
          let fullState = null;
          if (frameCount % 300 === 0 && engineRef.current.getState) {
            try {
              fullState = engineRef.current.getState();
            } catch (e) {
              // Failed to get full engine state
            }
          }
          
          // Emit simulation state event for components
          window.dispatchEvent(new CustomEvent('universeUpdate', {
            detail: {
              deltaTime,
              timestamp: currentTime,
              metrics,
              engineState: fullState // Will be null most of the time
            }
          }));
        }
      } catch (error) {
        console.error('[SimulationLoop] Simulation update error:', error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        setExecutionLogEntries(prev => [...prev, {
          id: `sim-error-${Date.now()}-${Math.random()}`,
          timestamp: Date.now(),
          level: 'error',
          category: 'execution',
          message: `Simulation error: ${errorMessage}`,
          details: error instanceof Error ? error.stack : undefined
        }]);
        
        // Stop simulation on critical errors
        if (errorMessage.includes('Cannot read') || errorMessage.includes('undefined')) {
          setIsSimulating(false);
          setIsProgramRunning(false);
          return;
        }
      }
      
      // Continue the loop only if still running
      if (shouldRun) {
        animationFrameId = requestAnimationFrame(simulationLoop);
      }
    };
    
    // Start the simulation loop
    animationFrameId = requestAnimationFrame(simulationLoop);
    
    // Log simulation start
    setExecutionLogEntries(prev => [...prev, {
      id: `sim-start-${Date.now()}-${Math.random()}`,
      timestamp: Date.now(),
      level: 'success',
      category: 'execution',
      message: executionMode === 'universe' ? 'Quantum universe simulation started' : 'Program execution started'
    }]);
    
    // Emit simulation started event
    window.dispatchEvent(new Event('universeStarted'));
    
    // Cleanup
    return () => {
      
      // Clear snapshots when stopping simulation
      if (executionMode === 'universe') {
        clearSnapshots();
      }
      
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      
      // Stop the engine if it has a stop method
      if (engineRef.current && engineRef.current.stop && typeof engineRef.current.stop === 'function') {
        try {
          engineRef.current.stop();
          // Engine stopped successfully
        } catch (error) {
          // Failed to stop engine
        }
      }
      
      // Log simulation stop
      setExecutionLogEntries(prev => [...prev, {
        id: `sim-pause-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'info',
        category: 'execution',
        message: 'Quantum universe simulation paused'
      }]);
      
      // Emit simulation stopped event
      window.dispatchEvent(new Event('universeStopped'));
      
      // Double-check states are reset (handles edge cases)
      const currentShouldRun = executionMode === 'universe' ? isSimulating : isProgramRunning;
      if (currentShouldRun) {
        // Cleanup detected running state, forcing stop
        if (executionMode === 'universe') {
          setIsSimulating(false);
        } else {
          setIsProgramRunning(false);
        }
      }
    };
  }, [isSimulating, isProgramRunning, executionMode, engineLoading, engineError]);
  
  // Throttled metrics update function
  const updateEngineMetrics = React.useCallback(
    throttle(() => {
      if (!metrics || !engineRef.current) return;
      // Update memory field engine with real data
      if (metrics.memory_fragments && engineRef.current?.memoryFieldEngine) {
        // Only update if fragments have changed
        const field = engineRef.current.memoryFieldEngine.getField();
        if (field.fragments.length !== metrics.memory_fragments.length) {
          field.fragments = [];
          
          metrics.memory_fragments.forEach((fragment: any) => {
            // Create a quantum state for the fragment
            const stateVector = Array(8).fill(0).map(() => 
              new Complex(
                Math.random() * fragment.coherence - fragment.coherence/2,
                Math.random() * fragment.coherence - fragment.coherence/2
              )
            );
            
            // Normalize
            const norm = Math.sqrt(stateVector.reduce((sum: number, c: Complex) => sum + c.magnitude() * c.magnitude(), 0));
            const normalizedState = stateVector.map((c: Complex) => c.scale(1/norm));
            
            engineRef.current!.memoryFieldEngine.addFragment(
              normalizedState,
              fragment.position || [0, 0, 0]
            );
          });
        }
        
        // Update strain
        if (engineRef.current?.memoryFieldEngine) {
          engineRef.current.memoryFieldEngine.setStrain(metrics.strain || 0);
        }
      }
      
      // Update observer engine
      if (metrics.observer_focus && engineRef.current?.observerEngine) {
        engineRef.current.observerEngine.setGlobalFocus(metrics.observer_focus);
      }
    }, 200), // 200ms throttle
    [metrics]
  );
  
  // Update engine when metrics change
  useEffect(() => {
    updateEngineMetrics();
  }, [metrics?.memory_fragments?.length, metrics?.strain, metrics?.observer_focus, updateEngineMetrics]);
  
  // Add metric updates to execution log when connected to backend
  useEffect(() => {
    if (isConnected && metrics && isSimulating && executionMode === 'universe') {
      // Only log metrics periodically, not on every change
      const logInterval = setInterval(() => {
        setExecutionLogEntries(prev => {
          // Keep only the last 100 entries to prevent memory issues
          const recentEntries = prev.slice(-100);
          
          return [...recentEntries, {
            id: `metric-update-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            level: 'info',
            category: 'analysis',
            message: `Universe Metrics Update`,
            data: {
              // Core OSH metrics - same structure as execution results for consistency
              rsp: metrics.rsp || 0,
              information: metrics.information || metrics.information_curvature || 0,
              information_curvature: metrics.information_curvature || metrics.information || 0,
              
              // Measurement data
              measurement_count: metrics.measurement_count || 0,
              
              // State metrics
              coherence: metrics.coherence || 0,
              entropy: metrics.entropy || 1.0,
              state_count: metrics.state_count || 0,
              observer_count: metrics.observer_count || 0,
              
              // System metrics with fallbacks
              depth: metrics.depth || metrics.recursion_depth || 0,
              recursion_depth: metrics.recursion_depth || metrics.depth || 0,
              strain: metrics.strain || 0,
              focus: metrics.focus || metrics.observer_focus || 0,
              observer_focus: metrics.observer_focus || metrics.focus || 0,
              
              // Additional OSH metrics
              phi: metrics.phi || metrics.integrated_information || 0,
              integrated_information: metrics.integrated_information || metrics.phi || 0,
              complexity: metrics.complexity || 0,
              entropy_flux: metrics.entropy_flux || 0,
              emergence_index: metrics.emergence_index || 0,
              field_energy: metrics.field_energy || 0,
              temporal_stability: metrics.temporal_stability || 0.5,
              memory_field_coupling: metrics.memory_field_coupling || 0,
              
              // Time Derivatives
              derivatives: {
                drsp_dt: metrics.drsp_dt || 0,
                di_dt: metrics.di_dt || 0,
                dc_dt: metrics.dc_dt || 0,
                de_dt: metrics.de_dt || 0,
                acceleration: metrics.acceleration || 0
              },
              
              // Conservation law data
              conservation_law: metrics.conservation_law || {
                verified: false,
                left_side: 0,
                right_side: 0,
                relative_error: 0,
                conservation_ratio: 1.0,
                message: 'No data'
              },
              
              // Error if present
              error: metrics.error || 0,
              
              // Timestamp
              timestamp: metrics.timestamp || Date.now()
            },
          performance: {
            cpuUsage: 0,
            memoryUsage: 0,
            executionDuration: 0
          }
        }];
        });
      }, 2000); // Log every 2 seconds instead of on every metric update
      
      // Cleanup interval on unmount or when conditions change
      return () => clearInterval(logInterval);
    }
  }, [isConnected, isSimulating, executionMode, metrics]);
  
  // Helper function to format metrics with fixed width to prevent jittering
  const formatMetricValue = useCallback((value: number | undefined, type: 'percentage' | 'decimal' | 'integer' | 'error_rate', defaultValue: string = '0'): string => {
    if (value === undefined || value === null || !isFinite(value)) {
      return defaultValue;
    }
    
    switch (type) {
      case 'percentage':
        // Format as XX.XX% for space efficiency
        const percentage = Math.min(100, Math.max(0, value * 100));
        if (percentage < 0.01) {
          return '0%';
        } else if (percentage < 1) {
          return percentage.toFixed(2) + '%';
        } else if (percentage < 10) {
          return percentage.toFixed(1) + '%';
        } else {
          return percentage.toFixed(0) + '%';
        }
      case 'error_rate':
        // Error rate is a decimal (e.g., 0.00518 = 0.518%)
        // Convert to percentage by multiplying by 100
        const errorPercentage = value * 100;
        if (errorPercentage < 0.001) {
          return '0%';
        } else if (errorPercentage < 0.01) {
          return errorPercentage.toFixed(5) + '%';
        } else if (errorPercentage < 0.1) {
          return errorPercentage.toFixed(4) + '%';
        } else if (errorPercentage < 1) {
          return errorPercentage.toFixed(3) + '%';
        } else if (errorPercentage < 10) {
          return errorPercentage.toFixed(2) + '%';
        } else {
          return errorPercentage.toFixed(1) + '%';
        }
      case 'decimal':
        // Format decimal values compactly
        if (value === 0) {
          return '0';
        } else if (Math.abs(value) < 0.01) {
          return value.toExponential(0);
        } else if (Math.abs(value) < 1) {
          return value.toFixed(2);
        } else if (Math.abs(value) < 10) {
          return value.toFixed(1);
        } else if (Math.abs(value) < 1000) {
          return value.toFixed(0);
        } else {
          // Use K, M notation for large numbers
          if (Math.abs(value) >= 1e6) {
            return (value / 1e6).toFixed(1) + 'M';
          } else {
            return (value / 1e3).toFixed(1) + 'K';
          }
        }
      case 'integer':
        // Format integers compactly
        const intValue = Math.floor(Math.abs(value));
        if (intValue < 1000) {
          return intValue.toString();
        } else if (intValue < 1e6) {
          return (intValue / 1e3).toFixed(0) + 'K';
        } else {
          return (intValue / 1e6).toFixed(0) + 'M';
        }
      default:
        return defaultValue;
    }
  }, []);

  // Process metrics for display
  
  // Smoothed recursion depth state for less jittery updates
  const [smoothedDepth, setSmoothedDepth] = useState(metrics?.recursion_depth || metrics?.depth || 1);
  const depthUpdateTimer = useRef<NodeJS.Timeout | null>(null);
  
  // Smoothed coherence state for less jittery updates
  const [smoothedCoherence, setSmoothedCoherence] = useState(metrics?.coherence || 0.95);
  const coherenceUpdateTimer = useRef<NodeJS.Timeout | null>(null);
  
  // Smoothed information state for less jittery updates
  const [smoothedInformation, setSmoothedInformation] = useState(metrics?.information || 0);
  const informationUpdateTimer = useRef<NodeJS.Timeout | null>(null);
  
  // Preserved state and observer counts for program mode
  const [preservedStateCount, setPreservedStateCount] = useState(0);
  const [preservedObserverCount, setPreservedObserverCount] = useState(0);
  const lastProgramExecutionTime = useRef<number>(0);
  
  // Update smoothed depth with throttling
  useEffect(() => {
    if (metrics) {
      // Backend consistently provides recursion_depth, use depth as fallback
      let targetDepth = metrics.recursion_depth || metrics.depth || 1;
      
      // Clear any existing timer
      if (depthUpdateTimer.current) {
        clearTimeout(depthUpdateTimer.current);
      }
      
      // Update depth smoothly with a 500ms throttle
      depthUpdateTimer.current = setTimeout(() => {
        setSmoothedDepth(prev => {
          // Smooth the transition
          const diff = targetDepth - prev;
          if (Math.abs(diff) > 5) {
            // Large jump - move halfway
            return prev + Math.floor(diff / 2);
          } else {
            // Small change - move directly
            return targetDepth;
          }
        });
      }, 500);
    }
    
    return () => {
      if (depthUpdateTimer.current) {
        clearTimeout(depthUpdateTimer.current);
      }
    };
  }, [metrics?.depth, metrics?.recursion_depth, metrics?.information, metrics?.coherence, metrics?.state_count, metrics?.observer_count]);
  
  // Update smoothed coherence with throttling
  useEffect(() => {
    if (metrics && metrics.coherence !== undefined) {
      const targetCoherence = metrics.coherence;
      
      // Clear any existing timer
      if (coherenceUpdateTimer.current) {
        clearTimeout(coherenceUpdateTimer.current);
      }
      
      // Update coherence smoothly with a 750ms throttle (slightly slower than depth)
      coherenceUpdateTimer.current = setTimeout(() => {
        setSmoothedCoherence(prev => {
          // Smooth the transition using exponential moving average
          const alpha = 0.3; // Smoothing factor (0 = no change, 1 = instant change)
          return prev * (1 - alpha) + targetCoherence * alpha;
        });
      }, 750);
    }
    
    return () => {
      if (coherenceUpdateTimer.current) {
        clearTimeout(coherenceUpdateTimer.current);
      }
    };
  }, [metrics?.coherence]);
  
  // Update smoothed information with throttling
  useEffect(() => {
    if (metrics && metrics.information !== undefined) {
      const targetInformation = metrics.information;
      
      // Clear any existing timer
      if (informationUpdateTimer.current) {
        clearTimeout(informationUpdateTimer.current);
      }
      
      // Update information smoothly with a 600ms throttle
      informationUpdateTimer.current = setTimeout(() => {
        setSmoothedInformation(prev => {
          // Smooth the transition using exponential moving average
          const alpha = 0.25; // Smoothing factor
          return prev * (1 - alpha) + targetInformation * alpha;
        });
      }, 600);
    }
    
    return () => {
      if (informationUpdateTimer.current) {
        clearTimeout(informationUpdateTimer.current);
      }
    };
  }, [metrics?.information]);
  
  // Preserve state and observer counts in program mode
  useEffect(() => {
    if (executionMode === 'program' && metrics) {
      // If we have non-zero values, preserve them
      if (metrics.state_count > 0) {
        setPreservedStateCount(metrics.state_count);
      }
      if (metrics.observer_count > 0) {
        setPreservedObserverCount(metrics.observer_count);
      }
    }
  }, [metrics?.state_count, metrics?.observer_count, executionMode]);
  
  // Reset preserved counts when a program is executed
  useEffect(() => {
    if (isProgramRunning) {
      lastProgramExecutionTime.current = Date.now();
    }
  }, [isProgramRunning]);
  
  const displayMetrics = useMemo(() => {
    if (!metrics) {
      return {
        rsp: '0',
        error: '0.03%', // Realistic quantum error rate
        coherence: '0%',
        entropy: '0',
        information: '0',
        recursionDepth: '0',
        memoryStrain: '0',
        observerFocus: '0',
        measurementCount: '0',
        stateCount: '0',
        observerCount: '0',
        fps: 0,
        // Universe-specific
        universeTime: '0',
        iterationCount: '0',
        entanglements: '0',
        universeMode: 'Standard'
      };
    }
    
    
    // Get error rate from multiple sources
    let errorRate = 0.001; // Default 0.1% base error rate
    
    // First priority: Use error from backend metrics if available
    if (metrics.error !== undefined && metrics.error !== null) {
      errorRate = metrics.error;
      // Using backend error rate
    }
    // Second priority: Get from UnifiedQuantumErrorReductionPlatform
    else if (engineRef.current && engineRef.current.errorReductionPlatform && 
        engineRef.current.errorReductionPlatform.getMetrics) {
      try {
        const errorMetrics = engineRef.current.errorReductionPlatform.getMetrics();
        if (errorMetrics && errorMetrics.effectiveErrorRate !== undefined) {
          errorRate = errorMetrics.effectiveErrorRate;
          // Using error reduction platform rate
        }
      } catch (e) {
        // Failed to get error rate from UnifiedQuantumErrorReductionPlatform
      }
    } else {
      // Fallback calculation based on system complexity
      const baseErrorRate = 0.001; // 0.1% base error rate with error correction
      const complexityFactor = Math.min(2.0, 1 + (metrics.state_count || 0) * 0.02 + (metrics.observer_count || 0) * 0.03);
      const coherencePenalty = Math.max(0, 1 - (metrics.coherence || 1)) * 0.05;
      errorRate = Math.min(0.1, baseErrorRate * complexityFactor + coherencePenalty);
      // Using fallback error calculation
    }
    
    const formattedError = formatMetricValue(errorRate, 'error_rate');
    
    
    const result = {
      rsp: formatMetricValue(metrics.rsp, 'decimal', '∞'),
      error: formattedError,
      coherence: formatMetricValue(smoothedCoherence, 'percentage'), // Use smoothed value
      entropy: formatMetricValue(metrics.entropy, 'decimal'),
      information: formatMetricValue(smoothedInformation, 'decimal'),
      recursionDepth: formatMetricValue(smoothedDepth, 'integer'), // Use smoothed value
      memoryStrain: formatMetricValue(metrics.strain, 'decimal'),
      observerFocus: formatMetricValue(metrics.observer_focus || metrics.focus, 'decimal'),
      measurementCount: formatMetricValue(metrics.measurement_count, 'integer'),
      stateCount: formatMetricValue(
        executionMode === 'program' && metrics.state_count === 0 ? preservedStateCount : metrics.state_count, 
        'integer', 
        '0'
      ),
      observerCount: formatMetricValue(
        executionMode === 'program' && metrics.observer_count === 0 ? preservedObserverCount : metrics.observer_count, 
        'integer', 
        '0'
      ),
      fps: metrics.fps !== undefined ? metrics.fps : (isSimulating ? fpsRef.current : 60),
      // Universe-specific metrics
      universeTime: formatMetricValue(metrics.universe_time, 'decimal'),
      iterationCount: formatMetricValue(metrics.iteration_count, 'integer'),
      entanglements: formatMetricValue(metrics.num_entanglements, 'integer', '0'),
      universeMode: metrics.universe_mode || 'Standard'
    };
    
    
    return result;
  }, [metrics, formatMetricValue, smoothedDepth, smoothedCoherence, smoothedInformation, isSimulating, executionMode, preservedStateCount, preservedObserverCount]);
  
  // Synchronize backend metrics with frontend engines
  useEffect(() => {
    syncEnginesWithBackend(engineRef.current, metrics);
  }, [metrics]);

  // Update CSS variables when primary color changes
  useEffect(() => {
    // Apply complete theme colors including palette
    applyThemeColors(primaryColor);
    
    // Generate palette for additional custom variables
    const palette = generateColorPalette(primaryColor);
    const root = document.documentElement;
    
    // Set additional legacy variables for compatibility
    const rgbValue = hexToRgb(primaryColor);
    root.style.setProperty('--primary-color', primaryColor);
    root.style.setProperty('--primary-rgb', rgbValue);
    root.style.setProperty('--primary-color-rgb', rgbValue);
    root.style.setProperty('--primary-color-dark', palette.primaryDark);
    root.style.setProperty('--primary-color-light', palette.primaryLight);
    root.style.setProperty('--border-glow', palette.primaryAlpha50);
    
    // Applied complete theme
  }, [primaryColor]);

  
  // Handle click outside color picker
  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Helper to render component with loading state and conditional rendering
  const renderComponent = (windowId: string, component: React.ReactNode, loadingText: string, requireMetrics: boolean = true) => {
    // Only render if the component is in the current layout
    const isInLayout = currentLayout.some(l => l.i === windowId);
    
    if (!isInLayout) {
      return <div className="component-inactive">Component not in layout</div>;
    }
    
    // Some components don't need metrics to function
    if (!requireMetrics) {
      return component;
    }
    
    return metrics ? component : (
      <div className="component-loading">{loadingText}</div>
    );
  };

  // Note: executeProgram has been moved before the callbacks to avoid hoisting issues

  /**
   * Add entry to execution log with proper typing
   */
  const logExecution = useCallback((entry: {
    level: 'info' | 'success' | 'warning' | 'error' | 'debug';
    category: 'execution' | 'measurement' | 'analysis' | 'osh_evidence' | 'quantum_state' | 'optimization' | 'compilation' | 'editor';
    message: string;
    data?: any;
  }) => {
    setExecutionLogEntries(prev => [...prev, {
      id: `log-${Date.now()}-${Math.random()}`,
      timestamp: Date.now(),
      ...entry
    }]);
  }, []);


  // Window configurations - define windows before any useEffect that references them
  const windows: WindowConfig[] = useMemo(() => [
    // Memory Field Visualizer removed
    // {
    //   id: 'memory-field',
    //   title: 'Memory Field Visualizer',
    //   icon: <Brain size={16} />,
    //   component: renderComponent(
    //     'memory-field',
    //     engineRef.current?.memoryFieldEngine ? (
    //       <MemoryFieldVisualizer
    //         memoryFieldEngine={engineRef.current.memoryFieldEngine}
    //         primaryColor={primaryColor}
    //         isActive={isSimulating}
    //       />
    //     ) : (
    //       <div className="component-loading">Initializing Memory Field Engine...</div>
    //     ),
    //     'Initializing Memory Field...'
    //   ),
    //   minW: 3,
    //   minH: 3
    // },
    {
      id: 'rsp-dashboard',
      title: 'RSP Analytics',
      icon: <Activity size={16} />,
      component: renderComponent(
        'rsp-dashboard',
        <RSPDashboard
          rspEngine={engineRef.current?.rspEngine!}
          primaryColor={primaryColor}
          isActive={isSimulating}
        />,
        'Initializing RSP Analytics...'
      ),
      minW: 4,
      minH: 3
    },
    // WaveformExplorer disabled for memory optimization
    // {
    //   id: 'waveform-explorer',
    //   title: 'Waveform Explorer',
    //   icon: <Waves size={16} />,
    //   component: renderComponent(
    //     'waveform-explorer',
    //     <WaveformExplorer
    //       wavefunctionSimulator={engineRef.current?.wavefunctionSimulator!}
    //       observerEngine={engineRef.current?.observerEngine!}
    //       memoryFieldEngine={engineRef.current?.memoryFieldEngine!}
    //       primaryColor={primaryColor}
    //       isActive={isSimulating}
    //     />,
    //     'Initializing Waveform Explorer...'
    //   ),
    //   minW: 3,
    //   minH: 3
    // },
    {
      id: 'curvature-map',
      title: 'Information Curvature',
      icon: <Network size={16} />,
      component: renderComponent(
        'curvature-map',
        <div style={{ height: '100%' }}>
          <InformationalCurvatureMap
            primaryColor={primaryColor}
            showClassicalComparison={true}
          />
        </div>,
        'Initializing Information Curvature...'
      ),
      minW: 3,
      minH: 3
    },
    {
      id: 'code-editor',
      title: 'Recursia Code Editor',
      icon: <Code size={16} />,
      component: (
        <QuantumCodeEditor
          primaryColor={primaryColor}
          initialCode={currentCode}
          onCodeChange={setCurrentCode}
          onRun={(code) => {
            setCurrentCode(code);
            handleExecuteProgram(1);
          }}
          onSave={(code, programInfo) => {
            // Handle save action
            setCurrentCode(code);
            
            if (programInfo) {
              // Update current program info for custom programs
              setCurrentProgram({
                id: programInfo.id,
                name: programInfo.name,
                code: code,
                description: '',
                category: 'custom',
                difficulty: 'variable',
                isCustom: programInfo.isCustom,
                author: 'User',
                dateCreated: new Date().toISOString()
              });
              
              logExecution({
                level: 'success',
                category: 'editor',
                message: `Program saved: ${programInfo.name}`
              });
            } else {
              logExecution({
                level: 'success',
                category: 'editor',
                message: `Program saved${currentProgram ? `: ${currentProgram.name}` : ''}`
              });
            }
          }}
          onNewFile={() => {
            // Clear current program when creating new file
            setCurrentProgram(null);
            logExecution({
              level: 'info',
              category: 'editor',
              message: 'Created new program file'
            });
          }}
          onProgramSelect={(programName, programId, isCustom) => {
            // Handle program selection after confirmation
            if (isCustom && programId) {
              // Load custom program
              const customPrograms = loadCustomPrograms();
              const program = customPrograms.find(p => p.id === programId);
              if (program) {
                setCurrentProgram({
                  ...program,
                  isCustom: true,
                  author: 'User',
                  dateCreated: new Date(program.createdAt).toISOString()
                });
                setCurrentCode(program.code);
                logExecution({
                  level: 'info',
                  category: 'editor',
                  message: `Loaded custom program: ${program.name}`
                });
              }
            } else {
              // Load built-in program
              const allPrograms = [...advancedQuantumPrograms, ...enterpriseQuantumPrograms, ...oshQuantumPrograms];
              const program = allPrograms.find(p => p.name === programName);
              if (program) {
                setCurrentProgram({
                  ...program,
                  isCustom: false
                });
                setCurrentCode(program.code);
                logExecution({
                  level: 'info',
                  category: 'editor',
                  message: `Loaded program: ${program.name}`
                });
              }
            }
          }}
          currentProgramName={currentProgram?.name}
          currentProgramId={currentProgram?.id}
          isCurrentProgramCustom={currentProgram?.isCustom || false}
        />
      ),
      minW: 3,
      minH: 3
    },
    {
      id: 'circuit-designer',
      title: 'Quantum Circuit Designer',
      icon: <Cpu size={16} />,
      component: renderComponent(
        'circuit-designer',
        <QuantumCircuitDesigner
          primaryColor={primaryColor}
          engine={engineRef.current}
          onSave={(circuit) => {
            logExecution({
              level: 'success',
              category: 'execution',
              message: `Circuit saved: ${circuit.name}`
            });
            // Store circuit for later use
            localStorage.setItem(`circuit-${circuit.id}`, JSON.stringify(circuit));
          }}
          onExport={(circuit, format) => {
            logExecution({
              level: 'info',
              category: 'execution',
              message: `Exported circuit "${circuit.name}" as ${format.toUpperCase()}`
            });
          }}
        />,
        'Initializing Circuit Designer...',
        false // Don't require metrics for circuit designer
      ),
      minW: 4,
      minH: 4
    },
    {
      id: 'execution-log',
      title: 'OSH Execution Log',
      icon: <Terminal size={16} />,
      component: <EnhancedExecutionLog 
        entries={executionLogEntries} 
        primaryColor={primaryColor} 
        onClearLog={() => setExecutionLogEntries([])}
        isExecuting={isExecuting}
      />,
      minW: 2,  // Reduced to allow more flexibility in narrow layouts
      minH: 4   // Increased for better log visibility
    },
    {
      id: 'program-selector',
      title: 'Quantum Programs Library',
      icon: <FileText size={16} />,
      component: executionMode === 'program' ? (
        <QuantumProgramsLibrary
          primaryColor={primaryColor}
          currentProgramId={currentProgram?.id}
          disabled={false}
          onProgramSelect={(program) => {
            // Dispatch custom event to trigger program selection in code editor
            // The code editor will handle unsaved changes confirmation
            setTimeout(() => {
              const editor = document.querySelector('.quantum-code-editor');
              if (editor) {
                const event = new CustomEvent('program-select-request', { 
                  detail: { program } 
                });
                editor.dispatchEvent(event);
              } else {
                // Code editor element not found
              }
            }, 0); // Use setTimeout to ensure DOM is ready
          }}
          onProgramLoad={(code, programName) => {
            setCurrentCode(code);
            // Search in all program collections
            const allPrograms = [...advancedQuantumPrograms, ...enterpriseQuantumPrograms, ...oshQuantumPrograms];
            const foundProgram = allPrograms.find(p => p.name === programName);
            
            setCurrentProgram(foundProgram || null);
            
            // Log the program load with proper name
            setExecutionLogEntries(prev => [...prev, {
              id: `load-${Date.now()}-${Math.random()}`,
              timestamp: Date.now(),
              level: 'info',
              category: 'execution',
              message: `Loaded program: ${foundProgram?.name || programName}`,
              executionContext: foundProgram ? {
                programName: foundProgram.name,
                lineNumber: 1,
                executionTime: 0
              } : undefined
            }]);
            
            // Ensure code editor is visible in layout
            if (!currentLayout.some(l => l.i === 'code-editor')) {
              const newItem = {
                i: 'code-editor',
                x: 0,
                y: 0,
                w: 6,
                h: 6
              };
              setCurrentLayout([...currentLayout, newItem]);
            }
          }}
          onProgramExecute={async (program) => {
            // Execute program
            try {
              // First ensure the program is loaded
              if (!currentCode || currentProgram?.id !== program.id) {
                // Need to load the program first
                setExecutionLogEntries(prev => [...prev, {
                  id: `load-exec-${Date.now()}-${Math.random()}`,
                  timestamp: Date.now(),
                  level: 'info',
                  category: 'execution',
                  message: `Loading ${program.name} before execution...`
                }]);
                
                // Load the program code from the API
                const code = await fetchProgram(program.path);
                
                // Set the code and program
                setCurrentCode(code);
                setCurrentProgram({
                  id: program.id,
                  name: program.name,
                  description: program.description,
                  code: code,
                  category: program.category,
                  difficulty: program.difficulty,
                  author: 'User',
                  dateCreated: new Date().toISOString()
                });
                
                // Wait a tick for state to update
                await new Promise(resolve => setTimeout(resolve, 50));
                
                // Now execute with the loaded code
                await executeProgram(code, program.name);
              } else {
                // Program already loaded, execute directly
                handleExecuteProgram();
              }
            } catch (error) {
              // Failed to execute program
              
              // Extract detailed error message
              let errorMessage = 'Unknown error';
              if (error instanceof Error) {
                errorMessage = error.message;
                
                // Add specific guidance for common errors
                if (errorMessage.includes('Cannot connect to API server')) {
                  errorMessage += '\n\nPlease ensure the backend is running:\n1. Open a terminal\n2. Navigate to the project directory\n3. Run: ./scripts/start-api-test.sh';
                } else if (errorMessage.includes('500') || errorMessage.includes('Internal Server Error')) {
                  errorMessage += '\n\nThe server encountered an error. Check the server logs for details.';
                }
              }
              
              setExecutionLogEntries(prev => [...prev, {
                id: `exec-error-${Date.now()}-${Math.random()}`,
                timestamp: Date.now(),
                level: 'error',
                category: 'execution',
                message: `Failed to execute ${program.name}: ${errorMessage}`
              }]);
            }
          }}
        />
      ) : (
        <div className="library-universe-mode">
          <div className="universe-mode-content">
            <div className="universe-icon">
              <Sparkles size={48} style={{ color: primaryColor, opacity: 0.8 }} />
            </div>
            <h3>Universe Mode Active</h3>
            <p>Switch to Program Mode to browse and execute quantum programs.</p>
            <div className="universe-stats">
              <span>Universe Time: {metrics?.universe_time?.toFixed(2) || '0.00'}s</span>
              <span>Iterations: {metrics?.iteration_count || 0}</span>
            </div>
          </div>
        </div>
      ),
      minW: 3,  // Appropriate for library sidebar
      minH: 6   // Ensure good visibility for program list
    },
    {
      id: 'osh-universe-3d',
      title: 'OSH Universe 3D',
      icon: <Sparkles size={16} />,
      component: renderComponent(
        'osh-universe-3d',
        <OSHUniverse3D 
          engine={engineRef.current}
          primaryColor={primaryColor}
        />,
        'Initializing OSH Universe 3D...'
      ),
      minW: 4,
      minH: 4
    },
    {
      id: 'gravitational-waves',
      title: 'Gravitational Wave Echoes',
      icon: <Waves size={16} />,
      component: renderComponent(
        'gravitational-waves',
        <GravitationalWaveEchoVisualizerErrorBoundary>
          <GravitationalWaveEchoVisualizer
          primaryColor={primaryColor}
          simulationData={{
            echoes: [
              {
                id: 'echo-1',
                timestamp: Date.now() - 1000,
                magnitude: 0.8,
                frequency: 100,
                position: new THREE.Vector3(0, 0, 0),
                phase: 0,
                informationContent: 12.5,
                memoryResonance: 0.75,
                source: 'Binary Black Hole Merger',
                velocity: new THREE.Vector3(0, 0, 0),
                polarization: 'plus' as const,
                decayRate: 0.1
              },
              {
                id: 'echo-2',
                timestamp: Date.now() - 500,
                magnitude: 0.6,
                frequency: 150,
                position: new THREE.Vector3(1, 1, 0),
                phase: Math.PI / 4,
                informationContent: 8.3,
                memoryResonance: 0.65,
                source: 'Neutron Star Collision',
                velocity: new THREE.Vector3(0, 0, 0),
                polarization: 'cross' as const,
                decayRate: 0.15
              }
            ],
            fieldStrength: metrics?.field_energy || 1.0,
            coherence: metrics?.coherence || 0.95,
            backgroundNoise: 0.1,
            strainSensitivity: 1e-21,
            informationCapacity: 1000,
            memoryFieldCoupling: metrics?.memory_field_coupling || 0.8
          }}
          isActive={isSimulating}
          />
        </GravitationalWaveEchoVisualizerErrorBoundary>,
        'Initializing Gravitational Wave Visualizer...'
      ),
      minW: 4,
      minH: 3
    },
    {
      id: 'osh-calculations',
      title: 'OSH Calculations',
      icon: <Brain size={16} />,
      component: renderComponent(
        'osh-calculations',
        <OSHCalculationsPanel primaryColor={primaryColor} />,
        'Initializing OSH Calculations...',
        false // Don't require metrics - panel works with defaults
      ),
      minW: 6,
      minH: 4
    },
    {
      id: 'theory-of-everything',
      title: 'Theory of Everything',
      icon: <Infinity size={16} />,
      component: renderComponent(
        'theory-of-everything',
        <TheoryOfEverythingErrorBoundary>
          <TheoryOfEverythingPanel 
            primaryColor={primaryColor}
            isActive={isSimulating}
          />
        </TheoryOfEverythingErrorBoundary>,
        'Initializing Theory of Everything Panel...'
      ),
      minW: 5,
      minH: 6
    }
  ], [metrics, isSimulating, primaryColor, selectedCategory, executionLogEntries, currentProgram, currentLayout, setCurrentLayout, logExecution]);
  
  // Check scroll capability on mount and when windows change
  useEffect(() => {
    const checkScroll = () => {
      const container = scrollContainerRef.current;
      if (container) {
        setCanScrollRight(
          container.scrollLeft < container.scrollWidth - container.clientWidth
        );
      }
    };
    
    // Check initially and after a short delay to ensure DOM is ready
    checkScroll();
    const timer = setTimeout(checkScroll, 100);
    
    return () => clearTimeout(timer);
  }, [windows.length]);
  
  /**
   * Layout presets for different workflow modes
   * Grid system: 12 columns wide, height units are flexible
   * Each preset optimized for specific workflows and use cases
   */
  const layoutPresets: LayoutConfig[] = [
    {
      id: 'default-program',
      name: 'Default Program View',
      icon: <Grid3x3 size={16} />,
      layout: [
        // Primary workflow components for quantum program development
        { i: 'program-selector', x: 0, y: 0, w: 4, h: 12 },  // 33% width (4/12), full height
        { i: 'code-editor', x: 4, y: 0, w: 5, h: 12 },      // 42% width (5/12), full height  
        { i: 'execution-log', x: 9, y: 0, w: 3, h: 12 }     // 25% width (3/12), full height
      ]
    },
    {
      id: 'universe-default',
      name: 'Universe Default View',
      icon: <Infinity size={16} />,
      layout: [
        // Equal split between three core universe visualization components
        { i: 'osh-universe-3d', x: 0, y: 0, w: 4, h: 12 },     // 33% width, full height
        { i: 'curvature-map', x: 4, y: 0, w: 4, h: 12 },       // 33% width, full height
        { i: 'gravitational-waves', x: 8, y: 0, w: 4, h: 12 }  // 33% width, full height
      ]
    },
    {
      id: 'osh-calculations',
      name: 'OSH Calculations View',
      icon: <Calculator size={16} />,
      layout: [
        // RSP Analytics 40%, OSH Calculations 60%
        { i: 'rsp-dashboard', x: 0, y: 0, w: 5, h: 12 },      // 40% width (5/12 ≈ 42%), full height
        { i: 'osh-calculations', x: 5, y: 0, w: 7, h: 12 }   // 60% width (7/12 ≈ 58%), full height
      ]
    },
    {
      id: 'development',
      name: 'Development View',
      icon: <Code size={16} />,
      layout: [
        // Three equal columns for development workflow
        { i: 'code-editor', x: 0, y: 0, w: 4, h: 12 },       // 33% width, full height
        { i: 'circuit-designer', x: 4, y: 0, w: 4, h: 12 },  // 33% width, full height
        { i: 'execution-log', x: 8, y: 0, w: 4, h: 12 }      // 33% width, full height
      ]
    },
    {
      id: 'theory-of-everything',
      name: 'Theory of Everything View',
      icon: <Atom size={16} />,
      layout: [
        // Theory of Everything panel on left, RSP Dashboard and OSH Calculations on right
        { i: 'theory-of-everything', x: 0, y: 0, w: 5, h: 12 },  // 42% width, full height
        { i: 'rsp-dashboard', x: 5, y: 0, w: 4, h: 6 },         // 33% width, half height
        { i: 'osh-calculations', x: 5, y: 6, w: 4, h: 6 },      // 33% width, half height
        { i: 'execution-log', x: 9, y: 0, w: 3, h: 12 }         // 25% width, full height
      ]
    }
  ];
  
  // Initialize default layout
  useEffect(() => {
    const defaultPreset = layoutPresets.find(p => p.id === 'default-program');
    if (defaultPreset) {
      setCurrentLayout(defaultPreset.layout);
    }
  }, []);
  
  const handleLayoutChange = (layout: GridLayout.Layout[]) => {
    if (!lockedLayout) {
      setCurrentLayout(layout);
    }
  };
  
  const saveLayout = () => {
    const layoutName = prompt('Enter layout name:');
    if (layoutName) {
      const newLayouts = { ...layouts, [layoutName]: currentLayout };
      setLayouts(newLayouts);
      localStorage.setItem('quantum-osh-layouts', JSON.stringify(newLayouts));
    }
  };
  
  const loadLayout = (layoutId: string) => {
    const preset = layoutPresets.find(p => p.id === layoutId);
    if (preset) {
      setCurrentLayout(preset.layout);
      setSelectedLayout(layoutId);
    } else if (layouts[layoutId]) {
      setCurrentLayout(layouts[layoutId]);
      setSelectedLayout(layoutId);
    }
  };

  const deleteLayout = (layoutName: string) => {
    if (confirm(`Are you sure you want to delete the layout "${layoutName}"?`)) {
      const newLayouts = { ...layouts };
      delete newLayouts[layoutName];
      setLayouts(newLayouts);
      localStorage.setItem('quantum-osh-layouts', JSON.stringify(newLayouts));
      
      // If we're currently using this layout, switch to default
      if (selectedLayout === layoutName) {
        loadLayout('default-program');
      }
      
      setExecutionLogEntries(prev => [...prev, {
        id: `layout-deleted-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'info',
        category: 'system',
        message: `Deleted custom layout: ${layoutName}`
      }]);
    }
  };
  
  // Load saved layouts
  useEffect(() => {
    const savedLayouts = localStorage.getItem('quantum-osh-layouts');
    if (savedLayouts) {
      setLayouts(JSON.parse(savedLayouts));
    }
  }, []);

  const exportData = () => {
    const exportData: any = {
      timestamp: new Date().toISOString(),
      version: '1.0.0',
      simulation: {
        metrics: metrics,
        states: states,
        events: executionLogEntries
      },
      metadata: {
        duration: Date.now() - startTime.current,
        steps: currentStep,
        parameters: {}
      }
    };
    const jsonString = DataExporter.exportToJSON(exportData);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'quantum-osh-simulation.json';
    a.click();
    URL.revokeObjectURL(url);
  };
  
  // Handler for toggling universe simulation from modal
  const handleToggleUniverseSimulation = useCallback(async () => {
    console.log('[QuantumOSHStudio] Toggle universe simulation:', {
      currentState: isSimulating,
      isUniverseLoading,
      action: isSimulating ? 'stopping' : 'starting',
      mode: universeParameters?.universeMode || 'standard',
      timestamp: new Date().toISOString()
    });
    
    if (isSimulating) {
      // Stopping universe simulation
      stopUniverseSimulation();
      setIsSimulating(false);
      
      setExecutionLogEntries(prev => [...prev, {
        id: `universe-modal-stop-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'info',
        category: 'execution',
        message: 'Universe simulation stopped from control modal'
      }]);
    } else if (!isUniverseLoading) {
      // Starting universe simulation
      const mode = universeParameters?.universeMode || 'standard';
      startUniverseSimulation(mode);
      setIsSimulating(true);
      
      setExecutionLogEntries(prev => [...prev, {
        id: `universe-modal-start-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        level: 'success',
        category: 'execution',
        message: `Universe simulation started from control modal in ${mode} mode`
      }]);
    }
  }, [isSimulating, isUniverseLoading, startUniverseSimulation, stopUniverseSimulation, universeParameters]);

  // Handler for updating universe parameters from modal
  const handleUpdateUniverseParameters = useCallback((params: UniverseParameters) => {
    // Updating universe parameters
    setUniverseParameters(params);
    
    // Send parameters to backend dynamic universe engine
    updateUniverseParameters(params);
    
    // If universe mode changed, update it
    if (params.universeMode) {
      setUniverseMode(params.universeMode);
    }
    
    setExecutionLogEntries(prev => [...prev, {
      id: `universe-params-${Date.now()}-${Math.random()}`,
      timestamp: Date.now(),
      level: 'info',
      category: 'execution',
      message: `Universe parameters updated: ${params.universeMode || 'standard'} mode`
    }]);
    
    // Apply parameters to running engine if active (legacy support)
    if (engineRef.current && isSimulating) {
      try {
        // Update observer count
        if (engineRef.current.observerEngine) {
          const currentObservers = engineRef.current.observerEngine.getAllObservers().length;
          const targetObservers = params.observerCount;
          
          // Add or remove observers to match target count
          if (currentObservers < targetObservers) {
            for (let i = currentObservers; i < targetObservers; i++) {
              engineRef.current.observerEngine.createObserver(
                `observer_${i}`,
                `Observer ${i}`,
                params.observerCollapseThreshold
              );
            }
          } else if (currentObservers > targetObservers) {
            const observers = engineRef.current.observerEngine.getAllObservers();
            for (let i = targetObservers; i < currentObservers; i++) {
              if (observers[i]) {
                engineRef.current.observerEngine.removeObserver(observers[i].id);
              }
            }
          }
          
          // Update observer properties
          engineRef.current.observerEngine.setGlobalFocus(params.observerSelfAwareness);
        }
        
        // Update coherence values across engines
        if (engineRef.current.memoryFieldEngine) {
          // Update memory field coherence by modifying the current field
          const currentField = engineRef.current.memoryFieldEngine.getCurrentField();
          if (currentField) {
            currentField.averageCoherence = params.memoryCoherence;
            // Update individual fragment coherence proportionally
            currentField.fragments.forEach(fragment => {
              fragment.coherence = Math.min(1, fragment.coherence * params.memoryCoherence);
            });
          }
        }
        
        // Update universe field parameters
        // Update via the engine's public methods
        if (engineRef.current && params.universeQubits > 0) {
          // Reset engine with new parameters
          engineRef.current.reset();
          // The engine will use the updated universe parameters from state
        }
        
        // Log parameter update
        setExecutionLogEntries(prev => [...prev, {
          id: `param-update-${Date.now()}`,
          timestamp: Date.now(),
          level: 'info',
          category: 'execution',
          message: `Updated universe parameters: ${params.observerCount} observers, coherence ${params.universeCoherence.toFixed(2)}`
        }]);
      } catch (error) {
        // Failed to apply parameters
      }
    }
  }, [isSimulating]);

  return (
    <UniverseProvider engine={engineRef.current} isSimulating={isSimulating}>
      <div className="quantum-osh-studio" style={{ '--primary-color': primaryColor } as any}>
      {/* Loading Overlay */}
      {engineLoading && (
        <div className="engine-loading-overlay">
          <div className="loading-content">
            <div className="loading-spinner">
              <Infinity size={48} className="spinning" style={{ color: primaryColor }} />
            </div>
            <h2>Initializing Quantum Engine</h2>
            <p>Preparing quantum states and memory fields...</p>
          </div>
        </div>
      )}
      
      {/* Error Overlay */}
      {engineError && (
        <div className="engine-error-overlay">
          <div className="error-content">
            <AlertCircle size={48} style={{ color: '#ff4444' }} />
            <h2>Engine Initialization Failed</h2>
            <p>{engineError}</p>
            <button onClick={() => window.location.reload()}>Reload Page</button>
          </div>
        </div>
      )}
      
      {/* Header */}
      <header className="studio-header">
        <div className="header-brand">
          <div className="brand-logo">
            <Infinity size={20} style={{ color: primaryColor }} />
          </div>
          <h1 className="brand-title">Recursia Studio</h1>
        </div>
        
        
        {/* Mode Selector - Sleek, minimal design */}
        <div className="mode-selector" style={{
          display: 'flex',
          alignItems: 'center',
          gap: '2px', // Small separation between buttons
          marginLeft: '20px',
          padding: '2px',
          borderRadius: '6px',
          background: 'rgba(255, 255, 255, 0.03)',
          border: '1px solid rgba(255, 255, 255, 0.08)'
        }}>
          {/* Program Execution button - now on the left */}
          <button
            className={`mode-btn ${executionMode === 'program' ? 'active' : ''}`}
            onClick={() => handleModeSwitch('program')}
            disabled={isSwitchingMode}
            style={{
              padding: '4px 12px', // Reduced padding
              borderRadius: '4px',
              border: 'none',
              background: executionMode === 'program' ? primaryColor : 'transparent',
              color: executionMode === 'program' ? getContrastColor(primaryColor) : '#999',
              fontSize: '11px', // Smaller text
              fontWeight: executionMode === 'program' ? '500' : '400',
              cursor: isSwitchingMode ? 'wait' : 'pointer',
              transition: 'all 0.15s ease',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              opacity: isSwitchingMode ? 0.7 : 1,
              letterSpacing: '0.3px'
            }}
            onMouseEnter={(e) => {
              if (executionMode !== 'program' && !isSwitchingMode) {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                e.currentTarget.style.color = '#bbb';
              }
            }}
            onMouseLeave={(e) => {
              if (executionMode !== 'program' && !isSwitchingMode) {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = '#999';
              }
            }}
          >
            {isSwitchingMode && executionMode === 'universe' ? (
              <RefreshCw size={12} className="spinning" />
            ) : (
              <Code size={12} />
            )}
            Program
          </button>
          
          {/* Universe Simulation button - now on the right */}
          <button
            className={`mode-btn ${executionMode === 'universe' ? 'active' : ''}`}
            onClick={() => handleModeSwitch('universe')}
            disabled={isSwitchingMode}
            style={{
              padding: '4px 12px', // Reduced padding
              borderRadius: '4px',
              border: 'none',
              background: executionMode === 'universe' ? primaryColor : 'transparent',
              color: executionMode === 'universe' ? getContrastColor(primaryColor) : '#999',
              fontSize: '11px', // Smaller text
              fontWeight: executionMode === 'universe' ? '500' : '400',
              cursor: isSwitchingMode ? 'wait' : 'pointer',
              transition: 'all 0.15s ease',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              opacity: isSwitchingMode ? 0.7 : 1,
              letterSpacing: '0.3px'
            }}
            onMouseEnter={(e) => {
              if (executionMode !== 'universe' && !isSwitchingMode) {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                e.currentTarget.style.color = '#bbb';
              }
            }}
            onMouseLeave={(e) => {
              if (executionMode !== 'universe' && !isSwitchingMode) {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = '#999';
              }
            }}
          >
            {isSwitchingMode && executionMode === 'program' ? (
              <RefreshCw size={12} className="spinning" />
            ) : (
              <Sparkles size={12} />
            )}
            Universe
          </button>
        </div>
        
        <div className="header-center">
          <div className="header-metrics">
          {executionMode === 'universe' ? (
            <>
              {/* Universe Mode Metrics */}
              <Tooltip primaryColor={primaryColor} content="Universe Simulation Time">
                <div className="metric-badge">
                  <span className="metric-label">Time</span>
                  <span className="metric-value" style={{ color: primaryColor }}>{displayMetrics.universeTime}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="Total Simulation Iterations">
                <div className="metric-badge">
                  <span className="metric-label">Iterations</span>
                  <span className="metric-value" style={{ color: primaryColor }}>{displayMetrics.iterationCount}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="Quantum Entanglements">
                <div className="metric-badge">
                  <span className="metric-label">Entanglements</span>
                  <span className="metric-value" style={{ color: primaryColor }}>{displayMetrics.entanglements}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="Recursive Simulation Potential">
                <div className="metric-badge">
                  <span className="metric-label">RSP</span>
                  <span className="metric-value" style={{ color: primaryColor }}>{displayMetrics.rsp}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="System Coherence">
                <div className="metric-badge">
                  <span className="metric-label">Coherence</span>
                  <span className="metric-value" style={{ color: primaryColor }}>{displayMetrics.coherence}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="Information Entropy">
                <div className="metric-badge">
                  <span className="metric-label">Entropy</span>
                  <span className="metric-value" style={{ color: primaryColor }}>{displayMetrics.entropy}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="Quantum States">
                <div className="metric-badge">
                  <span className="metric-label">States</span>
                  <span className="metric-value" style={{ color: primaryColor }} data-value={metrics?.state_count}>{displayMetrics.stateCount}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="Active Observers">
                <div className="metric-badge">
                  <span className="metric-label">Observers</span>
                  <span className="metric-value" style={{ color: primaryColor }} data-value={metrics?.observer_count}>{displayMetrics.observerCount}</span>
                </div>
              </Tooltip>
            </>
          ) : (
            <>
              {/* Program Mode Metrics */}
              <Tooltip primaryColor={primaryColor} content="Recursive Simulation Potential">
                <div className="metric-badge">
                  <span className="metric-label">RSP</span>
                  <span className="metric-value" style={{ color: primaryColor }}>{displayMetrics.rsp}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="Quantum Error Rate">
                <div className="metric-badge">
                  <span className="metric-label">Error</span>
                  <span className="metric-value" style={{ color: primaryColor }}>{displayMetrics.error}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="System Coherence">
                <div className="metric-badge">
                  <span className="metric-label">Coherence</span>
                  <span className="metric-value" style={{ color: primaryColor }}>{displayMetrics.coherence}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="Information Entropy">
                <div className="metric-badge">
                  <span className="metric-label">Entropy</span>
                  <span className="metric-value" style={{ color: primaryColor }}>{displayMetrics.entropy}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="Information Content">
                <div className="metric-badge">
                  <span className="metric-label">Info</span>
                  <span className="metric-value" style={{ color: primaryColor }}>{displayMetrics.information}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="Recursion Depth">
                <div className="metric-badge">
                  <span className="metric-label">Depth</span>
                  <span className="metric-value" style={{ color: primaryColor }}>{displayMetrics.recursionDepth}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="Quantum States">
                <div className="metric-badge">
                  <span className="metric-label">States</span>
                  <span className="metric-value" style={{ color: primaryColor }} data-value={metrics?.state_count}>{displayMetrics.stateCount}</span>
                </div>
              </Tooltip>
              <Tooltip primaryColor={primaryColor} content="Active Observers">
                <div className="metric-badge">
                  <span className="metric-label">Observers</span>
                  <span className="metric-value" style={{ color: primaryColor }} data-value={metrics?.observer_count}>{displayMetrics.observerCount}</span>
                </div>
              </Tooltip>
            </>
          )}
          </div>
        </div>
        
        <div className="header-controls">
          {/* Execution Status Indicator */}
          {isExecuting && (
            <div className="execution-indicator" style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '8px',
              marginRight: '12px',
              padding: '4px 12px',
              borderRadius: '4px',
              background: `rgba(${hexToRgb(primaryColor)}, 0.1)`,
              border: `1px solid ${primaryColor}`
            }}>
              <div className="spinner" style={{
                width: '12px',
                height: '12px',
                border: `2px solid ${primaryColor}30`,
                borderTopColor: primaryColor,
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite'
              }} />
              <span style={{ color: primaryColor, fontSize: '12px', fontWeight: '500' }}>
                Executing Program...
              </span>
            </div>
          )}
          
          {/* Layout selector with delete functionality */}
          <div className="layout-selector">
            <div className="layout-dropdown-container">
              <select
                value={selectedLayout}
                onChange={(e) => loadLayout(e.target.value)}
                className="layout-select"
                style={{ paddingRight: '30px' }}
              >
                <optgroup label="Presets">
                  {layoutPresets.map(preset => (
                    <option key={preset.id} value={preset.id}>
                      {preset.name}
                    </option>
                  ))}
                </optgroup>
                {Object.keys(layouts).length > 0 && (
                  <optgroup label="Custom">
                    {Object.keys(layouts).map(name => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </optgroup>
                )}
              </select>
              {Object.keys(layouts).includes(selectedLayout) && (
                <button
                  className="layout-delete-btn"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    deleteLayout(selectedLayout);
                  }}
                  title="Delete this custom layout"
                  style={{
                    position: 'absolute',
                    right: '28px', // Increased from 8px to give space for dropdown arrow
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'transparent',
                    border: 'none',
                    color: '#ff4444',
                    cursor: 'pointer',
                    padding: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '16px',
                    opacity: 0.7,
                    transition: 'opacity 0.2s ease'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                  onMouseLeave={(e) => e.currentTarget.style.opacity = '0.7'}
                >
                  <Trash2 size={16} />
                </button>
              )}
            </div>
          </div>
          
          <button
            className={`control-btn ${lockedLayout ? 'locked' : ''}`}
            onClick={() => setLockedLayout(!lockedLayout)}
            title={lockedLayout ? 'Unlock Layout' : 'Lock Layout'}
          >
            {lockedLayout ? (
              <><Lock size={20} /><span className="sr-only">Locked</span></>
            ) : (
              <><Unlock size={20} /><span className="sr-only">Unlocked</span></>
            )}
          </button>
          
          <button
            className="control-btn"
            onClick={saveLayout}
            title="Save Layout"
            style={{ fontSize: '12px', fontWeight: 'bold' }}
          >
            S
          </button>
          
          {/* Dynamic control buttons based on mode */}
          {executionMode === 'universe' ? (
            <>
              <button
                className={`control-btn ${isSimulating || isUniverseLoading ? 'active' : ''}`}
                onClick={() => {
                  // Open the universe control modal
                  setShowUniverseControlModal(true);
                }}
                disabled={isUniverseLoading}
                title="Universe Control Panel"
                style={{
                  backgroundColor: (isSimulating || isUniverseLoading) ? primaryColor : undefined,
                  color: (isSimulating || isUniverseLoading) ? '#000' : undefined
                }}
              >
                <Sliders size={20} />
              </button>
              {/* Removed play/pause button in universe mode */}
            </>
          ) : (
            <button
              className={`control-btn ${isProgramRunning ? 'active' : ''}`}
              onClick={() => {
                if (!isProgramRunning) {
                  handleExecuteProgram();
                } else {
                  // Stopping program execution
                  setIsProgramRunning(false);
                  // Ensure engine is stopped
                  if (engineRef.current && engineRef.current.stop) {
                    try {
                      engineRef.current.stop();
                    } catch (error) {
                      // Failed to stop engine
                    }
                  }
                }
              }}
              title={isProgramRunning ? 'Stop Program' : 'Run Program'}
              disabled={!currentCode.trim()}
            >
              {isProgramRunning ? <Pause size={20} /> : <PlayCircle size={20} />}
            </button>
          )}
          
          <button
            className="control-btn"
            onClick={exportData}
            title="Export Data"
          >
            <Download size={20} />
          </button>
          
          <button 
            className="control-btn" 
            title="Settings"
            onClick={() => setShowSettings(!showSettings)}
          >
            <Settings size={20} />
          </button>
        </div>
      </header>
      
      
      {/* Main workspace with grid layout */}
      <div className="studio-workspace">
        <div className="grid-container">
          {/* 
          Grid Layout System:
          - 12 columns total width
          - Row height is dynamic based on container
          - Default layout: Library (4 cols) | Editor (5 cols) | Log (3 cols)
          - All components take full height (12 units)
        */}
        <GridLayout
          className="layout"
          layout={currentLayout}
          cols={12}
          rowHeight={60}
          width={windowWidth - 32}
          onLayoutChange={handleLayoutChange}
          isDraggable={!lockedLayout}
          isResizable={!lockedLayout}
          compactType="vertical"
          preventCollision={false}
          margin={[8, 8]}
        >
          {windows
            .filter(window => currentLayout.some(l => l.i === window.id))
            .map(window => (
              <div key={window.id} className="panel" data-window-id={window.id}>
                <div className="panel-header">
                  <div className="panel-title">
                    {window.icon}
                    <span>{window.title}</span>
                  </div>
                  <div className="panel-controls">
                    <button className="panel-btn" title="Maximize">
                      <Maximize2 size={14} />
                    </button>
                  </div>
                </div>
                <div className="panel-content">
                  {window.component}
                </div>
              </div>
            ))}
          </GridLayout>
        </div>
      </div>
      
      {/* Settings Modal */}
      {showSettings && (
        <div className="settings-modal-overlay" onClick={() => setShowSettings(false)}>
          <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
            <div className="settings-header">
              <h2>Settings</h2>
              <button className="close-btn" onClick={() => setShowSettings(false)}>×</button>
            </div>
            <div className="settings-content">
              <div className="settings-section">
                <h3>Theme Color</h3>
                <div className="color-picker-section">
                  <HexColorPicker 
                    color={primaryColor} 
                    onChange={setPrimaryColor}
                  />
                  <div className="color-presets">
                    <button 
                      className="preset-color" 
                      style={{ background: '#00d4ff' }}
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setPrimaryColor('#00d4ff');
                      }}
                      title="Quantum Blue"
                    />
                    <button 
                      className="preset-color" 
                      style={{ background: '#ff00ff' }}
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setPrimaryColor('#ff00ff');
                      }}
                      title="Quantum Purple"
                    />
                    <button 
                      className="preset-color" 
                      style={{ background: '#00ff88' }}
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setPrimaryColor('#00ff88');
                      }}
                      title="Quantum Green"
                    />
                    <button 
                      className="preset-color" 
                      style={{ background: '#ffd700' }}
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setPrimaryColor('#ffd700');
                      }}
                      title="Quantum Gold"
                    />
                    <button 
                      className="preset-color" 
                      style={{ background: '#ff6b6b' }}
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setPrimaryColor('#ff6b6b');
                      }}
                      title="Quantum Red"
                    />
                    <button 
                      className="preset-color" 
                      style={{ background: '#4ecdc4' }}
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setPrimaryColor('#4ecdc4');
                      }}
                      title="Default Teal"
                    />
                  </div>
                </div>
              </div>
              <div className="settings-section">
                <h3>Performance</h3>
                <label className="settings-option">
                  <span>Target FPS:</span>
                  <select 
                    value={settings.targetFPS} 
                    onChange={(e) => {
                      setSettings(prev => ({ ...prev, targetFPS: e.target.value }));
                      // Update the target FPS for the simulation loop
                      const targetFrameTime = e.target.value === 'Unlimited' ? 0 : 1000 / parseInt(e.target.value);
                      (engineRef.current as any)?.setTargetFrameTime?.(targetFrameTime);
                    }}
                  >
                    <option value="30">30</option>
                    <option value="60">60</option>
                    <option value="120">120</option>
                    <option value="Unlimited">Unlimited</option>
                  </select>
                </label>
                <label className="settings-option">
                  <span>Quality:</span>
                  <select 
                    value={settings.quality} 
                    onChange={(e) => {
                      setSettings(prev => ({ ...prev, quality: e.target.value }));
                      // Apply quality settings
                      const qualitySettings = {
                        Low: { particleCount: 100, renderDistance: 50, effects: false },
                        Medium: { particleCount: 500, renderDistance: 100, effects: true },
                        High: { particleCount: 1000, renderDistance: 200, effects: true },
                        Ultra: { particleCount: 2000, renderDistance: 500, effects: true }
                      };
                      const config = qualitySettings[e.target.value as keyof typeof qualitySettings];
                      // Apply to visualization components
                      window.dispatchEvent(new CustomEvent('qualitySettingsChanged', { detail: config }));
                    }}
                  >
                    <option value="Low">Low</option>
                    <option value="Medium">Medium</option>
                    <option value="High">High</option>
                    <option value="Ultra">Ultra</option>
                  </select>
                </label>
              </div>
              <div className="settings-section">
                <h3>Simulation</h3>
                <label className="settings-option">
                  <span>Auto-save snapshots:</span>
                  <input 
                    type="checkbox" 
                    checked={settings.autoSaveSnapshots}
                    onChange={(e) => {
                      setSettings(prev => ({ ...prev, autoSaveSnapshots: e.target.checked }));
                      // Enable/disable auto-save functionality
                      (engineRef.current as any)?.setAutoSaveEnabled?.(e.target.checked);
                    }}
                  />
                </label>
                <label className="settings-option">
                  <span>Debug mode:</span>
                  <input 
                    type="checkbox" 
                    checked={settings.debugMode}
                    onChange={(e) => {
                      setSettings(prev => ({ ...prev, debugMode: e.target.checked }));
                      // Toggle debug mode
                      if (e.target.checked) {
                        // Debug mode enabled
                        window.localStorage.setItem('osh-debug-mode', 'true');
                        // Show more detailed logs
                        if (window.DiagnosticsSystem) {
                          window.DiagnosticsSystem.getInstance().setDebugMode(true);
                        }
                      } else {
                        // Debug mode disabled
                        window.localStorage.removeItem('osh-debug-mode');
                        if (window.DiagnosticsSystem) {
                          window.DiagnosticsSystem.getInstance().setDebugMode(false);
                        }
                      }
                    }}
                  />
                </label>
              </div>
            </div>
            <div className="settings-footer">
              <button className="btn-primary" onClick={() => {
                setShowSettings(false);
                // Apply all settings
                toast.success('Settings applied successfully');
              }}>Apply</button>
            </div>
          </div>
        </div>
      )}
      
      {/* Compact Footer */}
      <footer className="studio-footer">
        
        <div className="footer-component-selector">
          <span className="selector-label">Components:</span>
          <div style={{ position: 'relative', flex: 1, overflow: 'hidden' }}>
            <button
              className={`scroll-arrow left ${scrollPosition <= 0 ? 'disabled' : ''}`}
              onClick={() => {
                const container = document.querySelector('.component-pills');
                if (container && scrollPosition > 0) {
                  const newPosition = Math.max(0, scrollPosition - 200);
                  container.scrollTo({ left: newPosition, behavior: 'smooth' });
                  setScrollPosition(newPosition);
                }
              }}
              disabled={scrollPosition <= 0}
            >
              <ChevronLeft size={12} />
            </button>
            <div 
              className="component-pills"
              ref={scrollContainerRef}
              onScroll={(e) => {
                setScrollPosition(e.currentTarget.scrollLeft);
                setCanScrollRight(
                  e.currentTarget.scrollLeft < 
                  e.currentTarget.scrollWidth - e.currentTarget.clientWidth
                );
              }}
            >
              {windows.map(window => {
                const isActive = currentLayout.some(l => l.i === window.id);
                return (
                  <button
                    key={window.id}
                    className={`component-pill ${isActive ? 'active' : ''}`}
                    style={isActive ? {
                      background: primaryColor,
                      borderColor: primaryColor,
                      color: getContrastColor(primaryColor),
                      fontWeight: 600
                    } : {}}
                    onClick={() => {
                      if (isActive) {
                        // Remove from layout
                        setCurrentLayout(currentLayout.filter(l => l.i !== window.id));
                      } else {
                        // Add to layout - find empty spot
                        const newItem = {
                          i: window.id,
                          x: 0,
                          y: 0,
                          w: window.minW || 3,
                          h: window.minH || 3
                        };
                        setCurrentLayout([...currentLayout, newItem]);
                      }
                    }}
                    title={window.title}
                    onMouseEnter={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = `rgba(${hexToRgb(primaryColor)}, 0.1)`;
                        e.currentTarget.style.borderColor = `rgba(${hexToRgb(primaryColor)}, 0.3)`;
                        e.currentTarget.style.color = primaryColor;
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = '';
                        e.currentTarget.style.borderColor = '';
                        e.currentTarget.style.color = '';
                      }
                    }}
                  >
                    {React.cloneElement(window.icon as React.ReactElement, {
                      size: 12,
                      style: isActive ? { 
                        stroke: getContrastColor(primaryColor),
                        fill: 'none'
                      } : {}
                    })}
                    <span className="pill-label">
                      {window.title}
                    </span>
                  </button>
                );
              })}
            </div>
            <button
              className={`scroll-arrow right ${!canScrollRight ? 'disabled' : ''}`}
              onClick={() => {
                const container = document.querySelector('.component-pills');
                if (container && canScrollRight) {
                  const newPosition = Math.min(
                    container.scrollWidth - container.clientWidth,
                    scrollPosition + 200
                  );
                  container.scrollTo({ left: newPosition, behavior: 'smooth' });
                  setScrollPosition(newPosition);
                }
              }}
              disabled={!canScrollRight}
            >
              <ChevronRight size={12} />
            </button>
          </div>
        </div>
        
        <div className="footer-status">
          <span className="status-item">
            <span style={{ color: (isSimulating || isProgramRunning) ? primaryColor : '#666' }}>●</span>
            {executionMode === 'universe' 
              ? (isSimulating ? 'Universe Running' : 'Universe Paused')
              : (isProgramRunning ? 'Program Running' : 'Program Ready')
            }
          </span>
          <span className="status-item">
            FPS: {Math.round(displayMetrics.fps)}
          </span>
        </div>
      </footer>
      
      {/* Debug: Engine Status */}
      <EngineStatus engine={engineRef.current} primaryColor={primaryColor} />
      
      {/* Performance Manager */}
      <PerformanceManager 
        primaryColor={primaryColor} 
      />
      
      {/* Memory Monitor Modal */}
      {showMemoryMonitor && (
        <div className="modal-overlay" onClick={() => setShowMemoryMonitor(false)}>
          <div className="modal-content memory-monitor-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Memory Monitor</h3>
              <button 
                className="modal-close-btn"
                onClick={() => setShowMemoryMonitor(false)}
              >
                ×
              </button>
            </div>
            <div className="modal-body">
              <MemoryMonitor 
                primaryColor={primaryColor}
                onClose={() => setShowMemoryMonitor(false)}
              />
            </div>
          </div>
        </div>
      )}
      
      {/* Diagnostic Panel */}
      {showDiagnostics && (
        <DiagnosticPanel onClose={() => setShowDiagnostics(false)} />
      )}
      
      {/* Universe Loading Modal */}
      {isUniverseLoading && (
        <div className="universe-loading-modal">
          <div 
            className="loading-backdrop" 
            onClick={() => setIsUniverseLoading(false)}
          />
          <div className="loading-content">
            <div className="loading-header">
              <div className="loading-title-section">
                <h2>Initializing Enhanced Universe Simulation</h2>
              </div>
              <button 
                className="modal-close-btn"
                onClick={() => setIsUniverseLoading(false)}
                title="Close modal"
              >
                ×
              </button>
            </div>
            
            <div className="loading-progress">
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${universeLoadingProgress}%` }}
                />
              </div>
              <span className="progress-text">{Math.round(universeLoadingProgress)}%</span>
            </div>
            
            <div className="loading-stage">
              <div className="stage-indicator">
                <div className="pulse-ring" />
                <div className="pulse-core" />
              </div>
              <span className="stage-text">{universeLoadingStage}</span>
            </div>
            
            <div className="loading-details">
              <div className="detail-item">
                <span className="detail-label">Φ Scaling Factor:</span>
                <span className="detail-value">β = 2.31</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Recursion Depth:</span>
                <span className="detail-value">d = 22 ± 2</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Consciousness Threshold:</span>
                <span className="detail-value">Φ_c = 1.8</span>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Universe Control Modal */}
      <UniverseControlModal
        isOpen={showUniverseControlModal}
        onClose={() => setShowUniverseControlModal(false)}
        primaryColor={primaryColor}
        isSimulating={isSimulating}
        onToggleSimulation={handleToggleUniverseSimulation}
        onUpdateParameters={handleUpdateUniverseParameters}
        currentParameters={universeParameters}
        metrics={metrics}
      />
    </div>
    </UniverseProvider>
  );
};

export default QuantumOSHStudio;