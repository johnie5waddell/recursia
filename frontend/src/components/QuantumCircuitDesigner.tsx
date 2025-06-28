/**
 * Quantum Circuit Designer Component
 * A comprehensive visual quantum circuit builder with support for both simulation and hardware execution
 * Integrates with Recursia's OSH quantum computing framework
 */

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Cpu, Zap, GitBranch, Save, Download, Upload, Play, Pause, 
  RotateCcw, Settings, Info, Plus, Trash2, Copy, Grid,
  ChevronRight, ChevronDown, Layers, Activity, Eye, Lock,
  AlertCircle, CheckCircle, Loader, FileText, Code,
  HelpCircle, Maximize2, Minimize2, RefreshCw, Database, Minus
} from 'lucide-react';
import { Tooltip } from './ui/Tooltip';
import { Complex } from '../utils/complex';
import { OSHQuantumEngine } from '../engines/OSHQuantumEngine';
import { useEngineAPIContext } from '../contexts/EngineAPIContext';
import { convertToRecursia, convertToQASM, convertToQiskit } from '../utils/circuitIntegration';
import '../styles/quantum-circuit-designer.css';

// Gate types with comprehensive parameter support
export type GateType = 
  | 'H' | 'X' | 'Y' | 'Z' | 'S' | 'T' | 'CNOT' | 'CZ' | 'SWAP'
  | 'RX' | 'RY' | 'RZ' | 'U' | 'PHASE' | 'CRX' | 'CRY' | 'CRZ'
  | 'TOFFOLI' | 'FREDKIN' | 'QFT' | 'MEASURE' | 'BARRIER'
  | 'CUSTOM' | 'ORACLE' | 'GROVER' | 'TELEPORT';

// Hardware backend options
export type HardwareBackend = 'simulator' | 'ibm_quantum' | 'rigetti' | 'google' | 'ionq' | 'honeywell';

interface QuantumGate {
  id: string;
  type: GateType;
  qubits: number[]; // Target qubits
  params?: number[]; // Gate parameters (angles, phases)
  label?: string;
  color?: string;
  duration?: number; // Gate duration in ns for hardware
  error?: number; // Gate error rate
  custom?: {
    matrix?: Complex[][];
    unitary?: boolean;
    hermitian?: boolean;
  };
}

interface CircuitWire {
  qubit: number;
  label: string;
  classical?: boolean; // Classical bit for measurement outcomes
  ancilla?: boolean; // Ancilla qubit
}

interface QuantumCircuit {
  id: string;
  name: string;
  description?: string;
  wires: CircuitWire[];
  gates: QuantumGate[];
  metadata: {
    created: number;
    modified: number;
    author?: string;
    tags?: string[];
    hardware?: HardwareBackend;
    optimizationLevel?: number;
  };
}

interface CircuitExecution {
  id: string;
  circuitId: string;
  backend: HardwareBackend;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startTime: number;
  endTime?: number;
  shots: number;
  results?: {
    counts: { [outcome: string]: number };
    statevector?: Complex[];
    probabilities?: number[];
    expectationValues?: { [observable: string]: number };
    error?: string;
  };
  hardware?: {
    queuePosition?: number;
    estimatedTime?: number;
    device?: string;
    calibration?: any;
  };
}

interface QuantumCircuitDesignerProps {
  primaryColor: string;
  onSave?: (circuit: QuantumCircuit) => void;
  onExport?: (circuit: QuantumCircuit, format: 'qasm' | 'qiskit' | 'recursia') => void;
  initialCircuit?: QuantumCircuit;
  readOnly?: boolean;
  engine?: OSHQuantumEngine;
}

// Gate definitions with visual and quantum properties
const GATE_DEFINITIONS: Record<GateType, {
  name: string;
  symbol: string;
  qubits: number;
  params?: string[];
  color: string;
  description: string;
  matrix?: (params?: number[]) => Complex[][];
}> = {
  H: {
    name: 'Hadamard',
    symbol: 'H',
    qubits: 1,
    color: '#4fc3f7',
    description: 'Creates superposition: |0⟩ → (|0⟩+|1⟩)/√2',
    matrix: () => {
      const s = 1 / Math.sqrt(2);
      return [
        [new Complex(s, 0), new Complex(s, 0)],
        [new Complex(s, 0), new Complex(-s, 0)]
      ];
    }
  },
  X: {
    name: 'Pauli-X',
    symbol: 'X',
    qubits: 1,
    color: '#ff6b6b',
    description: 'Bit flip: |0⟩ ↔ |1⟩',
    matrix: () => [
      [new Complex(0, 0), new Complex(1, 0)],
      [new Complex(1, 0), new Complex(0, 0)]
    ]
  },
  Y: {
    name: 'Pauli-Y',
    symbol: 'Y',
    qubits: 1,
    color: '#ff9f40',
    description: 'Bit and phase flip',
    matrix: () => [
      [new Complex(0, 0), new Complex(0, -1)],
      [new Complex(0, 1), new Complex(0, 0)]
    ]
  },
  Z: {
    name: 'Pauli-Z',
    symbol: 'Z',
    qubits: 1,
    color: '#ab47bc',
    description: 'Phase flip: |1⟩ → -|1⟩',
    matrix: () => [
      [new Complex(1, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(-1, 0)]
    ]
  },
  S: {
    name: 'S Gate',
    symbol: 'S',
    qubits: 1,
    color: '#66bb6a',
    description: 'π/2 phase gate',
    matrix: () => [
      [new Complex(1, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(0, 1)]
    ]
  },
  T: {
    name: 'T Gate',
    symbol: 'T',
    qubits: 1,
    color: '#42a5f5',
    description: 'π/4 phase gate',
    matrix: () => {
      const phase = Math.PI / 4;
      return [
        [new Complex(1, 0), new Complex(0, 0)],
        [new Complex(0, 0), new Complex(Math.cos(phase), Math.sin(phase))]
      ];
    }
  },
  CNOT: {
    name: 'CNOT',
    symbol: '⊕',
    qubits: 2,
    color: '#ec407a',
    description: 'Controlled NOT gate',
    matrix: () => [
      [new Complex(1, 0), new Complex(0, 0), new Complex(0, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(1, 0), new Complex(0, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(0, 0), new Complex(0, 0), new Complex(1, 0)],
      [new Complex(0, 0), new Complex(0, 0), new Complex(1, 0), new Complex(0, 0)]
    ]
  },
  CZ: {
    name: 'CZ',
    symbol: 'CZ',
    qubits: 2,
    color: '#7e57c2',
    description: 'Controlled Z gate',
    matrix: () => [
      [new Complex(1, 0), new Complex(0, 0), new Complex(0, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(1, 0), new Complex(0, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(0, 0), new Complex(1, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(0, 0), new Complex(0, 0), new Complex(-1, 0)]
    ]
  },
  SWAP: {
    name: 'SWAP',
    symbol: '×',
    qubits: 2,
    color: '#26a69a',
    description: 'Swaps two qubits',
    matrix: () => [
      [new Complex(1, 0), new Complex(0, 0), new Complex(0, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(0, 0), new Complex(1, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(1, 0), new Complex(0, 0), new Complex(0, 0)],
      [new Complex(0, 0), new Complex(0, 0), new Complex(0, 0), new Complex(1, 0)]
    ]
  },
  RX: {
    name: 'RX',
    symbol: 'Rx',
    qubits: 1,
    params: ['θ'],
    color: '#ef5350',
    description: 'Rotation around X axis',
    matrix: (params) => {
      const theta = params?.[0] || 0;
      const cos = Math.cos(theta / 2);
      const sin = Math.sin(theta / 2);
      return [
        [new Complex(cos, 0), new Complex(0, -sin)],
        [new Complex(0, -sin), new Complex(cos, 0)]
      ];
    }
  },
  RY: {
    name: 'RY',
    symbol: 'Ry',
    qubits: 1,
    params: ['θ'],
    color: '#ffa726',
    description: 'Rotation around Y axis',
    matrix: (params) => {
      const theta = params?.[0] || 0;
      const cos = Math.cos(theta / 2);
      const sin = Math.sin(theta / 2);
      return [
        [new Complex(cos, 0), new Complex(-sin, 0)],
        [new Complex(sin, 0), new Complex(cos, 0)]
      ];
    }
  },
  RZ: {
    name: 'RZ',
    symbol: 'Rz',
    qubits: 1,
    params: ['φ'],
    color: '#9c27b0',
    description: 'Rotation around Z axis',
    matrix: (params) => {
      const phi = params?.[0] || 0;
      return [
        [new Complex(Math.cos(-phi/2), Math.sin(-phi/2)), new Complex(0, 0)],
        [new Complex(0, 0), new Complex(Math.cos(phi/2), Math.sin(phi/2))]
      ];
    }
  },
  U: {
    name: 'U Gate',
    symbol: 'U',
    qubits: 1,
    params: ['θ', 'φ', 'λ'],
    color: '#5c6bc0',
    description: 'General single-qubit unitary',
    matrix: (params) => {
      const [theta = 0, phi = 0, lambda = 0] = params || [];
      const cos = Math.cos(theta / 2);
      const sin = Math.sin(theta / 2);
      return [
        [
          new Complex(cos, 0),
          new Complex(-sin * Math.cos(lambda), -sin * Math.sin(lambda))
        ],
        [
          new Complex(sin * Math.cos(phi), sin * Math.sin(phi)),
          new Complex(cos * Math.cos(phi + lambda), cos * Math.sin(phi + lambda))
        ]
      ];
    }
  },
  PHASE: {
    name: 'Phase',
    symbol: 'P',
    qubits: 1,
    params: ['λ'],
    color: '#29b6f6',
    description: 'Phase gate',
    matrix: (params) => {
      const lambda = params?.[0] || 0;
      return [
        [new Complex(1, 0), new Complex(0, 0)],
        [new Complex(0, 0), new Complex(Math.cos(lambda), Math.sin(lambda))]
      ];
    }
  },
  CRX: {
    name: 'CRX',
    symbol: 'CRx',
    qubits: 2,
    params: ['θ'],
    color: '#e53935',
    description: 'Controlled RX rotation'
  },
  CRY: {
    name: 'CRY',
    symbol: 'CRy',
    qubits: 2,
    params: ['θ'],
    color: '#fb8c00',
    description: 'Controlled RY rotation'
  },
  CRZ: {
    name: 'CRZ',
    symbol: 'CRz',
    qubits: 2,
    params: ['φ'],
    color: '#8e24aa',
    description: 'Controlled RZ rotation'
  },
  TOFFOLI: {
    name: 'Toffoli',
    symbol: 'CCX',
    qubits: 3,
    color: '#d81b60',
    description: 'Controlled-controlled NOT gate'
  },
  FREDKIN: {
    name: 'Fredkin',
    symbol: 'CSWAP',
    qubits: 3,
    color: '#00897b',
    description: 'Controlled SWAP gate'
  },
  QFT: {
    name: 'QFT',
    symbol: 'QFT',
    qubits: 1, // Variable
    color: '#3949ab',
    description: 'Quantum Fourier Transform'
  },
  MEASURE: {
    name: 'Measure',
    symbol: 'M',
    qubits: 1,
    color: '#757575',
    description: 'Measurement in computational basis'
  },
  BARRIER: {
    name: 'Barrier',
    symbol: '||',
    qubits: 1, // Variable
    color: '#9e9e9e',
    description: 'Optimization barrier'
  },
  CUSTOM: {
    name: 'Custom',
    symbol: 'U',
    qubits: 1, // Variable
    color: '#ff6f00',
    description: 'Custom unitary gate'
  },
  ORACLE: {
    name: 'Oracle',
    symbol: 'O',
    qubits: 1, // Variable
    color: '#6a1b9a',
    description: 'Oracle function'
  },
  GROVER: {
    name: 'Grover',
    symbol: 'G',
    qubits: 1, // Variable
    color: '#1565c0',
    description: 'Grover diffusion operator'
  },
  TELEPORT: {
    name: 'Teleport',
    symbol: 'TP',
    qubits: 3,
    color: '#00838f',
    description: 'Quantum teleportation protocol'
  }
};


export function QuantumCircuitDesigner({
  primaryColor = '#4fc3f7',
  onSave,
  onExport,
  initialCircuit,
  readOnly = false,
  engine
}: QuantumCircuitDesignerProps) {
  // Notification state for user feedback
  const [notification, setNotification] = useState<{ message: string; type: 'success' | 'error' | 'info' } | null>(null);
  
  const showNotification = useCallback((message: string, type: 'success' | 'error' | 'info' = 'info') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 3000);
  }, []);
  
  // Circuit state
  const [circuit, setCircuit] = useState<QuantumCircuit>(initialCircuit || {
    id: `circuit-${Date.now()}`,
    name: 'New Circuit',
    description: '',
    wires: [
      { qubit: 0, label: 'q0' },
      { qubit: 1, label: 'q1' },
      { qubit: 2, label: 'q2' }
    ],
    gates: [],
    metadata: {
      created: Date.now(),
      modified: Date.now(),
      hardware: 'simulator',
      optimizationLevel: 1
    }
  });
  
  // UI state
  const [selectedGate, setSelectedGate] = useState<GateType | null>(null);
  const [selectedWires, setSelectedWires] = useState<number[]>([]);
  const [hoveredGate, setHoveredGate] = useState<string | null>(null);
  const [draggedGate, setDraggedGate] = useState<{ type: GateType; startPos: { x: number; y: number } } | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showGateLibrary, setShowGateLibrary] = useState(true);
  const [gridSize] = useState(50); // Grid size in pixels
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  
  // Execution state
  const [executions, setExecutions] = useState<CircuitExecution[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [selectedBackend, setSelectedBackend] = useState<HardwareBackend>('simulator');
  const [shots, setShots] = useState(1024);
  
  // Circuit canvas ref
  const canvasRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 400 });
  const [isCanvasReady, setIsCanvasReady] = useState(false);
  
  // Calculate dynamic canvas dimensions based on circuit content
  const calculateCanvasSize = useCallback(() => {
    const containerWidth = canvasRef.current?.clientWidth || 800;
    const containerHeight = canvasRef.current?.clientHeight || 400;
    
    // Calculate required width based on gates
    const gateCount = circuit.gates.length;
    const minWidth = Math.max(containerWidth, 150 + (gateCount + 2) * gridSize);
    
    // Calculate required height based on wires
    const wireCount = circuit.wires.length;
    const minHeight = Math.max(containerHeight, 100 + wireCount * gridSize);
    
    return { width: minWidth, height: minHeight };
  }, [circuit.gates.length, circuit.wires.length, gridSize]);
  
  // Connect to engine API
  const { isConnected, metrics, execute } = useEngineAPIContext();
  
  // Initialize quantum engine if not provided
  const quantumEngine = useMemo(() => {
    return engine || new OSHQuantumEngine();
  }, [engine]);
  
  // Component initialization
  useEffect(() => {
    // Component mounted
  }, []);
  
  // Update canvas size on mount, resize, and circuit changes
  useEffect(() => {
    const updateSize = () => {
      if (canvasRef.current) {
        const newSize = calculateCanvasSize();
        setCanvasSize(newSize);
      }
    };
    
    // Set canvas as ready immediately
    setIsCanvasReady(true);
    
    // Initial size calculation
    updateSize();
    
    // Update on resize
    window.addEventListener('resize', updateSize);
    
    // Use ResizeObserver for more accurate size tracking
    let resizeObserver: ResizeObserver | null = null;
    if (canvasRef.current && window.ResizeObserver) {
      resizeObserver = new ResizeObserver(updateSize);
      resizeObserver.observe(canvasRef.current);
    }
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', updateSize);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [calculateCanvasSize]);
  
  // Update canvas size when circuit changes
  useEffect(() => {
    if (canvasRef.current) {
      const newSize = calculateCanvasSize();
      setCanvasSize(newSize);
    }
  }, [circuit.gates.length, circuit.wires.length, calculateCanvasSize]);
  
  // Calculate circuit depth (maximum gate layer)
  const calculateCircuitDepth = useCallback(() => {
    if (circuit.gates.length === 0) return 0;
    
    // Track which time slots are occupied for each qubit
    const qubitTimelines: number[][] = circuit.wires.map(() => []);
    
    circuit.gates.forEach(gate => {
      // Find the latest time slot used by any of the gate's qubits
      let latestTime = 0;
      gate.qubits.forEach(qubit => {
        if (qubitTimelines[qubit] && qubitTimelines[qubit].length > 0) {
          latestTime = Math.max(latestTime, Math.max(...qubitTimelines[qubit]));
        }
      });
      
      // Place gate in the next available time slot
      const gateTime = latestTime + 1;
      gate.qubits.forEach(qubit => {
        if (qubitTimelines[qubit]) {
          qubitTimelines[qubit].push(gateTime);
        }
      });
    });
    
    // Return the maximum depth across all qubits
    return Math.max(...qubitTimelines.map(timeline => 
      timeline.length > 0 ? Math.max(...timeline) : 0
    ));
  }, [circuit]);
  
  // Gate placement calculation
  const calculateGatePosition = useCallback((gateIndex: number, qubitIndex: number) => {
    const x = 100 + gateIndex * gridSize;
    const y = 50 + qubitIndex * gridSize;
    return { x, y };
  }, [gridSize]);
  
  // Add gate to circuit
  const addGate = useCallback((type: GateType, qubits: number[]) => {
    if (readOnly) return;
    
    const gateId = `gate-${Date.now()}-${Math.random()}`;
    const gateDef = GATE_DEFINITIONS[type];
    
    const newGate: QuantumGate = {
      id: gateId,
      type,
      qubits,
      color: gateDef.color,
      params: gateDef.params ? new Array(gateDef.params.length).fill(0) : undefined
    };
    
    setCircuit(prev => ({
      ...prev,
      gates: [...prev.gates, newGate],
      metadata: { ...prev.metadata, modified: Date.now() }
    }));
    
    // Clear selection
    setSelectedGate(null);
    setSelectedWires([]);
  }, [readOnly]);
  
  // Remove gate from circuit
  const removeGate = useCallback((gateId: string) => {
    if (readOnly) return;
    
    setCircuit(prev => ({
      ...prev,
      gates: prev.gates.filter(g => g.id !== gateId),
      metadata: { ...prev.metadata, modified: Date.now() }
    }));
  }, [readOnly]);
  
  // Validate circuit
  const validateCircuit = useCallback((circuit: QuantumCircuit): { valid: boolean; errors: string[] } => {
    const errors: string[] = [];
    
    // Check if circuit has wires
    if (circuit.wires.length === 0) {
      errors.push('Circuit must have at least one qubit');
    }
    
    // Check if circuit has gates
    if (circuit.gates.length === 0) {
      errors.push('Circuit must have at least one gate');
    }
    
    // Check gate validity
    circuit.gates.forEach((gate, idx) => {
      const gateDef = GATE_DEFINITIONS[gate.type];
      
      // Check if gate type is valid
      if (!gateDef) {
        errors.push(`Unknown gate type "${gate.type}" at position ${idx}`);
      }
      
      // Check qubit count
      if (gateDef && gate.qubits.length !== gateDef.qubits) {
        errors.push(`Gate "${gate.type}" at position ${idx} requires ${gateDef.qubits} qubit(s), but has ${gate.qubits.length}`);
      }
      
      // Check qubit indices
      gate.qubits.forEach((qubit, qIdx) => {
        if (qubit < 0 || qubit >= circuit.wires.length) {
          errors.push(`Gate "${gate.type}" at position ${idx} references invalid qubit ${qubit}`);
        }
      });
      
      // Check for duplicate qubits in multi-qubit gates
      if (gate.qubits.length > 1) {
        const uniqueQubits = new Set(gate.qubits);
        if (uniqueQubits.size < gate.qubits.length) {
          errors.push(`Gate "${gate.type}" at position ${idx} has duplicate qubit references`);
        }
      }
    });
    
    return {
      valid: errors.length === 0,
      errors
    };
  }, []);
  
  // Update gate parameters
  const updateGateParams = useCallback((gateId: string, params: number[]) => {
    if (readOnly) return;
    
    setCircuit(prev => ({
      ...prev,
      gates: prev.gates.map(g => g.id === gateId ? { ...g, params } : g),
      metadata: { ...prev.metadata, modified: Date.now() }
    }));
  }, [readOnly]);
  
  // Add/remove wires (qubits)
  const addWire = useCallback(() => {
    if (readOnly) return;
    
    const newQubit = circuit.wires.length;
    setCircuit(prev => ({
      ...prev,
      wires: [...prev.wires, { qubit: newQubit, label: `q${newQubit}` }],
      metadata: { ...prev.metadata, modified: Date.now() }
    }));
  }, [readOnly, circuit.wires.length]);
  
  const removeWire = useCallback((qubit: number) => {
    if (readOnly || circuit.wires.length <= 1) return;
    
    setCircuit(prev => ({
      ...prev,
      wires: prev.wires.filter(w => w.qubit !== qubit),
      gates: prev.gates.filter(g => !g.qubits.includes(qubit)).map(g => ({
        ...g,
        qubits: g.qubits.map(q => q > qubit ? q - 1 : q)
      })),
      metadata: { ...prev.metadata, modified: Date.now() }
    }));
  }, [readOnly, circuit.wires.length]);
  
  // Execute circuit
  const executeCircuit = useCallback(async () => {
    if (isExecuting) return;
    
    // Validate circuit
    const validation = validateCircuit(circuit);
    if (!validation.valid) {
      // Circuit validation failed
      showNotification(`Validation failed: ${validation.errors[0]}`, 'error');
      return;
    }
    
    setIsExecuting(true);
    
    const execution: CircuitExecution = {
      id: `exec-${Date.now()}`,
      circuitId: circuit.id,
      backend: selectedBackend,
      status: 'pending',
      startTime: Date.now(),
      shots
    };
    
    setExecutions(prev => [execution, ...prev]);
    
    try {
      // Convert circuit to appropriate format based on backend
      let executionCode: string;
      
      if (selectedBackend === 'simulator') {
        executionCode = convertToRecursia(circuit);
      } else if (selectedBackend === 'ibm_quantum') {
        executionCode = convertToQiskit(circuit);
      } else {
        executionCode = convertToQASM(circuit);
      }
      
      // Execute the circuit using the backend API
      let result;
      
      if (selectedBackend === 'simulator' && execute) {
        // Execute using Recursia engine through API
        const executionResult = await execute(executionCode, shots);
        
        // Process the execution result
        if (executionResult.success) {
          const counts: { [key: string]: number } = {};
          const numQubits = circuit.wires.length;
          
          // Extract measurement results if available
          if (executionResult.measurements && executionResult.measurements.length > 0) {
            executionResult.measurements.forEach((measurement: any) => {
              const outcome = measurement.outcome.toString(2).padStart(numQubits, '0');
              counts[outcome] = (counts[outcome] || 0) + 1;
            });
          } else {
            // Default to ground state if no measurements
            counts['0'.repeat(numQubits)] = shots;
          }
          
          // Calculate probabilities
          const stateSize = Math.pow(2, numQubits);
          const probabilities = new Array(stateSize).fill(0);
          Object.entries(counts).forEach(([state, count]) => {
            const index = parseInt(state, 2);
            if (index < stateSize) {
              probabilities[index] = count / shots;
            }
          });
          
          result = {
            counts,
            probabilities,
            statevector: undefined,
            expectationValues: executionResult.metrics
          };
        } else {
          throw new Error(executionResult.error || 'Execution failed');
        }
      } else {
        // For hardware backends, create a placeholder result
        // In a real implementation, this would interface with quantum cloud services
        const numQubits = circuit.wires.length;
        const counts: { [key: string]: number } = {};
        counts['0'.repeat(numQubits)] = shots;
        
        result = {
          counts,
          probabilities: [1, ...new Array(Math.pow(2, numQubits) - 1).fill(0)],
          statevector: undefined,
          expectationValues: undefined
        };
        
        showNotification(`Hardware backend ${selectedBackend} execution is simulated`, 'info');
      }
      
      // Log the execution for debugging
      // Mock execution completed with results
      
      // Update execution with results
      const completedExecution: CircuitExecution = {
        ...execution,
        status: 'completed',
        endTime: Date.now(),
        results: {
          counts: result.counts || {},
          statevector: result.statevector,
          probabilities: result.probabilities,
          expectationValues: result.expectationValues
        }
      };
      
      setExecutions(prev => prev.map(e => e.id === execution.id ? completedExecution : e));
      showNotification('Execution completed successfully', 'success');
    } catch (error) {
      // Handle execution error
      showNotification('Execution failed', 'error');
      const failedExecution: CircuitExecution = {
        ...execution,
        status: 'failed',
        endTime: Date.now(),
        results: {
          counts: {},
          error: error instanceof Error ? error.message : 'Unknown error'
        }
      };
      
      setExecutions(prev => prev.map(e => e.id === execution.id ? failedExecution : e));
    } finally {
      setIsExecuting(false);
    }
  }, [circuit, selectedBackend, shots, isExecuting, execute]);
  
  // Convert circuit to various formats
  const convertToRecursia = useCallback((circuit: QuantumCircuit): string => {
    let code = `// ${circuit.name}\n`;
    if (circuit.description) {
      code += `// ${circuit.description}\n`;
    }
    code += '\n';
    
    // Create quantum states
    circuit.wires.forEach(wire => {
      code += `state ${wire.label} = |0⟩;\n`;
    });
    code += '\n';
    
    // Apply gates
    circuit.gates.forEach(gate => {
      const def = GATE_DEFINITIONS[gate.type];
      const wireLabels = gate.qubits.map(q => circuit.wires[q]?.label || `q${q}`);
      
      switch (gate.type) {
        case 'MEASURE':
          code += `measure ${wireLabels[0]};\n`;
          break;
        case 'BARRIER':
          code += `// Optimization barrier\n`;
          break;
        case 'CNOT':
          code += `CNOT ${wireLabels.join(', ')};\n`;
          break;
        case 'CZ':
          code += `CZ ${wireLabels.join(', ')};\n`;
          break;
        case 'SWAP':
          code += `SWAP ${wireLabels.join(', ')};\n`;
          break;
        case 'TOFFOLI':
          code += `CCNOT ${wireLabels.join(', ')};\n`;
          break;
        case 'FREDKIN':
          code += `CSWAP ${wireLabels.join(', ')};\n`;
          break;
        case 'QFT':
          code += `// Quantum Fourier Transform\n`;
          wireLabels.forEach((wire, idx) => {
            code += `  H ${wire};\n`;
            for (let j = idx + 1; j < wireLabels.length; j++) {
              const angle = Math.PI / Math.pow(2, j - idx);
              code += `  CPHASE(${angle.toFixed(4)}) ${wire}, ${wireLabels[j]};\n`;
            }
          });
          break;
        case 'GROVER':
          code += `// Grover operator - requires custom implementation\n`;
          code += `// Apply oracle and diffusion operator\n`;
          break;
        case 'ORACLE':
          code += `// Oracle function - custom implementation required\n`;
          break;
        case 'TELEPORT':
          code += `// Quantum teleportation protocol\n`;
          if (wireLabels.length >= 3) {
            code += `  H ${wireLabels[1]};\n`;
            code += `  CNOT ${wireLabels[1]}, ${wireLabels[2]};\n`;
            code += `  CNOT ${wireLabels[0]}, ${wireLabels[1]};\n`;
            code += `  H ${wireLabels[0]};\n`;
            code += `  measure ${wireLabels[0]};\n`;
            code += `  measure ${wireLabels[1]};\n`;
          }
          break;
        default:
          // Single-qubit gates
          if (gate.params && gate.params.length > 0) {
            const paramStr = gate.params.map(p => p.toFixed(4)).join(', ');
            code += `${gate.type}(${paramStr}) ${wireLabels.join(', ')};\n`;
          } else {
            code += `${gate.type} ${wireLabels.join(', ')};\n`;
          }
      }
    });
    
    return code;
  }, []);
  
  const convertToQiskit = useCallback((circuit: QuantumCircuit): string => {
    let code = '# Auto-generated Qiskit code\n';
    code += 'from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister\n';
    code += 'from qiskit import execute, Aer\n';
    code += 'from qiskit.visualization import plot_histogram\n';
    code += 'import numpy as np\n\n';
    
    // Circuit creation
    code += `# Create circuit: ${circuit.name}\n`;
    code += `qr = QuantumRegister(${circuit.wires.length}, 'q')\n`;
    
    const measureCount = circuit.gates.filter((g: QuantumGate) => g.type === 'MEASURE').length;
    if (measureCount > 0) {
      code += `cr = ClassicalRegister(${measureCount}, 'c')\n`;
      code += 'qc = QuantumCircuit(qr, cr)\n';
    } else {
      code += 'qc = QuantumCircuit(qr)\n';
    }
    
    code += '\n# Apply gates\n';
    
    // Gates
    let measureIndex = 0;
    circuit.gates.forEach((gate: QuantumGate) => {
      const qubits = gate.qubits.map(q => `qr[${q}]`).join(', ');
      
      switch (gate.type) {
        case 'H': code += `qc.h(${qubits})\n`; break;
        case 'X': code += `qc.x(${qubits})\n`; break;
        case 'Y': code += `qc.y(${qubits})\n`; break;
        case 'Z': code += `qc.z(${qubits})\n`; break;
        case 'S': code += `qc.s(${qubits})\n`; break;
        case 'T': code += `qc.t(${qubits})\n`; break;
        case 'CNOT': code += `qc.cx(${qubits})\n`; break;
        case 'CZ': code += `qc.cz(${qubits})\n`; break;
        case 'SWAP': code += `qc.swap(${qubits})\n`; break;
        case 'RX': code += `qc.rx(${gate.params?.[0] || 0}, ${qubits})\n`; break;
        case 'RY': code += `qc.ry(${gate.params?.[0] || 0}, ${qubits})\n`; break;
        case 'RZ': code += `qc.rz(${gate.params?.[0] || 0}, ${qubits})\n`; break;
        case 'PHASE': code += `qc.p(${gate.params?.[0] || 0}, ${qubits})\n`; break;
        case 'U':
          const [theta = 0, phi = 0, lambda = 0] = gate.params || [];
          code += `qc.u(${theta}, ${phi}, ${lambda}, ${qubits})\n`;
          break;
        case 'CRX': code += `qc.crx(${gate.params?.[0] || 0}, ${qubits})\n`; break;
        case 'CRY': code += `qc.cry(${gate.params?.[0] || 0}, ${qubits})\n`; break;
        case 'CRZ': code += `qc.crz(${gate.params?.[0] || 0}, ${qubits})\n`; break;
        case 'TOFFOLI': code += `qc.ccx(${qubits})\n`; break;
        case 'FREDKIN': code += `qc.cswap(${qubits})\n`; break;
        case 'MEASURE': 
          code += `qc.measure(qr[${gate.qubits[0]}], cr[${measureIndex}])\n`;
          measureIndex++;
          break;
        case 'BARRIER':
          code += `qc.barrier(${qubits})\n`;
          break;
        default:
          code += `# ${gate.type} gate not directly supported\n`;
      }
    });
    
    code += '\n# Execute circuit\n';
    code += 'backend = Aer.get_backend("qasm_simulator")\n';
    code += 'job = execute(qc, backend, shots=1024)\n';
    code += 'result = job.result()\n';
    code += 'counts = result.get_counts(qc)\n';
    code += 'print(counts)\n';
    code += 'plot_histogram(counts)\n';
    
    return code;
  }, []);
  
  const convertToQASM = useCallback((circuit: QuantumCircuit): string => {
    let qasm = 'OPENQASM 2.0;\n';
    qasm += 'include "qelib1.inc";\n\n';
    
    // Register declarations
    qasm += `qreg q[${circuit.wires.length}];\n`;
    const measureCount = circuit.gates.filter(g => g.type === 'MEASURE').length;
    if (measureCount > 0) {
      qasm += `creg c[${measureCount}];\n`;
    }
    qasm += '\n';
    
    // Gates
    let measureIndex = 0;
    circuit.gates.forEach(gate => {
      const def = GATE_DEFINITIONS[gate.type];
      const qubits = gate.qubits.map(q => `q[${q}]`).join(',');
      
      switch (gate.type) {
        case 'H': qasm += `h ${qubits};\n`; break;
        case 'X': qasm += `x ${qubits};\n`; break;
        case 'Y': qasm += `y ${qubits};\n`; break;
        case 'Z': qasm += `z ${qubits};\n`; break;
        case 'S': qasm += `s ${qubits};\n`; break;
        case 'T': qasm += `t ${qubits};\n`; break;
        case 'CNOT': qasm += `cx ${qubits};\n`; break;
        case 'CZ': qasm += `cz ${qubits};\n`; break;
        case 'SWAP': qasm += `swap ${qubits};\n`; break;
        case 'RX': qasm += `rx(${gate.params?.[0] || 0}) ${qubits};\n`; break;
        case 'RY': qasm += `ry(${gate.params?.[0] || 0}) ${qubits};\n`; break;
        case 'RZ': qasm += `rz(${gate.params?.[0] || 0}) ${qubits};\n`; break;
        case 'MEASURE': 
          qasm += `measure q[${gate.qubits[0]}] -> c[${measureIndex}];\n`;
          measureIndex++;
          break;
        case 'BARRIER':
          qasm += `barrier ${qubits};\n`;
          break;
        default:
          qasm += `// ${def.name} gate not supported in QASM\n`;
      }
    });
    
    return qasm;
  }, []);
  
  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (readOnly) return;
      
      // Ctrl/Cmd + S: Save
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        onSave?.(circuit);
      }
      
      // Delete: Remove selected gate
      if (e.key === 'Delete' && hoveredGate) {
        removeGate(hoveredGate);
      }
      
      // Escape: Clear selection
      if (e.key === 'Escape') {
        setSelectedGate(null);
        setSelectedWires([]);
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [circuit, hoveredGate, readOnly, onSave, removeGate]);
  
  // Listen for circuit updates from code editor
  useEffect(() => {
    const handleUpdateCircuit = (event: CustomEvent) => {
      const { circuit: newCircuit } = event.detail;
      if (newCircuit && !readOnly) {
        setCircuit(newCircuit);
        showNotification('Circuit updated from code editor', 'success');
      }
    };
    
    window.addEventListener('updateCircuitDesigner', handleUpdateCircuit as EventListener);
    return () => {
      window.removeEventListener('updateCircuitDesigner', handleUpdateCircuit as EventListener);
    };
  }, [readOnly]);
  
  // Component is always ready - no loading state needed
  
  try {
    return (
      <div className="quantum-circuit-designer" style={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        background: '#0a0a0a',
        color: '#fff',
        fontFamily: '"JetBrains Mono", monospace'
      }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: `1px solid ${primaryColor}30`,
        background: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Cpu size={20} color={primaryColor} />
          <input
            type="text"
            value={circuit.name}
            onChange={(e) => setCircuit(prev => ({ ...prev, name: e.target.value }))}
            readOnly={readOnly}
            style={{
              background: 'transparent',
              border: `1px solid ${primaryColor}30`,
              borderRadius: '4px',
              padding: '4px 8px',
              color: '#fff',
              fontSize: '14px',
              fontWeight: '600'
            }}
          />
          <span style={{ color: '#666', fontSize: '12px' }}>
            {circuit.wires.length} qubits, {circuit.gates.length} gates
          </span>
        </div>
        
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {/* Backend selector */}
          <select
            value={selectedBackend}
            onChange={(e) => setSelectedBackend(e.target.value as HardwareBackend)}
            style={{
              background: '#1a1a1a',
              border: `1px solid ${primaryColor}30`,
              borderRadius: '4px',
              padding: '6px 12px',
              color: '#fff',
              fontSize: '12px',
              cursor: 'pointer',
              outline: 'none'
            }}
          >
            <option value="simulator" style={{ background: '#1a1a1a', color: '#fff' }}>Simulator</option>
            <option value="ibm_quantum" style={{ background: '#1a1a1a', color: '#fff' }}>IBM Quantum</option>
            <option value="rigetti" style={{ background: '#1a1a1a', color: '#fff' }}>Rigetti</option>
            <option value="google" style={{ background: '#1a1a1a', color: '#fff' }}>Google</option>
            <option value="ionq" style={{ background: '#1a1a1a', color: '#fff' }}>IonQ</option>
            <option value="honeywell" style={{ background: '#1a1a1a', color: '#fff' }}>Honeywell</option>
          </select>
          
          {/* Shots input */}
          <input
            type="number"
            value={shots}
            onChange={(e) => setShots(parseInt(e.target.value) || 1024)}
            min={1}
            max={8192}
            style={{
              background: 'rgba(0, 0, 0, 0.5)',
              border: `1px solid ${primaryColor}30`,
              borderRadius: '4px',
              padding: '6px 12px',
              color: '#fff',
              fontSize: '12px',
              width: '80px'
            }}
          />
          
          {/* Action buttons */}
          <Tooltip primaryColor={primaryColor} content="Run Circuit">
            <button
              onClick={executeCircuit}
              disabled={isExecuting || circuit.gates.length === 0}
              style={{
                padding: '6px 12px',
                background: isExecuting ? '#666' : primaryColor,
                border: 'none',
                borderRadius: '4px',
                color: '#000',
                cursor: isExecuting ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '12px',
                fontWeight: '500'
              }}
            >
              {isExecuting ? (
                <div className="animate-spin" style={{ display: 'inline-flex' }}>
                  <Loader size={14} />
                </div>
              ) : (
                <Play size={14} />
              )}
              Run
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Save Circuit">
            <button
              onClick={() => {
                // Save to localStorage for integration with code editor
                localStorage.setItem(`circuit-${circuit.id}`, JSON.stringify(circuit));
                
                // Call parent save handler if provided
                if (onSave) {
                  onSave(circuit);
                }
                
                showNotification('Circuit saved', 'success');
              }}
              disabled={readOnly}
              style={{
                padding: '6px',
                background: 'transparent',
                border: `1px solid ${primaryColor}30`,
                borderRadius: '4px',
                color: primaryColor,
                cursor: readOnly ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              <Save size={14} />
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Export">
            <button
              onClick={() => {
                const format = prompt('Export format: qasm, qiskit, or recursia') as any;
                if (format && ['qasm', 'qiskit', 'recursia'].includes(format)) {
                  let content = '';
                  let filename = circuit.name.toLowerCase().replace(/\s+/g, '_');
                  
                  switch (format) {
                    case 'recursia':
                      content = convertToRecursia(circuit);
                      filename += '.recursia';
                      break;
                    case 'qasm':
                      content = convertToQASM(circuit);
                      filename += '.qasm';
                      break;
                    case 'qiskit':
                      content = convertToQiskit(circuit);
                      filename += '.py';
                      break;
                  }
                  
                  // Download the file
                  const blob = new Blob([content], { type: 'text/plain' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = filename;
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                  URL.revokeObjectURL(url);
                  
                  showNotification(`Circuit exported as ${format.toUpperCase()}`, 'success');
                  
                  // Call parent export handler if provided
                  onExport?.(circuit, format);
                }
              }}
              style={{
                padding: '6px',
                background: 'transparent',
                border: `1px solid ${primaryColor}30`,
                borderRadius: '4px',
                color: primaryColor,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              <Download size={14} />
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Open in Code Editor">
            <button
              onClick={() => {
                const recursiaCode = convertToRecursia(circuit);
                // Find the code editor window and update it
                const codeEditorWindow = document.querySelector('[data-window-id="code-editor"]');
                if (codeEditorWindow) {
                  // Dispatch custom event to update code editor
                  const event = new CustomEvent('updateCodeEditor', { 
                    detail: { code: recursiaCode, filename: `${circuit.name}.recursia` }
                  });
                  window.dispatchEvent(event);
                  showNotification('Circuit opened in Code Editor', 'success');
                } else {
                  // Copy to clipboard as fallback
                  navigator.clipboard.writeText(recursiaCode);
                  showNotification('Circuit code copied to clipboard. Open Code Editor to paste.', 'info');
                }
              }}
              style={{
                padding: '6px',
                background: 'transparent',
                border: `1px solid ${primaryColor}30`,
                borderRadius: '4px',
                color: primaryColor,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              <Code size={14} />
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Settings">
            <button
              onClick={() => setShowSettings(!showSettings)}
              style={{
                padding: '6px',
                background: showSettings ? `${primaryColor}20` : 'transparent',
                border: `1px solid ${primaryColor}30`,
                borderRadius: '4px',
                color: primaryColor,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              <Settings size={14} />
            </button>
          </Tooltip>
        </div>
      </div>
      
      {/* Main content */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Gate library */}
        <AnimatePresence>
          {showGateLibrary && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 200, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              style={{
                borderRight: `1px solid ${primaryColor}30`,
                background: 'rgba(0, 0, 0, 0.3)',
                padding: '16px',
                overflowY: 'auto'
              }}
            >
              <h3 style={{ fontSize: '14px', marginBottom: '12px', color: primaryColor }}>
                Gate Library
              </h3>
              
              {/* Gate categories */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {/* Single-qubit gates */}
                <div>
                  <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
                    Single-Qubit
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px' }}>
                    {['H', 'X', 'Y', 'Z', 'S', 'T', 'RX', 'RY', 'RZ'].map(type => {
                      const def = GATE_DEFINITIONS[type as GateType];
                      return (
                        <Tooltip key={type} primaryColor={primaryColor} content={def.description}>
                          <button
                            onClick={() => setSelectedGate(type as GateType)}
                            style={{
                              width: '100%',
                              height: '36px',
                              padding: '8px',
                              background: selectedGate === type ? def.color : 'rgba(255, 255, 255, 0.05)',
                              border: `1px solid ${selectedGate === type ? def.color : '#333'}`,
                              borderRadius: '4px',
                              color: selectedGate === type ? '#000' : def.color,
                              cursor: 'pointer',
                              fontSize: '12px',
                              fontWeight: '600',
                              transition: 'all 0.2s',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center'
                            }}
                          >
                            {def.symbol}
                          </button>
                        </Tooltip>
                      );
                    })}
                  </div>
                </div>
                
                {/* Two-qubit gates */}
                <div>
                  <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
                    Two-Qubit
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px' }}>
                    {['CNOT', 'CZ', 'SWAP', 'CRX', 'CRY', 'CRZ'].map(type => {
                      const def = GATE_DEFINITIONS[type as GateType];
                      return (
                        <Tooltip key={type} primaryColor={primaryColor} content={def.description}>
                          <button
                            onClick={() => setSelectedGate(type as GateType)}
                            style={{
                              width: '100%',
                              height: '36px',
                              padding: '8px',
                              background: selectedGate === type ? def.color : 'rgba(255, 255, 255, 0.05)',
                              border: `1px solid ${selectedGate === type ? def.color : '#333'}`,
                              borderRadius: '4px',
                              color: selectedGate === type ? '#000' : def.color,
                              cursor: 'pointer',
                              fontSize: '12px',
                              fontWeight: '600',
                              transition: 'all 0.2s',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center'
                            }}
                          >
                            {def.symbol}
                          </button>
                        </Tooltip>
                      );
                    })}
                  </div>
                </div>
                
                {/* Multi-qubit and special gates */}
                <div>
                  <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
                    Multi-Qubit & Special
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px' }}>
                    {['TOFFOLI', 'FREDKIN', 'QFT', 'GROVER', 'ORACLE', 'TELEPORT'].map(type => {
                      const def = GATE_DEFINITIONS[type as GateType];
                      return (
                        <Tooltip key={type} primaryColor={primaryColor} content={def.description}>
                          <button
                            onClick={() => setSelectedGate(type as GateType)}
                            style={{
                              width: '100%',
                              height: '36px',
                              padding: '8px',
                              background: selectedGate === type ? def.color : 'rgba(255, 255, 255, 0.05)',
                              border: `1px solid ${selectedGate === type ? def.color : '#333'}`,
                              borderRadius: '4px',
                              color: selectedGate === type ? '#000' : def.color,
                              cursor: 'pointer',
                              fontSize: '12px',
                              fontWeight: '600',
                              transition: 'all 0.2s',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center'
                            }}
                          >
                            {def.symbol}
                          </button>
                        </Tooltip>
                      );
                    })}
                  </div>
                </div>
                
                {/* Tools */}
                <div>
                  <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
                    Tools
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px' }}>
                    {['MEASURE', 'BARRIER'].map(type => {
                      const def = GATE_DEFINITIONS[type as GateType];
                      return (
                        <Tooltip key={type} primaryColor={primaryColor} content={def.description}>
                          <button
                            onClick={() => setSelectedGate(type as GateType)}
                            style={{
                              width: '100%',
                              height: '36px',
                              padding: '8px',
                              background: selectedGate === type ? def.color : 'rgba(255, 255, 255, 0.05)',
                              border: `1px solid ${selectedGate === type ? def.color : '#333'}`,
                              borderRadius: '4px',
                              color: selectedGate === type ? '#000' : def.color,
                              cursor: 'pointer',
                              fontSize: '12px',
                              fontWeight: '600',
                              transition: 'all 0.2s',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center'
                            }}
                          >
                            {def.symbol}
                          </button>
                        </Tooltip>
                      );
                    })}
                  </div>
                </div>
              </div>
              
              {/* Instructions */}
              <div style={{
                marginTop: '24px',
                padding: '12px',
                background: 'rgba(255, 255, 255, 0.05)',
                borderRadius: '6px',
                fontSize: '11px',
                color: '#888'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                  <Info size={12} />
                  <strong>Quick Guide</strong>
                </div>
                <ul style={{ margin: 0, paddingLeft: '16px', lineHeight: 1.6 }}>
                  <li>Select a gate type</li>
                  <li>Click on wires to place</li>
                  <li>Hover + Delete to remove</li>
                  <li>Drag to reorder gates</li>
                  <li>Double-click to edit params</li>
                </ul>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Circuit canvas */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          {/* Toolbar */}
          <div style={{
            padding: '8px 16px',
            borderBottom: `1px solid ${primaryColor}30`,
            background: 'rgba(0, 0, 0, 0.3)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div style={{ display: 'flex', gap: '8px' }}>
              <Tooltip primaryColor={primaryColor} content="Toggle Gate Library">
                <button
                  onClick={() => setShowGateLibrary(!showGateLibrary)}
                  style={{
                    padding: '6px',
                    background: showGateLibrary ? `${primaryColor}20` : 'transparent',
                    border: `1px solid ${primaryColor}30`,
                    borderRadius: '4px',
                    color: primaryColor,
                    cursor: 'pointer'
                  }}
                >
                  <Layers size={14} />
                </button>
              </Tooltip>
              
              <Tooltip primaryColor={primaryColor} content="Add Qubit">
                <button
                  onClick={addWire}
                  disabled={readOnly || circuit.wires.length >= 50}
                  style={{
                    padding: '6px',
                    background: 'transparent',
                    border: `1px solid ${primaryColor}30`,
                    borderRadius: '4px',
                    color: primaryColor,
                    cursor: readOnly ? 'not-allowed' : 'pointer'
                  }}
                >
                  <Plus size={14} />
                </button>
              </Tooltip>
              
              <Tooltip primaryColor={primaryColor} content="Clear Circuit">
                <button
                  onClick={() => setCircuit(prev => ({ ...prev, gates: [] }))}
                  disabled={readOnly || circuit.gates.length === 0}
                  style={{
                    padding: '6px',
                    background: 'transparent',
                    border: `1px solid ${primaryColor}30`,
                    borderRadius: '4px',
                    color: primaryColor,
                    cursor: readOnly ? 'not-allowed' : 'pointer'
                  }}
                >
                  <Trash2 size={14} />
                </button>
              </Tooltip>
              
              <div style={{ width: '1px', height: '20px', background: '#333', margin: '0 4px' }} />
              
              <Tooltip primaryColor={primaryColor} content="Zoom In">
                <button
                  onClick={() => setZoom(prev => Math.min(prev + 0.1, 2))}
                  style={{
                    padding: '6px',
                    background: 'transparent',
                    border: `1px solid ${primaryColor}30`,
                    borderRadius: '4px',
                    color: primaryColor,
                    cursor: 'pointer'
                  }}
                >
                  <Plus size={14} />
                </button>
              </Tooltip>
              
              <span style={{ color: '#666', fontSize: '12px', padding: '0 8px' }}>
                {Math.round(zoom * 100)}%
              </span>
              
              <Tooltip primaryColor={primaryColor} content="Zoom Out">
                <button
                  onClick={() => setZoom(prev => Math.max(prev - 0.1, 0.5))}
                  style={{
                    padding: '6px',
                    background: 'transparent',
                    border: `1px solid ${primaryColor}30`,
                    borderRadius: '4px',
                    color: primaryColor,
                    cursor: 'pointer'
                  }}
                >
                  <Minus size={14} />
                </button>
              </Tooltip>
              
              <Tooltip primaryColor={primaryColor} content="Reset View">
                <button
                  onClick={() => {
                    setZoom(1);
                    setPanOffset({ x: 0, y: 0 });
                  }}
                  style={{
                    padding: '6px',
                    background: 'transparent',
                    border: `1px solid ${primaryColor}30`,
                    borderRadius: '4px',
                    color: primaryColor,
                    cursor: 'pointer'
                  }}
                >
                  <RefreshCw size={14} />
                </button>
              </Tooltip>
            </div>
            
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              {/* Circuit info */}
              <span style={{ fontSize: '11px', color: '#666' }}>
                Depth: {calculateCircuitDepth()}
              </span>
              
              {/* Connection status */}
              {isConnected && (
                <span style={{
                  padding: '2px 6px',
                  background: `${primaryColor}30`,
                  borderRadius: '3px',
                  fontSize: '10px',
                  color: primaryColor
                }}>
                  CONNECTED
                </span>
              )}
            </div>
          </div>
          
          {/* Circuit visualization */}
          <div 
            ref={canvasRef}
            style={{
              flex: 1,
              position: 'relative',
              overflow: 'hidden',
              background: '#0a0a0a',
              minHeight: '300px',
              backgroundImage: `
                linear-gradient(${primaryColor}10 1px, transparent 1px),
                linear-gradient(90deg, ${primaryColor}10 1px, transparent 1px)
              `,
              backgroundSize: `${gridSize * zoom}px ${gridSize * zoom}px`,
              backgroundPosition: `${panOffset.x}px ${panOffset.y}px`,
              cursor: selectedGate ? 'crosshair' : isPanning ? 'grabbing' : 'grab',
              userSelect: 'none'
            }}
            onMouseDown={(e) => {
              if (!selectedGate && e.button === 0) {
                setIsPanning(true);
                setPanStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
              }
            }}
            onMouseMove={(e) => {
              if (isPanning) {
                setPanOffset({
                  x: e.clientX - panStart.x,
                  y: e.clientY - panStart.y
                });
              }
            }}
            onMouseUp={() => setIsPanning(false)}
            onMouseLeave={() => setIsPanning(false)}
            onWheel={(e) => {
              e.preventDefault();
              const delta = e.deltaY > 0 ? -0.1 : 0.1;
              setZoom(prev => Math.max(0.5, Math.min(2, prev + delta)));
            }}
            onClick={(e) => {
              if (!selectedGate || readOnly || !isCanvasReady) return;
              
              const rect = canvasRef.current?.getBoundingClientRect();
              if (!rect || rect.width === 0 || rect.height === 0) return;
              
              const x = (e.clientX - rect.left - panOffset.x) / zoom;
              const y = (e.clientY - rect.top - panOffset.y) / zoom;
              
              // Find nearest wire
              const wireIndex = Math.round((y - 50) / gridSize);
              if (wireIndex >= 0 && wireIndex < circuit.wires.length) {
                const gateDef = GATE_DEFINITIONS[selectedGate];
                
                if (gateDef.qubits === 1) {
                  addGate(selectedGate, [wireIndex]);
                } else if (selectedWires.length === 0) {
                  setSelectedWires([wireIndex]);
                } else if (selectedWires.length < gateDef.qubits) {
                  const newWires = [...selectedWires, wireIndex];
                  if (newWires.length === gateDef.qubits) {
                    addGate(selectedGate, newWires);
                    setSelectedWires([]);
                  } else {
                    setSelectedWires(newWires);
                  }
                }
              }
            }}
          >
            <svg
              width={canvasSize.width}
              height={canvasSize.height}
              viewBox={`0 0 ${canvasSize.width} ${canvasSize.height}`}
              preserveAspectRatio="xMidYMid meet"
              style={{
                position: 'absolute',
                top: panOffset.y,
                left: panOffset.x,
                width: canvasSize.width * zoom,
                height: canvasSize.height * zoom,
                pointerEvents: isPanning ? 'none' : 'auto',
                transition: isPanning ? 'none' : 'transform 0.1s ease-out'
              }}
            >
              {/* Draw wires */}
              {circuit.wires.map((wire, idx) => {
                const y = 50 + idx * gridSize;
                const isSelected = selectedWires.includes(idx);
                
                // Validate y coordinate
                if (!isFinite(y)) {
                  // Invalid y coordinate for wire
                  return null;
                }
                
                return (
                  <g key={`wire-${wire.qubit}-${idx}`}>
                    {/* Wire line */}
                    <line
                      x1={20}
                      y1={y}
                      x2={canvasSize.width - 20}
                      y2={y}
                      stroke={isSelected ? primaryColor : '#333'}
                      strokeWidth={isSelected ? 2 : 1}
                      strokeDasharray={wire.classical ? '5,5' : undefined}
                    />
                    
                    {/* Wire label */}
                    <text
                      x={10}
                      y={y + 4}
                      fill={isSelected ? primaryColor : '#666'}
                      fontSize="12"
                      fontFamily="monospace"
                      textAnchor="end"
                    >
                      {wire.label}
                    </text>
                    
                    {/* Remove wire button */}
                    {!readOnly && circuit.wires.length > 1 && (
                      <g
                        onClick={(e) => {
                          e.stopPropagation();
                          removeWire(idx);
                        }}
                        style={{ cursor: 'pointer' }}
                      >
                        <circle
                          cx={canvasSize.width - 40}
                          cy={y}
                          r={8}
                          fill="#333"
                          stroke="#666"
                          strokeWidth={1}
                        />
                        <line
                          x1={canvasSize.width - 44}
                          y1={y}
                          x2={canvasSize.width - 36}
                          y2={y}
                          stroke="#999"
                          strokeWidth={1.5}
                        />
                      </g>
                    )}
                  </g>
                );
              })}
              
              {/* Draw gates */}
              {circuit.gates.map((gate, gateIdx) => {
                const def = GATE_DEFINITIONS[gate.type];
                if (!def) {
                  // Unknown gate type
                  return null;
                }
                
                const isHovered = hoveredGate === gate.id;
                
                // Calculate gate position
                const x = 100 + gateIdx * gridSize;
                const minQubit = gate.qubits.length > 0 ? Math.min(...gate.qubits) : 0;
                const maxQubit = gate.qubits.length > 0 ? Math.max(...gate.qubits) : 0;
                
                // Validate positions
                if (!isFinite(x) || !isFinite(minQubit) || !isFinite(maxQubit)) {
                  // Invalid gate position
                  return null;
                }
                
                return (
                  <g
                    key={gate.id}
                    onMouseEnter={() => setHoveredGate(gate.id)}
                    onMouseLeave={() => setHoveredGate(null)}
                    style={{ cursor: readOnly ? 'default' : 'pointer' }}
                  >
                    {/* Multi-qubit connection line */}
                    {gate.qubits.length > 1 && (
                      <line
                        x1={x}
                        y1={50 + minQubit * gridSize}
                        x2={x}
                        y2={50 + maxQubit * gridSize}
                        stroke={def.color}
                        strokeWidth={2}
                        opacity={0.5}
                      />
                    )}
                    
                    {/* Gate boxes */}
                    {gate.qubits.map((qubit, idx) => {
                      // Validate qubit index
                      if (typeof qubit !== 'number' || !isFinite(qubit)) {
                        // Invalid qubit index in gate
                        return null;
                      }
                      
                      const y = 50 + qubit * gridSize;
                      const isControl = gate.type.startsWith('C') && idx === 0;
                      
                      if (isControl && gate.type !== 'CUSTOM') {
                        // Control dot
                        return (
                          <circle
                            key={`${gate.id}-${qubit}`}
                            cx={x}
                            cy={y}
                            r={6}
                            fill={def.color}
                            stroke={isHovered ? '#fff' : def.color}
                            strokeWidth={isHovered ? 2 : 0}
                          />
                        );
                      } else if (gate.type === 'MEASURE') {
                        // Measurement gate
                        return (
                          <g key={`${gate.id}-${qubit}`}>
                            <rect
                              x={x - 20}
                              y={y - 20}
                              width={40}
                              height={40}
                              fill="#333"
                              stroke={isHovered ? '#fff' : def.color}
                              strokeWidth={2}
                              rx={4}
                            />
                            <path
                              d={`M ${x-10} ${y+5} Q ${x} ${y-10} ${x+10} ${y+5}`}
                              fill="none"
                              stroke={def.color}
                              strokeWidth={2}
                            />
                            <line
                              x1={x}
                              y1={y-10}
                              x2={x}
                              y2={y+10}
                              stroke={def.color}
                              strokeWidth={2}
                            />
                          </g>
                        );
                      } else {
                        // Regular gate box
                        return (
                          <g key={`${gate.id}-${qubit}`}>
                            <rect
                              x={x - 20}
                              y={y - 15}
                              width={40}
                              height={30}
                              fill={def.color}
                              stroke={isHovered ? '#fff' : def.color}
                              strokeWidth={isHovered ? 2 : 0}
                              rx={4}
                              opacity={0.9}
                            />
                            <text
                              x={x}
                              y={y + 4}
                              fill="#000"
                              fontSize="12"
                              fontWeight="600"
                              fontFamily="monospace"
                              textAnchor="middle"
                            >
                              {def.symbol}
                            </text>
                          </g>
                        );
                      }
                    })}
                    
                    {/* Gate parameters */}
                    {gate.params && gate.params.length > 0 && (
                      <text
                        x={x}
                        y={50 + Math.min(...gate.qubits) * gridSize - 20}
                        fill="#888"
                        fontSize="10"
                        fontFamily="monospace"
                        textAnchor="middle"
                      >
                        {gate.params.map(p => p.toFixed(2)).join(',')}
                      </text>
                    )}
                    
                    {/* Delete button on hover */}
                    {isHovered && !readOnly && (
                      <g
                        onClick={(e) => {
                          e.stopPropagation();
                          removeGate(gate.id);
                        }}
                      >
                        <circle
                          cx={x + 25}
                          cy={50 + Math.min(...gate.qubits) * gridSize - 25}
                          r={8}
                          fill="#ff4444"
                          stroke="#fff"
                          strokeWidth={1}
                        />
                        <line
                          x1={x + 22}
                          y1={50 + Math.min(...gate.qubits) * gridSize - 28}
                          x2={x + 28}
                          y2={50 + Math.min(...gate.qubits) * gridSize - 22}
                          stroke="#fff"
                          strokeWidth={2}
                        />
                        <line
                          x1={x + 28}
                          y1={50 + Math.min(...gate.qubits) * gridSize - 28}
                          x2={x + 22}
                          y2={50 + Math.min(...gate.qubits) * gridSize - 22}
                          stroke="#fff"
                          strokeWidth={2}
                        />
                      </g>
                    )}
                  </g>
                );
              })}
              
              {/* Selection preview */}
              {selectedGate && selectedWires.length > 0 && (
                <g opacity={0.5}>
                  {selectedWires.map(wire => (
                    <circle
                      key={`preview-${wire}`}
                      cx={100 + circuit.gates.length * gridSize}
                      cy={50 + wire * gridSize}
                      r={10}
                      fill={GATE_DEFINITIONS[selectedGate].color}
                      stroke="#fff"
                      strokeWidth={2}
                      strokeDasharray="3,3"
                    />
                  ))}
                </g>
              )}
            </svg>
          </div>
          
          {/* Results panel */}
          {executions.length > 0 && (
            <div style={{
              minHeight: '200px',
              maxHeight: '350px',
              borderTop: `1px solid ${primaryColor}30`,
              background: 'rgba(0, 0, 0, 0.3)',
              padding: '16px',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <h3 style={{ fontSize: '14px', margin: 0, color: primaryColor }}>
                  Execution Results
                </h3>
                <button
                  onClick={() => setExecutions([])}
                  style={{
                    padding: '4px 8px',
                    background: 'transparent',
                    border: `1px solid ${primaryColor}30`,
                    borderRadius: '4px',
                    color: '#666',
                    cursor: 'pointer',
                    fontSize: '11px'
                  }}
                >
                  Clear
                </button>
              </div>
              
              {executions.map(exec => (
                <div
                  key={exec.id}
                  style={{
                    marginBottom: '12px',
                    padding: '12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '6px',
                    border: `1px solid ${
                      exec.status === 'completed' ? '#4caf50' :
                      exec.status === 'failed' ? '#f44336' :
                      exec.status === 'running' ? primaryColor :
                      '#666'
                    }30`
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      {exec.status === 'completed' && <CheckCircle size={14} color="#4caf50" />}
                      {exec.status === 'failed' && <AlertCircle size={14} color="#f44336" />}
                      {exec.status === 'running' && (
                        <div className="animate-spin" style={{ display: 'inline-flex' }}>
                          <Loader size={14} color={primaryColor} />
                        </div>
                      )}
                      {exec.status === 'pending' && <Activity size={14} color="#666" />}
                      
                      <span style={{ fontSize: '12px', color: '#fff' }}>
                        {exec.backend} - {exec.shots} shots
                      </span>
                    </div>
                    
                    <span style={{ fontSize: '11px', color: '#666' }}>
                      {exec.endTime ? `${((exec.endTime - exec.startTime) / 1000).toFixed(2)}s` : 'Running...'}
                    </span>
                  </div>
                  
                  {exec.results && (
                    <div style={{ fontSize: '11px', color: '#888' }}>
                      {exec.results.error ? (
                        <div style={{ color: '#f44336' }}>Error: {exec.results.error}</div>
                      ) : (
                        <div>
                          {/* Show measurement counts */}
                          {Object.keys(exec.results.counts).length > 0 && (
                            <div style={{ marginTop: '12px' }}>
                              <strong style={{ color: primaryColor, fontSize: '12px' }}>Measurement Results:</strong>
                              <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                {Object.entries(exec.results.counts)
                                  .sort(([, a], [, b]) => b - a)
                                  .slice(0, 8)
                                  .map(([outcome, count]) => {
                                    const percentage = (count / exec.shots) * 100;
                                    return (
                                      <div key={outcome} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <span style={{
                                          fontFamily: 'monospace',
                                          fontSize: '12px',
                                          color: primaryColor,
                                          minWidth: '80px'
                                        }}>
                                          |{outcome}⟩
                                        </span>
                                        <div style={{
                                          flex: 1,
                                          height: '16px',
                                          background: 'rgba(255, 255, 255, 0.05)',
                                          borderRadius: '3px',
                                          overflow: 'hidden',
                                          position: 'relative'
                                        }}>
                                          <div style={{
                                            position: 'absolute',
                                            left: 0,
                                            top: 0,
                                            height: '100%',
                                            width: `${percentage}%`,
                                            background: `linear-gradient(90deg, ${primaryColor}40, ${primaryColor})`,
                                            transition: 'width 0.3s ease'
                                          }} />
                                          <span style={{
                                            position: 'absolute',
                                            left: '8px',
                                            top: '50%',
                                            transform: 'translateY(-50%)',
                                            fontSize: '10px',
                                            color: '#fff',
                                            fontWeight: '600'
                                          }}>
                                            {percentage.toFixed(1)}%
                                          </span>
                                        </div>
                                        <span style={{
                                          fontSize: '11px',
                                          color: '#666',
                                          minWidth: '40px',
                                          textAlign: 'right'
                                        }}>
                                          {count}
                                        </span>
                                      </div>
                                    );
                                  })}
                                {Object.keys(exec.results.counts).length > 8 && (
                                  <div style={{ fontSize: '10px', color: '#666', marginTop: '4px', fontStyle: 'italic' }}>
                                    ... and {Object.keys(exec.results.counts).length - 8} more outcomes
                                  </div>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      
      {/* Settings modal */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0, 0, 0, 0.8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000
            }}
            onClick={() => setShowSettings(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              style={{
                background: '#1a1a1a',
                border: `1px solid ${primaryColor}30`,
                borderRadius: '8px',
                padding: '24px',
                maxWidth: '500px',
                width: '90%'
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <h2 style={{ fontSize: '18px', marginBottom: '16px', color: primaryColor }}>
                Circuit Settings
              </h2>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {/* Circuit metadata */}
                <div>
                  <label style={{ fontSize: '12px', color: '#888', display: 'block', marginBottom: '4px' }}>
                    Description
                  </label>
                  <textarea
                    value={circuit.description || ''}
                    onChange={(e) => setCircuit(prev => ({ ...prev, description: e.target.value }))}
                    style={{
                      width: '100%',
                      height: '80px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: `1px solid ${primaryColor}30`,
                      borderRadius: '4px',
                      padding: '8px',
                      color: '#fff',
                      fontSize: '12px',
                      resize: 'vertical'
                    }}
                  />
                </div>
                
                {/* Optimization level */}
                <div>
                  <label style={{ fontSize: '12px', color: '#888', display: 'block', marginBottom: '4px' }}>
                    Optimization Level
                  </label>
                  <select
                    value={circuit.metadata.optimizationLevel || 1}
                    onChange={(e) => setCircuit(prev => ({
                      ...prev,
                      metadata: { ...prev.metadata, optimizationLevel: parseInt(e.target.value) }
                    }))}
                    style={{
                      width: '100%',
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: `1px solid ${primaryColor}30`,
                      borderRadius: '4px',
                      padding: '8px',
                      color: '#fff',
                      fontSize: '12px'
                    }}
                  >
                    <option value={0}>No optimization</option>
                    <option value={1}>Light optimization</option>
                    <option value={2}>Heavy optimization</option>
                    <option value={3}>Maximum optimization</option>
                  </select>
                </div>
                
                {/* Export options */}
                <div>
                  <label style={{ fontSize: '12px', color: '#888', display: 'block', marginBottom: '8px' }}>
                    Export Circuit
                  </label>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                      onClick={() => {
                        const code = convertToRecursia(circuit);
                        const blob = new Blob([code], { type: 'text/plain' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${circuit.name.replace(/\s+/g, '_')}.recursia`;
                        a.click();
                        URL.revokeObjectURL(url);
                        showNotification('Exported as Recursia code', 'success');
                        if (onExport) onExport(circuit, 'recursia');
                      }}
                      style={{
                        flex: 1,
                        padding: '8px',
                        background: `${primaryColor}20`,
                        border: `1px solid ${primaryColor}`,
                        borderRadius: '4px',
                        color: primaryColor,
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                    >
                      Recursia
                    </button>
                    <button
                      onClick={() => {
                        const qasm = convertToQASM(circuit);
                        const blob = new Blob([qasm], { type: 'text/plain' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${circuit.name.replace(/\s+/g, '_')}.qasm`;
                        a.click();
                        URL.revokeObjectURL(url);
                        showNotification('Exported as QASM', 'success');
                        if (onExport) onExport(circuit, 'qasm');
                      }}
                      style={{
                        flex: 1,
                        padding: '8px',
                        background: 'transparent',
                        border: `1px solid ${primaryColor}30`,
                        borderRadius: '4px',
                        color: primaryColor,
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                    >
                      QASM
                    </button>
                    <button
                      onClick={() => {
                        const json = JSON.stringify(circuit, null, 2);
                        const blob = new Blob([json], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${circuit.name.replace(/\s+/g, '_')}.json`;
                        a.click();
                        URL.revokeObjectURL(url);
                        showNotification('Exported as JSON', 'success');
                      }}
                      style={{
                        flex: 1,
                        padding: '8px',
                        background: 'transparent',
                        border: `1px solid ${primaryColor}30`,
                        borderRadius: '4px',
                        color: primaryColor,
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                    >
                      JSON
                    </button>
                  </div>
                </div>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '24px' }}>
                <button
                  onClick={() => setShowSettings(false)}
                  style={{
                    padding: '8px 16px',
                    background: primaryColor,
                    border: 'none',
                    borderRadius: '4px',
                    color: '#000',
                    cursor: 'pointer',
                    fontSize: '12px',
                    fontWeight: '500'
                  }}
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
    );
  } catch (error) {
    // Render error in QuantumCircuitDesigner
    return (
      <div style={{ 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0a0a0a',
        color: '#fff',
        padding: '24px'
      }}>
        <div style={{ textAlign: 'center' }}>
          <h3 style={{ color: '#ff4444' }}>Circuit Designer Error</h3>
          <p style={{ color: '#888' }}>{error instanceof Error ? error.message : 'Unknown error'}</p>
          <pre style={{ fontSize: '11px', color: '#666', marginTop: '16px' }}>
            {error instanceof Error ? error.stack : ''}
          </pre>
        </div>
      </div>
    );
  }
}