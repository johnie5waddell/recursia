/**
 * Circuit Integration Utilities
 * Functions to integrate quantum circuits with the code editor and simulation engine
 */

import { Complex } from './complex';

export interface CircuitBlock {
  id: string;
  name: string;
  description: string;
  code: string;
  category: 'basic' | 'intermediate' | 'advanced' | 'custom';
  tags: string[];
}

/**
 * Convert a saved circuit to a reusable code block
 */
export function circuitToCodeBlock(circuit: any): CircuitBlock {
  const code = generateCircuitCode(circuit);
  
  return {
    id: circuit.id,
    name: circuit.name,
    description: circuit.description || 'Quantum circuit',
    code,
    category: determineCategory(circuit),
    tags: generateTags(circuit)
  };
}

/**
 * Generate Recursia code from circuit definition
 */
function generateCircuitCode(circuit: any): string {
  let code = `// ${circuit.name}\n`;
  
  if (circuit.description) {
    code += `// ${circuit.description}\n`;
  }
  
  code += `// Auto-generated from Quantum Circuit Designer\n\n`;
  
  // Function wrapper for reusability
  code += `function ${sanitizeName(circuit.name)}() {\n`;
  
  // Create quantum states
  circuit.wires.forEach((wire: any) => {
    code += `  state ${wire.label} = |0⟩;\n`;
  });
  
  if (circuit.gates.length > 0) {
    code += '\n';
  }
  
  // Apply gates
  circuit.gates.forEach((gate: any) => {
    const wireLabels = gate.qubits.map((q: number) => 
      circuit.wires[q]?.label || `q${q}`
    );
    
    code += '  ';
    
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
        code += `// Quantum Fourier Transform on ${wireLabels.join(', ')}\n`;
        code += `  QFT ${wireLabels.join(', ')};\n`;
        break;
        
      case 'GROVER':
        code += `// Grover operator on ${wireLabels.join(', ')}\n`;
        code += `  Grover ${wireLabels.join(', ')};\n`;
        break;
        
      case 'ORACLE':
        code += `// Oracle function on ${wireLabels.join(', ')}\n`;
        code += `  Oracle ${wireLabels.join(', ')};\n`;
        break;
        
      case 'TELEPORT':
        code += `// Quantum teleportation protocol\n`;
        code += `  Teleport ${wireLabels.join(', ')};\n`;
        break;
        
      default:
        // Single-qubit gates and parameterized gates
        if (gate.params && gate.params.length > 0) {
          const paramStr = gate.params.map((p: number) => p.toFixed(4)).join(', ');
          code += `${gate.type}(${paramStr}) ${wireLabels.join(', ')};\n`;
        } else {
          code += `${gate.type} ${wireLabels.join(', ')};\n`;
        }
    }
  });
  
  // Return quantum states
  code += '\n';
  circuit.wires.forEach((wire: any) => {
    code += `  return ${wire.label};\n`;
  });
  
  code += '}\n\n';
  code += `// Execute circuit\n`;
  code += `${sanitizeName(circuit.name)}();\n`;
  
  return code;
}

/**
 * Determine circuit complexity category
 */
function determineCategory(circuit: any): 'basic' | 'intermediate' | 'advanced' | 'custom' {
  const gateCount = circuit.gates.length;
  const qubitCount = circuit.wires.length;
  const hasParameterized = circuit.gates.some((g: any) => g.params && g.params.length > 0);
  const hasMultiQubit = circuit.gates.some((g: any) => g.qubits.length > 2);
  
  if (gateCount <= 5 && qubitCount <= 3 && !hasParameterized) {
    return 'basic';
  } else if (gateCount <= 15 && qubitCount <= 5 && !hasMultiQubit) {
    return 'intermediate';
  } else if (hasMultiQubit || hasParameterized || qubitCount > 5) {
    return 'advanced';
  }
  
  return 'custom';
}

/**
 * Generate tags based on circuit properties
 */
function generateTags(circuit: any): string[] {
  const tags: string[] = [];
  
  // Gate type tags
  const gateTypes = new Set(circuit.gates.map((g: any) => g.type));
  
  if (gateTypes.has('H')) tags.push('superposition');
  if (gateTypes.has('CNOT') || gateTypes.has('CZ')) tags.push('entanglement');
  if (gateTypes.has('MEASURE')) tags.push('measurement');
  if (gateTypes.has('QFT')) tags.push('fourier');
  if (gateTypes.has('GROVER')) tags.push('search');
  if (gateTypes.has('TELEPORT')) tags.push('teleportation');
  
  // Parameterized gates
  if (circuit.gates.some((g: any) => ['RX', 'RY', 'RZ', 'U', 'PHASE'].includes(g.type))) {
    tags.push('rotation');
  }
  
  // Circuit size
  if (circuit.wires.length <= 2) tags.push('small');
  else if (circuit.wires.length <= 5) tags.push('medium');
  else tags.push('large');
  
  // Special properties
  if (circuit.gates.some((g: any) => g.type === 'ORACLE')) tags.push('oracle');
  if (circuit.gates.some((g: any) => g.type === 'CUSTOM')) tags.push('custom');
  
  return tags;
}

/**
 * Sanitize circuit name for use as function name
 */
function sanitizeName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9_]/g, '_')
    .replace(/^[0-9]/, '_$&')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '') || 'circuit';
}

/**
 * Load saved circuits from localStorage
 */
export function loadSavedCircuits(): any[] {
  const circuits: any[] = [];
  
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.startsWith('circuit-')) {
      try {
        const circuit = JSON.parse(localStorage.getItem(key) || '{}');
        if (circuit.id && circuit.name) {
          circuits.push(circuit);
        }
      } catch (error) {
        console.error('Error loading circuit:', key, error);
      }
    }
  }
  
  return circuits.sort((a, b) => b.metadata.modified - a.metadata.modified);
}

/**
 * Export circuit to various formats
 */
export function exportCircuit(circuit: any, format: 'qasm' | 'qiskit' | 'json' | 'svg'): string {
  switch (format) {
    case 'qasm':
      return convertToQASM(circuit);
    case 'qiskit':
      return convertToQiskit(circuit);
    case 'json':
      return JSON.stringify(circuit, null, 2);
    case 'svg':
      return generateCircuitSVG(circuit);
    default:
      throw new Error(`Unsupported export format: ${format}`);
  }
}

/**
 * Convert circuit to OpenQASM 2.0
 */
export function convertToQASM(circuit: any): string {
  let qasm = 'OPENQASM 2.0;\n';
  qasm += 'include "qelib1.inc";\n\n';
  
  // Register declarations
  qasm += `qreg q[${circuit.wires.length}];\n`;
  const measureCount = circuit.gates.filter((g: any) => g.type === 'MEASURE').length;
  if (measureCount > 0) {
    qasm += `creg c[${measureCount}];\n`;
  }
  qasm += '\n';
  
  // Gates
  let measureIndex = 0;
  circuit.gates.forEach((gate: any) => {
    const qubits = gate.qubits.map((q: number) => `q[${q}]`).join(',');
    
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
      case 'PHASE': qasm += `u1(${gate.params?.[0] || 0}) ${qubits};\n`; break;
      case 'U':
        const [theta = 0, phi = 0, lambda = 0] = gate.params || [];
        qasm += `u3(${theta},${phi},${lambda}) ${qubits};\n`;
        break;
      case 'TOFFOLI': qasm += `ccx ${qubits};\n`; break;
      case 'MEASURE': 
        qasm += `measure q[${gate.qubits[0]}] -> c[${measureIndex}];\n`;
        measureIndex++;
        break;
      case 'BARRIER':
        qasm += `barrier ${qubits};\n`;
        break;
      default:
        qasm += `// ${gate.type} gate not supported in QASM\n`;
    }
  });
  
  return qasm;
}

/**
 * Convert circuit to Qiskit Python code
 */
export function convertToQiskit(circuit: any): string {
  let code = '# Auto-generated Qiskit code\n';
  code += 'from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister\n';
  code += 'from qiskit import execute, Aer\n';
  code += 'from qiskit.visualization import plot_histogram\n';
  code += 'import numpy as np\n\n';
  
  // Circuit creation
  code += `# Create circuit: ${circuit.name}\n`;
  code += `qr = QuantumRegister(${circuit.wires.length}, 'q')\n`;
  
  const measureCount = circuit.gates.filter((g: any) => g.type === 'MEASURE').length;
  if (measureCount > 0) {
    code += `cr = ClassicalRegister(${measureCount}, 'c')\n`;
    code += 'qc = QuantumCircuit(qr, cr)\n';
  } else {
    code += 'qc = QuantumCircuit(qr)\n';
  }
  
  code += '\n# Apply gates\n';
  
  // Gates
  let measureIndex = 0;
  circuit.gates.forEach((gate: any) => {
    const qubits = gate.qubits.map((q: number) => `qr[${q}]`).join(', ');
    
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
}

/**
 * Generate SVG representation of circuit
 */
function generateCircuitSVG(circuit: any): string {
  const width = 800;
  const height = 50 + circuit.wires.length * 50 + 50;
  const gridSize = 50;
  
  let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">\n`;
  svg += '  <defs>\n';
  svg += '    <style>\n';
  svg += '      .wire { stroke: #333; stroke-width: 1; }\n';
  svg += '      .gate { stroke-width: 2; }\n';
  svg += '      .gate-text { font-family: monospace; font-size: 12px; text-anchor: middle; }\n';
  svg += '      .wire-label { font-family: monospace; font-size: 12px; fill: #666; }\n';
  svg += '    </style>\n';
  svg += '  </defs>\n';
  
  // Background
  svg += `  <rect width="${width}" height="${height}" fill="white"/>\n`;
  
  // Draw wires
  circuit.wires.forEach((wire: any, idx: number) => {
    const y = 50 + idx * gridSize;
    svg += `  <line x1="20" y1="${y}" x2="${width - 20}" y2="${y}" class="wire"/>\n`;
    svg += `  <text x="10" y="${y + 4}" class="wire-label" text-anchor="end">${wire.label}</text>\n`;
  });
  
  // Draw gates
  circuit.gates.forEach((gate: any, gateIdx: number) => {
    const x = 100 + gateIdx * gridSize;
    const color = gate.color || '#4fc3f7';
    
    gate.qubits.forEach((qubit: number, idx: number) => {
      const y = 50 + qubit * gridSize;
      
      if (gate.type === 'CNOT' && idx === 0) {
        // Control dot
        svg += `  <circle cx="${x}" cy="${y}" r="6" fill="${color}" class="gate"/>\n`;
      } else if (gate.type === 'MEASURE') {
        // Measurement symbol
        svg += `  <rect x="${x - 20}" y="${y - 20}" width="40" height="40" fill="none" stroke="${color}" class="gate" rx="4"/>\n`;
        svg += `  <path d="M ${x-10} ${y+5} Q ${x} ${y-10} ${x+10} ${y+5}" fill="none" stroke="${color}" stroke-width="2"/>\n`;
        svg += `  <line x1="${x}" y1="${y-10}" x2="${x}" y2="${y+10}" stroke="${color}" stroke-width="2"/>\n`;
      } else {
        // Regular gate box
        svg += `  <rect x="${x - 20}" y="${y - 15}" width="40" height="30" fill="${color}" class="gate" rx="4" opacity="0.9"/>\n`;
        svg += `  <text x="${x}" y="${y + 4}" class="gate-text" fill="black" font-weight="600">${gate.type}</text>\n`;
      }
    });
    
    // Multi-qubit connections
    if (gate.qubits.length > 1) {
      const minY = 50 + Math.min(...gate.qubits) * gridSize;
      const maxY = 50 + Math.max(...gate.qubits) * gridSize;
      svg += `  <line x1="${x}" y1="${minY}" x2="${x}" y2="${maxY}" stroke="${color}" stroke-width="2" opacity="0.5"/>\n`;
    }
  });
  
  svg += '</svg>';
  
  return svg;
}

/**
 * Validate circuit against hardware constraints
 */
export function validateForHardware(circuit: any, backend: string): {
  valid: boolean;
  errors: string[];
  warnings: string[];
} {
  const errors: string[] = [];
  const warnings: string[] = [];
  
  // Backend-specific constraints
  const constraints: Record<string, any> = {
    'ibm_quantum': {
      maxQubits: 127,
      supportedGates: ['H', 'X', 'Y', 'Z', 'S', 'T', 'CNOT', 'RX', 'RY', 'RZ', 'MEASURE'],
      connectivity: 'limited',
      gateTime: { '1q': 35, '2q': 300 } // ns
    },
    'rigetti': {
      maxQubits: 80,
      supportedGates: ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'CNOT', 'CZ', 'MEASURE'],
      connectivity: 'limited',
      gateTime: { '1q': 40, '2q': 180 }
    },
    'google': {
      maxQubits: 72,
      supportedGates: ['H', 'X', 'Y', 'Z', 'S', 'T', 'CNOT', 'CZ', 'SWAP', 'MEASURE'],
      connectivity: 'grid',
      gateTime: { '1q': 25, '2q': 40 }
    },
    'ionq': {
      maxQubits: 32,
      supportedGates: ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'CNOT', 'MEASURE'],
      connectivity: 'all-to-all',
      gateTime: { '1q': 100, '2q': 600 }
    }
  };
  
  const constraint = constraints[backend];
  if (!constraint) {
    return { valid: true, errors: [], warnings: [] };
  }
  
  // Check qubit count
  if (circuit.wires.length > constraint.maxQubits) {
    errors.push(`Circuit uses ${circuit.wires.length} qubits, but ${backend} supports max ${constraint.maxQubits}`);
  }
  
  // Check supported gates
  circuit.gates.forEach((gate: any, idx: number) => {
    if (!constraint.supportedGates.includes(gate.type)) {
      warnings.push(`Gate ${gate.type} at position ${idx} may require transpilation for ${backend}`);
    }
  });
  
  // Check connectivity (simplified)
  if (constraint.connectivity === 'limited') {
    const twoQubitGates = circuit.gates.filter((g: any) => g.qubits.length === 2);
    const farConnections = twoQubitGates.filter((g: any) => 
      Math.abs(g.qubits[0] - g.qubits[1]) > 1
    );
    
    if (farConnections.length > 0) {
      warnings.push(`${farConnections.length} two-qubit gates may require SWAP gates due to limited connectivity`);
    }
  }
  
  // Estimate execution time
  let totalTime = 0;
  circuit.gates.forEach((gate: any) => {
    const isOneQubit = gate.qubits.length === 1;
    totalTime += constraint.gateTime[isOneQubit ? '1q' : '2q'] || 100;
  });
  
  if (totalTime > 100000) { // 100 microseconds
    warnings.push(`Circuit depth may lead to decoherence (estimated time: ${(totalTime / 1000).toFixed(1)}μs)`);
  }
  
  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}

/**
 * Convert a quantum circuit to Recursia code
 * @param circuit - The quantum circuit to convert
 * @returns Recursia code as a string
 */
export function convertToRecursia(circuit: any): string {
  let code = `// ${circuit.name}\n`;
  if (circuit.description) {
    code += `// ${circuit.description}\n`;
  }
  code += `// Generated by Quantum Circuit Designer\n\n`;
  
  // Create quantum states
  circuit.wires.forEach((wire: any) => {
    if (!wire.classical) {
      code += `state ${wire.label} = |0⟩;\n`;
    }
  });
  
  if (circuit.wires.length > 0) {
    code += '\n';
  }
  
  // Add gates
  circuit.gates.forEach((gate: any) => {
    const wireLabels = gate.qubits.map((q: number) => circuit.wires[q]?.label || `q${q}`);
    
    switch (gate.type) {
      case 'H':
        code += `H ${wireLabels[0]};\n`;
        break;
      case 'X':
        code += `X ${wireLabels[0]};\n`;
        break;
      case 'Y':
        code += `Y ${wireLabels[0]};\n`;
        break;
      case 'Z':
        code += `Z ${wireLabels[0]};\n`;
        break;
      case 'S':
        code += `S ${wireLabels[0]};\n`;
        break;
      case 'T':
        code += `T ${wireLabels[0]};\n`;
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
      case 'RX':
        code += `RX(${gate.params?.[0]?.toFixed(4) || 0}) ${wireLabels[0]};\n`;
        break;
      case 'RY':
        code += `RY(${gate.params?.[0]?.toFixed(4) || 0}) ${wireLabels[0]};\n`;
        break;
      case 'RZ':
        code += `RZ(${gate.params?.[0]?.toFixed(4) || 0}) ${wireLabels[0]};\n`;
        break;
      case 'MEASURE':
        code += `measure ${wireLabels[0]};\n`;
        break;
      case 'BARRIER':
        code += `// Optimization barrier\n`;
        break;
      case 'TOFFOLI':
        code += `TOFFOLI ${wireLabels.join(', ')};\n`;
        break;
      case 'FREDKIN':
        code += `FREDKIN ${wireLabels.join(', ')};\n`;
        break;
      case 'QFT':
        code += `QFT ${wireLabels.join(', ')};\n`;
        break;
      case 'GROVER':
        code += `// Grover operator\n`;
        code += `GROVER ${wireLabels.join(', ')};\n`;
        break;
      case 'TELEPORT':
        code += `// Quantum teleportation protocol\n`;
        code += `TELEPORT ${wireLabels.join(', ')};\n`;
        break;
      default:
        if (gate.params && gate.params.length > 0) {
          const paramStr = gate.params.map((p: number) => p.toFixed(4)).join(', ');
          code += `${gate.type}(${paramStr}) ${wireLabels.join(', ')};\n`;
        } else {
          code += `${gate.type} ${wireLabels.join(', ')};\n`;
        }
    }
  });
  
  return code;
}

/**
 * Parse Recursia code to extract quantum circuit information
 * @param code - The Recursia code to parse
 * @returns A quantum circuit object or null if parsing fails
 */
export function parseRecursiaToCircuit(code: string): any | null {
  try {
    const lines = code.split('\n').filter(line => line.trim() && !line.trim().startsWith('//'));
    
    // Extract circuit name from comments
    const nameMatch = code.match(/\/\/\s*(.+?)(?:\n|$)/);
    const circuitName = nameMatch ? nameMatch[1].trim() : 'Imported Circuit';
    
    // Track quantum states and their indices
    const wireMap = new Map<string, number>();
    const wires: any[] = [];
    const gates: any[] = [];
    
    // Parse state declarations
    lines.forEach(line => {
      const stateMatch = line.match(/state\s+(\w+)\s*=\s*\|0⟩/);
      if (stateMatch) {
        const label = stateMatch[1];
        const index = wires.length;
        wireMap.set(label, index);
        wires.push({
          qubit: index,
          label: label,
          classical: false,
          ancilla: false
        });
      }
    });
    
    // If no states found, return null
    if (wires.length === 0) {
      return null;
    }
    
    // Parse gate operations
    lines.forEach(line => {
      // Single-qubit gates
      const singleQubitMatch = line.match(/^(H|X|Y|Z|S|T)\s+(\w+)/);
      if (singleQubitMatch) {
        const [, gateType, wireName] = singleQubitMatch;
        const qubitIndex = wireMap.get(wireName);
        if (qubitIndex !== undefined) {
          gates.push({
            id: `gate-${Date.now()}-${Math.random()}`,
            type: gateType,
            qubits: [qubitIndex],
            label: gateType
          });
        }
        return;
      }
      
      // Parameterized single-qubit gates
      const paramGateMatch = line.match(/^(RX|RY|RZ)\s*\(\s*([\d.-]+)\s*\)\s+(\w+)/);
      if (paramGateMatch) {
        const [, gateType, param, wireName] = paramGateMatch;
        const qubitIndex = wireMap.get(wireName);
        if (qubitIndex !== undefined) {
          gates.push({
            id: `gate-${Date.now()}-${Math.random()}`,
            type: gateType,
            qubits: [qubitIndex],
            params: [parseFloat(param)],
            label: `${gateType}(${parseFloat(param).toFixed(2)})`
          });
        }
        return;
      }
      
      // Two-qubit gates
      const twoQubitMatch = line.match(/^(CNOT|CZ|SWAP)\s+(\w+)\s*,\s*(\w+)/);
      if (twoQubitMatch) {
        const [, gateType, wire1, wire2] = twoQubitMatch;
        const qubit1 = wireMap.get(wire1);
        const qubit2 = wireMap.get(wire2);
        if (qubit1 !== undefined && qubit2 !== undefined) {
          gates.push({
            id: `gate-${Date.now()}-${Math.random()}`,
            type: gateType,
            qubits: [qubit1, qubit2],
            label: gateType
          });
        }
        return;
      }
      
      // Three-qubit gates
      const threeQubitMatch = line.match(/^(TOFFOLI|FREDKIN|CCNOT|CSWAP)\s+(\w+)\s*,\s*(\w+)\s*,\s*(\w+)/);
      if (threeQubitMatch) {
        const [, gateType, wire1, wire2, wire3] = threeQubitMatch;
        const qubit1 = wireMap.get(wire1);
        const qubit2 = wireMap.get(wire2);
        const qubit3 = wireMap.get(wire3);
        if (qubit1 !== undefined && qubit2 !== undefined && qubit3 !== undefined) {
          gates.push({
            id: `gate-${Date.now()}-${Math.random()}`,
            type: gateType === 'CCNOT' ? 'TOFFOLI' : gateType === 'CSWAP' ? 'FREDKIN' : gateType,
            qubits: [qubit1, qubit2, qubit3],
            label: gateType
          });
        }
        return;
      }
      
      // Measurement
      const measureMatch = line.match(/measure\s+(\w+)/);
      if (measureMatch) {
        const wireName = measureMatch[1];
        const qubitIndex = wireMap.get(wireName);
        if (qubitIndex !== undefined) {
          gates.push({
            id: `gate-${Date.now()}-${Math.random()}`,
            type: 'MEASURE',
            qubits: [qubitIndex],
            label: 'M'
          });
        }
        return;
      }
    });
    
    // Create circuit object
    return {
      id: `circuit-${Date.now()}`,
      name: circuitName,
      description: 'Imported from Recursia code',
      wires: wires,
      gates: gates,
      metadata: {
        created: Date.now(),
        modified: Date.now(),
        author: 'Code Import',
        tags: ['imported'],
        hardware: 'simulator',
        optimizationLevel: 0
      }
    };
  } catch (error) {
    console.error('Error parsing Recursia code:', error);
    return null;
  }
}