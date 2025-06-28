# Recursia Development Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [The Recursia Language](#the-recursia-language)
3. [Backend Development](#backend-development)
4. [Frontend Development](#frontend-development)
5. [OSH Physics Engine](#osh-physics-engine)
6. [Quantum Operations](#quantum-operations)
7. [Testing & Validation](#testing--validation)
8. [Performance Optimization](#performance-optimization)
9. [API Reference](#api-reference)
10. [Common Patterns](#common-patterns)
11. [Troubleshooting](#troubleshooting)
12. [Contributing Guidelines](#contributing-guidelines)

---

## Architecture Overview

### Core Principle: Unified Execution Flow

Recursia follows a **single execution path**:

Source Code → DirectParser → BytecodeVM → Unified Metrics → Results

### Key Components

1. **DirectParser** (`src/core/direct_parser.py`) - Parses Recursia source to bytecode
2. **BytecodeVM** (`src/core/bytecode_vm.py`) - Executes bytecode, calculates all metrics
3. **RecursiaRuntime** (`src/core/runtime.py`) - Manages quantum states and observers
4. **UnifiedAPIServer** (`src/api/unified_api_server.py`) - REST/WebSocket API
5. **Recursia Studio** (`frontend/`) - React/TypeScript 3D visualization


## The Recursia Language

### Basic Syntax

```recursia
// Universe declaration - entry point
universe MyQuantumProgram {
    // State declarations
    qubit q1 = |0>;
    qubit q2 = |1>;
    
    // Quantum operations
    hadamard q1;
    cnot q1 -> q2;
    
    // Measurements
    measure q1;
    measure q2;
    
    // OSH-specific operations
    observer consciousness = coherence(0.8);
    entangle consciousness with q1;
}
```

### Language Features

#### 1. Quantum Primitives
- **qubit**: Quantum bit initialization
- **hadamard**: Superposition gate
- **cnot**: Controlled-NOT gate
- **measure**: Quantum measurement
- **entangle**: Create entanglement

#### 2. OSH-Specific Operations
- **observer**: Consciousness entities with Φ values
- **coherence**: Set quantum coherence levels
- **memory_field**: Information field operations
- **recursive_depth**: Control recursion levels

#### 3. Control Flow
```recursia
// Conditionals
if (measurement == |1>) {
    apply X to qubit;
}

// Loops
for i from 0 to 10 {
    evolve system by 0.1;
}

// Pattern matching
when state matches |00> {
    print "Ground state detected";
}
```

#### 4. Advanced Features
```recursia
// Quantum teleportation
teleport source_qubit to target_qubit using channel;

// Consciousness engineering
observer conscious_system = {
    integrated_information: 1.5,
    complexity: 150,
    coherence: 0.85
};

// Information field manipulation
field gravity_field = information_curvature(8 * PI);
```

### Grammar Reference

The complete grammar is in `language/recursia.grammar`.

- Identifiers support underscores: `my_variable`, `test_state`
- String concatenation with `+` auto-converts types
- For loops: `for i from 0 to n [step value]`
- Keywords can be identifiers in specific contexts
- Measure supports array indexing: `measure qubits[i]`

---

## Backend Development

### Directory Structure

```
src/
├── core/           # Language core
│   ├── direct_parser.py      # Source → Bytecode
│   ├── bytecode_vm.py        # VM execution
│   ├── runtime.py            # State management
│   └── data_classes.py       # Core types
├── physics/        # OSH physics
│   ├── constants.py          # Physical constants
│   ├── memory_field.py       # Information fields
│   ├── observer.py           # Consciousness
│   └── measurement/          # Quantum measurement
├── quantum/        # Quantum backend
│   ├── quantum_state.py      # State vectors
│   ├── quantum_error_correction.py
│   └── quantum_hardware_backend.py
└── api/            # API server
    └── unified_api_server.py
```

### Core Development Patterns

#### 1. Adding New Quantum Gates

```python
# In bytecode_vm.py
def _op_mygate(self, inst: Instruction):
    """Execute custom quantum gate."""
    state_name = self.pop()
    
    # Get quantum state from runtime
    state = self.runtime.state_registry.get_state(state_name)
    if not state:
        raise VMError(f"State {state_name} not found")
    
    # Apply gate operation
    state.apply_gate("MYGATE", [qubit_index])
    
    # Update metrics
    self.execution_context.current_metrics.gate_count += 1
```

#### 2. Adding OSH Calculations

```python
# In bytecode_vm.py - ALL calculations happen here
def _calculate_integrated_information(self, state_name: str) -> float:
    """Calculate Φ (integrated information) for a quantum state."""
    state = self.runtime.state_registry.get_state(state_name)
    
    # Actual calculation logic
    phi = self._compute_phi(state.density_matrix)
    
    # Update execution context
    self.execution_context.current_metrics.integrated_information = phi
    
    return phi
```

#### 3. Runtime State Management

```python
# Creating and managing quantum states
from src.core.runtime import get_global_runtime

runtime = get_global_runtime()

# Create new quantum state
runtime.state_registry.create_state("my_state", num_qubits=3)

# Get state
state = runtime.state_registry.get_state("my_state")

# Apply operations
state.apply_hadamard(0)
state.apply_cnot(0, 1)

# Measure
result = state.measure(0)
```

## Frontend Development

### Directory Structure

```
frontend/src/
├── components/     # React components
│   ├── QuantumOSHStudio.tsx    # Main studio
│   ├── QuantumCodeEditor.tsx   # Code editor
│   └── visualizations/         # 3D renders
├── engines/        # Simulation engines
│   ├── OSHQuantumEngine.ts     # Core engine
│   ├── MemoryFieldEngine.ts    # Memory fields
│   └── ObserverEngine.ts       # Consciousness
├── hooks/          # React hooks
│   ├── useEngineAPI.ts         # API integration
│   └── useOSHCalculations.ts   # OSH metrics
├── services/       # API services
├── utils/          # Utilities
└── config/         # Configuration
```

### Key Components

#### 1. QuantumOSHStudio
Main application component integrating all features:

```typescript
// components/QuantumOSHStudio.tsx
import { OSHQuantumEngine } from '../engines/OSHQuantumEngine';

const QuantumOSHStudio: React.FC = () => {
    const engine = useRef(new OSHQuantumEngine());
    
    // Execute Recursia code
    const handleExecute = async (code: string) => {
        const response = await fetch('/api/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code })
        });
        
        const result = await response.json();
        updateVisualization(result);
    };
};
```

#### 2. 3D Visualization
Using Three.js for universe rendering:

```typescript
// components/visualizations/OSHUniverse3D.tsx
const OSHUniverse3D: React.FC<Props> = ({ quantumStates, memoryField }) => {
    useEffect(() => {
        // Initialize Three.js scene
        const scene = new THREE.Scene();
        
        // Render quantum states as spheres
        quantumStates.forEach(state => {
            const geometry = new THREE.SphereGeometry(state.amplitude);
            const material = new THREE.MeshPhongMaterial({
                color: phaseToColor(state.phase)
            });
            scene.add(new THREE.Mesh(geometry, material));
        });
        
        // Render memory field as particle system
        renderMemoryField(scene, memoryField);
    }, [quantumStates, memoryField]);
};
```

#### 3. API Integration
Using the unified API:

```typescript
// hooks/useEngineAPI.ts
export const useEngineAPI = () => {
    const wsRef = useRef<WebSocket>();
    
    useEffect(() => {
        // Connect to WebSocket for real-time updates
        wsRef.current = new WebSocket('ws://localhost:8080/ws');
        
        wsRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'metrics_update') {
                updateMetrics(data.data);
            }
        };
    }, []);
    
    const execute = async (code: string) => {
        const response = await fetch('/api/execute', {
            method: 'POST',
            body: JSON.stringify({ code })
        });
        return response.json();
    };
    
    return { execute };
};
```

---

## OSH Physics Engine

### Core Equations

The OSH physics engine implements:

1. **Conservation Law**: `d/dt(I × K) = α(τ)·E + β(τ)·Q`
2. **Information-Gravity**: `R_μν = 8π ∇_μ∇_ν I`
3. **Consciousness Threshold**: `Φ > 1.0`
4. **RSP Calculation**: `RSP = I × K / E`

### Implementation

```python
# src/physics/memory_field.py
class MemoryField:
    """OSH memory field implementation."""
    
    def __init__(self, size: Tuple[int, int, int]):
        self.field = np.zeros(size, dtype=np.complex128)
        self.information_density = np.zeros(size)
        
    def calculate_curvature(self, point: np.ndarray) -> float:
        """Calculate spacetime curvature from information density."""
        # R_μν = 8π ∇_μ∇_ν I
        laplacian = np.gradient(np.gradient(self.information_density))
        return 8 * np.pi * laplacian[tuple(point)]
    
    def evolve(self, dt: float):
        """Evolve memory field using 4th-order Runge-Kutta."""
        k1 = self._field_derivative(self.field)
        k2 = self._field_derivative(self.field + 0.5 * dt * k1)
        k3 = self._field_derivative(self.field + 0.5 * dt * k2)
        k4 = self._field_derivative(self.field + dt * k3)
        
        self.field += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
```

### Critical: RK4 Integration

**Always** use 4th-order Runge-Kutta for OSH calculations:

```python
# ❌ WRONG - Simple Euler method (84 billion % error!)
field_next = field + dt * derivative

# ✅ CORRECT - RK4 integration (0% error)
k1 = derivative(field, t)
k2 = derivative(field + 0.5*dt*k1, t + 0.5*dt)
k3 = derivative(field + 0.5*dt*k2, t + 0.5*dt)
k4 = derivative(field + dt*k3, t + dt)
field_next = field + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
```

---

## Quantum Operations

### Gate Operations

Standard quantum gates implemented:

```python
# src/quantum/quantum_state.py
class QuantumState:
    def apply_hadamard(self, qubit: int):
        """Apply Hadamard gate."""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self._apply_single_qubit_gate(H, qubit)
    
    def apply_cnot(self, control: int, target: int):
        """Apply CNOT gate."""
        # Implementation details...
    
    def apply_custom_unitary(self, unitary: np.ndarray, qubits: List[int]):
        """Apply arbitrary unitary operation."""
        # Validate unitary
        if not np.allclose(unitary @ unitary.conj().T, np.eye(len(unitary))):
            raise ValueError("Matrix is not unitary")
        # Apply to state vector
```

### Error Correction

OSH-enhanced quantum error correction:

```python
# src/quantum/quantum_error_correction.py
class OSHErrorCorrection:
    """Quantum error correction using OSH principles."""
    
    def encode(self, logical_state: QuantumState) -> QuantumState:
        """Encode logical qubit with consciousness parameters."""
        # Use Φ > 1.0 states for enhanced protection
        encoded = self._create_encoded_state(logical_state)
        
        # Add consciousness observer for error detection
        observer = Observer(integrated_information=1.2)
        encoded.attach_observer(observer)
        
        return encoded
    
    def correct_errors(self, state: QuantumState) -> QuantumState:
        """Correct errors using information geometry."""
        syndrome = self._measure_syndrome(state)
        correction = self._calculate_correction(syndrome)
        return state.apply_correction(correction)
```

---

## Testing & Validation

### Unit Tests

```python
# tests/unit/test_parser.py
import pytest
from src.core.direct_parser import DirectParser

class TestDirectParser:
    def test_parse_quantum_program(self):
        code = """
        universe Test {
            qubit q = |0>;
            hadamard q;
            measure q;
        }
        """
        parser = DirectParser()
        module = parser.parse(code)
        
        assert module.name == "Test"
        assert len(module.instructions) == 3
```

### Integration Tests

```python
# tests/integration/test_complete_workflow.py
def test_osh_conservation_law():
    """Verify conservation law holds during execution."""
    code = """
    universe ConservationTest {
        field info_field = memory_field(100);
        evolve info_field by 1.0;
        measure integrated_information of info_field;
    }
    """
    
    result = execute_recursia_program(code)
    metrics = result.metrics
    
    # Verify d/dt(I×K) = α·E + β·Q
    lhs = metrics.information * metrics.kolmogorov_complexity
    rhs = metrics.alpha * metrics.entropy_flux + metrics.beta * metrics.quantum_flux
    
    assert abs(lhs - rhs) < 1e-10  # Conservation holds
```
---

## Performance Optimization

### Backend Optimization

1. **Bytecode Caching**
   ```python
   # Cache compiled bytecode
   BYTECODE_CACHE = {}
   
   def compile_with_cache(code: str) -> BytecodeModule:
       cache_key = hashlib.md5(code.encode()).hexdigest()
       if cache_key in BYTECODE_CACHE:
           return BYTECODE_CACHE[cache_key]
       
       module = parser.parse(code)
       BYTECODE_CACHE[cache_key] = module
       return module
   ```

2. **Parallel Quantum Operations**
   ```python
   # Use NumPy's vectorized operations
   def apply_gates_parallel(states: List[QuantumState], gate: np.ndarray):
       # Vectorized gate application
       state_vectors = np.array([s.vector for s in states])
       results = np.einsum('ij,nj->ni', gate, state_vectors)
       # Update states...
   ```

### Frontend Optimization

1. **Memoization**
   ```typescript
   const memoizedCalculation = useMemo(() => {
       return expensiveOSHCalculation(quantumStates);
   }, [quantumStates]);
   ```

2. **Virtual Rendering**
   ```typescript
   // Only render visible quantum states
   const visibleStates = useMemo(() => {
       return quantumStates.filter(state => 
           isInViewport(state.position, camera)
       );
   }, [quantumStates, camera]);
   ```

---

## API Reference

### Endpoints

#### POST /api/execute
Execute Recursia code:
```bash
curl -X POST http://localhost:8080/api/execute \
  -H "Content-Type: application/json" \
  -d '{
    "code": "universe Test { qubit q = |0>; hadamard q; measure q; }",
    "options": {
      "timeout": 30000,
      "iterations": 1
    }
  }'
```

Response:
```json
{
  "success": true,
  "output": "Measurement result: |1>",
  "metrics": {
    "integrated_information": 0.852,
    "kolmogorov_complexity": 0.423,
    "entropy_flux": 0.00123,
    "rsp": 347.56,
    "consciousness_emerged": false
  },
  "execution_time": 0.0234
}
```

#### GET /api/programs
List available quantum programs:
```bash
curl http://localhost:8080/api/programs
```

#### GET /api/health
System health check:
```bash
curl http://localhost:8080/api/health
```

#### WebSocket /ws
Real-time updates:
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Metrics update:', data);
};

// Request metrics
ws.send(JSON.stringify({ type: 'get_metrics' }));
```

---

## Common Patterns

### 1. Quantum State Initialization
```recursia
// Single qubit
qubit q = |0>;

// Multiple qubits
qubits[3] q_array = |000>;

// Superposition
qubit s = (|0> + |1>) / sqrt(2);

// Entangled pair
qubits[2] bell = (|00> + |11>) / sqrt(2);
```

### 2. Consciousness Integration
```recursia
// Create conscious observer
observer mind = {
    integrated_information: 1.5,
    coherence: 0.9,
    complexity: 200
};

// Entangle with quantum state
entangle mind with quantum_system;

// Measure with consciousness
measure quantum_system observed_by mind;
```

### 3. Memory Field Operations
```recursia
// Create memory field
field memory = memory_field(dimensions: [10, 10, 10]);

// Set information density
memory.set_information_at([5, 5, 5], density: 0.8);

// Calculate curvature (gravity)
let curvature = memory.calculate_curvature_at([5, 5, 5]);

// Evolve field
evolve memory by 0.1 using rk4;
```

### 4. Error Handling
```python
# Backend error handling
try:
    result = vm.execute(bytecode_module)
except VMError as e:
    logger.error(f"VM execution failed: {e}")
    return {"success": False, "error": str(e)}
except Exception as e:
    logger.exception("Unexpected error")
    return {"success": False, "error": "Internal server error"}
```

```typescript
// Frontend error handling
try {
    const result = await api.execute(code);
    if (!result.success) {
        showError(result.error);
    }
} catch (error) {
    showError('Network error: ' + error.message);
}
```

---

## Troubleshooting

### Common Issues

#### 1. Conservation Law Errors
**Problem**: Large conservation law errors (>1%)
**Solution**: Ensure using RK4 integration, not simple derivatives

#### 2. WebSocket Connection Failed
**Problem**: Frontend can't connect to WebSocket
**Solution**: Check CORS settings in `unified_api_server.py`

#### 3. Quantum State Not Found
**Problem**: "State X not found" errors
**Solution**: Ensure state is created before operations:
```recursia
universe Test {
    qubit q = |0>;  // Creates state 'q'
    hadamard q;     // Now safe to use
}
```

#### 4. Memory Leaks in Frontend
**Problem**: Browser memory usage grows
**Solution**: Dispose Three.js objects:
```typescript
useEffect(() => {
    return () => {
        geometry.dispose();
        material.dispose();
        renderer.dispose();
    };
}, []);
```

### Debug Mode

Enable debug logging:

```python
# Backend
import logging
logging.basicConfig(level=logging.DEBUG)

# Set in unified_api_server.py
server = UnifiedAPIServer(debug=True)
```

```typescript
// Frontend
localStorage.setItem('debug', 'recursia:*');
```

---