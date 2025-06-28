# Unified Architecture - Recursia

## Table of Contents
1. [Overview](#overview)
2. [Core Design Principles](#core-design-principles)
3. [System Architecture](#system-architecture)
4. [Data Flow](#data-flow)
5. [Component Interactions](#component-interactions)
6. [Execution Pipeline](#execution-pipeline)
7. [Metrics Flow](#metrics-flow)
8. [Memory Management](#memory-management)
9. [Error Handling](#error-handling)
10. [Performance Characteristics](#performance-characteristics)

## Overview

Recursia v3 implements a unified architecture that ensures single-path execution, centralized metric calculation, and consistent data flow throughout the system. This document defines the canonical architecture that all components must follow.

## Core Design Principles

### 1. Single Execution Path
```
Source Code → DirectParser → BytecodeVM → Unified Metrics → Results
```
- **NO** parallel execution paths
- **NO** AST intermediate representation
- **NO** external calculators during execution

### 2. Centralized Calculation
All physics and OSH calculations happen in ONE place:
- `src/core/bytecode_vm.py` - Primary execution and calculation
- `src/core/unified_vm_calculations.py` - Unified calculation methods

### 3. Immutable Metrics Flow
Metrics flow in one direction only:
```
VM Execution → ExecutionContext → API Response → Frontend Display
```
- **NO** post-processing of metrics
- **NO** recalculation in different components
- **NO** metric modification after VM execution

## Data Flow

### 1. Program Execution Flow

```mermaid
graph LR
    A[Recursia Code] --> B[DirectParser]
    B --> C[BytecodeModule]
    C --> D[BytecodeVM]
    D --> E[ExecutionContext]
    E --> F[VMExecutionResult]
    F --> G[API Response]
    G --> H[Frontend Display]
```

### 2. Metrics Calculation Flow

All metrics are calculated during VM execution:

```python
# In bytecode_vm.py
def _calculate_metrics(self):
    """All metric calculations happen here during execution."""
    
    # Get unified calculator instance
    calculator = self.execution_context.unified_calculator
    
    # Calculate OSH metrics
    phi = calculator.calculate_integrated_information(state_name, self.runtime)
    complexity = calculator.calculate_kolmogorov_complexity(state_name, self.runtime)
    entropy_flux = calculator.calculate_entropy_flux(state_name, self.runtime)
    
    # Update execution context
    self.execution_context.current_metrics.integrated_information = phi
    self.execution_context.current_metrics.kolmogorov_complexity = complexity
    self.execution_context.current_metrics.entropy_flux = entropy_flux
    
    # Calculate derived metrics
    rsp = calculator.calculate_recursive_simulation_potential(
        phi, complexity, entropy_flux
    )
    self.execution_context.current_metrics.rsp = rsp
```

### 3. WebSocket Message Flow

```
Frontend Request → WebSocket → API Handler → VM Execution → Metrics → WebSocket → Frontend Update
```

Message types:
- `get_metrics` - Request current metrics
- `get_states` - Request quantum state data
- `start_universe` - Begin universe simulation
- `metrics_update` - Real-time metric updates

## Component Interactions

### API Server ↔ VM

```python
# unified_api_server.py
@app.post("/api/execute")
async def execute_code(request: ExecuteRequest):
    # Parse code
    parser = DirectParser()
    bytecode_module = parser.parse(request.code)
    
    # Execute in VM
    runtime = get_global_runtime()
    vm = RecursiaVM(runtime)
    result = vm.execute(bytecode_module)
    
    # Return metrics directly from execution
    return {
        "success": result.success,
        "metrics": result.metrics,  # Direct from VM, no modification
        "output": result.output
    }
```

### Frontend ↔ API

```typescript
// useEngineAPI.ts
const execute = async (code: string) => {
    const response = await fetch('/api/execute', {
        method: 'POST',
        body: JSON.stringify({ code })
    });
    
    const result = await response.json();
    // Use metrics exactly as returned by VM
    updateMetrics(result.metrics);
};
```

## Execution Pipeline

### Phase 1: Parsing
```python
DirectParser.parse(code) → BytecodeModule
```
- Tokenization
- Syntax validation
- Bytecode generation
- **NO** AST generation

### Phase 2: Execution
```python
BytecodeVM.execute(module) → VMExecutionResult
```
- Instruction dispatch
- State management
- Metric calculation
- Result generation

### Phase 3: Response
```python
VMExecutionResult → API Response → Frontend
```
- Direct metric pass-through
- No transformation
- No recalculation

## Metrics Flow

### Core Metrics (Calculated in VM)

1. **Integrated Information (Φ)**
   - Calculated in `unified_vm_calculations.py`
   - Includes time evolution factors
   - Bimodal distribution: ~0.009 or >2.5

2. **Kolmogorov Complexity (K)**
   - Lempel-Ziv compression approximation
   - Normalized 0-1 range

3. **Entropy Flux (E)**
   - Physical entropy production rate
   - Units: bits/second

4. **Recursive Simulation Potential (RSP)**
   - RSP = Φ × K / E
   - Units: bit-seconds

### Metric Guarantees

- **Consistency**: Same input → same metrics (modulo quantum randomness)
- **Accuracy**: Conservation law holds to 10^-4
- **Performance**: All calculations < 50ms

## Memory Management

### Backend Memory Strategy

```python
# Memory limits enforced at multiple levels
class MemoryManager:
    MAX_STATE_VECTORS = 1000
    MAX_VECTOR_SIZE = 2**20  # ~1M complex numbers
    CACHE_TIMEOUT = 50  # milliseconds
```

### Frontend Memory Management

```typescript
// SystemResourceMonitor.ts
const getMemorySafeQubitLimit = (availableMemoryGB: number): number => {
    // 2^n complex amplitudes, 16 bytes each
    const bytesPerAmplitude = 16;
    const safetyFactor = 0.5; // Use only 50% of available memory
    
    const maxQubits = Math.floor(
        Math.log2(availableMemoryGB * 1e9 * safetyFactor / bytesPerAmplitude)
    );
    
    return Math.min(maxQubits, 25); // Hard limit at 25 qubits
};
```

## Error Handling

### Unified Error Types

```python
# data_classes.py
class RecursiaError(Exception):
    """Base error class"""
    pass

class ParseError(RecursiaError):
    """Parsing failures"""
    pass

class VMError(RecursiaError):
    """Execution failures"""
    pass

class QuantumError(RecursiaError):
    """Quantum operation failures"""
    pass
```

### Error Propagation

```
VM Error → ExecutionResult.error → API Response → Frontend Error Display
```

## Performance Characteristics

### Validated Performance Metrics

| Metric | Target | Achieved | Notes |
|--------|---------|----------|-------|
| API Response Time | <10ms | ✓ 8ms avg | Excluding computation |
| Frontend FPS | 60 FPS | ✓ 60 FPS | With 1000 particles |
| Validation Speed | >100k iter/s | ✓ 150,544 iter/s | 10M iterations |
| Memory Usage | <500MB | ✓ 380MB typical | 10 qubit systems |
| Φ Calculation | <100ms | ✓ 50ms cached | Dynamic evolution |
| Conservation Error | <0.01% | ✓ 10^-4 | Quantum noise limit |

### Scalability Limits

- **Qubits**: 25 maximum (memory constraint and even 25 requies a powerhouse. Stick with 10-12 qubits for commercial pc's)
- **Observers**: 100 simultaneous (3-5 for commercial pc's is optimal)
- **Memory Fields**: 100×100×100 grid

## Architectural Invariants

These properties MUST be maintained:

1. **Single Source of Truth**: VM execution context contains all metrics
2. **Immutable Results**: VMExecutionResult is never modified after creation
3. **Direct Pass-Through**: Metrics flow from VM → API → Frontend unchanged
4. **No Dual Calculation**: Each metric calculated exactly once
5. **Synchronized State**: Runtime state consistent across all components

## Future Architecture Considerations

### Planned Enhancements

1. **Distributed Execution**: Multi-node quantum simulation
2. **GPU Acceleration**: CUDA/WebGL for large systems
3. **Real-time Collaboration**: Shared universe states
4. **Persistent Storage**: Quantum state checkpointing

### Architecture Stability

The core architecture is designed for stability:
- Interfaces are versioned
- Breaking changes require major version bump
- Backward compatibility for 2 major versions

---

*This document defines the canonical architecture for Recursia. All implementations must conform to these specifications.*