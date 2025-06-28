# Recursia Frontend Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Technology Stack](#technology-stack)
4. [Architecture Patterns](#architecture-patterns)
5. [Engine Architecture](#engine-architecture)
6. [Component Hierarchy](#component-hierarchy)
7. [State Management](#state-management)
8. [Memory Management](#memory-management)
9. [Performance Optimization](#performance-optimization)
10. [Testing Strategy](#testing-strategy)
11. [Build System](#build-system)
12. [Development Workflow](#development-workflow)

## Overview

Recursia is an enterprise-grade quantum computing studio built with React and TypeScript, featuring a sophisticated simulation engine architecture. The frontend provides an immersive 3D visualization environment for quantum computing simulations, focusing on the Organic Simulation Hypothesis (OSH) framework.

### Key Features
- **Real-time Quantum Simulation**: Live physics simulation with WebSocket connectivity
- **3D Visualization**: Three.js-powered immersive quantum universe visualization
- **Enterprise Memory Management**: Comprehensive resource tracking and automatic cleanup
- **Distributed Computing**: Web Worker-based parallel processing
- **Advanced UI Components**: Monaco Editor integration, drag-and-drop layouts
- **Real-time Metrics**: Live performance monitoring and diagnostics

## Project Structure

```
src/
├── components/           # React components
│   ├── debug/           # Debug utilities
│   ├── osh/             # OSH-specific components
│   ├── ui/              # Reusable UI components
│   └── visualizations/  # 3D visualizations
├── engines/             # Quantum simulation engines
├── hooks/               # Custom React hooks
├── contexts/            # React context providers
├── workers/             # Web workers for parallel processing
├── utils/               # Utility functions and helpers
├── config/              # Configuration files
├── data/                # Static data and quantum programs
├── services/            # API services
├── store/               # Redux store configuration
├── styles/              # CSS and styling
├── tests/               # Test files
├── types/               # TypeScript type definitions
└── validation/          # Code validation utilities
```

## Technology Stack

### Core Technologies
- **React 18.2.0**: Component framework with Concurrent Mode
- **TypeScript 5.7.2**: Type-safe JavaScript with strict configuration
- **Vite 6.0.1**: Build tool with HMR and optimized bundling
- **Three.js 0.171.0**: 3D graphics and WebGL rendering

### UI Libraries
- **React Three Fiber**: React renderer for Three.js
- **React Three Drei**: Three.js utilities and helpers
- **Framer Motion**: Animation library
- **Lucide React**: Icon library
- **React Grid Layout**: Drag-and-drop dashboard layouts

### State Management
- **React Query (TanStack)**: Server state management
- **Redux Toolkit**: Global state management
- **React Context**: Component-level state

### Development Tools
- **Vitest**: Unit testing framework
- **Testing Library**: Component testing utilities
- **ESLint**: Code linting
- **Prettier**: Code formatting
- **Storybook**: Component documentation

### Performance & Optimization
- **Web Workers**: Parallel computation
- **Monaco Editor**: Code editing with syntax highlighting
- **Chart.js**: Data visualization
- **Memory Management**: Custom resource tracking

## Architecture Patterns

### 1. Enterprise Component Architecture
```typescript
// Component with memory management and error boundaries
export const EnterpriseComponent: React.FC<Props> = ({ ... }) => {
  const { track } = useMemoryManager('ComponentName');
  
  useEffect(() => {
    const resource = initializeResource();
    track('resource-id', resource, ResourceType.COMPONENT);
    
    return () => {
      // Automatic cleanup handled by memory manager
    };
  }, []);
  
  return <ErrorBoundary>{/* component content */}</ErrorBoundary>;
};
```

### 2. Engine Orchestration Pattern
```typescript
// Central engine coordinator with resource management
export class EngineOrchestrator extends EventEmitter {
  private engines: Map<string, ManagedEngine> = new Map();
  private resourceManager: ResourceManager;
  
  registerEngine(config: EngineConfig): void {
    // Register engines with priority-based scheduling
  }
  
  update(deltaTime: number): void {
    // Coordinated updates with resource throttling
  }
}
```

### 3. Memory-Aware Component Pattern
```typescript
// Components that automatically clean up resources
export const MemoryManagedComponent = <T extends {}>(
  Component: React.ComponentType<T>
) => {
  return (props: T) => {
    const memoryManager = useMemoryManager(Component.name);
    
    useEffect(() => {
      return () => memoryManager.cleanup();
    }, []);
    
    return <Component {...props} />;
  };
});
```

## Engine Architecture

### Core Engine Hierarchy

```typescript
OSHQuantumEngine (Central Orchestrator)
├── MemoryFieldEngine           # Quantum memory field simulation
├── EntropyCoherenceSolver      # Entropy and coherence calculations
├── RSPEngine                   # Recursive Self-Improvement calculations
├── ObserverEngine              # Quantum observation mechanics
├── MLAssistedObserver          # Machine learning optimization
├── MacroTeleportationEngine    # Quantum teleportation simulation
├── CurvatureTensorGenerator    # Spacetime curvature calculations
├── CoherenceFieldLockingEngine # Field stabilization
├── RecursiveTensorFieldEngine  # Tensor field recursion
└── SubstrateIntrospectionEngine # System self-analysis
```

### Engine Orchestration System

The `EngineOrchestrator` manages engine lifecycle and resource allocation:

```typescript
interface EngineConfig {
  id: string;
  priority: number;        // 1-10, higher is more important
  resourceWeight: number;  // Estimated resource consumption (0-1)
  required: boolean;       // Core engines that must always run
  updateFrequency: number; // How often to update (ms)
  dependencies?: string[]; // Other engines this depends on
}
```

### Resource Management Strategy

1. **Priority-Based Scheduling**: Critical engines get CPU time first
2. **Adaptive Resource Allocation**: Engines throttled based on available resources
3. **Dependency Resolution**: Engines updated in dependency order
4. **Performance Monitoring**: Real-time tracking of engine performance

## Component Hierarchy

### Main Application Structure

```
App
├── ErrorBoundary
├── QueryClientProvider
├── Provider (Redux)
├── Router
│   └── QuantumOSHStudio (Main Component)
│       ├── UniverseProvider
│       ├── MemoryMonitor
│       ├── EngineStatus
│       ├── QuantumCodeEditor
│       ├── QuantumProgramsLibrary
│       ├── OSHCalculationsPanel
│       ├── DataFlowMonitor
│       ├── DiagnosticPanel
│       └── Visualizations
│           ├── OSHUniverse3D
│           ├── MemoryFieldVisualizer
│           ├── RSPDashboard
│           ├── InformationalCurvatureMap
│           └── GravitationalWaveEchoVisualizer
└── Toaster (Notifications)
```

### Key Component Patterns

#### 1. Context-Aware Components
```typescript
// Components that consume universe context
const Component = () => {
  const { engine, universeState, isSimulating } = useUniverse();
  // Component logic
};
```

#### 2. Hook-Based Data Fetching
```typescript
// API integration with caching and error handling
const { metrics, execute, isConnected } = useEngineAPI();
const { data, isLoading, error } = useOSHCalculations();
```

#### 3. Memory-Safe 3D Components
```typescript
// Three.js components with automatic resource disposal
export const OSHUniverse3D = ({ engine, primaryColor }) => {
  const { track } = useMemoryManager('OSHUniverse3D');
  const rendererRef = useRef<WebGLRenderer | null>(null);
  
  useEffect(() => {
    return () => {
      if (rendererRef.current) {
        disposeRenderer(rendererRef.current);
      }
    };
  }, []);
};
```

## State Management

### Multi-Layer State Architecture

1. **Global State (Redux)**: Application-wide settings and theme
2. **Server State (React Query)**: API data with caching and synchronization
3. **Component State (React State)**: Local component state
4. **Context State**: Shared state between related components

### State Flow Pattern

```typescript
// Server state with automatic synchronization
const useEngineAPI = () => {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  
  // WebSocket for real-time updates
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'metrics_update') {
        setMetrics(message.data);
      }
    };
  }, []);
};
```

### Universe Context Provider

The `UniverseProvider` manages the quantum simulation state:

```typescript
interface UniverseState {
  timestamp: number;
  initialized: boolean;
  memoryField: {
    fragments: number;
    totalCoherence: number;
    strain: number;
  };
  wavefunction: {
    dimensions: number;
    coherence: number;
  };
  rsp: {
    value: number;
    information: number;
    complexity: number;
  };
  performance: {
    fps: number;
    memoryUsage: number;
    updateTime: number;
  };
}
```

## Memory Management

### Enterprise Memory Management System

The application implements a comprehensive memory management system:

#### 1. Memory Manager
```typescript
interface TrackedResource {
  id: string;
  type: ResourceType;
  component?: string;
  size: number;
  created: number;
  cleanup: () => void;
  priority: 'low' | 'medium' | 'high' | 'critical';
}
```

#### 2. Three.js Memory Management
```typescript
export class ThreeMemoryManager {
  track(id: string, resource: DisposableThreeObject): void;
  dispose(id: string): void;
  disposeAll(): void;
  getMemoryUsage(): number;
}
```

#### 3. Memory Configuration
```typescript
export function getMemorySafeGridSize(): number {
  // Dynamic grid sizing based on available memory
  const memory = (performance as any).memory;
  const availableMB = (memory.jsHeapSizeLimit - memory.usedJSHeapSize) / (1024 * 1024);
  
  // Conservative grid sizes for stability
  if (availableMB < 1000) return 3;      // 27 cells
  if (availableMB < 2000) return 4;      // 64 cells
  if (availableMB < 3000) return 5;      // 125 cells
  return Math.min(8, maxGridSize);       // Max 512 cells
}
```

### Memory Management Strategies

1. **Automatic Resource Tracking**: Components automatically register resources
2. **Threshold-Based Cleanup**: Automatic cleanup when memory thresholds are reached
3. **Component Lifecycle Integration**: Resources cleaned up with component unmounting
4. **WebGL Resource Management**: Specialized handling for Three.js resources
5. **Worker Pool Management**: Automatic worker cleanup and recycling

## Performance Optimization

### 1. Code Splitting and Lazy Loading
```typescript
// Vite configuration for optimal bundling
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          redux: ['@reduxjs/toolkit', 'react-redux'],
          query: ['@tanstack/react-query'],
          three: ['three', '@react-three/fiber', '@react-three/drei'],
        },
      },
    },
  },
});
```

### 2. Web Worker Architecture
```typescript
export class ViteWorkerPool {
  private workers: Worker[] = [];
  private taskQueue: WorkerTask[] = [];
  
  execute<T>(task: WorkerTask<T>): Promise<WorkerResult<T>> {
    // Distribute tasks across available workers
  }
}
```

### 3. Performance Monitoring
```typescript
// Real-time performance tracking
interface PerformanceMetrics {
  fps: number;
  memoryUsage: number;
  updateTime: number;
  renderTime: number;
  workerUtilization: number;
}
```

### 4. Adaptive Rendering
- **LOD (Level of Detail)**: Dynamic geometry complexity based on distance
- **Frustum Culling**: Skip rendering of off-screen objects
- **Instance Rendering**: Efficient rendering of repeated geometry
- **Texture Compression**: Optimized texture formats for WebGL

## Testing Strategy

### Test Architecture
```
tests/
├── components/          # Component tests
│   └── osh/            # OSH-specific component tests
├── memoryStabilityTest.ts  # Memory leak detection
└── oshEngines.test.ts      # Engine functionality tests
```

### Testing Patterns

#### 1. Component Testing
```typescript
import { render, screen } from '@testing-library/react';
import { CMBAnalysisPanel } from '../components/osh/CMBAnalysisPanel';

describe('CMBAnalysisPanel', () => {
  it('renders analysis results', () => {
    render(<CMBAnalysisPanel data={mockData} />);
    expect(screen.getByText('CMB Analysis')).toBeInTheDocument();
  });
});
```

#### 2. Engine Testing
```typescript
describe('OSH Engines', () => {
  it('should maintain coherence under load', async () => {
    const engine = new OSHQuantumEngine();
    const result = await engine.runSimulation(1000);
    expect(result.coherence).toBeGreaterThan(0.8);
  });
});
```

#### 3. Memory Stability Testing
```typescript
describe('Memory Stability', () => {
  it('should not leak memory during simulation', async () => {
    const initialMemory = performance.memory.usedJSHeapSize;
    // Run simulation...
    const finalMemory = performance.memory.usedJSHeapSize;
    expect(finalMemory - initialMemory).toBeLessThan(MEMORY_THRESHOLD);
  });
});
```

### Test Configuration
```typescript
// vitest.config.ts
export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
});
```

## Build System

### Vite Configuration
The project uses Vite for optimal development experience and production builds:

```typescript
export default defineConfig({
  plugins: [react({ jsxRuntime: 'automatic' })],
  worker: { format: 'es' },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@utils': path.resolve(__dirname, './src/utils'),
    },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8080',
      '/ws': { target: 'ws://localhost:8080', ws: true },
    },
  },
});
```

### Build Optimization Features
- **Hot Module Replacement (HMR)**: Instant development updates
- **Tree Shaking**: Eliminate unused code
- **Code Splitting**: Automatic bundle optimization
- **Worker Support**: ES module workers for parallel processing
- **Proxy Configuration**: Seamless API integration

### TypeScript Configuration
```json
{
  "compilerOptions": {
    "target": "ES2021",
    "strict": false,
    "noImplicitAny": false,
    "jsx": "react-jsx",
    "baseUrl": "src",
    "paths": {
      "@/*": ["*"],
      "@components/*": ["components/*"],
      "@hooks/*": ["hooks/*"]
    }
  }
}
```

## Development Workflow

### 1. Local Development Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm run test

# Type checking
npm run type-check

# Build for production
npm run build
```

### 2. Code Quality Standards
- **ESLint**: Enforces coding standards
- **Prettier**: Code formatting
- **TypeScript**: Type safety (with relaxed configuration for research code)
- **Git Hooks**: Pre-commit quality checks

### 3. Performance Monitoring
- **Memory Usage Tracking**: Real-time memory monitoring
- **FPS Monitoring**: Frame rate tracking for 3D visualizations
- **Bundle Analysis**: Build size optimization
- **WebGL Performance**: GPU utilization monitoring

### 4. Debugging Tools
- **React DevTools**: Component inspection
- **Redux DevTools**: State debugging
- **Performance API**: Memory and timing analysis
- **WebGL Inspector**: 3D rendering debugging

### 5. Deployment Strategy
- **Production Build**: Optimized bundle with source maps
- **Static Asset Optimization**: Compressed images and fonts
- **Service Worker**: Offline support and caching
- **CDN Integration**: Fast global content delivery

## Key Architectural Decisions

### 1. Memory-First Design
Every component is designed with memory management as a primary concern, ensuring stability during long-running simulations.

### 2. Engine Modularity
The quantum simulation engines are modular and independently manageable, allowing for selective activation based on available resources.

### 3. Real-time Communication
WebSocket integration provides low-latency communication with the backend simulation engine.

### 4. Adaptive Performance
The system automatically adjusts rendering quality and computational load based on available system resources.

### 5. Enterprise Error Handling
Comprehensive error boundaries and validation systems ensure graceful failure handling.

This architecture provides a robust foundation for quantum computing visualization while maintaining enterprise-grade performance and reliability standards.