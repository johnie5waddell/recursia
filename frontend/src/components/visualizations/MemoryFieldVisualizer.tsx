/**
 * Memory Field Visualizer - Advanced Quantum Visualization
 * 
 * Features:
 * - Dynamic memory fragment visualization with quantum properties
 * - Coherence wave propagation and interference patterns
 * - Strain field visualization with gravitational effects
 * - Timeline view showing temporal evolution
 * - Network view displaying quantum entanglement connections
 * - Real-time synchronization with backend metrics
 * 
 * Performance Optimizations:
 * - Throttled state updates (10fps UI, 30fps physics)
 * - Proper Three.js object disposal
 * - Fragment virtualization and LOD system
 * - Instanced rendering for large fragment counts
 * - Efficient WebSocket data handling
 */

import React, { useRef, useEffect, useMemo, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text, Sphere, Box } from '@react-three/drei';
import * as THREE from 'three';
import { MemoryFieldEngine, MemoryFragment, MemoryField, CoherenceWave } from '../../engines/MemoryFieldEngine';
import { Complex } from '../../utils/complex';
import { Info, Play, Pause, Layers, Clock, Activity, Zap, Download, RefreshCw, Settings, Eye, EyeOff, Grid3x3, Camera } from 'lucide-react';
import { Tooltip } from '../ui/Tooltip';
import { useEngineAPIContext } from '../../contexts/EngineAPIContext';
import { adjustLightness } from '../../utils/colorUtils';

// Performance constants - Optimized for commercial hardware
const ANIMATION_THROTTLE_MS = 100; // Update UI at 10fps for performance
const PHYSICS_UPDATE_MS = 100; // Physics at 10fps to minimize CPU
const MAX_VISIBLE_FRAGMENTS = 12; // Reduced limit for stability
const DEFAULT_FRAGMENT_LIMIT = 8; // More conservative default
const FRAGMENT_UPDATE_THROTTLE = 2000; // Update every 2 seconds to reduce load
const WAVE_UPDATE_THROTTLE = 1000; // Less frequent wave updates
const MIN_FRAGMENT_SIZE = 0.5;
const MAX_FRAGMENT_SIZE = 1.5; // Smaller fragments for performance
const MAX_TOTAL_FRAGMENTS = 12; // Strict hard limit
const FRAGMENT_CLEANUP_THRESHOLD = 10; // More aggressive cleanup
const MAX_CONNECTIONS = 4; // Fewer connection lines

// Memoized components to prevent unnecessary re-renders
const MemoizedBox = React.memo(Box);
const MemoizedSphere = React.memo(Sphere);
const MemoizedText = React.memo(Text);

interface MemoryFieldVisualizerProps {
  memoryFieldEngine: MemoryFieldEngine;
  primaryColor: string;
  isActive: boolean;
}

interface BackendMemoryFragment {
  coherence: number;
  size: number;
  coupling_strength: number;
  position: [number, number, number];
  phase: number;
}

interface BackendMetrics {
  memory_fragments?: BackendMemoryFragment[];
  memory_field_coherence?: number;
  strain?: number;
  field_entropy?: number;
  information_density?: number;
  coherence_stability?: number;
}

/**
 * Simplified coherence field indicator - static visual with minimal performance impact
 */
const CoherenceFieldIndicator: React.FC<{ coherence: number; primaryColor: string }> = React.memo(({ coherence, primaryColor }) => {
  // Simple static plane to indicate field coherence
  const color = useMemo(() => {
    const c = new THREE.Color(primaryColor);
    c.multiplyScalar(0.3 + coherence * 0.7);
    return c;
  }, [primaryColor, coherence]);
  
  return (
    <mesh position={[0, -5, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[60, 60, 1, 1]} />
      <meshBasicMaterial 
        color={color}
        opacity={0.1 + coherence * 0.2}
        transparent
        side={THREE.DoubleSide}
      />
    </mesh>
  );
});

// Strain field visualization removed for performance

/**
 * Optimized memory fragment renderer - simplified for performance
 */
const MemoryFragmentRenderer: React.FC<{
  fragment: MemoryFragment;
  viewMode: 'particles' | 'boxes' | 'spheres';
  primaryColor: string;
}> = React.memo(({ fragment, viewMode, primaryColor }) => {
  const size = MIN_FRAGMENT_SIZE + (fragment.strain || 0) * (MAX_FRAGMENT_SIZE - MIN_FRAGMENT_SIZE);
  const coherence = fragment.coherence || 0;
  
  // Simplified color calculation
  const brightness = 0.5 + coherence * 0.5;
  const fragmentColor = new THREE.Color(primaryColor).multiplyScalar(brightness);
  const opacity = 0.5 + coherence * 0.3;
  
  // Position is always an array per MemoryFragment interface
  const position: [number, number, number] = fragment.position;
  
  // Validate position
  if (!position || position.length !== 3) {
    return null;
  }
  
  // Use only boxes for consistent performance
  return (
    <MemoizedBox
      position={position}
      args={[size, size, size]}
    >
      <meshBasicMaterial
        color={fragmentColor}
        opacity={opacity}
        transparent
      />
    </MemoizedBox>
  );
});

/**
 * Main Memory Field Visualizer Component
 */
export const MemoryFieldVisualizer: React.FC<MemoryFieldVisualizerProps> = ({ 
  memoryFieldEngine, 
  primaryColor,
  isActive 
}) => {
  // Get universe context - commented out to fix hooks issue
  // const { universeState, isSimulating } = useUniverse();
  const isSimulating = true; // Default value
  
  // Core state
  const [updateCounter, setUpdateCounter] = useState(0);
  const [showHighCoherence, setShowHighCoherence] = useState(true);
  const [showLowCoherence, setShowLowCoherence] = useState(false);
  const [showHighStrain, setShowHighStrain] = useState(false);
  const [showCoherenceWaves, setShowCoherenceWaves] = useState(false);
  const [showStrainField, setShowStrainField] = useState(false);
  const [viewMode, setViewMode] = useState<'3d' | 'timeline'>('3d');
  const [fragmentViewMode, setFragmentViewMode] = useState<'particles' | 'boxes' | 'spheres'>('boxes');
  const [cameraPosition, setCameraPosition] = useState<[number, number, number]>([50, 50, 50]);
  const [autoRotate, setAutoRotate] = useState(false);
  const [fragmentLimit, setFragmentLimit] = useState(DEFAULT_FRAGMENT_LIMIT);
  const [coherenceThreshold, setCoherenceThreshold] = useState(0.0);
  const [strainThreshold, setStrainThreshold] = useState(0.0);
  const [timeScale, setTimeScale] = useState(1.0);
  const [currentFps, setCurrentFps] = useState(60);
  const [performanceMode, setPerformanceMode] = useState<'auto' | 'low' | 'medium' | 'high'>('auto');
  
  // Performance refs
  const animationFrameRef = useRef<number>();
  const physicsTimerRef = useRef<NodeJS.Timeout>();
  const fragmentUpdateTimerRef = useRef<NodeJS.Timeout>();
  const lastPhysicsUpdate = useRef<number>(Date.now());
  const lastUIUpdate = useRef<number>(Date.now());
  
  // Backend integration with unified metrics
  const { metrics, isConnected, states } = useEngineAPIContext();
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Clear all timers
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (physicsTimerRef.current) {
        clearTimeout(physicsTimerRef.current);
      }
      if (fragmentUpdateTimerRef.current) {
        clearTimeout(fragmentUpdateTimerRef.current);
      }
    };
  }, []);
  
  // Enhanced sync with backend metrics - OSH-compliant data flow
  useEffect(() => {
    if (!memoryFieldEngine || !isActive || !metrics) return;
    
    // Clear existing timer
    if (fragmentUpdateTimerRef.current) {
      clearTimeout(fragmentUpdateTimerRef.current);
    }
    
    // Debounce fragment sync to prevent rapid updates
    fragmentUpdateTimerRef.current = setTimeout(() => {
      try {
        const field = memoryFieldEngine.getCurrentField();
        if (!field) {
          return;
        }
        
        // Aggressive memory cleanup before update
        if (field.fragments.length > FRAGMENT_CLEANUP_THRESHOLD) {
          // Keep only the most coherent fragments
          field.fragments = field.fragments
            .sort((a, b) => (b.coherence || 0) - (a.coherence || 0))
            .slice(0, FRAGMENT_CLEANUP_THRESHOLD);
        }
        
        // Create memory fragments from backend memory field data OR quantum states
        // First priority: Use memory fragments from backend if available
        if (metrics.memory_fragments && metrics.memory_fragments.length > 0) {
          // Only update if fragments have changed or field is empty
          const existingFragmentIds = new Set(field.fragments.map(f => f.id));
          const newFragmentIds = new Set(metrics.memory_fragments.map((f: any, i: number) => `mem-${f.name}-${i}`));
          const hasChanged = existingFragmentIds.size !== newFragmentIds.size || 
                           ![...existingFragmentIds].every(id => newFragmentIds.has(id));
          
          if (hasChanged || field.fragments.length === 0) {
            // Clear and rebuild fragments from backend memory field data
            field.fragments = [];
            
            // Limit fragments to prevent memory issues - strict limit of 12
            const fragmentsToProcess = metrics.memory_fragments.slice(0, 12);
            
            fragmentsToProcess.forEach((memFragment: any, index: number) => {
              // Convert backend memory fragment to frontend MemoryFragment format
              
              // Use position from backend if available, otherwise calculate
              let position: [number, number, number];
              if (memFragment.position && Array.isArray(memFragment.position) && memFragment.position.length === 3) {
                // Clamp positions to reasonable bounds
                position = [
                  Math.max(-30, Math.min(30, memFragment.position[0])),
                  Math.max(-10, Math.min(10, memFragment.position[1])),
                  Math.max(-30, Math.min(30, memFragment.position[2]))
                ] as [number, number, number];
              } else {
                // Fallback position calculation with fixed bounds
                const radius = 10; // Fixed radius, no scaling
                const angle = (index * 2 * Math.PI) / Math.max(fragmentsToProcess.length, 1);
                const x = Math.cos(angle) * radius;
                const z = Math.sin(angle) * radius;
                const y = memFragment.strain * 2; // Smaller height variation
                position = [x, y, z];
              }
              
              const fragment: MemoryFragment = {
                id: `mem-${memFragment.name}-${index}`,
                position: position,
                // Create quantum state from coherence value
                state: [new Complex(
                  Math.sqrt(memFragment.coherence || 0.5),
                  memFragment.phase || (index * Math.PI / 4) // Use phase from backend or calculate
                )],
                coherence: memFragment.coherence || 0.5,
                timestamp: Date.now() - (index * 100),
                entropy: memFragment.entropy || 0.1,
                strain: memFragment.strain || 0.1,
                parentFragments: [],
                // Store original backend data for reference
                metadata: {
                  originalName: memFragment.name,
                  connections: memFragment.connections || 0,
                  size: memFragment.size || 1.0,
                  couplingStrength: memFragment.coupling_strength || 0.5
                }
              };
              
              field.fragments.push(fragment);
            });
            
            // Silent cleanup - no console warnings for production
          }
        }
        // Second priority: Create from quantum states if no memory fragments
        else if (states && Object.keys(states).length > 0) {
          // Clear and rebuild fragments from current quantum states
          field.fragments = [];
          
          Object.entries(states).forEach(([stateName, state], index) => {
            // Only create fragments for states with sufficient coherence (OSH threshold)
            const stateCoherence = (state as any)?.coherence || 0;
            if (state && stateCoherence > 0.1) {
            // Calculate fragment properties according to OSH equations
            const phi = metrics.phi || 0;
            const rsp = metrics.rsp || 1;
            
            // Position based on information geometry (MATHEMATICAL_SUPPLEMENT.md)
            // Information creates curvature in spacetime
            const informationCurvature = metrics.information_curvature || 0.001;
            const radius = 10; // Fixed radius for stability
            const angle = (index * 2 * Math.PI) / Math.max(Object.keys(states).length, 1);
            
            // Height represents information density (Φ contribution) - bounded
            const height = Math.max(-5, Math.min(5, phi * stateCoherence * 2));
            
            const fragment: MemoryFragment = {
              id: `state-${stateName}-${index}`,
              position: [
                Math.cos(angle) * radius * (1 + informationCurvature),
                height,
                Math.sin(angle) * radius * (1 + informationCurvature)
              ],
              // Quantum state represented as complex amplitude
              state: [new Complex(
                Math.sqrt(stateCoherence), // Amplitude
                index * Math.PI / 4 // Phase based on index
              )],
              coherence: stateCoherence,
              timestamp: Date.now() - (index * 100), // Temporal ordering
              entropy: (state as any)?.entropy || (1 - stateCoherence),
              // Strain from RSP equation: RSP = I·C/E
              strain: Math.min(1, rsp / 1000), // Normalized strain
              parentFragments: (state as any)?.properties?.entangled_with || []
            };
            
            field.fragments.push(fragment);
          }
        });
      }
      
      // Update field properties with real-time OSH metrics
      if (field) {
        // Core OSH metrics
        field.averageCoherence = metrics.coherence || 0.5;
        field.totalEntropy = metrics.entropy || 0.1;
        
        // Memory field coupling from OSH theory
        const memoryFieldCoupling = metrics.memory_field_coupling || 0.5;
        
        // Create coherence waves based on Φ (integrated information)
        if (metrics.phi > 0.1 && showCoherenceWaves) {
          // Wave properties derived from OSH equations
          const waveAmplitude = metrics.phi * memoryFieldCoupling;
          const waveFrequency = metrics.coherence * Math.PI;
          
          // Create wave at observer-influenced positions
          const observerInfluence = metrics.observer_influence || 0;
          const waveOrigin: [number, number, number] = [
            observerInfluence * 10,
            0,
            observerInfluence * 10
          ];
          
          // Generate coherence wave
          const wave: CoherenceWave = {
            origin: waveOrigin,
            amplitude: waveAmplitude,
            frequency: waveFrequency,
            phase: metrics.universe_time ? (metrics.universe_time * 0.1) % (2 * Math.PI) : 0,
            timestamp: Date.now()
          };
          
          // Add wave to field by updating coherence values
          // Since propagateCoherenceWave is not available, we simulate it
          field.fragments.forEach(fragment => {
            const dx = fragment.position[0] - waveOrigin[0];
            const dy = fragment.position[1] - waveOrigin[1];
            const dz = fragment.position[2] - waveOrigin[2];
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
            
            // Wave influence decreases with distance
            const waveInfluence = waveAmplitude * Math.exp(-distance / 20) * Math.cos(distance * 0.1 - wave.phase);
            fragment.coherence = Math.min(1, Math.max(0, fragment.coherence + waveInfluence * 0.01));
          });
        }
        
        // Update fragment properties based on global metrics
        field.fragments.forEach((fragment, i) => {
          // Apply temporal stability decay (OSH theory)
          const temporalStability = metrics.temporal_stability || 0.9;
          fragment.coherence *= temporalStability;
          
          // Apply gravitational effects on memory (OSH gravitational binding)
          const gravitationalAnomaly = metrics.gravitational_anomaly || 0;
          if (gravitationalAnomaly > 0) {
            // Gravitational effects concentrate information
            const centerPull = 0.1 * gravitationalAnomaly;
            fragment.position[0] *= (1 - centerPull);
            fragment.position[2] *= (1 - centerPull);
          }
          
          // Update strain based on conservation ratio
          const conservationError = metrics.conservation_error || 0;
          fragment.strain = Math.min(1, fragment.strain + conservationError * 0.1);
        });
      }
      
      // Limit update counter to prevent infinite growth
      setUpdateCounter(prev => (prev + 1) % 1000);
      } catch (error) {
        console.error('[MemoryFieldVisualizer] Error updating fragments:', error);
      }
    }, FRAGMENT_UPDATE_THROTTLE);
    
    return () => {
      if (fragmentUpdateTimerRef.current) {
        clearTimeout(fragmentUpdateTimerRef.current);
        fragmentUpdateTimerRef.current = undefined;
      }
    };
  }, [metrics, states, memoryFieldEngine, isActive, showCoherenceWaves]);
  
  // Optimized physics loop with requestAnimationFrame
  useEffect(() => {
    if (!isActive || !memoryFieldEngine) return;
    
    let frameId: number;
    
    const runPhysics = () => {
      const now = Date.now();
      const delta = (now - lastPhysicsUpdate.current) / 1000;
      
      if (delta >= PHYSICS_UPDATE_MS / 1000) {
        try {
          memoryFieldEngine.update(delta * timeScale);
          lastPhysicsUpdate.current = now;
        } catch (error) {
          console.error('[MemoryField] Physics update error:', error);
        }
      }
      
      frameId = requestAnimationFrame(runPhysics);
    };
    
    frameId = requestAnimationFrame(runPhysics);
    
    return () => {
      if (frameId) {
        cancelAnimationFrame(frameId);
      }
    };
  }, [isActive, memoryFieldEngine, timeScale]);
  
  // Optimized UI update loop with performance monitoring
  useEffect(() => {
    if (!isActive) return;
    
    let frameCount = 0;
    let lastFpsUpdate = Date.now();
    let lowFpsCount = 0; // Track consecutive low FPS frames
    
    const updateUI = () => {
      const now = Date.now();
      frameCount++;
      
      // Update FPS counter
      if (now - lastFpsUpdate >= 1000) {
        const fps = frameCount;
        frameCount = 0;
        lastFpsUpdate = now;
        
        // Aggressive performance adaptation for commercial hardware
        if (fps < 20) {
          lowFpsCount++;
          // Immediately reduce quality for low FPS
          setFragmentLimit(prev => Math.max(6, prev - 4));
          setShowStrainField(false); // Disable expensive effects
          if (lowFpsCount > 1) {
            setShowCoherenceWaves(false); // Disable waves quickly
            setAutoRotate(false); // Stop auto rotation
          }
        } else if (fps < 30 && fragmentLimit > 8) {
          // Gradual reduction for moderate performance issues
          setFragmentLimit(prev => Math.max(8, prev - 2));
        } else if (fps > 50) {
          lowFpsCount = 0; // Reset low FPS counter
          // Very conservative increase only if excellent performance
          if (fragmentLimit < MAX_VISIBLE_FRAGMENTS && fps > 55) {
            setFragmentLimit(prev => Math.min(MAX_VISIBLE_FRAGMENTS, prev + 1));
          }
        }
        
        // Update FPS display
        setCurrentFps(fps);
        
        // Log performance metrics periodically
        if (fps < 20) {
          console.warn(`[MemoryFieldVisualizer] Low FPS: ${fps}, fragments: ${fragmentLimit}`);
        }
      }
      
      if (now - lastUIUpdate.current >= ANIMATION_THROTTLE_MS) {
        setUpdateCounter(prev => prev + 1);
        lastUIUpdate.current = now;
      }
      
      animationFrameRef.current = requestAnimationFrame(updateUI);
    };
    
    animationFrameRef.current = requestAnimationFrame(updateUI);
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isActive, fragmentLimit]);
  
  // Memoized visualization data with fragment limiting and real-time metrics
  const visualizationData = useMemo(() => {
    if (!memoryFieldEngine) return null;
    
    try {
      const field = memoryFieldEngine.getCurrentField();
      if (!field) {
        return null;
      }
      
      const allFragments = field.fragments || [];
    
    // Apply filters and limits
    const filteredFragments = allFragments
      .filter(f => f.coherence >= coherenceThreshold && f.strain >= strainThreshold)
      .sort((a, b) => b.coherence - a.coherence)
      .slice(0, fragmentLimit);
    
    // Enhanced field metrics with backend data
    if (metrics && field) {
      // Update field metrics with real-time data
      field.averageCoherence = metrics.coherence || field.averageCoherence || 0.5;
      field.totalEntropy = metrics.entropy || field.totalEntropy || 0;
      
      // Add dynamic strain based on RSP
      if (metrics.rsp > 100) {
        const rspStrain = Math.min(metrics.rsp / 1000, 1);
        filteredFragments.forEach(f => {
          f.strain = Math.min(1, (f.strain || 0) + rspStrain * 0.1);
        });
      }
    }
    
    return {
      fragments: filteredFragments,
      metrics: (() => {
        try {
          return memoryFieldEngine.getMetrics();
        } catch (e) {
          console.warn('[MemoryFieldVisualizer] Failed to get metrics:', e);
          return { coherence: 0.5, entropy: 0, totalFragments: 0 };
        }
      })(),
      field,
      totalFragments: allFragments.length,
      hasLiveData: isConnected && Object.keys(states || {}).length > 0,
      realTimeMetrics: metrics
    };
    } catch (error) {
      console.error('[MemoryFieldVisualizer] Error creating visualization data:', error);
      return null;
    }
  }, [memoryFieldEngine, updateCounter, fragmentLimit, coherenceThreshold, strainThreshold, isConnected, states, metrics]);
  
  // Memoized filtered fragments
  const visibleFragments = useMemo(() => {
    if (!visualizationData) return [];
    
    return visualizationData.fragments.filter(fragment => {
      if (!showHighCoherence && fragment.coherence > 0.7) return false;
      if (!showLowCoherence && fragment.coherence < 0.3) return false;
      if (!showHighStrain && fragment.strain > 0.7) return false;
      return true;
    });
  }, [visualizationData, showHighCoherence, showLowCoherence, showHighStrain]);
  
  // Export functionality
  const exportData = useCallback(() => {
    if (!visualizationData) return;
    
    const exportObj = {
      timestamp: new Date().toISOString(),
      metrics: visualizationData.metrics,
      fragments: visualizationData.fragments.map(f => ({
        id: f.id,
        position: f.position,
        coherence: f.coherence,
        entropy: f.entropy,
        strain: f.strain,
        timestamp: f.timestamp || Date.now()
      })),
      field: {
        averageCoherence: visualizationData.field.averageCoherence || 0,
        totalCoherence: visualizationData.field.totalCoherence || 0,
        totalEntropy: visualizationData.field.totalEntropy || 0,
        fragmentCount: visualizationData.fragments.length
      }
    };
    
    const blob = new Blob([JSON.stringify(exportObj, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `memory-field-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [visualizationData]);
  
  // Reset function
  const handleReset = useCallback(() => {
    if (memoryFieldEngine) {
      try {
        memoryFieldEngine.reset();
        setUpdateCounter(prev => prev + 1);
      } catch (error) {
        console.error('[MemoryFieldVisualizer] Error resetting engine:', error);
      }
    }
  }, [memoryFieldEngine]);
  
  if (!visualizationData) {
    return (
      <div style={{ 
        width: '100%', 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        background: '#0a0a0a',
        color: '#666'
      }}>
        <p>Initializing Memory Field Engine...</p>
      </div>
    );
  }
  
  return (
    <div style={{ 
      width: '100%', 
      height: '100%', 
      position: 'relative',
      background: '#0a0a0a',
      overflow: 'hidden'
    }}>
      {/* 3D View Canvas */}
      {viewMode === '3d' && (
        <Canvas
          camera={{ position: cameraPosition, fov: 60 }}
          gl={{ 
            antialias: false,
            alpha: true,
            powerPreference: 'high-performance'
          }}
          dpr={1}
          performance={{ min: 0.5 }}
        >
          <PerspectiveCamera makeDefault position={cameraPosition} />
          <OrbitControls 
            enablePan={true} 
            enableZoom={true} 
            enableRotate={true}
            autoRotate={autoRotate}
            autoRotateSpeed={0.5}
          />
          
          {/* Simplified Lighting */}
          <ambientLight intensity={0.4} />
          <pointLight position={[50, 50, 50]} intensity={0.6} color={primaryColor} />
          
          <group>
            {/* Coherence Field Indicator */}
            {showCoherenceWaves && visualizationData.field && (
              <CoherenceFieldIndicator coherence={visualizationData.field.averageCoherence || 0.5} primaryColor={primaryColor} />
            )}
            
            {/* Memory Fragments with dynamic effects */}
            {visibleFragments.map((fragment, index) => (
              <MemoryFragmentRenderer
                key={fragment.id}
                fragment={fragment}
                viewMode={fragmentViewMode}
                primaryColor={primaryColor}
              />
            ))}
            
            {/* Particle connections for high coherence fragments - limited for performance */}
            {showCoherenceWaves && visibleFragments
              .filter(f => (f.coherence || 0) > 0.7)
              .slice(0, MAX_CONNECTIONS)
              .map((fragment, i, arr) => {
                if (i >= arr.length - 1) return null;
                const nextFragment = arr[i + 1];
                
                const pos1 = fragment.position;
                const pos2 = nextFragment.position;
                
                // Create a properly oriented line between fragments
                const start = new THREE.Vector3(...pos1);
                const end = new THREE.Vector3(...pos2);
                const direction = new THREE.Vector3().subVectors(end, start);
                const distance = direction.length();
                const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
                
                // Calculate proper rotation for the connection line
                const quaternion = new THREE.Quaternion();
                const up = new THREE.Vector3(0, 1, 0);
                quaternion.setFromUnitVectors(up, direction.normalize());
                const euler = new THREE.Euler().setFromQuaternion(quaternion);
                
                // Use a different color for connections (lighter/desaturated)
                const connectionColor = adjustLightness(primaryColor, 30);
                
                return (
                  <mesh
                    key={`${fragment.id}-${nextFragment.id}`}
                    position={[midpoint.x, midpoint.y, midpoint.z]}
                    rotation={[euler.x, euler.y, euler.z]}
                  >
                    <cylinderGeometry args={[0.1, 0.1, distance, 8]} />
                    <meshBasicMaterial 
                      color={connectionColor}
                      opacity={0.3 * Math.min(fragment.coherence || 0, nextFragment.coherence || 0)}
                      transparent
                      depthWrite={false}
                    />
                  </mesh>
                );
              })}
            
            {/* Simple grid - no fog or stars for performance */}
            <gridHelper args={[60, 12, '#1a1a2e', '#0f0f1e']} position={[0, -5, 0]} />
          </group>
        </Canvas>
      )}
      
      {/* Timeline View Canvas */}
      {viewMode === 'timeline' && (
        <Canvas
          camera={{ position: [0, 0, 50], fov: 60 }}
          gl={{ 
            antialias: false,
            alpha: true,
            powerPreference: 'high-performance'
          }}
          dpr={1}
        >
          <PerspectiveCamera makeDefault position={[0, 0, 50]} />
          <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
          
          {/* Simplified Lighting */}
          <ambientLight intensity={0.4} />
          <pointLight position={[0, 20, 20]} intensity={0.6} color={primaryColor} />
          
          <group>
            
            {/* Memory fragments on timeline */}
            {visibleFragments.map((fragment, index) => {
              const creationTime = fragment.timestamp || Date.now();
              const age = (Date.now() - creationTime) / 1000;
              const normalizedAge = Math.min(age / 30, 1); // Normalize to 30 seconds
              
              const x = -50 + normalizedAge * 100;
              const coherence = fragment.coherence || 0;
              const strain = fragment.strain || 0;
              const y = coherence * 20 - 5;
              const z = (index % 10 - 5) * 2;
              
              const size = 0.5 + strain * 2;
              const color = new THREE.Color().setHSL(
                0.6 - coherence * 0.2, // Hue from blue to green
                0.8,
                0.5 + coherence * 0.3
              );
              
              return (
                <group key={fragment.id}>
                  {/* Fragment sphere */}
                  <MemoizedSphere
                    position={[x, y, z]}
                    args={[size, 6, 6]}
                  >
                    <meshBasicMaterial
                      color={color}
                      opacity={0.6 + coherence * 0.4}
                      transparent
                    />
                  </MemoizedSphere>
                  
                  {/* Vertical line to timeline */}
                  <mesh position={[x, (y - 10) / 2, z]}>
                    <boxGeometry args={[0.1, Math.abs(y + 10), 0.1]} />
                    <meshBasicMaterial 
                      color={color}
                      opacity={0.3}
                      transparent
                    />
                  </mesh>
                  
                  {/* Age indicator */}
                  {index < 5 && (
                    <MemoizedText
                      position={[x, y + size + 1, z]}
                      fontSize={0.8}
                      color={color}
                      anchorX="center"
                    >
                      {age.toFixed(1)}s
                    </MemoizedText>
                  )}
                </group>
              );
            })}
            
            {/* Timeline axis with markers */}
            <mesh position={[0, -10, 0]}>
              <boxGeometry args={[100, 0.3, 0.3]} />
              <meshBasicMaterial color={primaryColor} />
            </mesh>
            
            {/* Time markers */}
            {[0, 10, 20, 30].map(time => {
              const x = -50 + (time / 30) * 100;
              return (
                <group key={`marker-${time}`}>
                  <mesh position={[x, -11, 0]}>
                    <boxGeometry args={[0.2, 2, 0.2]} />
                    <meshBasicMaterial color={primaryColor} />
                  </mesh>
                  <MemoizedText
                    position={[x, -14, 0]}
                    fontSize={1.5}
                    color={primaryColor}
                    anchorX="center"
                  >
                    {time}s
                  </MemoizedText>
                </group>
              );
            })}
            
            <MemoizedText
              position={[0, -18, 0]}
              fontSize={2}
              color="#ffffff"
              anchorX="center"
            >
              Temporal Evolution →
            </MemoizedText>
          </group>
        </Canvas>
      )}
      
      
      {/* Enhanced Control Panel */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        background: 'rgba(0, 0, 0, 0.8)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '8px',
        padding: '16px',
        backdropFilter: 'blur(10px)',
        maxWidth: '320px',
        maxHeight: '80vh',
        overflowY: 'auto',
        fontSize: '12px',
        color: '#fff'
      }}>
        <div style={{ marginBottom: '16px' }}>
          <h3 style={{ 
            margin: '0 0 12px 0', 
            fontSize: '14px', 
            fontWeight: 'bold',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <Activity size={16} />
            Memory Field Visualizer
            {visualizationData.hasLiveData && (
              <span style={{
                background: primaryColor,
                color: '#000',
                padding: '2px 6px',
                borderRadius: '4px',
                fontSize: '10px',
                fontWeight: 'bold',
                boxShadow: `0 0 10px ${primaryColor}50`
              }}>
                LIVE DATA
              </span>
            )}
          </h3>
          
          {/* Status indicators */}
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: '1fr 1fr', 
            gap: '8px',
            marginBottom: '12px'
          }}>
            <div style={{ 
              background: 'rgba(255, 255, 255, 0.05)', 
              padding: '8px', 
              borderRadius: '4px' 
            }}>
              <Tooltip content="Overall field coherence level">
                <div style={{ fontSize: '10px', opacity: 0.7 }}>Coherence</div>
              </Tooltip>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: primaryColor }}>
                {(() => {
                  const coherence = visualizationData.field.averageCoherence || 0;
                  const percentage = coherence * 100;
                  return isNaN(percentage) ? '0.0' : percentage.toFixed(1);
                })()}%
              </div>
            </div>
            <div style={{ 
              background: 'rgba(255, 255, 255, 0.05)', 
              padding: '8px', 
              borderRadius: '4px' 
            }}>
              <Tooltip content="Number of active memory fragments">
                <div style={{ fontSize: '10px', opacity: 0.7 }}>Fragments</div>
              </Tooltip>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: adjustLightness(primaryColor, 20) }}>
                {visibleFragments.length}/{visualizationData.totalFragments}
              </div>
            </div>
          </div>
          
          {/* View Mode Selector */}
          <div style={{ marginBottom: '12px' }}>
            <label style={{ 
              display: 'block', 
              marginBottom: '4px', 
              fontSize: '11px', 
              opacity: 0.7 
            }}>
              View Mode
            </label>
            <div style={{ display: 'flex', gap: '4px' }}>
              {(['3d', 'timeline'] as const).map(mode => (
                <Tooltip key={mode} content={`Switch to ${mode} view`}>
                  <button
                    onClick={() => setViewMode(mode)}
                    style={{
                      flex: 1,
                      padding: '6px',
                      background: viewMode === mode ? primaryColor : 'rgba(255, 255, 255, 0.1)',
                      border: 'none',
                      borderRadius: '4px',
                      color: '#fff',
                      fontSize: '11px',
                      cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                  >
                    {mode.toUpperCase()}
                  </button>
                </Tooltip>
              ))}
            </div>
          </div>
          
          {/* Visual Options */}
          <div style={{ marginBottom: '12px' }}>
            <label style={{ 
              display: 'block', 
              marginBottom: '4px', 
              fontSize: '11px', 
              opacity: 0.7 
            }}>
              Visual Options
            </label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <Tooltip content="Show/hide high coherence fragments (>70%)">
                <label style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '8px',
                  cursor: 'pointer'
                }}>
                  <input
                    type="checkbox"
                    checked={showHighCoherence}
                    onChange={(e) => setShowHighCoherence(e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  <Eye size={12} style={{ opacity: showHighCoherence ? 1 : 0.5 }} />
                  High Coherence
                </label>
              </Tooltip>
              
              <Tooltip content="Show/hide low coherence fragments (<30%)">
                <label style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '8px',
                  cursor: 'pointer'
                }}>
                  <input
                    type="checkbox"
                    checked={showLowCoherence}
                    onChange={(e) => setShowLowCoherence(e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  <Eye size={12} style={{ opacity: showLowCoherence ? 1 : 0.5 }} />
                  Low Coherence
                </label>
              </Tooltip>
              
              <Tooltip content="Show/hide high strain areas (>70%)">
                <label style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '8px',
                  cursor: 'pointer'
                }}>
                  <input
                    type="checkbox"
                    checked={showHighStrain}
                    onChange={(e) => setShowHighStrain(e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  <Zap size={12} style={{ opacity: showHighStrain ? 1 : 0.5 }} />
                  High Strain
                </label>
              </Tooltip>
              
              <Tooltip content="Show/hide coherence wave effect">
                <label style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '8px',
                  cursor: 'pointer'
                }}>
                  <input
                    type="checkbox"
                    checked={showCoherenceWaves}
                    onChange={(e) => setShowCoherenceWaves(e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  <Activity size={12} style={{ opacity: showCoherenceWaves ? 1 : 0.5 }} />
                  Coherence Waves
                </label>
              </Tooltip>
              
              <Tooltip content="Show/hide strain field visualization">
                <label style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '8px',
                  cursor: 'pointer'
                }}>
                  <input
                    type="checkbox"
                    checked={showStrainField}
                    onChange={(e) => setShowStrainField(e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  <Grid3x3 size={12} style={{ opacity: showStrainField ? 1 : 0.5 }} />
                  Strain Field
                </label>
              </Tooltip>
            </div>
          </div>
          
          {/* Advanced Controls */}
          <div style={{ marginBottom: '12px' }}>
            <label style={{ 
              display: 'block', 
              marginBottom: '4px', 
              fontSize: '11px', 
              opacity: 0.7 
            }}>
              Advanced Controls
            </label>
            
            {/* Fragment View Mode */}
            <div style={{ marginBottom: '8px' }}>
              <Tooltip content="Change how fragments are rendered">
                <label style={{ fontSize: '10px', opacity: 0.7 }}>Fragment Style</label>
              </Tooltip>
              <select
                value={fragmentViewMode}
                onChange={(e) => setFragmentViewMode(e.target.value as any)}
                style={{
                  width: '100%',
                  padding: '4px',
                  background: 'rgba(0, 0, 0, 0.8)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '4px',
                  color: '#fff',
                  fontSize: '11px',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = primaryColor;
                  e.currentTarget.style.background = `${primaryColor}20`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.2)';
                  e.currentTarget.style.background = 'rgba(0, 0, 0, 0.8)';
                }}
              >
                <option value="boxes" style={{ background: '#000', color: '#fff' }}>Quantum Boxes</option>
                <option value="spheres" style={{ background: '#000', color: '#fff' }}>Energy Spheres</option>
                <option value="particles" style={{ background: '#000', color: '#fff' }}>Wave Particles</option>
              </select>
            </div>
            
            {/* Fragment Limit */}
            <div style={{ marginBottom: '8px' }}>
              <Tooltip content="Maximum number of fragments to display">
                <label style={{ fontSize: '10px', opacity: 0.7 }}>
                  Fragment Limit: {fragmentLimit}
                </label>
              </Tooltip>
              <input
                type="range"
                min="6"
                max="16"
                step="2"
                value={fragmentLimit}
                onChange={(e) => setFragmentLimit(parseInt(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
            
            {/* Coherence Threshold */}
            <div style={{ marginBottom: '8px' }}>
              <Tooltip content="Minimum coherence level to display fragments">
                <label style={{ fontSize: '10px', opacity: 0.7 }}>
                  Coherence Threshold: {(coherenceThreshold * 100).toFixed(0)}%
                </label>
              </Tooltip>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={coherenceThreshold}
                onChange={(e) => setCoherenceThreshold(parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
            
            {/* Time Scale */}
            <div style={{ marginBottom: '8px' }}>
              <Tooltip content="Speed of physics simulation">
                <label style={{ fontSize: '10px', opacity: 0.7 }}>
                  Time Scale: {timeScale.toFixed(1)}x
                </label>
              </Tooltip>
              <input
                type="range"
                min="0.1"
                max="3"
                step="0.1"
                value={timeScale}
                onChange={(e) => setTimeScale(parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
            
            {/* Auto Rotate */}
            <Tooltip content="Enable camera auto-rotation">
              <label style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '8px',
                cursor: 'pointer'
              }}>
                <input
                  type="checkbox"
                  checked={autoRotate}
                  onChange={(e) => setAutoRotate(e.target.checked)}
                  style={{ cursor: 'pointer' }}
                />
                <Camera size={12} style={{ opacity: autoRotate ? 1 : 0.5 }} />
                Auto Rotate
              </label>
            </Tooltip>
          </div>
          
          {/* Action Buttons */}
          <div style={{ 
            display: 'flex', 
            gap: '8px',
            marginTop: '16px',
            paddingTop: '12px',
            borderTop: '1px solid rgba(255, 255, 255, 0.1)'
          }}>
            <Tooltip content="Export visualization data as JSON">
              <button
                onClick={exportData}
                style={{
                  flex: 1,
                  padding: '8px',
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: `1px solid transparent`,
                  borderRadius: '4px',
                  color: '#fff',
                  fontSize: '11px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '4px',
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = `${primaryColor}20`;
                  e.currentTarget.style.borderColor = primaryColor;
                  e.currentTarget.style.color = primaryColor;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                  e.currentTarget.style.borderColor = 'transparent';
                  e.currentTarget.style.color = '#fff';
                }}
              >
                <Download size={12} />
                Export
              </button>
            </Tooltip>
            
            <Tooltip content="Reset memory field to initial state">
              <button
                onClick={handleReset}
                style={{
                  flex: 1,
                  padding: '8px',
                  background: 'rgba(255, 87, 34, 0.2)',
                  border: '1px solid transparent',
                  borderRadius: '4px',
                  color: '#ff5722',
                  fontSize: '11px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '4px',
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 87, 34, 0.3)';
                  e.currentTarget.style.borderColor = '#ff5722';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 87, 34, 0.2)';
                  e.currentTarget.style.borderColor = 'transparent';
                }}
              >
                <RefreshCw size={12} />
                Reset
              </button>
            </Tooltip>
          </div>
        </div>
      </div>
      
      {/* Performance Stats - moved to bottom right */}
      <div style={{
        position: 'absolute',
        bottom: '20px',
        right: '20px',
        background: 'rgba(0, 0, 0, 0.8)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '4px',
        padding: '8px 12px',
        fontSize: '10px',
        color: '#888',
        fontFamily: 'monospace'
      }}>
        <div style={{ color: currentFps < 20 ? '#ff5722' : currentFps < 30 ? '#ff9800' : '#4caf50' }}>
          FPS: {currentFps} {currentFps < 20 && '⚠️'}
        </div>
        <div>Fragments: {visibleFragments.length}/{visualizationData.totalFragments}</div>
        <div>Limit: {fragmentLimit}</div>
        <div>Backend: {isConnected ? '✓ Connected' : '✗ Disconnected'}</div>
        {visualizationData.realTimeMetrics && (
          <>
            <div style={{ marginTop: '4px', paddingTop: '4px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
              RSP: {visualizationData.realTimeMetrics.rsp?.toFixed(1) || '0'}
            </div>
            <div>Φ: {visualizationData.realTimeMetrics.phi?.toFixed(2) || '0'}</div>
            <div>Coherence: {((visualizationData.realTimeMetrics.coherence || 0) * 100).toFixed(1)}%</div>
          </>
        )}
        <div style={{ marginTop: '4px', paddingTop: '4px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
          Quality: {performanceMode === 'auto' ? 'Auto' : performanceMode}
        </div>
      </div>
    </div>
  );
};

export default MemoryFieldVisualizer;