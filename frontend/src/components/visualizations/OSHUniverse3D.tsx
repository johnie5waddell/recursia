/**
 * OSH Universe 3D Visualization Component
 * Enterprise-grade 3D visualization of the Organic Simulation Hypothesis universe
 * with full error handling, memory management, and performance optimization
 */

import React, { useState, useCallback, useEffect, useRef, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import type { WebGLRenderer } from 'three';
import { 
  Info, 
  Zap, 
  Activity, 
  Layers, 
  GitBranch,
  Maximize2,
  Move3D,
  Timer,
  Gauge,
  Palette
} from 'lucide-react';
import { Tooltip } from '../ui/Tooltip';
import { VisualizationControls, VisualizationSettings, defaultVisualizationSettings } from './VisualizationControls';
import { OSHUniverse3DScene } from './OSHUniverse3DScene';
import { OSHUniverse3DErrorBoundary } from './OSHUniverse3DErrorBoundary';
import { useMemoryManager, ResourceType } from '../../utils/memoryManager';
import { disposeRenderer } from '../../utils/ThreeMemoryManager';
import '../../styles/osh-universe-3d.css';

interface OSHUniverse3DProps {
  engine: any;
  primaryColor: string;
}

interface EngineMetrics {
  rsp: number;
  coherence: number;
  entropy: number;
  recursionDepth: number;
  fps: number;
}

/**
 * OSH Universe 3D Component
 * Provides immersive 3D visualization of quantum simulation state
 * with real-time metrics and interactive controls
 */
export const OSHUniverse3D: React.FC<OSHUniverse3DProps> = ({ 
  engine, 
  primaryColor 
}) => {
  // Memory management
  const { track } = useMemoryManager('OSHUniverse3D');
  const rendererRef = useRef<WebGLRenderer | null>(null);
  
  // Visualization state
  const [showInfo, setShowInfo] = useState(false);
  const [showLattice, setShowLattice] = useState(true);
  const [showMemoryField, setShowMemoryField] = useState(true);
  const [showEntanglements, setShowEntanglements] = useState(true);
  const [showBoundary, setShowBoundary] = useState(true);
  const [showWavefronts, setShowWavefronts] = useState(true);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [observerPOV, setObserverPOV] = useState<'global' | 'observer1' | 'observer2'>('global');
  const [timeScale, setTimeScale] = useState(1);
  const [recursiveZoom, setRecursiveZoom] = useState(1);
  const [showControls, setShowControls] = useState(false);
  const [visualizationSettings, setVisualizationSettings] = useState<VisualizationSettings>(defaultVisualizationSettings);
  const [engineData, setEngineData] = useState<EngineMetrics | null>(null);
  
  /**
   * Handle engine data updates from scene
   */
  const handleEngineData = useCallback((data: EngineMetrics) => {
    setEngineData(data);
  }, []);

  /**
   * Cleanup WebGL resources on unmount
   */
  useEffect(() => {
    return () => {
      if (rendererRef.current) {
        disposeRenderer(rendererRef.current);
        rendererRef.current = null;
      }
    };
  }, []);
  
  // Loading state when engine not ready
  if (!engine) {
    return (
      <div style={{ 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        background: '#0a0a0a',
        color: primaryColor 
      }}>
        <div style={{ textAlign: 'center' }}>
          <Activity size={48} style={{ marginBottom: '16px', opacity: 0.5 }} />
          <p>Initializing OSH Universe...</p>
        </div>
      </div>
    );
  }
  
  // CSS variables for theming
  const cssVars = {
    '--primary-color': primaryColor,
    '--primary-color-rgb': `${parseInt(primaryColor.slice(1,3), 16)}, ${parseInt(primaryColor.slice(3,5), 16)}, ${parseInt(primaryColor.slice(5,7), 16)}`
  } as React.CSSProperties;
  
  return (
    <OSHUniverse3DErrorBoundary 
      primaryColor={primaryColor}
      onError={(error, errorInfo) => {
        console.error('OSH Universe 3D Error:', error, errorInfo);
      }}
    >
      <div style={{ 
        height: '100%', 
        position: 'relative', 
        background: '#0a0a0a',
        ...cssVars
      }}>
        <Canvas
          camera={{ position: [20, 20, 20], fov: 60 }}
          gl={{ 
            antialias: true, 
            alpha: true,
            powerPreference: "high-performance",
            preserveDrawingBuffer: false,
            failIfMajorPerformanceCaveat: false
          }}
          onCreated={({ gl }) => {
            rendererRef.current = gl;
            track('webgl-renderer', ResourceType.WEBGL, () => {
              if (rendererRef.current) {
                disposeRenderer(rendererRef.current);
              }
            }, { size: 8388608 }); // 8MB estimate
          }}
        >
          <Suspense fallback={
            <mesh>
              <boxGeometry args={[1, 1, 1]} />
              <meshBasicMaterial color={primaryColor} wireframe />
            </mesh>
          }>
            <OSHUniverse3DScene
              engine={engine}
              primaryColor={primaryColor}
              showLattice={showLattice}
              showMemoryField={showMemoryField}
              showEntanglements={showEntanglements}
              showBoundary={showBoundary}
              showWavefronts={showWavefronts}
              observerPOV={observerPOV}
              timeScale={timeScale}
              recursiveZoom={recursiveZoom}
              visualizationSettings={visualizationSettings}
              onSelectNode={setSelectedNode}
              onEngineData={handleEngineData}
            />
          </Suspense>
        </Canvas>
      
        {/* Info Overlay */}
        <div className={`info-overlay ${showInfo ? 'visible' : ''}`} style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          width: '350px',
          background: 'rgba(0,0,0,0.95)',
          padding: '24px',
          borderRadius: '12px',
          border: `2px solid ${primaryColor}`,
          color: 'white',
          backdropFilter: 'blur(20px)',
          transition: 'all 0.3s ease',
          opacity: showInfo ? 1 : 0,
          transform: showInfo ? 'translateX(0)' : 'translateX(20px)',
          pointerEvents: showInfo ? 'all' : 'none',
          maxHeight: '80vh',
          overflowY: 'auto'
        }}>
          <h3 style={{ color: primaryColor, marginBottom: '20px', fontSize: '20px' }}>
            OSH Universe Visualization
          </h3>
          
          <div style={{ marginBottom: '20px', fontSize: '14px', lineHeight: '1.6' }}>
            <p style={{ marginBottom: '12px' }}>
              This visualization represents the OSH Universe as a recursive information fabric, 
              where reality emerges from memory strain, coherence loops, and observer collapses.
            </p>
            
            <h4 style={{ color: primaryColor, marginTop: '16px', marginBottom: '8px' }}>
              Visual Elements:
            </h4>
            <ul style={{ paddingLeft: '20px', marginBottom: '12px' }}>
              <li><strong>Lattice Nodes:</strong> Information-processing centers with recursive depth</li>
              <li><strong>Memory Field:</strong> Pulsing membrane showing entropy and strain</li>
              <li><strong>Entanglements:</strong> Quantum, classical, and consciousness connections</li>
              <li><strong>Boundary Zone:</strong> Simulation rendering limits and decoherence</li>
              <li><strong>Wavefronts:</strong> Coherence propagation after observations</li>
            </ul>
          </div>
          
          {/* Real-time Metrics */}
          {engineData && (
            <div style={{ 
              background: 'rgba(255,255,255,0.05)', 
              padding: '16px', 
              borderRadius: '8px',
              marginBottom: '16px'
            }}>
              <h4 style={{ color: primaryColor, marginBottom: '12px' }}>Universe State</h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div>
                  <span style={{ fontSize: '12px', opacity: 0.7 }}>RSP Value</span>
                  <div style={{ fontSize: '18px', color: primaryColor, fontWeight: 'bold' }}>
                    {engineData.rsp > 1000 ? '∞' : engineData.rsp.toFixed(2)}
                  </div>
                </div>
                <div>
                  <span style={{ fontSize: '12px', opacity: 0.7 }}>Coherence</span>
                  <div style={{ fontSize: '18px', color: primaryColor, fontWeight: 'bold' }}>
                    {(engineData.coherence * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <span style={{ fontSize: '12px', opacity: 0.7 }}>Entropy</span>
                  <div style={{ fontSize: '18px', color: primaryColor, fontWeight: 'bold' }}>
                    {engineData.entropy.toFixed(3)}
                  </div>
                </div>
                <div>
                  <span style={{ fontSize: '12px', opacity: 0.7 }}>Recursion</span>
                  <div style={{ fontSize: '18px', color: primaryColor, fontWeight: 'bold' }}>
                    {Math.floor(engineData.recursionDepth)}
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Selected Node Info */}
          {selectedNode && (
            <div style={{ 
              background: 'rgba(255,255,255,0.05)', 
              padding: '16px', 
              borderRadius: '8px',
              marginBottom: '16px'
            }}>
              <h4 style={{ color: primaryColor, marginBottom: '8px' }}>
                Selected Node: {selectedNode}
              </h4>
              <p style={{ fontSize: '12px', opacity: 0.8 }}>
                Information processing center in the recursive lattice structure.
              </p>
            </div>
          )}
        </div>
        
        {/* Interactive Controls */}
        <div className="interactive-controls" style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          display: 'flex',
          gap: '8px',
          flexWrap: 'wrap',
          maxWidth: '600px'
        }}>
          {/* Visualization Toggles */}
          <Tooltip primaryColor={primaryColor} content="Toggle information overlay">
            <button 
              className={`interactive-btn ${showInfo ? 'active' : ''}`}
              onClick={() => setShowInfo(!showInfo)}
              style={{
                borderColor: showInfo ? primaryColor : 'rgba(255,255,255,0.2)',
                ...(showInfo && { 
                  background: primaryColor,
                  color: '#0a0a0a'
                })
              }}
            >
              <Info size={14} />
              Info
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Toggle recursive lattice">
            <button 
              className={`interactive-btn ${showLattice ? 'active' : ''}`}
              onClick={() => setShowLattice(!showLattice)}
              style={{
                borderColor: showLattice ? primaryColor : 'rgba(255,255,255,0.2)',
                ...(showLattice && { 
                  background: primaryColor,
                  color: '#0a0a0a'
                })
              }}
            >
              <Layers size={14} />
              Lattice
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Toggle memory field">
            <button 
              className={`interactive-btn ${showMemoryField ? 'active' : ''}`}
              onClick={() => setShowMemoryField(!showMemoryField)}
              style={{
                borderColor: showMemoryField ? primaryColor : 'rgba(255,255,255,0.2)',
                ...(showMemoryField && { 
                  background: primaryColor,
                  color: '#0a0a0a'
                })
              }}
            >
              <Activity size={14} />
              Memory
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Toggle entanglements">
            <button 
              className={`interactive-btn ${showEntanglements ? 'active' : ''}`}
              onClick={() => setShowEntanglements(!showEntanglements)}
              style={{
                borderColor: showEntanglements ? primaryColor : 'rgba(255,255,255,0.2)',
                ...(showEntanglements && { 
                  background: primaryColor,
                  color: '#0a0a0a'
                })
              }}
            >
              <GitBranch size={14} />
              Entangle
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Toggle simulation boundary">
            <button 
              className={`interactive-btn ${showBoundary ? 'active' : ''}`}
              onClick={() => setShowBoundary(!showBoundary)}
              style={{
                borderColor: showBoundary ? primaryColor : 'rgba(255,255,255,0.2)',
                ...(showBoundary && { 
                  background: primaryColor,
                  color: '#0a0a0a'
                })
              }}
            >
              <Maximize2 size={14} />
              Boundary
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Toggle coherence wavefronts">
            <button 
              className={`interactive-btn ${showWavefronts ? 'active' : ''}`}
              onClick={() => setShowWavefronts(!showWavefronts)}
              style={{
                borderColor: showWavefronts ? primaryColor : 'rgba(255,255,255,0.2)',
                ...(showWavefronts && { 
                  background: primaryColor,
                  color: '#0a0a0a'
                })
              }}
            >
              <Zap size={14} />
              Waves
            </button>
          </Tooltip>
          
          {/* View Controls */}
          <div style={{ width: '100%', height: '0' }} />
          
          <Tooltip primaryColor={primaryColor} content="Observer POV">
            <select
              value={observerPOV}
              onChange={(e) => setObserverPOV(e.target.value as any)}
              className="interactive-select"
              style={{
                background: 'rgba(0,0,0,0.6)',
                border: `1px solid ${primaryColor}`,
                color: 'white',
                padding: '6px 12px',
                borderRadius: '6px',
                fontSize: '12px'
              }}
            >
              <option value="global" style={{ background: '#1a1a1a', color: '#fff' }}>Global View</option>
              <option value="observer1" style={{ background: '#1a1a1a', color: '#fff' }}>Observer 1</option>
              <option value="observer2" style={{ background: '#1a1a1a', color: '#fff' }}>Observer 2</option>
            </select>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Time scale">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Timer size={14} />
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={timeScale}
                onChange={(e) => setTimeScale(parseFloat(e.target.value))}
                style={{ width: '80px' }}
              />
              <span style={{ fontSize: '12px', minWidth: '30px' }}>{timeScale.toFixed(1)}x</span>
            </div>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Recursive zoom">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Move3D size={14} />
              <input
                type="range"
                min="0.1"
                max="3"
                step="0.1"
                value={recursiveZoom}
                onChange={(e) => setRecursiveZoom(parseFloat(e.target.value))}
                style={{ width: '80px' }}
              />
              <span style={{ fontSize: '12px', minWidth: '30px' }}>{recursiveZoom.toFixed(1)}x</span>
            </div>
          </Tooltip>
        </div>
        
        {/* Performance Indicator and Controls Row */}
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          right: '10px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: '12px'
        }}>
          {/* FPS Display */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'rgba(0,0,0,0.6)',
            padding: '8px 12px',
            borderRadius: '6px',
            fontSize: '12px',
            color: engineData && engineData.fps > 30 ? '#00ff00' : engineData && engineData.fps > 15 ? '#ffff00' : '#ff0000'
          }}>
            <Gauge size={14} />
            {engineData ? `${engineData.fps.toFixed(0)} FPS` : '-- FPS'}
          </div>
          
          {/* Visual Controls Button and Panel */}
          <div style={{ maxWidth: '320px' }}>
            <button
              onClick={() => setShowControls(!showControls)}
              style={{
                marginBottom: showControls ? '8px' : '0',
                padding: '8px 12px',
                background: 'rgba(0,0,0,0.6)',
                border: `1px solid ${showControls ? primaryColor : 'rgba(255,255,255,0.2)'}`,
                borderRadius: '6px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                transition: 'all 0.2s ease',
                ...(showControls && {
                  background: `rgba(${parseInt(primaryColor.slice(1,3), 16)}, ${parseInt(primaryColor.slice(3,5), 16)}, ${parseInt(primaryColor.slice(5,7), 16)}, 0.1)`,
                  borderColor: primaryColor
                })
              }}
            >
              <Palette size={14} />
              {showControls ? 'Hide' : 'Show'} Visual Controls
            </button>
            
            {showControls && (
              <div style={{
                animation: 'slideDown 0.2s ease-out',
                transformOrigin: 'top'
              }}>
                <VisualizationControls
                  settings={visualizationSettings}
                  onSettingsChange={setVisualizationSettings}
                  primaryColor={primaryColor}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </OSHUniverse3DErrorBoundary>
  );
};

export default OSHUniverse3D;