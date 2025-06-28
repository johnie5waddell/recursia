/**
 * OSH Universe 3D Scene Component
 * Pure Three.js scene elements for use inside Canvas
 */

import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  PerspectiveCamera, 
  Html, 
  Float
} from '@react-three/drei';
import * as THREE from 'three';
// Temporarily disabled for debugging
// import { 
//   EffectComposer, 
//   Bloom, 
//   ChromaticAberration,
//   DepthOfField,
//   Vignette 
// } from '@react-three/postprocessing';
import { useEngineData } from '../../hooks/useEngineData';
import { useEngineAPIContext } from '../../contexts/EngineAPIContext';
import { VisualizationSettings, defaultVisualizationSettings } from './VisualizationControls';
import { useThreeMemoryManager } from '../../utils/ThreeMemoryManager';
// import { useManagedRAF } from '../../utils/MemoryManagedComponent'; // Not needed for useFrame

interface OSHUniverse3DSceneProps {
  engine: any;
  primaryColor: string;
  showLattice: boolean;
  showMemoryField: boolean;
  showEntanglements: boolean;
  showBoundary: boolean;
  showWavefronts: boolean;
  observerPOV: 'global' | 'observer1' | 'observer2';
  timeScale: number;
  recursiveZoom: number;
  visualizationSettings: VisualizationSettings;
  onSelectNode: (id: string | null) => void;
  onEngineData?: (data: any) => void;
}

/**
 * Recursive Information Lattice Node
 * Represents information-processing centers with recursive depth
 */
const LatticeNode: React.FC<{
  position: THREE.Vector3;
  coherence: number;
  strain: number;
  recursionDepth: number;
  connections: THREE.Vector3[];
  primaryColor: string;
  intensity: number;
  glow: number;
  isCollapsing: boolean;
  nodeId: string;
  onSelect: (id: string) => void;
}> = ({ position, coherence, strain, recursionDepth, connections, primaryColor, intensity, glow, isCollapsing, nodeId, onSelect }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const threeManager = useThreeMemoryManager('LatticeNode');
  
  useFrame((state) => {
    if (meshRef.current) {
      // Oscillation based on information transfer
      const time = state.clock.elapsedTime;
      meshRef.current.scale.setScalar(
        1 + Math.sin(time * 2 + position.x) * 0.1 * coherence
      );
      
      // Discrete jumps for observer collapses
      if (isCollapsing) {
        meshRef.current.scale.setScalar(2);
        meshRef.current.rotation.x += 0.1;
        meshRef.current.rotation.y += 0.1;
      }
      
      // Recursive depth visualization through rotation
      meshRef.current.rotation.z = recursionDepth * 0.1 + time * 0.1;
    }
  });

  // Track mesh resources
  useEffect(() => {
    if (meshRef.current && threeManager) {
      threeManager.track(`node-${nodeId}`, meshRef.current);
    }
  }, [nodeId, threeManager]);
  
  // Node appearance based on coherence and strain
  const nodeColor = new THREE.Color(primaryColor);
  nodeColor.multiplyScalar(coherence * intensity);
  if (strain > 0.7) {
    nodeColor.lerp(new THREE.Color('#ff0000'), strain - 0.7);
  }
  
  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onClick={() => onSelect(nodeId)}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <icosahedronGeometry args={[0.3 + recursionDepth * 0.05, Math.min(recursionDepth, 3)]} />
        <meshPhysicalMaterial
          color={nodeColor}
          emissive={nodeColor}
          emissiveIntensity={coherence * glow * intensity}
          metalness={0.8}
          roughness={0.2}
          clearcoat={1}
          clearcoatRoughness={0.1}
          transparent
          opacity={0.7 + coherence * 0.3}
          wireframe={recursionDepth > 3}
        />
      </mesh>
      {/* Glow effect */}
      <mesh scale={[1.2, 1.2, 1.2]}>
        <icosahedronGeometry args={[0.3 + recursionDepth * 0.05, 1]} />
        <meshBasicMaterial
          color={nodeColor}
          transparent
          opacity={coherence * glow * 0.3}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      {hovered && (
        <Html center>
          <div style={{
            background: 'rgba(0,0,0,0.95)',
            color: 'white',
            padding: '12px',
            borderRadius: '8px',
            border: `1px solid ${primaryColor}`,
            fontSize: '12px',
            whiteSpace: 'nowrap'
          }}>
            <div style={{ color: primaryColor, fontWeight: 'bold' }}>Node {nodeId}</div>
            <div>Coherence: {(coherence * 100).toFixed(1)}%</div>
            <div>Memory Strain: {(strain * 100).toFixed(1)}%</div>
            <div>Recursion Depth: {recursionDepth}</div>
          </div>
        </Html>
      )}
    </group>
  );
};

/**
 * Memory Field Topology
 * Semi-transparent membrane representing memory field dynamics
 */
const MemoryFieldTopology: React.FC<{
  entropy: number;
  coherence: number;
  strain: number;
  primaryColor: string;
  secondaryColor: string;
  opacity: number;
  distortion: number;
  time: number;
}> = ({ entropy, coherence, strain, primaryColor, secondaryColor, opacity, distortion, time }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const threeManager = useThreeMemoryManager('MemoryFieldTopology');
  
  const materialRef = useRef<THREE.ShaderMaterial>();
  
  useFrame((state) => {
    if (materialRef.current && materialRef.current.uniforms) {
      materialRef.current.uniforms.time.value = state.clock.elapsedTime * time;
      materialRef.current.uniforms.entropy.value = entropy;
      materialRef.current.uniforms.coherence.value = coherence;
      materialRef.current.uniforms.strain.value = strain;
    }
  });

  // Track mesh resources
  useEffect(() => {
    if (meshRef.current && threeManager) {
      threeManager.track('memory-field-mesh', meshRef.current);
    }
  }, [threeManager]);
  
  const vertexShader = `
    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;
    
    uniform float time;
    uniform float entropy;
    uniform float strain;
    uniform float distortion;
    
    // Simplex noise for organic deformation
    vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
    
    float snoise(vec3 v) {
      const vec2 C = vec2(1.0/6.0, 1.0/3.0);
      const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
      
      vec3 i = floor(v + dot(v, C.yyy));
      vec3 x0 = v - i + dot(i, C.xxx);
      
      vec3 g = step(x0.yzx, x0.xyz);
      vec3 l = 1.0 - g;
      vec3 i1 = min(g.xyz, l.zxy);
      vec3 i2 = max(g.xyz, l.zxy);
      
      vec3 x1 = x0 - i1 + C.xxx;
      vec3 x2 = x0 - i2 + C.yyy;
      vec3 x3 = x0 - D.yyy;
      
      i = mod289(i);
      vec4 p = permute(permute(permute(
        i.z + vec4(0.0, i1.z, i2.z, 1.0))
        + i.y + vec4(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4(0.0, i1.x, i2.x, 1.0));
      
      float n_ = 0.142857142857;
      vec3 ns = n_ * D.wyz - D.xzx;
      
      vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
      
      vec4 x_ = floor(j * ns.z);
      vec4 y_ = floor(j - 7.0 * x_);
      
      vec4 x = x_ * ns.x + ns.yyyy;
      vec4 y = y_ * ns.x + ns.yyyy;
      vec4 h = 1.0 - abs(x) - abs(y);
      
      vec4 b0 = vec4(x.xy, y.xy);
      vec4 b1 = vec4(x.zw, y.zw);
      
      vec4 s0 = floor(b0) * 2.0 + 1.0;
      vec4 s1 = floor(b1) * 2.0 + 1.0;
      vec4 sh = -step(h, vec4(0.0));
      
      vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
      vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
      
      vec3 p0 = vec3(a0.xy, h.x);
      vec3 p1 = vec3(a0.zw, h.y);
      vec3 p2 = vec3(a1.xy, h.z);
      vec3 p3 = vec3(a1.zw, h.w);
      
      vec4 norm = 1.79284291400159 - 0.85373472095314 * vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3));
      p0 *= norm.x;
      p1 *= norm.y;
      p2 *= norm.z;
      p3 *= norm.w;
      
      vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
      m = m * m;
      return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
    }
    
    void main() {
      vUv = uv;
      vPosition = position;
      vNormal = normal;
      
      // Deform based on memory strain and entropy
      vec3 pos = position;
      float noise1 = snoise(position * 2.0 + time * 0.5);
      float noise2 = snoise(position * 4.0 - time * 0.3);
      
      // Memory saturation deformation with adjustable distortion
      float deformation = (noise1 * 0.5 + noise2 * 0.5) * strain * distortion;
      pos += normal * deformation * 0.3;
      
      // Entropy-based undulation
      pos.y += sin(position.x * 3.0 + time) * sin(position.z * 3.0 - time) * entropy * 0.2;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `;
  
  const fragmentShader = `
    uniform vec3 primaryColor;
    uniform vec3 secondaryColor;
    uniform float time;
    uniform float entropy;
    uniform float coherence;
    uniform float strain;
    uniform float opacity;
    
    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;
    
    void main() {
      // Color gradient based on entropy, coherence, and strain
      vec3 entropyColor = secondaryColor; // Secondary color for high entropy
      vec3 coherenceColor = primaryColor; // Primary color for coherence
      vec3 strainColor = vec3(0.8, 0.1, 0.8); // Purple for high strain
      
      // Mix colors based on field state
      vec3 color = mix(coherenceColor, entropyColor, entropy);
      color = mix(color, strainColor, strain * strain);
      
      // Pulsing effect
      float pulse = sin(time * 2.0 + vPosition.x * 0.5) * 0.5 + 0.5;
      color *= 0.8 + pulse * 0.2;
      
      // Edge glow
      float edge = 1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0)));
      color += primaryColor * pow(edge, 2.0) * 0.3;
      
      // Transparency based on coherence and user-defined opacity
      float alpha = opacity * (0.5 + coherence * 0.5 - entropy * 0.2);
      
      gl_FragColor = vec4(color, alpha);
    }
  `;
  
  return (
    <mesh ref={meshRef} name="memory-field-mesh">
      <sphereGeometry args={[8, 64, 64]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={{
          primaryColor: { value: new THREE.Color(primaryColor) },
          secondaryColor: { value: new THREE.Color(secondaryColor) },
          time: { value: 0 },
          entropy: { value: entropy },
          coherence: { value: coherence },
          strain: { value: strain },
          distortion: { value: distortion },
          opacity: { value: opacity }
        }}
        transparent
        side={THREE.DoubleSide}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
};

/**
 * Entanglement Threads
 * Dynamic connections between entangled nodes
 */
const EntanglementThreads: React.FC<{
  connections: Array<{
    start: THREE.Vector3;
    end: THREE.Vector3;
    strength: number;
    type: 'classical' | 'quantum' | 'consciousness';
  }>;
  primaryColor: string;
  intensity: number;
  width: number;
  time: number;
}> = ({ connections, primaryColor, intensity, width, time }) => {
  const linesRef = useRef<THREE.Group>(null);
  const threeManager = useThreeMemoryManager('EntanglementThreads');
  
  useFrame((state) => {
    if (linesRef.current) {
      linesRef.current.children.forEach((child, index) => {
        if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshBasicMaterial) {
          const connection = connections[index];
          if (connection) {
            // Phase-shift animation
            child.material.opacity = 0.3 + Math.sin(state.clock.elapsedTime * 2 + index) * 0.2 * connection.strength;
            
            // Vibration based on fidelity
            const vibration = Math.sin(state.clock.elapsedTime * 10 * connection.strength) * 0.02;
            child.position.x = connection.start.x + vibration;
          }
        }
      });
    }
  });
  
  // Memoize geometries and materials
  const connectionElements = useMemo(() => {
    // Skip if no connections or manager not ready
    if (!connections || connections.length === 0) {
      return [];
    }
    
    return connections.map((connection, index) => {
      const points = [connection.start, connection.end];
      const curve = new THREE.CatmullRomCurve3(points);
      const tubeGeometry = new THREE.TubeGeometry(curve, 20, width, 8, false);
      
      const color = connection.type === 'quantum' 
        ? new THREE.Color(primaryColor)
        : connection.type === 'consciousness'
        ? new THREE.Color('#ff00ff')
        : new THREE.Color('#00ffff');
      
      // Track geometry if manager is available
      if (threeManager) {
        threeManager.track(`entanglement-geo-${index}`, tubeGeometry);
      }
      
      return {
        geometry: tubeGeometry,
        color,
        opacity: 0.3 * intensity + connection.strength * 0.2 * intensity,
        key: `${index}-${connection.start.x}-${connection.end.x}`
      };
    });
  }, [connections, primaryColor, intensity, width, threeManager]);
  
  return (
    <group ref={linesRef}>
      {connectionElements.map((element) => (
        <mesh key={element.key} geometry={element.geometry}>
          <meshBasicMaterial
            color={element.color}
            transparent
            opacity={element.opacity}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      ))}
    </group>
  );
};

/**
 * Simulation Boundary Zone
 * Represents the rendering limits and phase decoherence zones
 */
const SimulationBoundary: React.FC<{
  radius: number;
  coherence: number;
  primaryColor: string;
  opacity: number;
  distortion: number;
  showDistortion: boolean;
}> = ({ radius, coherence, primaryColor, opacity, distortion, showDistortion }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshPhysicalMaterial>();
  
  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.opacity = 0.1 + (1 - coherence) * 0.3;
    }
    
    if (meshRef.current && showDistortion) {
      meshRef.current.scale.setScalar(
        1 + Math.sin(state.clock.elapsedTime) * 0.1
      );
    }
  });
  
  return (
    <mesh ref={meshRef} scale={radius}>
      <sphereGeometry args={[1, 32, 32]} />
      <meshPhysicalMaterial
        ref={materialRef}
        color={primaryColor}
        transparent
        opacity={opacity}
        side={THREE.BackSide}
        roughness={1}
        metalness={0}
        emissive={primaryColor}
        emissiveIntensity={0.1 * distortion}
        wireframe={showDistortion}
      />
    </mesh>
  );
};

/**
 * Coherence Wavefront
 * Animated fields propagating after observations
 */
const CoherenceWavefront: React.FC<{
  origin: THREE.Vector3;
  time: number;
  primaryColor: string;
  intensity: number;
  speed: number;
  maxRadius: number;
}> = ({ origin, time, primaryColor, intensity, speed, maxRadius }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const radius = ((time * speed) % 2) * maxRadius;
  const opacity = Math.max(0, 1 - radius / maxRadius) * intensity;
  
  if (opacity <= 0) return null;
  
  return (
    <mesh ref={meshRef} position={origin}>
      <sphereGeometry args={[radius, 32, 32]} />
      <meshBasicMaterial
        color={primaryColor}
        transparent
        opacity={opacity * 0.3}
        side={THREE.BackSide}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
};

/**
 * Main OSH Universe 3D Scene Component
 */
export const OSHUniverse3DScene: React.FC<OSHUniverse3DSceneProps> = ({ 
  engine, 
  primaryColor,
  showLattice,
  showMemoryField,
  showEntanglements,
  showBoundary,
  showWavefronts,
  observerPOV,
  timeScale,
  recursiveZoom,
  visualizationSettings,
  onSelectNode,
  onEngineData
}) => {
  // Track universe animation state
  const universeTimeRef = useRef(0);
  const sceneGroupRef = useRef<THREE.Group>(null);
  // Subscribe to real-time engine data
  const { data: engineData, isConnected } = useEngineData(engine, {
    updateInterval: 50,
    includeHistory: true,
    historySize: 100
  });
  
  // Get backend API metrics
  const { metrics, states } = useEngineAPIContext();
  
  // Use backend metrics if available, otherwise use engine data
  const effectiveData = useMemo(() => {
    const defaultData = {
      rsp: 0,
      coherence: 0.5,
      entropy: 0.3,
      information: 0,
      strain: 0.2,
      observerFocus: 0.5,
      recursionDepth: 1,
      fieldCoherence: 0.5,
      observers: [],
      memoryFragments: [],
      fps: 60,
      universeTime: 0,
      iterationCount: 0,
      universeRunning: false
    };
    
    
    if (metrics && Object.keys(metrics).length > 0) {
      return {
        ...defaultData,
        rsp: metrics.rsp || defaultData.rsp,
        coherence: metrics.coherence || defaultData.coherence,
        entropy: metrics.entropy || defaultData.entropy,
        information: metrics.information || defaultData.information,
        strain: metrics.strain || defaultData.strain,
        observerFocus: metrics.observer_focus || defaultData.observerFocus,
        recursionDepth: metrics.recursion_depth || defaultData.recursionDepth,
        fieldCoherence: metrics.coherence || defaultData.fieldCoherence,
        observers: engineData?.observers || [],
        memoryFragments: metrics.memory_fragments || engineData?.memoryFragments || [],
        fps: metrics.fps || defaultData.fps,
        universeTime: metrics.universe_time || defaultData.universeTime,
        iterationCount: metrics.iteration_count || defaultData.iterationCount,
        universeRunning: metrics.universe_running || defaultData.universeRunning
      };
    }
    
    if (engineData) {
      return {
        ...defaultData,
        ...engineData
      };
    }
    
    return defaultData;
  }, [metrics, engineData]);
  
  // Update universe time reference when data changes
  useEffect(() => {
    if (effectiveData.universeRunning && effectiveData.universeTime > 0) {
      universeTimeRef.current = effectiveData.universeTime;
      
    }
  }, [effectiveData.universeTime, effectiveData.universeRunning]);
  
  // Animate scene based on universe time
  useFrame((state) => {
    // Always animate for visual feedback, but change behavior when universe is running
    const baseTime = state.clock.elapsedTime;
    const animTime = effectiveData.universeRunning ? 
      (universeTimeRef.current * timeScale + baseTime * 0.1) : // Universe time + slow drift
      baseTime; // Regular clock time
    
    // Rotate the entire scene slightly for visual feedback
    if (sceneGroupRef.current && observerPOV === 'global') {
      const rotationSpeed = effectiveData.universeRunning ? 0.02 : 0.005;
      sceneGroupRef.current.rotation.y = animTime * rotationSpeed;
    }
    
    // Update shader uniforms if memory field is visible
    const memoryFieldMesh = state.scene.getObjectByName('memory-field-mesh');
    if (memoryFieldMesh && memoryFieldMesh instanceof THREE.Mesh) {
      const material = memoryFieldMesh.material as THREE.ShaderMaterial;
      if (material.uniforms) {
        material.uniforms.time.value = animTime;
      }
    }
  });
  
  // Pass engine data to parent
  useEffect(() => {
    if (onEngineData && effectiveData) {
      onEngineData(effectiveData);
    }
  }, [effectiveData, onEngineData]);
  
  // Force scene update when universe state changes
  const sceneKey = useMemo(() => {
    return `universe-${effectiveData.universeRunning}-${Math.floor(effectiveData.iterationCount / 100)}`;
  }, [effectiveData.universeRunning, effectiveData.iterationCount]);
  
  // Generate lattice nodes from engine data
  const latticeNodes = useMemo(() => {
    if (!effectiveData) return [];
    
    const nodes: any[] = [];
    const gridSize = 5;
    const spacing = 3;
    
    // Create a recursive lattice structure
    for (let x = -gridSize; x <= gridSize; x++) {
      for (let y = -gridSize; y <= gridSize; y++) {
        for (let z = -gridSize; z <= gridSize; z++) {
          // Skip some nodes for visual clarity and performance
          const skipPattern = Math.abs(x) + Math.abs(y) + Math.abs(z);
          if (skipPattern % 2 === 0 && skipPattern < 10) {
            const position = new THREE.Vector3(
              x * spacing * (1 + effectiveData.entropy * 0.2),
              y * spacing * (1 + effectiveData.entropy * 0.2),
              z * spacing * (1 + effectiveData.entropy * 0.2)
            );
            
            // Calculate node properties based on position and engine state
            const distanceFromCenter = position.length();
            const coherence = Math.max(0, Math.min(1, effectiveData.coherence * Math.exp(-distanceFromCenter * 0.1)));
            const fragmentCount = Array.isArray(effectiveData.memoryFragments) ? effectiveData.memoryFragments.length : 0;
            const strain = Math.min(1, fragmentCount * 0.01 + distanceFromCenter * 0.05);
            const recursionDepth = Math.floor(Math.max(0, effectiveData.recursionDepth * (1 - distanceFromCenter / 20)));
            
            nodes.push({
              id: `node-${x}-${y}-${z}`,
              position,
              coherence,
              strain,
              recursionDepth: Math.max(0, recursionDepth),
              connections: [],
              isCollapsing: Math.random() < 0.01 // Random collapse events
            });
          }
        }
      }
    }
    
    // Create connections between nearby nodes
    nodes.forEach((node, i) => {
      nodes.forEach((otherNode, j) => {
        if (i !== j) {
          const distance = node.position.distanceTo(otherNode.position);
          if (distance < spacing * 1.5) {
            node.connections.push(otherNode.position);
          }
        }
      });
    });
    
    return nodes;
  }, [effectiveData]);
  
  // Generate entanglement connections
  const entanglements = useMemo(() => {
    if (!effectiveData || latticeNodes.length < 2) return [];
    
    const connections = [];
    const observerCount = Array.isArray(effectiveData.observers) ? effectiveData.observers.length : 0;
    const numConnections = Math.min(20, Math.max(5, observerCount * 3));
    
    for (let i = 0; i < numConnections; i++) {
      const node1 = latticeNodes[Math.floor(Math.random() * latticeNodes.length)];
      const node2 = latticeNodes[Math.floor(Math.random() * latticeNodes.length)];
      
      if (node1 !== node2) {
        connections.push({
          start: node1.position,
          end: node2.position,
          strength: Math.random() * 0.5 + 0.5,
          type: ['classical', 'quantum', 'consciousness'][Math.floor(Math.random() * 3)] as any
        });
      }
    }
    
    return connections;
  }, [effectiveData, latticeNodes]);
  
  // Generate wavefront origins from collapse events
  const wavefrontOrigins = useMemo(() => {
    if (!effectiveData) return [];
    
    return latticeNodes
      .filter(node => node.isCollapsing)
      .map(node => node.position);
  }, [effectiveData, latticeNodes]);
  
  // Camera positions for different POVs
  const cameraPositions = {
    global: [20, 20, 20] as [number, number, number],
    observer1: [10, 0, 0] as [number, number, number],
    observer2: [0, 10, 0] as [number, number, number]
  };
  
  // Return minimal scene if not ready
  if (!engine) {
    return (
      <>
        <PerspectiveCamera makeDefault position={[20, 20, 20]} fov={60} />
        <OrbitControls enableDamping dampingFactor={0.05} />
        <ambientLight intensity={0.15} />
        <pointLight position={[0, 0, 0]} intensity={0.8} color={primaryColor} />
        <mesh>
          <sphereGeometry args={[30, 32, 32]} />
          <meshBasicMaterial 
            color={primaryColor} 
            transparent 
            opacity={0.02} 
            side={THREE.BackSide}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      </>
    );
  }
  
  return (
    <>
      <PerspectiveCamera 
        makeDefault 
        position={cameraPositions[observerPOV]} 
        fov={60}
      />
      <OrbitControls 
        enableDamping 
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={50}
        autoRotate={observerPOV === 'global' && effectiveData.universeRunning}
        autoRotateSpeed={0.2 * timeScale}
      />
      
      {/* Environment and Lighting */}
      <ambientLight intensity={0.15} />
      <pointLight position={[0, 0, 0]} intensity={0.8} color={primaryColor} />
      <pointLight position={[10, 10, 10]} intensity={0.3} color="#ffffff" />
      <pointLight position={[-10, -10, -10]} intensity={0.3} color="#ffffff" />
      <fog attach="fog" args={['#0a0a0a', 20, 60]} />
      
      {/* Add bloom-like glow using emissive materials and fog */}
      <mesh>
        <sphereGeometry args={[30, 32, 32]} />
        <meshBasicMaterial 
          color={primaryColor} 
          transparent 
          opacity={0.02} 
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      
      {/* Main scene group for universe-wide animations */}
      <group ref={sceneGroupRef} key={sceneKey}>
        {/* Universe time indicator - visual feedback for updates */}
        {effectiveData.universeRunning && (
          <group position={[0, 15, 0]}>
            <mesh>
              <torusGeometry args={[2, 0.1, 16, 100]} />
              <meshBasicMaterial 
                color={primaryColor} 
                transparent 
                opacity={0.5 + Math.sin(effectiveData.universeTime * 0.5) * 0.3}
              />
            </mesh>
            <mesh>
              <torusGeometry args={[2.5, 0.05, 16, 100]} />
              <meshBasicMaterial 
                color="#ffffff" 
                transparent 
                opacity={0.2 + Math.sin(effectiveData.universeTime * 0.3) * 0.1}
              />
            </mesh>
            {/* Floating text indicator */}
            <Html center>
              <div style={{
                color: primaryColor,
                fontSize: '12px',
                fontWeight: 'bold',
                textAlign: 'center',
                background: 'rgba(0,0,0,0.7)',
                padding: '4px 8px',
                borderRadius: '4px',
                border: `1px solid ${primaryColor}`,
                pointerEvents: 'none'
              }}>
                UNIVERSE ACTIVE<br/>
                T: {effectiveData.universeTime.toFixed(2)}<br/>
                I: {effectiveData.iterationCount}
              </div>
            </Html>
          </group>
        )}
        
        {/* Recursive Information Lattice */}
        {showLattice && visualizationSettings.latticeVisible && (
          <group scale={[recursiveZoom, recursiveZoom, recursiveZoom]}>
          {latticeNodes.map(node => (
            <LatticeNode
              key={node.id}
              nodeId={node.id}
              position={node.position}
              coherence={node.coherence}
              strain={node.strain}
              recursionDepth={node.recursionDepth}
              connections={node.connections}
              primaryColor={visualizationSettings.latticeNodeColor}
              intensity={visualizationSettings.latticeNodeIntensity}
              glow={visualizationSettings.latticeNodeGlow}
              isCollapsing={node.isCollapsing}
              onSelect={onSelectNode}
            />
          ))}
        </group>
      )}
      
      {/* Memory Field Topology */}
      {showMemoryField && visualizationSettings.memoryFieldVisible && effectiveData && (
        <MemoryFieldTopology
          entropy={effectiveData.entropy}
          coherence={effectiveData.coherence}
          strain={Math.min(1, (Array.isArray(effectiveData.memoryFragments) ? effectiveData.memoryFragments.length : 0) / 100)}
          primaryColor={visualizationSettings.memoryFieldColor1}
          secondaryColor={visualizationSettings.memoryFieldColor2}
          opacity={visualizationSettings.memoryFieldOpacity}
          distortion={visualizationSettings.memoryFieldDistortion}
          time={Date.now() * 0.001 * timeScale}
        />
      )}
      
      {/* Entanglement Threads */}
      {showEntanglements && visualizationSettings.entanglementVisible && (
        <EntanglementThreads
          connections={entanglements}
          primaryColor={visualizationSettings.entanglementColor}
          intensity={visualizationSettings.entanglementIntensity}
          width={visualizationSettings.entanglementWidth}
          time={Date.now() * 0.001 * timeScale}
        />
      )}
      
      {/* Simulation Boundary */}
      {showBoundary && visualizationSettings.boundaryVisible && effectiveData && (
        <SimulationBoundary
          radius={25}
          coherence={effectiveData.coherence}
          primaryColor={visualizationSettings.boundaryColor}
          opacity={visualizationSettings.boundaryOpacity}
          distortion={visualizationSettings.boundaryDistortion}
          showDistortion={effectiveData.entropy > 0.7}
        />
      )}
      
      {/* Coherence Wavefronts - Limit to first 5 for performance */}
      {showWavefronts && visualizationSettings.wavefrontVisible && wavefrontOrigins.slice(0, 5).map((origin, index) => (
        <CoherenceWavefront
          key={index}
          origin={origin}
          time={Date.now() * 0.001 * timeScale}
          primaryColor={visualizationSettings.wavefrontColor}
          intensity={visualizationSettings.wavefrontIntensity}
          speed={visualizationSettings.wavefrontSpeed}
          maxRadius={10}
        />
      ))}
      
        {/* Post-processing Effects - Temporarily disabled for debugging
        {effectiveData.fps > 30 && (
          <EffectComposer>
          <Bloom 
            intensity={visualizationSettings.bloomIntensity} 
            luminanceThreshold={0.8} 
            luminanceSmoothing={0.9} 
          />
          <ChromaticAberration 
            offset={[visualizationSettings.chromaticAberration, visualizationSettings.chromaticAberration]}
            radialModulation={false}
            modulationOffset={0}
          />
          <DepthOfField 
            focusDistance={0} 
            focalLength={0.02} 
            bokehScale={2} 
            height={480}
          />
          <Vignette offset={0.1} darkness={visualizationSettings.vignetteIntensity} />
          </EffectComposer>
        )} */}
      </group>
    </>
  );
};

export default OSHUniverse3DScene;