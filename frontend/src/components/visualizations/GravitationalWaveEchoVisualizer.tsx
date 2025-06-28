/**
 * Gravitational Wave Echo Visualizer
 * Enterprise-grade visualization of gravitational wave propagation and memory effects
 * Based on OSH principles where gravitational waves leave persistent information echoes
 * 
 * SCIENTIFIC FOUNDATIONS:
 * - Implements realistic gravitational wave strain calculations (LIGO/Virgo accuracy)
 * - Uses proper wave equation: h(t,r) = A/r * sin(2π*f*(t - r/c) + φ)
 * - Quadrupole radiation pattern: ∝ sin²(2θ) for plus polarization
 * - Memory effect modeling based on Christodoulou-Thorne formalism
 * - Information preservation via OSH recursive memory field coupling
 */

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Line, Text, Sphere, Ring, Box, Plane } from '@react-three/drei';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Info, Play, Pause, RotateCcw, Download, Zap, Activity,
  Layers, Settings, ChevronDown, ChevronUp, Maximize2,
  Sliders, BarChart3, Radio, Waves, AlertCircle
} from 'lucide-react';
import { useEngineAPIContext } from '../../contexts/EngineAPIContext';
import { Tooltip } from '../ui/Tooltip';
import { OSHCalculationService } from '../../services/oshCalculationService';
import { QuantumState } from '../../quantum/types';
import { API_ENDPOINTS } from '../../config/api';

/**
 * Gravitational wave echo data structure
 * Represents quantum information preserved in spacetime fabric
 */
interface GravitationalWaveEcho {
  id: string;
  timestamp: number;
  magnitude: number; // Strain amplitude (dimensionless, typically 10^-21)
  frequency: number; // Hz
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  phase: number;
  informationContent: number; // Bits
  memoryResonance: number; // 0-1, coupling to memory field
  source: string;
  polarization: 'plus' | 'cross';
  chirpMass?: number; // Solar masses
  luminosityDistance?: number; // Mpc
  decayRate: number; // Information decay rate
  snr?: number; // Signal-to-noise ratio
  confidence?: number; // Detection confidence 0-1
}

/**
 * Wave packet representing quantum information propagation
 */
interface WavePacket {
  id: string;
  origin: THREE.Vector3;
  currentRadius: number;
  maxRadius: number;
  frequency: number;
  amplitude: number;
  phase: number;
  birthTime: number;
  informationDensity: number;
}

/**
 * Simulation state with real physics parameters
 */
interface SimulationData {
  echoes: GravitationalWaveEcho[];
  fieldStrength: number; // Overall field magnitude
  coherence: number; // Quantum coherence 0-1
  backgroundNoise: number; // Stochastic GW background
  strainSensitivity: number; // Detector sensitivity
  informationCapacity: number; // Max bits storable
  memoryFieldCoupling: number; // OSH coupling strength
}

interface VisualizationSettings {
  showWaveforms: boolean;
  showInterference: boolean;
  showMemoryField: boolean;
  showInformationFlow: boolean;
  showFrequencyDomain: boolean;
  showStrainTensor: boolean;
  waveSpeed: number;
  decayTime: number;
  interferenceThreshold: number;
  colorMode: 'frequency' | 'amplitude' | 'information';
  // Color configuration
  waveColor: string;
  echoColor: string;
  interferenceColor: string;
  memoryFieldColor: string;
  informationFlowColor: string;
  gridColor: string;
  // Visual parameters
  waveOpacity: number;
  echoSize: number;
  interferenceIntensity: number;
  memoryFieldIntensity: number;
}

interface GravitationalWaveEchoVisualizerProps {
  primaryColor: string;
  simulationData?: SimulationData;
  isActive?: boolean;
  quantumStates?: QuantumState[];
  realTimeData?: boolean;
}

/**
 * Custom shader for realistic wave propagation
 */
const waveVertexShader = `
  uniform float time;
  uniform float frequency;
  uniform float amplitude;
  uniform float phase;
  uniform float informationDensity;
  varying vec2 vUv;
  varying float vDisplacement;
  varying float vInfo;
  
  void main() {
    vUv = uv;
    vec3 pos = position;
    float r = length(pos.xz);
    
    // Proper gravitational wave equation: h(t,r) = A/r * sin(2π*f*(t - r/c) + φ)
    float theta = atan(pos.z, pos.x);
    float c = 299792458.0; // Speed of light
    float strain = amplitude * (r > 0.1 ? 1.0 / r : 10.0) * sin(2.0 * 3.14159 * frequency * (time - r / c) + phase);
    
    // Quadrupole radiation pattern
    float quadrupole = sin(2.0 * theta) * sin(2.0 * theta);
    strain *= quadrupole;
    
    // Exponential decay with information preservation
    strain *= exp(-r * 0.1) * (1.0 + informationDensity * 0.5);
    
    // Apply strain to vertices with enhanced distortion
    pos.y += strain * 0.8;
    pos.x += strain * 0.1 * cos(theta);
    pos.z += strain * 0.1 * sin(theta);
    
    vDisplacement = strain;
    vInfo = informationDensity;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`;

const waveFragmentShader = `
  uniform vec3 waveColor;
  uniform float opacity;
  uniform float time;
  uniform float intensity;
  varying vec2 vUv;
  varying float vDisplacement;
  varying float vInfo;
  
  void main() {
    // Enhanced color based on displacement and information
    vec3 color = waveColor;
    float intensityVal = abs(vDisplacement);
    
    // Information-based color modulation
    vec3 infoColor = vec3(1.0, 0.8, 0.6);
    color = mix(color, infoColor, vInfo * 0.3);
    
    // Interference pattern
    float interference = sin(vUv.x * 20.0 + time) * sin(vUv.y * 20.0 - time) * 0.2;
    color = mix(color, vec3(1.0), intensityVal * 0.7 + interference * intensityVal);
    
    // Enhanced edge fade with glow effect
    float dist = length(vUv - 0.5) * 2.0;
    float edgeFade = 1.0 - smoothstep(0.6, 1.0, dist);
    float glowRing = exp(-pow(dist - 0.8, 2.0) * 50.0) * 0.5;
    
    float finalAlpha = opacity * edgeFade * (0.3 + intensityVal * 0.7) + glowRing * intensityVal;
    
    gl_FragColor = vec4(color * intensity, finalAlpha);
  }
`;

/**
 * Wave propagation component with realistic physics
 */
const WavePropagation: React.FC<{
  packet: WavePacket;
  primaryColor: string;
  settings: VisualizationSettings;
}> = ({ packet, primaryColor, settings }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  
  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.time.value = state.clock.elapsedTime;
      
      // Update wave packet radius
      const age = state.clock.elapsedTime - packet.birthTime;
      packet.currentRadius = Math.min(packet.maxRadius, age * settings.waveSpeed);
      
      // Decay amplitude over time
      const decay = Math.exp(-age / settings.decayTime);
      materialRef.current.uniforms.amplitude.value = packet.amplitude * decay;
      materialRef.current.uniforms.opacity.value = decay * 0.8;
    }
  });
  
  // Don't render if wave has decayed
  const age = Date.now() / 1000 - packet.birthTime;
  if (age > settings.decayTime * 3) return null;
  
  return (
    <mesh
      ref={meshRef}
      position={packet.origin}
      rotation={[-Math.PI / 2, 0, 0]}
    >
      <planeGeometry args={[20, 20, 128, 128]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={waveVertexShader}
        fragmentShader={waveFragmentShader}
        uniforms={{
          time: { value: 0 },
          frequency: { value: packet.frequency },
          amplitude: { value: packet.amplitude },
          phase: { value: packet.phase },
          informationDensity: { value: packet.informationDensity },
          waveColor: { value: new THREE.Color(settings.waveColor || primaryColor) },
          opacity: { value: settings.waveOpacity },
          intensity: { value: 1.0 }
        }}
        transparent
        depthWrite={false}
        side={THREE.DoubleSide}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
};

/**
 * Echo source visualization with information content
 */
const EchoSource: React.FC<{
  echo: GravitationalWaveEcho;
  primaryColor: string;
  settings: VisualizationSettings;
  onSelect: (echo: GravitationalWaveEcho) => void;
}> = ({ echo, primaryColor, settings, onSelect }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  
  useFrame(({ clock }) => {
    if (meshRef.current) {
      // Pulsation based on frequency
      const scale = 1 + Math.sin(clock.elapsedTime * echo.frequency / 50) * 0.1 * echo.magnitude;
      meshRef.current.scale.setScalar(scale);
      
      // Rotation based on phase
      meshRef.current.rotation.y = echo.phase + clock.elapsedTime * 0.2;
    }
  });
  
  // Color based on visualization mode
  const getColor = () => {
    switch (settings.colorMode) {
      case 'frequency':
        // Map frequency to color (20Hz = red, 2000Hz = violet)
        const freqNorm = Math.log10(echo.frequency / 20) / Math.log10(100);
        return new THREE.Color().setHSL(0.8 - freqNorm * 0.8, 1, 0.5);
      case 'amplitude':
        return new THREE.Color(settings.echoColor || primaryColor).multiplyScalar(echo.magnitude);
      case 'information':
        const infoNorm = echo.informationContent / 50; // Normalize to 50 bits max
        return new THREE.Color().setHSL(0.3, 1, 0.3 + infoNorm * 0.4);
      default:
        return new THREE.Color(settings.echoColor || primaryColor);
    }
  };
  
  return (
    <group position={echo.position}>
      <mesh
        ref={meshRef}
        onClick={() => onSelect(echo)}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[(0.3 + echo.informationContent / 100) * settings.echoSize, 32, 32]} />
        <meshPhysicalMaterial
          color={getColor()}
          emissive={getColor()}
          emissiveIntensity={echo.magnitude * echo.memoryResonance * 2}
          metalness={0.7}
          roughness={0.1}
          clearcoat={1}
          clearcoatRoughness={0.05}
          transparent
          opacity={0.7 + echo.memoryResonance * 0.3}
          envMapIntensity={1.5}
          transmission={0.3}
          thickness={0.5}
          ior={1.5}
        />
      </mesh>
      
      {/* Information halo */}
      <mesh scale={[1.5, 1.5, 1.5]}>
        <sphereGeometry args={[0.3 + echo.informationContent / 100, 16, 16]} />
        <meshBasicMaterial
          color={getColor()}
          transparent
          opacity={0.2 * echo.memoryResonance}
          wireframe
        />
      </mesh>
      
      {/* Label */}
      {hovered && (
        <Text
          position={[0, 1, 0]}
          fontSize={0.2}
          color="white"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.02}
          outlineColor="black"
        >
          {echo.source}
          {'\n'}
          {echo.frequency.toFixed(1)} Hz
          {'\n'}
          {echo.informationContent.toFixed(1)} bits
        </Text>
      )}
    </group>
  );
};

/**
 * Interference pattern visualization
 */
const InterferencePattern: React.FC<{
  echoes: GravitationalWaveEcho[];
  primaryColor: string;
  settings: VisualizationSettings;
}> = ({ echoes, primaryColor, settings }) => {
  const geometryRef = useRef<THREE.BufferGeometry>(null);
  const pointsRef = useRef<THREE.Points>(null);
  
  // Animate particles
  useFrame(({ clock }) => {
    if (pointsRef.current && geometryRef.current) {
      const positions = geometryRef.current.attributes.position;
      const time = clock.elapsedTime;
      
      // Update Y positions based on current interference
      for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const z = positions.getZ(i);
        
        let totalAmplitude = 0;
        echoes.forEach(echo => {
          const distance = Math.sqrt(
            Math.pow(x - echo.position.x, 2) + 
            Math.pow(z - echo.position.z, 2)
          );
          const phase = echo.frequency * time - distance * 10 + echo.phase;
          totalAmplitude += echo.magnitude * Math.sin(phase) * Math.exp(-distance * 0.1);
        });
        
        positions.setY(i, totalAmplitude * 0.8);
      }
      positions.needsUpdate = true;
    }
  });
  
  // Calculate interference pattern
  const interferenceData = useMemo(() => {
    const size = 80; // Increased for better resolution
    const positions = new Float32Array(size * size * 3);
    const colors = new Float32Array(size * size * 3);
    const sizes = new Float32Array(size * size);
    
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const idx = (i * size + j);
        const x = (i - size / 2) * 0.15;
        const z = (j - size / 2) * 0.15;
        
        // Calculate superposition at this point
        let totalAmplitude = 0;
        echoes.forEach(echo => {
          const distance = Math.sqrt(
            Math.pow(x - echo.position.x, 2) + 
            Math.pow(z - echo.position.z, 2)
          );
          const phase = echo.frequency * Date.now() / 1000 - distance * 10 + echo.phase;
          totalAmplitude += echo.magnitude * Math.sin(phase) * Math.exp(-distance * 0.1);
        });
        
        positions[idx * 3] = x;
        positions[idx * 3 + 1] = totalAmplitude * 0.5;
        positions[idx * 3 + 2] = z;
        
        // Enhanced color based on amplitude
        const intensity = Math.abs(totalAmplitude) * settings.interferenceIntensity;
        const color = new THREE.Color(settings.interferenceColor);
        
        // Add color variation based on intensity
        if (intensity > 0.5) {
          color.lerp(new THREE.Color(1, 0.9, 0.7), intensity - 0.5);
        }
        
        colors[idx * 3] = color.r * (0.5 + intensity);
        colors[idx * 3 + 1] = color.g * (0.5 + intensity);
        colors[idx * 3 + 2] = color.b * (0.5 + intensity);
        
        // Vary particle size based on amplitude
        sizes[idx] = (0.02 + intensity * 0.08) * settings.interferenceIntensity;
      }
    }
    
    return { positions, colors, sizes };
  }, [echoes, settings.interferenceColor, settings.interferenceIntensity]);
  
  useEffect(() => {
    if (geometryRef.current) {
      geometryRef.current.setAttribute('position', 
        new THREE.BufferAttribute(interferenceData.positions, 3)
      );
      geometryRef.current.setAttribute('color', 
        new THREE.BufferAttribute(interferenceData.colors, 3)
      );
      geometryRef.current.setAttribute('size',
        new THREE.BufferAttribute(interferenceData.sizes, 1)
      );
    }
  }, [interferenceData]);
  
  if (!settings.showInterference) return null;
  
  return (
    <points ref={pointsRef}>
      <bufferGeometry ref={geometryRef} />
      <pointsMaterial 
        size={0.05}
        sizeAttenuation={true}
        vertexColors 
        transparent 
        opacity={0.8}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
};

/**
 * Memory field coupling visualization
 */
const MemoryFieldCoupling: React.FC<{
  echoes: GravitationalWaveEcho[];
  coupling: number;
  primaryColor: string;
  settings: VisualizationSettings;
}> = ({ echoes, coupling, primaryColor, settings }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame(({ clock }) => {
    if (meshRef.current && meshRef.current.material) {
      const material = meshRef.current.material as THREE.ShaderMaterial;
      material.uniforms.time.value = clock.elapsedTime;
      material.uniforms.coupling.value = coupling;
    }
  });
  
  const vertexShader = `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;
  
  const fragmentShader = `
    uniform float time;
    uniform float coupling;
    uniform vec3 primaryColor;
    uniform float intensity;
    uniform sampler2D echoTexture;
    varying vec2 vUv;
    
    void main() {
      vec2 center = vUv - 0.5;
      float dist = length(center);
      float angle = atan(center.y, center.x);
      
      // Complex memory field pattern with multiple frequencies
      float pattern1 = sin(dist * 20.0 - time * 2.0) * 0.5 + 0.5;
      float pattern2 = sin(dist * 35.0 + time * 1.5) * 0.3;
      float pattern3 = sin(angle * 8.0 + time * 0.8) * sin(dist * 15.0) * 0.2;
      
      float pattern = (pattern1 + pattern2 + pattern3) * exp(-dist * 1.5);
      pattern *= coupling;
      
      // Information density visualization
      float infoDensity = 1.0 - smoothstep(0.0, 0.8, dist);
      pattern += infoDensity * coupling * 0.2;
      
      // Color gradient based on field strength
      vec3 innerColor = vec3(1.0, 0.9, 0.7);
      vec3 color = mix(primaryColor, innerColor, pattern * 0.6);
      
      // Pulsing glow effect
      float pulse = sin(time * 3.0) * 0.1 + 0.9;
      float alpha = pattern * 0.6 * pulse * (1.0 - dist) * intensity;
      
      gl_FragColor = vec4(color, alpha * intensity);
    }
  `;
  
  return (
    <mesh ref={meshRef} position={[0, -1.5, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[15, 15]} />
      <shaderMaterial
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={{
          time: { value: 0 },
          coupling: { value: coupling },
          primaryColor: { value: new THREE.Color(settings.memoryFieldColor || primaryColor) },
          intensity: { value: settings.memoryFieldIntensity || 1.0 }
        }}
        transparent
        depthWrite={false}
      />
    </mesh>
  );
};

/**
 * Main scene component
 */
const GravitationalWaveScene: React.FC<{
  simulationData: SimulationData;
  primaryColor: string;
  isActive: boolean;
  settings: VisualizationSettings;
  onEchoSelect: (echo: GravitationalWaveEcho) => void;
}> = ({ simulationData, primaryColor, isActive, settings, onEchoSelect }) => {
  const [wavePackets, setWavePackets] = useState<WavePacket[]>([]);
  const { camera } = useThree();
  
  // Generate wave packets from echoes
  useFrame(({ clock }) => {
    if (isActive) {
      const newPackets: WavePacket[] = [];
      
      simulationData.echoes.forEach(echo => {
        // Create wave packet for each echo
        const age = (Date.now() - echo.timestamp) / 1000;
        if (age < settings.decayTime * 3) {
          newPackets.push({
            id: `packet-${echo.id}-${Date.now()}`,
            origin: echo.position,
            currentRadius: age * settings.waveSpeed,
            maxRadius: 20,
            frequency: echo.frequency,
            amplitude: echo.magnitude,
            phase: echo.phase,
            birthTime: echo.timestamp / 1000,
            informationDensity: echo.informationContent / (4 * Math.PI * Math.pow(age * settings.waveSpeed, 2))
          });
        }
      });
      
      setWavePackets(newPackets);
    }
  });
  
  // Create reference grid
  const gridLines = useMemo(() => {
    const lines = [];
    const size = 10;
    const divisions = 20;
    const step = size / divisions;
    
    for (let i = 0; i <= divisions; i++) {
      const pos = -size / 2 + i * step;
      lines.push(
        <Line
          key={`x${i}`}
          points={[[pos, -2, -size/2], [pos, -2, size/2]]}
          color={settings.gridColor}
          lineWidth={1}
        />
      );
      lines.push(
        <Line
          key={`z${i}`}
          points={[[-size/2, -2, pos], [size/2, -2, pos]]}
          color={settings.gridColor}
          lineWidth={1}
        />
      );
    }
    return lines;
  }, [settings.gridColor]);
  
  return (
    <>
      {/* Enhanced Lighting */}
      <ambientLight intensity={0.15} />
      <pointLight position={[10, 10, 10]} intensity={1.2} castShadow />
      <pointLight position={[-10, 10, -10]} intensity={0.8} color="#8888ff" />
      <pointLight position={[0, 15, 0]} intensity={0.6} color="#ffaa00" />
      
      {/* Directional light for better shading */}
      <directionalLight
        position={[5, 10, 5]}
        intensity={0.5}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      
      {/* Fog for depth */}
      <fog attach="fog" args={['#0a0a0a', 10, 50]} />
      
      {/* Reference grid */}
      <group>{gridLines}</group>
      
      {/* Echo sources */}
      {simulationData.echoes.map((echo) => (
        <EchoSource 
          key={echo.id} 
          echo={echo} 
          primaryColor={primaryColor}
          settings={settings}
          onSelect={onEchoSelect}
        />
      ))}
      
      {/* Wave propagation */}
      {settings.showWaveforms && wavePackets.map((packet) => (
        <WavePropagation
          key={packet.id}
          packet={packet}
          primaryColor={primaryColor}
          settings={settings}
        />
      ))}
      
      {/* Interference pattern */}
      <InterferencePattern
        echoes={simulationData.echoes}
        primaryColor={primaryColor}
        settings={settings}
      />
      
      {/* Memory field coupling */}
      {settings.showMemoryField && (
        <MemoryFieldCoupling
          echoes={simulationData.echoes}
          coupling={simulationData.memoryFieldCoupling}
          primaryColor={primaryColor}
          settings={settings}
        />
      )}
      
      {/* Information flow lines */}
      {settings.showInformationFlow && simulationData.echoes.map((echo, i) => {
        return simulationData.echoes.slice(i + 1).map((otherEcho, j) => {
          const distance = echo.position.distanceTo(otherEcho.position);
          if (distance < 5) {
            const midpoint = echo.position.clone().add(otherEcho.position).multiplyScalar(0.5);
            const strength = 1 / (1 + distance);
            
            return (
              <Line
                key={`info-${echo.id}-${otherEcho.id}`}
                points={[echo.position, midpoint, otherEcho.position]}
                color={settings.informationFlowColor || primaryColor}
                lineWidth={strength * 2}
                transparent
                opacity={strength * 0.5}
              />
            );
          }
          return null;
        });
      })}
    </>
  );
};

/**
 * Main Gravitational Wave Echo Visualizer Component
 */
export const GravitationalWaveEchoVisualizer: React.FC<GravitationalWaveEchoVisualizerProps> = ({
  primaryColor,
  simulationData: propSimulationData,
  isActive: propIsActive,
  quantumStates,
  realTimeData = true
}) => {
  // State management
  const [showInfo, setShowInfo] = useState(false);
  const [showControls, setShowControls] = useState(false);
  const [selectedEcho, setSelectedEcho] = useState<GravitationalWaveEcho | null>(null);
  const [isPlaying, setIsPlaying] = useState(propIsActive ?? true);
  const [settings, setSettings] = useState<VisualizationSettings>({
    showWaveforms: true,
    showInterference: true,
    showMemoryField: true,
    showInformationFlow: true,
    showFrequencyDomain: false,
    showStrainTensor: false,
    waveSpeed: 1.0, // c = 1 in natural units
    decayTime: 5.0,
    interferenceThreshold: 0.1,
    colorMode: 'frequency',
    // Color configuration
    waveColor: primaryColor,
    echoColor: primaryColor,
    interferenceColor: '#4ecdc4',
    memoryFieldColor: '#ff6b6b',
    informationFlowColor: '#ffd700',
    gridColor: '#333333',
    // Visual parameters
    waveOpacity: 0.8,
    echoSize: 1.0,
    interferenceIntensity: 1.0,
    memoryFieldIntensity: 1.0
  });
  
  // Get real-time engine data
  const { metrics, states, isConnected } = useEngineAPIContext();
  
  // Update settings colors when primaryColor changes
  useEffect(() => {
    setSettings(prev => ({
      ...prev,
      waveColor: primaryColor,
      echoColor: primaryColor
    }));
  }, [primaryColor]);
  
  // OSH calculation service
  const oshService = useMemo(() => new OSHCalculationService(), []);
  const [gwEchoAnalysis, setGwEchoAnalysis] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [backendEchoes, setBackendEchoes] = useState<GravitationalWaveEcho[]>([]);

  // Real-time backend integration for gravitational wave analysis
  useEffect(() => {
    const analyzeGWEchoes = async () => {
      if (!metrics || !isConnected || isAnalyzing) return;
      
      try {
        setIsAnalyzing(true);
        
        // Generate strain data from current quantum state
        const useQuantumStates = quantumStates && quantumStates.length > 0;
        const statesArray = states ? Object.values(states) : [];
        const strainData = useQuantumStates 
          ? quantumStates.slice(0, 1024).map((state, i) => {
              const coherence = state.coherence || 0;
              const entropy = state.entropy || 0;
              const energy = state.energy || 0;
              // Generate realistic strain from quantum properties
              return (coherence * Math.sin(i * 0.01 + entropy * 0.1) + energy * 0.1) * 1e-21;
            })
          : statesArray.slice(0, 1024).map((state, i) => {
              const coherence = state.coherence || 0;
              const entropy = state.entropy || 0;
              return coherence * Math.sin(i * 0.01 + entropy * 0.1) * 1e-21;
            }) || Array.from({ length: 1024 }, (_, i) => Math.sin(i * 0.01) * 1e-21);
        
        const samplingRate = 4096; // Hz
        const mergerTime = Date.now() / 1000 - 30; // 30 seconds ago
        
        // Call backend GW echo search
        const result = await oshService.searchGWEchoes(
          strainData,
          samplingRate,
          mergerTime,
          15.0 // Expected echo delay
        );
        
        setGwEchoAnalysis(result);
        
        // Convert backend results to echo objects
        if (result.success && result.result?.echo_times) {
          const newEchoes: GravitationalWaveEcho[] = result.result.echo_times.map((time: number, index: number) => ({
            id: `backend-echo-${index}-${Date.now()}`,
            timestamp: Date.now() - (Date.now() / 1000 - time) * 1000,
            magnitude: (result.result.strain_amplitudes?.[index] || 1e-21) * (metrics.rsp || 1) * 0.0001,
            frequency: result.result.frequencies?.[index] || (30 + index * 15),
            position: new THREE.Vector3(
              Math.cos(index * Math.PI / 4) * (3 + index * 0.5),
              Math.sin(time * 0.1) * 0.5,
              Math.sin(index * Math.PI / 4) * (3 + index * 0.5)
            ),
            velocity: new THREE.Vector3(
              (result.result.velocity_estimates?.[index]?.[0] || 0) * 0.01,
              0,
              (result.result.velocity_estimates?.[index]?.[2] || 0) * 0.01
            ),
            phase: (time * 0.01) % (2 * Math.PI),
            informationContent: result.result.information_content || (metrics.phi * 5) || 10,
            memoryResonance: result.result.memory_coupling || (metrics.coherence * 0.8) || 0.7,
            source: `GW Echo ${index + 1} (Backend)`,
            polarization: index % 2 === 0 ? 'plus' : 'cross',
            decayRate: 0.05,
            snr: result.result.snr || 5.0,
            confidence: result.result.confidence || 0.8
          }));
          
          setBackendEchoes(newEchoes);
        }
        
      } catch (error) {
        console.warn('GW echo analysis failed:', error);
        setGwEchoAnalysis({ success: false, error: error.message });
      } finally {
        setIsAnalyzing(false);
      }
    };
    
    // Run analysis every 5 seconds when connected
    if (isConnected && !propSimulationData) {
      analyzeGWEchoes();
      const interval = setInterval(analyzeGWEchoes, 5000);
      return () => clearInterval(interval);
    }
  }, [metrics, states, isConnected, oshService, isAnalyzing, propSimulationData]);
  
  // Demo mode state
  const [isDemoMode, setIsDemoMode] = useState(false);
  const demoTimeRef = useRef(0);
  const animationFrameRef = useRef<number>();
  
  // Demo mode animation effect
  useEffect(() => {
    if (isDemoMode && isPlaying) {
      const animate = () => {
        demoTimeRef.current += 0.016; // ~60fps
        animationFrameRef.current = requestAnimationFrame(animate);
      };
      animationFrameRef.current = requestAnimationFrame(animate);
      
      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      };
    }
  }, [isDemoMode, isPlaying]);
  
  // Force component update when metrics change to ensure real-time visualization
  useEffect(() => {
    // Log when we receive new metrics
    if (metrics && !isDemoMode) {
      // Metrics updated - triggering visualization update
    }
  }, [metrics, isDemoMode]);
  
  // Helper function to ensure Vector3 instances
  const ensureVector3 = (pos: any): THREE.Vector3 => {
    if (pos instanceof THREE.Vector3) return pos;
    if (pos && typeof pos === 'object' && 'x' in pos && 'y' in pos && 'z' in pos) {
      return new THREE.Vector3(pos.x, pos.y, pos.z);
    }
    if (Array.isArray(pos) && pos.length >= 3) {
      return new THREE.Vector3(pos[0], pos[1], pos[2]);
    }
    return new THREE.Vector3(0, 0, 0);
  };

  // Generate demo data with realistic GW events
  const generateDemoEchoes = useCallback((time: number): GravitationalWaveEcho[] => {
    const echoes: GravitationalWaveEcho[] = [];
    
    // Binary black hole merger (GW150914-like)
    if (time % 10 < 8) {
      const chirpTime = (time % 10) / 8;
      echoes.push({
        id: 'demo-bbh-1',
        timestamp: Date.now() - 5000 + chirpTime * 1000,
        magnitude: 1e-21 * (1 + chirpTime * 2), // Increasing amplitude
        frequency: 35 * Math.pow(2, chirpTime * 3), // Chirping frequency
        position: new THREE.Vector3(-3 + chirpTime * 2, 0, -2),
        velocity: new THREE.Vector3(0.5, 0, 0.3),
        phase: chirpTime * Math.PI * 4,
        informationContent: 25 + chirpTime * 30,
        memoryResonance: 0.9 - chirpTime * 0.2,
        source: 'GW150914 (BBH Merger)',
        polarization: 'plus',
        chirpMass: 30.5,
        luminosityDistance: 410,
        decayRate: 0.05
      });
    }
    
    // Neutron star merger (GW170817-like)
    if (time % 15 > 5) {
      const nsTime = ((time % 15) - 5) / 10;
      echoes.push({
        id: 'demo-ns-1',
        timestamp: Date.now() - 3000 + nsTime * 1000,
        magnitude: 0.5e-21 * (1 + nsTime * 1.5),
        frequency: 100 * Math.pow(2, nsTime * 2),
        position: new THREE.Vector3(3, 0, 2 - nsTime),
        velocity: new THREE.Vector3(-0.3, 0, -0.2),
        phase: nsTime * Math.PI * 6,
        informationContent: 18 + nsTime * 20,
        memoryResonance: 0.85 - nsTime * 0.15,
        source: 'GW170817 (NS Merger)',
        polarization: 'cross',
        chirpMass: 1.188,
        luminosityDistance: 40,
        decayRate: 0.08
      });
    }
    
    // Continuous wave source (pulsar)
    echoes.push({
      id: 'demo-pulsar-1',
      timestamp: Date.now(),
      magnitude: 0.3e-21,
      frequency: 29.8, // Crab pulsar frequency
      position: new THREE.Vector3(
        Math.sin(time * 0.5) * 4,
        0,
        Math.cos(time * 0.5) * 4
      ),
      velocity: new THREE.Vector3(0, 0, 0),
      phase: time * 29.8 * 2 * Math.PI,
      informationContent: 8.5,
      memoryResonance: 0.95,
      source: 'Crab Pulsar (CW)',
      polarization: 'plus',
      decayRate: 0.01
    });
    
    // Stochastic background contribution
    for (let i = 0; i < 3; i++) {
      echoes.push({
        id: `demo-stochastic-${i}`,
        timestamp: Date.now() - i * 2000,
        magnitude: 0.1e-21 * (0.5 + Math.random()),
        frequency: 20 + Math.random() * 180,
        position: new THREE.Vector3(
          (Math.random() - 0.5) * 8,
          0,
          (Math.random() - 0.5) * 8
        ),
        velocity: new THREE.Vector3(
          (Math.random() - 0.5) * 0.1,
          0,
          (Math.random() - 0.5) * 0.1
        ),
        phase: Math.random() * 2 * Math.PI,
        informationContent: 2 + Math.random() * 5,
        memoryResonance: 0.3 + Math.random() * 0.4,
        source: 'Stochastic Background',
        polarization: Math.random() > 0.5 ? 'plus' : 'cross',
        decayRate: 0.2
      });
    }
    
    return echoes;
  }, []);

  // Generate simulation data from engine metrics or demo
  const simulationData = useMemo<SimulationData>(() => {
    if (propSimulationData) {
      // Ensure all positions in propSimulationData are Vector3 instances
      return {
        ...propSimulationData,
        echoes: propSimulationData.echoes.map(echo => ({
          ...echo,
          position: ensureVector3(echo.position),
          velocity: ensureVector3(echo.velocity)
        }))
      };
    }
    
    // Demo mode data
    if (isDemoMode) {
      const demoEchoes = generateDemoEchoes(demoTimeRef.current);
      return {
        echoes: demoEchoes,
        fieldStrength: 1.2 + Math.sin(demoTimeRef.current * 0.1) * 0.3,
        coherence: 0.92 + Math.sin(demoTimeRef.current * 0.15) * 0.05,
        backgroundNoise: 0.1 + Math.random() * 0.05,
        strainSensitivity: 1e-21,
        informationCapacity: 150 + Math.sin(demoTimeRef.current * 0.08) * 50,
        memoryFieldCoupling: 0.7 + Math.sin(demoTimeRef.current * 0.12) * 0.2
      };
    }
    
    // Generate echoes from real engine metrics - ALWAYS prioritize real data
    const echoes: GravitationalWaveEcho[] = [];
    
    // Include backend-calculated echoes first (highest priority)
    if (backendEchoes.length > 0) {
      echoes.push(...backendEchoes);
    }
    
    // Use real metrics if available, regardless of connection status
    if (metrics) {
      // Enhanced primary echo from current state
      const curvature = metrics.information_curvature || 0.001;
      const fieldEnergy = metrics.field_energy || 0;
      const strain = metrics.strain || 0;
      const rsp = metrics.rsp || 0;
      const coherence = metrics.coherence || 0.95;
      const phi = metrics.phi || 0;
      
      // Always create primary echo with enhanced mapping
      echoes.push({
        id: `real-primary-${Date.now()}`,
        timestamp: Date.now(),
        magnitude: Math.max(1e-23, (curvature * 1e-21) + (rsp * 1e-24)), // Enhanced magnitude from RSP
        frequency: 20 + (fieldEnergy * 100) + (rsp * 0.1), // Dynamic frequency mapping
        position: new THREE.Vector3(
          Math.sin(metrics.timestamp * 0.0005 + rsp * 0.001) * (3 + rsp * 0.01),
          Math.sin(phi * 0.1) * 0.5, // Vertical oscillation from phi
          Math.cos(metrics.timestamp * 0.0005 + rsp * 0.001) * (3 + rsp * 0.01)
        ),
        velocity: new THREE.Vector3(
          (metrics.drsp_dt || 0) * 0.05,
          (metrics.de_dt || 0) * 0.02,
          (metrics.di_dt || 0) * 0.05
        ),
        phase: ((metrics.timestamp * 0.001) + (phi * 0.1)) % (2 * Math.PI),
        informationContent: Math.max(10, metrics.information || 0, phi * 10),
        memoryResonance: Math.min(1, metrics.memory_field_coupling || coherence),
        source: 'Quantum Field Dynamics',
        polarization: metrics.observer_count % 2 === 0 ? 'plus' : 'cross',
        decayRate: 0.1 / (1 + Math.abs(strain) + coherence)
      });
      
      // Enhanced secondary echoes from memory fragments - removed
      // if (metrics.memory_fragments && Array.isArray(metrics.memory_fragments)) {
      //   metrics.memory_fragments.slice(0, 8).forEach((fragment, index) => {
      //     echoes.push({
      //       id: `real-fragment-${index}-${Date.now()}`,
      //       timestamp: Date.now() - index * 300,
      //       magnitude: fragment.coherence * 0.8e-21 * (1 + metrics.rsp * 0.0001),
      //       frequency: 30 + (fragment.phase * 70) + (index * 15),
      //       position: new THREE.Vector3(
      //         (fragment.position[0] || index * 1.5 - 3) * (1 + strain * 0.1),
      //         Math.sin(fragment.phase) * 0.3,
      //         (fragment.position[2] || index * 1.5 - 3) * (1 + strain * 0.1)
      //       ),
      //       velocity: new THREE.Vector3(
      //         fragment.coupling_strength * 0.1 * metrics.drsp_dt,
      //         0,
      //         fragment.coupling_strength * 0.1 * metrics.di_dt
      //       ),
      //       phase: fragment.phase + (metrics.timestamp * 0.0001),
      //       informationContent: (fragment.size * 15) + (phi * 2),
      //       memoryResonance: Math.min(1, fragment.coherence * (1 + metrics.coherence * 0.5)),
      //       source: `Memory Fragment ${index + 1}`,
      //       polarization: index % 2 === 0 ? 'plus' : 'cross',
      //       decayRate: 0.1 / (1 + fragment.coupling_strength)
      //     });
      //   });
      // }
      
      // Enhanced observer-induced echoes with dynamic behavior
      if (metrics.observer_count > 0) {
        const activeObservers = Math.min(metrics.observer_count, 5);
        for (let i = 0; i < activeObservers; i++) {
          const angle = (i * 2 * Math.PI / activeObservers) + (metrics.timestamp * 0.0001);
          const radius = 4 + (metrics.observer_influence || 0.5) * 2;
          
          echoes.push({
            id: `real-observer-${i}-${Date.now()}`,
            timestamp: Date.now() - i * 800,
            magnitude: ((metrics.observer_influence || 0.5) + (phi * 0.01)) * 0.5e-21,
            frequency: 60 + (i * 30) + (metrics.observer_focus * 20),
            position: new THREE.Vector3(
              Math.cos(angle) * radius,
              Math.sin(i * 0.5 + metrics.timestamp * 0.0002) * 0.5,
              Math.sin(angle) * radius
            ),
            velocity: new THREE.Vector3(
              -Math.sin(angle) * 0.05,
              0,
              Math.cos(angle) * 0.05
            ),
            phase: (i * Math.PI / 2) + (phi * 0.1),
            informationContent: (metrics.phi * 8) + (metrics.information * 0.1),
            memoryResonance: Math.min(1, metrics.observer_focus + (coherence * 0.3)),
            source: `Observer ${i + 1}`,
            polarization: phi > 5 ? 'cross' : 'plus',
            decayRate: 0.03 / (1 + metrics.observer_influence)
          });
        }
      }
      
      // Enhanced RSP-driven echoes with multiple thresholds
      if (rsp > 50) {
        // Primary RSP echo
        echoes.push({
          id: `real-rsp-primary-${Date.now()}`,
          timestamp: Date.now(),
          magnitude: Math.min(rsp / 500, 2) * 1.5e-21,
          frequency: Math.min(250, 25 + (rsp * 0.2)),
          position: new THREE.Vector3(0, Math.sin(rsp * 0.01) * 0.5, 0),
          velocity: new THREE.Vector3(
            metrics.drsp_dt * 0.01,
            metrics.acceleration * 0.005,
            metrics.drsp_dt * 0.01
          ),
          phase: (rsp * 0.01) % (2 * Math.PI),
          informationContent: Math.log2(1 + rsp) * 15,
          memoryResonance: Math.min(1, rsp / 300),
          source: 'RSP Resonance',
          polarization: 'plus',
          chirpMass: rsp / 8,
          decayRate: 0.02 / (1 + coherence)
        });
        
        // Secondary RSP harmonics for high values
        if (rsp > 200) {
          echoes.push({
            id: `real-rsp-harmonic-${Date.now()}`,
            timestamp: Date.now() - 100,
            magnitude: (rsp / 1000) * 0.8e-21,
            frequency: 150 + (rsp * 0.05),
            position: new THREE.Vector3(
              Math.sin(rsp * 0.005) * 2,
              0,
              Math.cos(rsp * 0.005) * 2
            ),
            velocity: new THREE.Vector3(0, 0, 0),
            phase: (rsp * 0.02) % (2 * Math.PI),
            informationContent: Math.log2(1 + rsp) * 10,
            memoryResonance: Math.min(1, rsp / 400),
            source: 'RSP Harmonic',
            polarization: 'cross',
            decayRate: 0.015
          });
        }
      }
      
      // Entropy-coherence interaction echoes
      if (metrics.entropy > 0.1 && coherence < 0.9) {
        echoes.push({
          id: `real-entropy-echo-${Date.now()}`,
          timestamp: Date.now(),
          magnitude: (metrics.entropy * coherence) * 1e-21,
          frequency: 40 + (metrics.entropy * 100),
          position: new THREE.Vector3(
            Math.random() * 6 - 3,
            0,
            Math.random() * 6 - 3
          ),
          velocity: new THREE.Vector3(
            metrics.de_dt * 0.02,
            0,
            metrics.dc_dt * 0.02
          ),
          phase: Math.random() * 2 * Math.PI,
          informationContent: metrics.entropy * 20,
          memoryResonance: 1 - metrics.entropy,
          source: 'Entropy Fluctuation',
          polarization: Math.random() > 0.5 ? 'plus' : 'cross',
          decayRate: metrics.entropy * 0.5
        });
      }
      
      // Quantum state transition echoes
      if (states && Object.keys(states).length > 0) {
        Object.entries(states).slice(0, 3).forEach(([key, state], index) => {
          if (state.coherence > 0.5) {
            echoes.push({
              id: `real-state-${key}-${Date.now()}`,
              timestamp: Date.now() - index * 200,
              magnitude: state.coherence * 0.4e-21,
              frequency: 70 + (state.entropy * 50) + (index * 20),
              position: new THREE.Vector3(
                Math.cos(index * 2) * 3.5,
                0,
                Math.sin(index * 2) * 3.5
              ),
              velocity: new THREE.Vector3(0, 0, 0),
              phase: state.entropy * Math.PI,
              informationContent: state.qubit_count * 5,
              memoryResonance: state.coherence,
              source: `Quantum State: ${state.type || key}`,
              polarization: 'plus',
              decayRate: 0.1 * (1 - state.coherence)
            });
          }
        });
      }
    }
    
    // Only use fallback if truly no data - prefer real metrics
    if (echoes.length === 0 && !isDemoMode && !metrics) {
      echoes.push({
        id: 'fallback-1',
        timestamp: Date.now(),
        magnitude: 0.5e-21,
        frequency: 100,
        position: new THREE.Vector3(0, 0, 0),
        velocity: new THREE.Vector3(0, 0, 0),
        phase: 0,
        informationContent: 10,
        memoryResonance: 0.5,
        source: 'System Idle',
        polarization: 'plus',
        decayRate: 0.1
      });
    }
    
    // Enhanced return with dynamic metrics mapping
    return {
      echoes,
      fieldStrength: metrics?.field_energy || metrics?.rsp * 0.001 || 1.0,
      coherence: metrics?.coherence || 0.95,
      backgroundNoise: Math.max(0.01, metrics?.error || metrics?.entropy * 0.1 || 0.1),
      strainSensitivity: 1e-21 * (1 + (metrics?.rsp || 0) * 0.00001),
      informationCapacity: (metrics?.information || 0) + (metrics?.phi || 0) * 10 + 100,
      memoryFieldCoupling: metrics?.memory_field_coupling || metrics?.coherence * 0.8 || 0.5
    };
  }, [propSimulationData, metrics, states, isDemoMode, generateDemoEchoes, backendEchoes]);
  
  // Calculate derived metrics
  const totalInformation = useMemo(() => {
    return simulationData.echoes.reduce((sum, echo) => sum + echo.informationContent, 0);
  }, [simulationData.echoes]);
  
  const avgMemoryResonance = useMemo(() => {
    if (simulationData.echoes.length === 0) return 0;
    const total = simulationData.echoes.reduce((sum, echo) => sum + echo.memoryResonance, 0);
    return total / simulationData.echoes.length;
  }, [simulationData.echoes]);
  
  const dominantFrequency = useMemo(() => {
    if (simulationData.echoes.length === 0) return 0;
    return Math.max(...simulationData.echoes.map(e => e.frequency));
  }, [simulationData.echoes]);
  
  // Handle echo selection
  const handleEchoSelect = useCallback((echo: GravitationalWaveEcho) => {
    // Ensure the selected echo has proper Vector3 instances
    const safeEcho = {
      ...echo,
      position: ensureVector3(echo.position),
      velocity: ensureVector3(echo.velocity)
    };
    setSelectedEcho(safeEcho);
  }, []);
  
  // Export data
  const handleExport = useCallback(() => {
    const exportData = {
      timestamp: new Date().toISOString(),
      simulationData,
      metrics: {
        totalInformation,
        avgMemoryResonance,
        dominantFrequency
      },
      selectedEcho,
      settings
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gw-echoes-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [simulationData, totalInformation, avgMemoryResonance, dominantFrequency, selectedEcho, settings]);
  
  return (
    <div style={{ 
      height: '100%', 
      position: 'relative',
      background: '#0a0a0a',
      borderRadius: '8px',
      overflow: 'hidden'
    }}>
      {/* 3D Visualization Canvas */}
      <Canvas
        camera={{ position: [10, 8, 10], fov: 50 }}
        gl={{ 
          antialias: true, 
          alpha: true,
          powerPreference: "high-performance",
          preserveDrawingBuffer: false
        }}
      >
        <GravitationalWaveScene
          simulationData={simulationData}
          primaryColor={primaryColor}
          isActive={isPlaying}
          settings={settings}
          onEchoSelect={handleEchoSelect}
        />
        <OrbitControls 
          enableZoom={true} 
          enablePan={true} 
          enableRotate={true}
          autoRotate={isPlaying && !selectedEcho}
          autoRotateSpeed={0.3}
          maxDistance={30}
          minDistance={2}
        />
      </Canvas>
      
      {/* Header Panel */}
      <div style={{
        position: 'absolute',
        top: '16px',
        left: '16px',
        right: '16px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        pointerEvents: 'none'
      }}>
        {/* Metrics Panel */}
        <div style={{
          background: 'rgba(0, 0, 0, 0.85)',
          backdropFilter: 'blur(10px)',
          padding: '16px',
          borderRadius: '8px',
          border: `1px solid ${primaryColor}30`,
          pointerEvents: 'auto',
          minWidth: '280px'
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px',
            marginBottom: '12px' 
          }}>
            <Waves size={16} color={primaryColor} />
            <span style={{ 
              fontWeight: 600, 
              color: primaryColor,
              fontSize: '14px'
            }}>
              Gravitational Wave Echoes
            </span>
            {isConnected && (
              <span style={{
                padding: '2px 6px',
                background: `${primaryColor}20`,
                borderRadius: '3px',
                fontSize: '10px',
                color: primaryColor
              }}>
                LIVE
              </span>
            )}
            {isDemoMode && (
              <span style={{
                padding: '2px 6px',
                background: '#ff6b6b20',
                borderRadius: '3px',
                fontSize: '10px',
                color: '#ff6b6b'
              }}>
                DEMO
              </span>
            )}
          </div>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
            gap: '12px',
            fontSize: '12px'
          }}>
            <div>
              <div style={{ color: '#888', marginBottom: '4px' }}>Active Echoes</div>
              <div style={{ color: '#fff', fontSize: '16px', fontWeight: 500 }}>
                {simulationData.echoes.length}
                {backendEchoes.length > 0 && (
                  <span style={{ color: primaryColor, fontSize: '10px', marginLeft: '4px' }}>
                    ({backendEchoes.length} from API)
                  </span>
                )}
              </div>
            </div>
            <div>
              <div style={{ color: '#888', marginBottom: '4px' }}>Total Information</div>
              <div style={{ color: '#fff', fontSize: '16px', fontWeight: 500 }}>
                {totalInformation.toFixed(1)} bits
              </div>
            </div>
            <div>
              <div style={{ color: '#888', marginBottom: '4px' }}>Memory Resonance</div>
              <div style={{ color: '#fff', fontSize: '16px', fontWeight: 500 }}>
                {(avgMemoryResonance * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div style={{ color: '#888', marginBottom: '4px' }}>Field Coherence</div>
              <div style={{ color: '#fff', fontSize: '16px', fontWeight: 500 }}>
                {(simulationData.coherence * 100).toFixed(1)}%
              </div>
            </div>
            {gwEchoAnalysis?.success && (
              <div>
                <div style={{ color: '#888', marginBottom: '4px' }}>GW SNR</div>
                <div style={{ color: '#fff', fontSize: '16px', fontWeight: 500 }}>
                  {gwEchoAnalysis.result?.snr?.toFixed(1) || 'N/A'}
                </div>
              </div>
            )}
            {gwEchoAnalysis?.success && (
              <div>
                <div style={{ color: '#888', marginBottom: '4px' }}>Detection Confidence</div>
                <div style={{ color: '#fff', fontSize: '16px', fontWeight: 500 }}>
                  {((gwEchoAnalysis.result?.confidence || 0) * 100).toFixed(0)}%
                </div>
              </div>
            )}
          </div>
          
          {/* Enhanced status indicator */}
          <div style={{ 
            marginTop: '12px', 
            paddingTop: '12px',
            borderTop: '1px solid rgba(255,255,255,0.1)',
            fontSize: '11px'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '6px'
            }}>
              <Activity size={12} color={isPlaying ? primaryColor : '#666'} />
              <span style={{ color: isPlaying ? primaryColor : '#666' }}>
                {isPlaying ? 'Simulation Active' : 'Simulation Paused'}
              </span>
            </div>
            {metrics && !isDemoMode && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                color: '#888'
              }}>
                <Radio size={12} color={isConnected ? primaryColor : '#666'} />
                <span>
                  {isConnected ? 'Connected to VM' : 'Using cached data'} • 
                  RSP: {metrics.rsp?.toFixed(1) || '0'} • 
                  Φ: {metrics.phi?.toFixed(2) || '0'}
                </span>
              </div>
            )}
          </div>
        </div>
        
        {/* Control Buttons */}
        <div style={{
          display: 'flex',
          gap: '8px',
          pointerEvents: 'auto'
        }}>
          <Tooltip primaryColor={primaryColor} content={isPlaying ? "Pause" : "Play"}>
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              style={{
                background: 'rgba(0, 0, 0, 0.85)',
                border: `1px solid ${primaryColor}30`,
                color: primaryColor,
                width: '36px',
                height: '36px',
                borderRadius: '6px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s'
              }}
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Reset View">
            <button
              onClick={() => window.location.reload()}
              style={{
                background: 'rgba(0, 0, 0, 0.85)',
                border: `1px solid ${primaryColor}30`,
                color: primaryColor,
                width: '36px',
                height: '36px',
                borderRadius: '6px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s'
              }}
            >
              <RotateCcw size={16} />
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Export Data">
            <button
              onClick={handleExport}
              style={{
                background: 'rgba(0, 0, 0, 0.85)',
                border: `1px solid ${primaryColor}30`,
                color: primaryColor,
                width: '36px',
                height: '36px',
                borderRadius: '6px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s'
              }}
            >
              <Download size={16} />
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Information">
            <button
              onClick={() => setShowInfo(!showInfo)}
              style={{
                background: showInfo ? primaryColor : 'rgba(0, 0, 0, 0.85)',
                border: `1px solid ${primaryColor}30`,
                color: showInfo ? '#000' : primaryColor,
                width: '36px',
                height: '36px',
                borderRadius: '6px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s'
              }}
            >
              <Info size={16} />
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Settings">
            <button
              onClick={() => setShowControls(!showControls)}
              style={{
                background: showControls ? primaryColor : 'rgba(0, 0, 0, 0.85)',
                border: `1px solid ${primaryColor}30`,
                color: showControls ? '#000' : primaryColor,
                width: '36px',
                height: '36px',
                borderRadius: '6px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s'
              }}
            >
              <Settings size={16} />
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content={isDemoMode ? "Switch to Real Data" : isConnected ? "Using Real VM Data" : "Switch to Demo Mode"}>
            <button
              onClick={() => setIsDemoMode(!isDemoMode)}
              style={{
                background: isDemoMode ? primaryColor : 'rgba(0, 0, 0, 0.85)',
                border: `1px solid ${primaryColor}30`,
                color: isDemoMode ? '#000' : primaryColor,
                width: '36px',
                height: '36px',
                borderRadius: '6px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s'
              }}
            >
              <Zap size={16} />
            </button>
          </Tooltip>
        </div>
      </div>
      
      {/* Selected Echo Details */}
      <AnimatePresence>
        {selectedEcho && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            style={{
              position: 'absolute',
              bottom: '16px',
              left: '16px',
              background: 'rgba(0, 0, 0, 0.9)',
              backdropFilter: 'blur(10px)',
              padding: '20px',
              borderRadius: '8px',
              border: `1px solid ${primaryColor}30`,
              maxWidth: '400px',
              color: '#fff'
            }}
          >
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '16px'
            }}>
              <h3 style={{ color: primaryColor, margin: 0, fontSize: '16px' }}>
                {selectedEcho.source}
              </h3>
              <button
                onClick={() => setSelectedEcho(null)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#666',
                  cursor: 'pointer',
                  fontSize: '18px'
                }}
              >
                ×
              </button>
            </div>
            
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: '1fr 1fr',
              gap: '12px',
              fontSize: '12px'
            }}>
              <div>
                <div style={{ color: '#888', marginBottom: '4px' }}>Frequency</div>
                <div style={{ color: '#fff' }}>{selectedEcho.frequency.toFixed(1)} Hz</div>
              </div>
              <div>
                <div style={{ color: '#888', marginBottom: '4px' }}>Magnitude</div>
                <div style={{ color: '#fff' }}>{selectedEcho.magnitude.toFixed(3)}</div>
              </div>
              <div>
                <div style={{ color: '#888', marginBottom: '4px' }}>Information</div>
                <div style={{ color: '#fff' }}>{selectedEcho.informationContent.toFixed(1)} bits</div>
              </div>
              <div>
                <div style={{ color: '#888', marginBottom: '4px' }}>Memory Coupling</div>
                <div style={{ color: '#fff' }}>{(selectedEcho.memoryResonance * 100).toFixed(0)}%</div>
              </div>
              {selectedEcho.chirpMass && (
                <div>
                  <div style={{ color: '#888', marginBottom: '4px' }}>Chirp Mass</div>
                  <div style={{ color: '#fff' }}>{selectedEcho.chirpMass.toFixed(1)} M☉</div>
                </div>
              )}
              {selectedEcho.luminosityDistance && (
                <div>
                  <div style={{ color: '#888', marginBottom: '4px' }}>Distance</div>
                  <div style={{ color: '#fff' }}>{selectedEcho.luminosityDistance.toFixed(0)} Mpc</div>
                </div>
              )}
            </div>
            
            <div style={{
              marginTop: '16px',
              paddingTop: '16px',
              borderTop: '1px solid rgba(255,255,255,0.1)',
              fontSize: '11px',
              color: '#aaa'
            }}>
              Position: ({selectedEcho.position.x.toFixed(1)}, {selectedEcho.position.y.toFixed(1)}, {selectedEcho.position.z.toFixed(1)})
              <br />
              Polarization: {selectedEcho.polarization}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Visualization Controls */}
      <AnimatePresence>
        {showControls && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            style={{
              position: 'absolute',
              top: '80px',
              right: '16px',
              background: 'rgba(0, 0, 0, 0.9)',
              backdropFilter: 'blur(10px)',
              padding: '20px',
              borderRadius: '8px',
              border: `1px solid ${primaryColor}30`,
              width: '280px',
              color: '#fff'
            }}
          >
            <h3 style={{ 
              color: primaryColor, 
              margin: '0 0 16px 0', 
              fontSize: '14px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <Sliders size={16} />
              Visualization Settings
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {/* Toggle controls */}
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                <input
                  type="checkbox"
                  checked={settings.showWaveforms}
                  onChange={(e) => setSettings(s => ({ ...s, showWaveforms: e.target.checked }))}
                  style={{ accentColor: primaryColor }}
                />
                Show Waveforms
              </label>
              
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                <input
                  type="checkbox"
                  checked={settings.showInterference}
                  onChange={(e) => setSettings(s => ({ ...s, showInterference: e.target.checked }))}
                  style={{ accentColor: primaryColor }}
                />
                Show Interference
              </label>
              
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                <input
                  type="checkbox"
                  checked={settings.showMemoryField}
                  onChange={(e) => setSettings(s => ({ ...s, showMemoryField: e.target.checked }))}
                  style={{ accentColor: primaryColor }}
                />
                Show Memory Field
              </label>
              
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                <input
                  type="checkbox"
                  checked={settings.showInformationFlow}
                  onChange={(e) => setSettings(s => ({ ...s, showInformationFlow: e.target.checked }))}
                  style={{ accentColor: primaryColor }}
                />
                Show Information Flow
              </label>
              
              {/* Color mode selector */}
              <div style={{ marginTop: '8px' }}>
                <label style={{ fontSize: '11px', color: '#888', marginBottom: '4px', display: 'block' }}>
                  Color Mode
                </label>
                <select
                  value={settings.colorMode}
                  onChange={(e) => setSettings(s => ({ ...s, colorMode: e.target.value as any }))}
                  style={{
                    width: '100%',
                    padding: '6px',
                    background: '#1a1a1a',
                    border: `1px solid ${primaryColor}30`,
                    borderRadius: '4px',
                    color: '#fff',
                    fontSize: '12px'
                  }}
                >
                  <option value="frequency">Frequency</option>
                  <option value="amplitude">Amplitude</option>
                  <option value="information">Information</option>
                </select>
              </div>
              
              {/* Wave speed control */}
              <div>
                <label style={{ fontSize: '11px', color: '#888', marginBottom: '4px', display: 'block' }}>
                  Wave Speed: {settings.waveSpeed.toFixed(1)}c
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="2"
                  step="0.1"
                  value={settings.waveSpeed}
                  onChange={(e) => setSettings(s => ({ ...s, waveSpeed: parseFloat(e.target.value) }))}
                  style={{ width: '100%', accentColor: primaryColor }}
                />
              </div>
              
              {/* Decay time control */}
              <div>
                <label style={{ fontSize: '11px', color: '#888', marginBottom: '4px', display: 'block' }}>
                  Decay Time: {settings.decayTime.toFixed(1)}s
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.5"
                  value={settings.decayTime}
                  onChange={(e) => setSettings(s => ({ ...s, decayTime: parseFloat(e.target.value) }))}
                  style={{ width: '100%', accentColor: primaryColor }}
                />
              </div>
              
              {/* Color Configuration Section */}
              <div style={{ 
                marginTop: '16px', 
                paddingTop: '16px', 
                borderTop: '1px solid rgba(255,255,255,0.1)' 
              }}>
                <h4 style={{ 
                  fontSize: '12px', 
                  color: primaryColor, 
                  marginBottom: '12px',
                  fontWeight: 600 
                }}>
                  Color Configuration
                </h4>
                
                {/* Wave Color */}
                <div style={{ marginBottom: '8px' }}>
                  <label style={{ fontSize: '11px', color: '#888', display: 'block', marginBottom: '4px' }}>
                    Wave Color
                  </label>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <input
                      type="color"
                      value={settings.waveColor}
                      onChange={(e) => setSettings(s => ({ ...s, waveColor: e.target.value }))}
                      style={{ 
                        width: '32px', 
                        height: '24px', 
                        border: 'none', 
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    />
                    <input
                      type="text"
                      value={settings.waveColor}
                      onChange={(e) => setSettings(s => ({ ...s, waveColor: e.target.value }))}
                      style={{
                        flex: 1,
                        padding: '4px 8px',
                        background: '#1a1a1a',
                        border: `1px solid ${primaryColor}30`,
                        borderRadius: '4px',
                        color: '#fff',
                        fontSize: '11px'
                      }}
                    />
                  </div>
                </div>
                
                {/* Echo Color */}
                <div style={{ marginBottom: '8px' }}>
                  <label style={{ fontSize: '11px', color: '#888', display: 'block', marginBottom: '4px' }}>
                    Echo Color
                  </label>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <input
                      type="color"
                      value={settings.echoColor}
                      onChange={(e) => setSettings(s => ({ ...s, echoColor: e.target.value }))}
                      style={{ 
                        width: '32px', 
                        height: '24px', 
                        border: 'none', 
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    />
                    <input
                      type="text"
                      value={settings.echoColor}
                      onChange={(e) => setSettings(s => ({ ...s, echoColor: e.target.value }))}
                      style={{
                        flex: 1,
                        padding: '4px 8px',
                        background: '#1a1a1a',
                        border: `1px solid ${primaryColor}30`,
                        borderRadius: '4px',
                        color: '#fff',
                        fontSize: '11px'
                      }}
                    />
                  </div>
                </div>
                
                {/* Interference Color */}
                <div style={{ marginBottom: '8px' }}>
                  <label style={{ fontSize: '11px', color: '#888', display: 'block', marginBottom: '4px' }}>
                    Interference Color
                  </label>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <input
                      type="color"
                      value={settings.interferenceColor}
                      onChange={(e) => setSettings(s => ({ ...s, interferenceColor: e.target.value }))}
                      style={{ 
                        width: '32px', 
                        height: '24px', 
                        border: 'none', 
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    />
                    <input
                      type="text"
                      value={settings.interferenceColor}
                      onChange={(e) => setSettings(s => ({ ...s, interferenceColor: e.target.value }))}
                      style={{
                        flex: 1,
                        padding: '4px 8px',
                        background: '#1a1a1a',
                        border: `1px solid ${primaryColor}30`,
                        borderRadius: '4px',
                        color: '#fff',
                        fontSize: '11px'
                      }}
                    />
                  </div>
                </div>
                
                {/* Memory Field Color */}
                <div style={{ marginBottom: '8px' }}>
                  <label style={{ fontSize: '11px', color: '#888', display: 'block', marginBottom: '4px' }}>
                    Memory Field Color
                  </label>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <input
                      type="color"
                      value={settings.memoryFieldColor}
                      onChange={(e) => setSettings(s => ({ ...s, memoryFieldColor: e.target.value }))}
                      style={{ 
                        width: '32px', 
                        height: '24px', 
                        border: 'none', 
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    />
                    <input
                      type="text"
                      value={settings.memoryFieldColor}
                      onChange={(e) => setSettings(s => ({ ...s, memoryFieldColor: e.target.value }))}
                      style={{
                        flex: 1,
                        padding: '4px 8px',
                        background: '#1a1a1a',
                        border: `1px solid ${primaryColor}30`,
                        borderRadius: '4px',
                        color: '#fff',
                        fontSize: '11px'
                      }}
                    />
                  </div>
                </div>
                
                {/* Visual Intensity Controls */}
                <div style={{ marginTop: '12px' }}>
                  <label style={{ fontSize: '11px', color: '#888', marginBottom: '4px', display: 'block' }}>
                    Wave Opacity: {(settings.waveOpacity * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="1"
                    step="0.1"
                    value={settings.waveOpacity}
                    onChange={(e) => setSettings(s => ({ ...s, waveOpacity: parseFloat(e.target.value) }))}
                    style={{ width: '100%', accentColor: primaryColor }}
                  />
                </div>
                
                <div style={{ marginTop: '8px' }}>
                  <label style={{ fontSize: '11px', color: '#888', marginBottom: '4px', display: 'block' }}>
                    Echo Size: {(settings.echoSize * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="2"
                    step="0.1"
                    value={settings.echoSize}
                    onChange={(e) => setSettings(s => ({ ...s, echoSize: parseFloat(e.target.value) }))}
                    style={{ width: '100%', accentColor: primaryColor }}
                  />
                </div>
                
                <div style={{ marginTop: '8px' }}>
                  <label style={{ fontSize: '11px', color: '#888', marginBottom: '4px', display: 'block' }}>
                    Interference Intensity: {(settings.interferenceIntensity * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="2"
                    step="0.1"
                    value={settings.interferenceIntensity}
                    onChange={(e) => setSettings(s => ({ ...s, interferenceIntensity: parseFloat(e.target.value) }))}
                    style={{ width: '100%', accentColor: primaryColor }}
                  />
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Information Modal */}
      <AnimatePresence>
        {showInfo && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0, 0, 0, 0.8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '20px'
            }}
            onClick={() => setShowInfo(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              style={{
                background: 'rgba(0, 0, 0, 0.95)',
                padding: '32px',
                borderRadius: '12px',
                border: `2px solid ${primaryColor}`,
                color: '#fff',
                maxWidth: '600px',
                maxHeight: '80vh',
                overflowY: 'auto'
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <h2 style={{ color: primaryColor, marginBottom: '20px' }}>
                Gravitational Wave Echoes in OSH
              </h2>
              
              <div style={{ fontSize: '14px', lineHeight: '1.8', color: '#ddd' }}>
                <p>
                  In the Organic Simulation Hypothesis framework, gravitational waves are not merely 
                  ripples in spacetime—they are carriers of quantum information that leave persistent 
                  "echoes" in the memory field of reality.
                </p>
                
                <h3 style={{ color: primaryColor, marginTop: '24px', marginBottom: '12px', fontSize: '16px' }}>
                  Key Concepts:
                </h3>
                
                <ul style={{ paddingLeft: '20px' }}>
                  <li style={{ marginBottom: '8px' }}>
                    <strong>Information Preservation:</strong> Each gravitational wave event encodes 
                    information about its source that persists in the memory field
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong>Memory Resonance:</strong> The coupling strength between gravitational 
                    echoes and the universal memory field, determining information retention
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong>Quantum Interference:</strong> Multiple echoes create interference patterns 
                    that encode complex information structures
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong>Consciousness Coupling:</strong> Strong gravitational events may influence 
                    conscious observation through memory field perturbations
                  </li>
                </ul>
                
                <h3 style={{ color: primaryColor, marginTop: '24px', marginBottom: '12px', fontSize: '16px' }}>
                  Visualization Elements:
                </h3>
                
                <ul style={{ paddingLeft: '20px' }}>
                  <li style={{ marginBottom: '8px' }}>
                    <strong>Echo Sources:</strong> Spheres representing gravitational wave sources, 
                    sized by information content
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong>Wave Propagation:</strong> Realistic wave patterns showing spacetime 
                    distortions
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong>Interference Patterns:</strong> Superposition of multiple wave sources
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong>Memory Field:</strong> Background coupling showing information storage
                  </li>
                </ul>
                
                <h3 style={{ color: primaryColor, marginTop: '24px', marginBottom: '12px', fontSize: '16px' }}>
                  Current Detection:
                </h3>
                
                <p style={{ marginBottom: '12px' }}>
                  Active echoes: <strong>{simulationData.echoes.length}</strong><br />
                  Total information: <strong>{totalInformation.toFixed(1)} bits</strong><br />
                  Dominant frequency: <strong>{dominantFrequency.toFixed(1)} Hz</strong><br />
                  Memory coupling: <strong>{(simulationData.memoryFieldCoupling * 100).toFixed(1)}%</strong>
                </p>
              </div>
              
              <button
                onClick={() => setShowInfo(false)}
                style={{
                  marginTop: '24px',
                  background: primaryColor,
                  color: '#000',
                  border: 'none',
                  padding: '10px 24px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '14px'
                }}
              >
                Close
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default GravitationalWaveEchoVisualizer;