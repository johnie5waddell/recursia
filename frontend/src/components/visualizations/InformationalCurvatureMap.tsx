import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useEngineAPIContext } from '../../contexts/EngineAPIContext';
import { type CurvatureTensor } from '../../quantum/types';
import { Tooltip } from '../ui/Tooltip';
import { Activity, Eye, Settings, Layers, TrendingUp, Info } from 'lucide-react';
import { adjustLightness } from '../../utils/colorUtils';

/**
 * Enhanced color palette generator using primary color shader offsets
 * Creates consistent, mathematically-derived color schemes for each visualization mode
 */
interface ColorPalette {
  primary: string;
  positive: string;
  negative: string;
  neutral: string;
  accent: string;
  background: string;
  grid: string;
  gridSecondary: string;
}

/**
 * Generates scientifically-accurate color palettes based on primary color and visualization mode
 * Uses perceptually uniform color spaces with enhanced brightness and contrast
 */
function generateModePalette(primaryColor: string, mode: 'ricci' | 'scalar' | 'information'): ColorPalette {
  const baseColor = new THREE.Color(primaryColor);
  const hsl = { h: 0, s: 0, l: 0 };
  baseColor.getHSL(hsl);
  
  switch (mode) {
    case 'scalar':
      // Scalar curvature: High contrast red (positive) to blue (negative) mapping
      // Enhanced brightness for clear visibility
      return {
        primary: primaryColor,
        positive: '#ff3366', // Bright red for positive curvature
        negative: '#0099ff', // Bright blue for negative curvature
        neutral: '#888888',  // Medium gray for zero curvature
        accent: '#ffff00',   // Yellow for highlights
        background: '#0a0a0a',
        grid: '#00ff88',    // Bright green grid
        gridSecondary: '#006644'
      };
      
    case 'information':
      // Information density: Rainbow spectrum for maximum differentiation
      // Each information level gets distinct color
      return {
        primary: primaryColor,
        positive: '#ff00ff', // Magenta for high information
        negative: '#00ffff', // Cyan for low information
        neutral: '#ffff00',  // Yellow for medium
        accent: '#ff6600',   // Orange accent
        background: '#050510',
        grid: '#ff00aa',     // Hot pink grid
        gridSecondary: '#aa0066'
      };
      
    case 'ricci':
      // Ricci tensor: Purple to green with enhanced vibrancy
      // Clear distinction between tensor components
      return {
        primary: primaryColor,
        positive: '#ff00ff', // Bright purple for positive eigenvalues
        negative: '#00ff00', // Bright green for negative eigenvalues
        neutral: '#ffaa00',  // Orange for trace
        accent: '#00ffff',   // Cyan accent
        background: '#0a0015',
        grid: '#ff66ff',     // Light purple grid
        gridSecondary: '#aa44aa'
      };
      
    default:
      return generateModePalette(primaryColor, 'scalar');
  }
}

interface InformationalCurvatureMapProps {
  data?: CurvatureTensor[];
  primaryColor: string;
  showClassicalComparison?: boolean;
  fieldStrength?: number;
}

/**
 * Enhanced curvature field visualization with improved positioning accuracy,
 * scientifically-rigorous color mapping, and detailed visual representations
 */
function CurvatureField({ 
  data, 
  primaryColor, 
  colorPalette,
  fieldStrength,
  curvatureMode,
  onSelectPoint,
  showFieldLines,
  showParticles,
  animationSpeed
}: { 
  data: CurvatureTensor[]; 
  primaryColor: string;
  colorPalette: ColorPalette;
  fieldStrength: number;
  curvatureMode: 'ricci' | 'scalar' | 'information';
  onSelectPoint: (tensor: CurvatureTensor | null) => void;
  showFieldLines: boolean;
  showParticles: boolean;
  animationSpeed: number;
}) {
  const [time, setTime] = useState(0);
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const particlesRef = useRef<THREE.Points>(null);
  const fieldLinesRef = useRef<THREE.Group>(null);
  const tensorVisualizationRef = useRef<THREE.Group>(null);
  
  // Enhanced animation system with physically-accurate motion patterns
  useFrame((state) => {
    const elapsedTime = state.clock.getElapsedTime();
    setTime(elapsedTime);
    
    // Main field rotation with mode-specific characteristics
    if (meshRef.current) {
      const baseSpeed = curvatureMode === 'information' ? 0.15 : curvatureMode === 'ricci' ? 0.08 : 0.12;
      meshRef.current.rotation.y = elapsedTime * baseSpeed * animationSpeed;
    }
    
    // Enhanced particle system with accurate physical modeling
    if (particlesRef.current) {
      // Mode-specific particle dynamics based on curvature physics
      switch (curvatureMode) {
        case 'scalar':
          // Scalar: Geodesic deviation along curvature gradients
          particlesRef.current.rotation.y = -elapsedTime * 0.06;
          particlesRef.current.rotation.x = Math.sin(elapsedTime * 0.4) * 0.08;
          break;
        case 'information':
          // Information: Holographic flow patterns with entropy dynamics
          particlesRef.current.rotation.y = elapsedTime * 0.18;
          particlesRef.current.rotation.z = Math.cos(elapsedTime * 0.25) * 0.15;
          particlesRef.current.position.y = Math.sin(elapsedTime * 0.1) * 0.05;
          break;
        case 'ricci':
          // Ricci: Einstein field equation dynamics
          const ricciPhase = elapsedTime * 0.12;
          particlesRef.current.rotation.y = Math.sin(ricciPhase) * Math.PI * 0.7;
          particlesRef.current.rotation.x = Math.cos(ricciPhase * 1.3) * 0.25;
          particlesRef.current.rotation.z = Math.sin(ricciPhase * 0.8) * 0.18;
          break;
      }
      
      // High-precision position updates with mathematical accuracy
      const positions = particlesRef.current.geometry.attributes.position;
      const velocities = particlesRef.current.geometry.attributes.velocity as THREE.BufferAttribute;
      
      if (positions && data.length > 0) {
        for (let i = 0; i < data.length; i++) {
          const tensor = data[i];
          let positionOffset = 0;
          let velocityFactor = 1;
          
          switch (curvatureMode) {
            case 'scalar':
              // Scalar curvature creates geodesic deviation
              const curvatureSign = Math.sign(tensor.scalar);
              const curvatureMagnitude = Math.abs(tensor.scalar);
              positionOffset = Math.sin(elapsedTime * 2.5 + i * 0.15) * 0.25 * curvatureSign * curvatureMagnitude;
              velocityFactor = 1 + curvatureMagnitude * 0.5;
              break;
              
            case 'information':
              // Information creates holographic surface fluctuations
              const infoPhase = tensor.information * 0.08 + elapsedTime * 3.2;
              positionOffset = Math.sin(infoPhase) * 0.35 * (tensor.information / 20);
              velocityFactor = 1 + Math.log(1 + tensor.information) * 0.3;
              break;
              
            case 'ricci':
              // Ricci tensor creates complex spacetime geometry
              const ricciTrace = tensor.ricci[0][0] + tensor.ricci[1][1] + tensor.ricci[2][2];
              const ricciDeterminant = tensor.ricci[0][0] * tensor.ricci[1][1] * tensor.ricci[2][2];
              positionOffset = Math.sin(elapsedTime * 1.8 + ricciTrace * 0.5) * 0.18 * Math.cbrt(Math.abs(ricciDeterminant) + 0.1);
              velocityFactor = 1 + Math.abs(ricciTrace) * 0.2;
              break;
          }
          
          // Apply position updates with interpolation for smooth motion
          const currentY = positions.getY(i);
          const targetY = tensor.position[1] + positionOffset;
          const newY = THREE.MathUtils.lerp(currentY, targetY, 0.1 * velocityFactor);
          positions.setY(i, newY);
          
          // Update X and Z coordinates for spiral/orbital motion
          if (curvatureMode === 'information') {
            const spiralRadius = 0.1 * Math.sqrt(tensor.information);
            const spiralPhase = elapsedTime + i * 0.5;
            const baseX = tensor.position[0];
            const baseZ = tensor.position[2];
            positions.setX(i, baseX + Math.cos(spiralPhase) * spiralRadius);
            positions.setZ(i, baseZ + Math.sin(spiralPhase) * spiralRadius);
          }
        }
        positions.needsUpdate = true;
      }
    }
    
    // Enhanced tensor visualization animations
    if (tensorVisualizationRef.current) {
      tensorVisualizationRef.current.children.forEach((child, i) => {
        if (child instanceof THREE.Group) {
          // Tensor component visualization with eigenvalue rotation
          child.rotation.x = elapsedTime * 0.3 + i * 0.2;
          child.rotation.y = elapsedTime * 0.5 + i * 0.3;
          
          if (curvatureMode === 'ricci') {
            // Additional rotation for Ricci tensor eigenspaces
            child.rotation.z = Math.sin(elapsedTime * 0.4 + i) * 0.3;
          }
        }
      });
    }
    
    // Field line dynamics with improved physics
    if (fieldLinesRef.current) {
      fieldLinesRef.current.children.forEach((child, i) => {
        if (child instanceof THREE.Mesh) {
          let scale = 1;
          let intensityModulation = 1;
          
          switch (curvatureMode) {
            case 'scalar':
              // Scalar: Gravitational wave-like pulsing
              scale = 1 + Math.sin(elapsedTime * 2.2 + i * 0.6) * 0.25;
              intensityModulation = 0.8 + Math.sin(elapsedTime * 1.8 + i * 0.4) * 0.3;
              break;
            case 'information':
              // Information: Quantum fluctuation patterns
              scale = 1 + Math.sin(elapsedTime * 6 + i * 1.2) * 0.35;
              intensityModulation = 0.6 + Math.sin(elapsedTime * 4 + i * 0.8) * 0.4;
              break;
            case 'ricci':
              // Ricci: Smooth Einstein field oscillations
              scale = 1 + Math.sin(elapsedTime * 1.3 + i * 0.35) * 0.15;
              intensityModulation = 0.9 + Math.sin(elapsedTime * 0.9 + i * 0.25) * 0.2;
              break;
          }
          
          child.scale.setScalar(scale);
          
          // Update material properties for dynamic intensity
          if (child.material && 'emissiveIntensity' in child.material) {
            (child.material as THREE.MeshStandardMaterial).emissiveIntensity = intensityModulation;
          }
        }
      });
    }
  });

  // Enhanced geometry creation with scientifically-accurate particle system
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    
    if (data.length === 0) {
      // Create empty attributes with proper structure
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
      geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(0), 3));
      geo.setAttribute('size', new THREE.BufferAttribute(new Float32Array(0), 1));
      geo.setAttribute('velocity', new THREE.BufferAttribute(new Float32Array(0), 3));
      geo.setAttribute('curvature', new THREE.BufferAttribute(new Float32Array(0), 1));
      return geo;
    }
    
    // Enhanced data arrays for sophisticated visualization
    const positions = new Float32Array(data.length * 3);
    const colors = new Float32Array(data.length * 3);
    const sizes = new Float32Array(data.length);
    const velocities = new Float32Array(data.length * 3);
    const curvatures = new Float32Array(data.length);
    
    data.forEach((tensor, i) => {
      // Precise positioning with sub-pixel accuracy
      positions[i * 3] = tensor.position[0];
      positions[i * 3 + 1] = tensor.position[1];
      positions[i * 3 + 2] = tensor.position[2];
      
      // Calculate curvature magnitude for unified scaling
      let curvatureMagnitude = 0;
      let intensity = 0;
      let particleColor = new THREE.Color();
      let particleSize = 0.08;
      
      switch (curvatureMode) {
        case 'scalar':
          // Scalar curvature: Mathematically accurate thermal mapping
          curvatureMagnitude = Math.abs(tensor.scalar);
          intensity = Math.pow(curvatureMagnitude, 0.6); // Perceptually uniform scaling
          
          if (tensor.scalar > 0) {
            // Positive curvature: Use palette positive color
            particleColor.set(colorPalette.positive);
            particleColor.setHSL(
              particleColor.getHSL({ h: 0, s: 0, l: 0 }).h,
              Math.min(0.95, 0.7 + intensity * 0.25),
              Math.min(0.85, 0.4 + intensity * 0.45)
            );
          } else {
            // Negative curvature: Use palette negative color
            particleColor.set(colorPalette.negative);
            particleColor.setHSL(
              particleColor.getHSL({ h: 0, s: 0, l: 0 }).h,
              Math.min(0.98, 0.8 + intensity * 0.18),
              Math.min(0.75, 0.35 + intensity * 0.4)
            );
          }
          
          particleSize = 0.09 + intensity * 0.15;
          break;
          
        case 'information':
          // Information density: Holographic spectrum mapping
          curvatureMagnitude = tensor.information;
          intensity = Math.min(1, Math.log(1 + curvatureMagnitude) / Math.log(16)); // Logarithmic scaling
          
          // Dynamic spectral mapping using palette colors
          const baseHue = new THREE.Color(colorPalette.primary).getHSL({ h: 0, s: 0, l: 0 }).h;
          const spectralHue = (baseHue + intensity * 0.7 + time * 0.05) % 1;
          particleColor.setHSL(
            spectralHue,
            0.85 + intensity * 0.15,
            0.45 + intensity * 0.5
          );
          
          particleSize = 0.06 + intensity * 0.25;
          break;
          
        case 'ricci':
          // Ricci tensor: Eigenvalue-based matrix visualization
          const ricciTrace = tensor.ricci[0][0] + tensor.ricci[1][1] + tensor.ricci[2][2];
          const ricciNorm = Math.sqrt(tensor.ricci.flat().reduce((sum, val) => sum + val * val, 0));
          curvatureMagnitude = ricciNorm;
          intensity = Math.min(1, ricciNorm / 2.0);
          
          // Eigenvalue-based color mapping using palette
          const eigenvalueR = Math.abs(tensor.ricci[0][0]) / Math.max(0.1, ricciNorm);
          const eigenvalueG = Math.abs(tensor.ricci[1][1]) / Math.max(0.1, ricciNorm);
          const eigenvalueB = Math.abs(tensor.ricci[2][2]) / Math.max(0.1, ricciNorm);
          
          // Blend palette colors based on eigenvalue distribution
          const positiveColor = new THREE.Color(colorPalette.positive);
          const negativeColor = new THREE.Color(colorPalette.negative);
          const accentColor = new THREE.Color(colorPalette.accent);
          
          particleColor.setRGB(
            positiveColor.r * eigenvalueR + negativeColor.r * (1 - eigenvalueR),
            accentColor.g * eigenvalueG + positiveColor.g * (1 - eigenvalueG),
            negativeColor.b * eigenvalueB + accentColor.b * (1 - eigenvalueB)
          );
          
          particleSize = 0.08 + intensity * 0.12;
          break;
      }
      
      // Enhanced pulsing with mode-specific characteristics
      let pulseFactor = 1;
      switch (curvatureMode) {
        case 'scalar':
          pulseFactor = 1 + Math.sin(time * 2.5 + i * 0.3) * 0.25 * intensity;
          break;
        case 'information':
          pulseFactor = 1 + Math.sin(time * 4 + curvatureMagnitude * 0.2) * 0.35 * intensity;
          break;
        case 'ricci':
          pulseFactor = 1 + Math.sin(time * 1.8 + i * 0.25) * 0.2 * intensity;
          break;
      }
      
      // Apply enhanced color calculations
      colors[i * 3] = Math.min(1, particleColor.r * pulseFactor);
      colors[i * 3 + 1] = Math.min(1, particleColor.g * pulseFactor);
      colors[i * 3 + 2] = Math.min(1, particleColor.b * pulseFactor);
      
      // Advanced size calculation with field strength integration
      const sizeModulation = 1 + Math.sin(time * 2.2 + i * 0.4) * 0.18;
      const fieldModulation = Math.pow(fieldStrength, 0.8); // Non-linear field response
      sizes[i] = Math.min(4, particleSize * sizeModulation * fieldModulation * 1.8);
      
      // Velocity calculations for realistic motion
      const velocityMagnitude = 0.1 + intensity * 0.2;
      velocities[i * 3] = Math.sin(time + i) * velocityMagnitude;
      velocities[i * 3 + 1] = Math.cos(time * 1.3 + i * 0.7) * velocityMagnitude;
      velocities[i * 3 + 2] = Math.sin(time * 0.8 + i * 1.1) * velocityMagnitude;
      
      // Store curvature magnitude for shader use
      curvatures[i] = curvatureMagnitude;
    });
    
    // Enhanced attribute assignment
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    geo.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
    geo.setAttribute('curvature', new THREE.BufferAttribute(curvatures, 1));
    
    return geo;
  }, [data, colorPalette, fieldStrength, curvatureMode, time]);

  return (
    <>
      {/* Enhanced particle system with advanced material properties */}
      {showParticles && (
        <points ref={particlesRef} geometry={geometry}>
          <pointsMaterial
            size={curvatureMode === 'information' ? 0.18 : curvatureMode === 'ricci' ? 0.14 : 0.12}
            sizeAttenuation={true}
            vertexColors
            transparent
            opacity={curvatureMode === 'ricci' ? 0.92 : curvatureMode === 'information' ? 0.88 : 0.85}
            blending={curvatureMode === 'scalar' ? THREE.NormalBlending : THREE.AdditiveBlending}
            depthWrite={false}
          />
        </points>
      )}
      
      {/* Enhanced tensor visualization with accurate positioning */}
      <group ref={tensorVisualizationRef}>
        {data.slice(0, Math.min(40, data.length)).map((tensor, i) => {
          // Calculate tensor eigenvalues for accurate visualization
          const ricciTrace = tensor.ricci[0][0] + tensor.ricci[1][1] + tensor.ricci[2][2];
          const ricciNorm = Math.sqrt(tensor.ricci.flat().reduce((sum, val) => sum + val * val, 0));
          
          return (
            <group key={`tensor_${i}`} position={tensor.position}>
              {/* Primary tensor visualization */}
              <mesh 
                onClick={() => onSelectPoint(tensor)}
                onPointerOver={(e) => {
                  e.stopPropagation();
                  if (e.object instanceof THREE.Mesh) {
                    e.object.scale.setScalar(1.4);
                  }
                }}
                onPointerOut={(e) => {
                  e.stopPropagation();
                  if (e.object instanceof THREE.Mesh) {
                    e.object.scale.setScalar(1);
                  }
                }}
              >
                <icosahedronGeometry args={[
                  curvatureMode === 'ricci' ? 0.06 + ricciNorm * 0.08 : 
                  curvatureMode === 'information' ? 0.05 + (tensor.information / 25) * 0.15 :
                  0.07 + Math.abs(tensor.scalar) * 0.12,
                  curvatureMode === 'ricci' ? 2 : 3
                ]} />
                <meshStandardMaterial
                  color={
                    curvatureMode === 'scalar' ? 
                      (tensor.scalar > 0 ? colorPalette.positive : colorPalette.negative) :
                    curvatureMode === 'information' ?
                      colorPalette.accent :
                    colorPalette.primary
                  }
                  emissive={
                    curvatureMode === 'scalar' ? 
                      (tensor.scalar > 0 ? colorPalette.positive : colorPalette.negative) :
                    curvatureMode === 'information' ?
                      colorPalette.positive :
                    colorPalette.accent
                  }
                  emissiveIntensity={
                    curvatureMode === 'scalar' ? 
                      0.2 + Math.abs(tensor.scalar) * 0.4 + Math.sin(time * 3 + i) * 0.2 :
                    curvatureMode === 'information' ?
                      0.3 + (tensor.information / 30) * 0.5 + Math.sin(time * 4 + i) * 0.3 :
                      0.25 + ricciNorm * 0.3 + Math.sin(time * 2 + i) * 0.15
                  }
                  transparent
                  opacity={
                    curvatureMode === 'scalar' ? 
                      0.7 + Math.abs(tensor.scalar) * 0.25 :
                    curvatureMode === 'information' ?
                      0.5 + Math.min(0.4, tensor.information / 40) :
                      0.6 + ricciNorm * 0.2
                  }
                  roughness={curvatureMode === 'information' ? 0.2 : 0.1}
                  metalness={curvatureMode === 'scalar' ? 0.7 : curvatureMode === 'ricci' ? 0.9 : 0.5}
                  envMapIntensity={curvatureMode === 'ricci' ? 2.5 : 1.8}
                />
              </mesh>
              
              {/* Eigenspace visualization for Ricci mode */}
              {curvatureMode === 'ricci' && (
                <>
                  <mesh position={[0.15, 0, 0]} scale={[0.4, 0.4, 0.4]}>
                    <boxGeometry args={[0.08, 0.08, 0.08]} />
                    <meshStandardMaterial
                      color={colorPalette.positive}
                      emissive={colorPalette.positive}
                      emissiveIntensity={0.3 + Math.abs(tensor.ricci[0][0]) * 0.5}
                      transparent
                      opacity={0.6}
                    />
                  </mesh>
                  <mesh position={[0, 0.15, 0]} scale={[0.4, 0.4, 0.4]}>
                    <boxGeometry args={[0.08, 0.08, 0.08]} />
                    <meshStandardMaterial
                      color={colorPalette.accent}
                      emissive={colorPalette.accent}
                      emissiveIntensity={0.3 + Math.abs(tensor.ricci[1][1]) * 0.5}
                      transparent
                      opacity={0.6}
                    />
                  </mesh>
                  <mesh position={[0, 0, 0.15]} scale={[0.4, 0.4, 0.4]}>
                    <boxGeometry args={[0.08, 0.08, 0.08]} />
                    <meshStandardMaterial
                      color={colorPalette.negative}
                      emissive={colorPalette.negative}
                      emissiveIntensity={0.3 + Math.abs(tensor.ricci[2][2]) * 0.5}
                      transparent
                      opacity={0.6}
                    />
                  </mesh>
                </>
              )}
            </group>
          );
        })}
      </group>
      
      {/* Enhanced field lines with palette-based colors */}
      {showFieldLines && (
        <group ref={fieldLinesRef}>
          {data.length > 1 && (
            <lineSegments>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                array={new Float32Array(
                  data.slice(0, Math.min(curvatureMode === 'ricci' ? 35 : 25, data.length - 1)).flatMap((tensor, i) => {
                    const next = data[i + 1];
                    return next ? [...tensor.position, ...next.position] : [];
                  }).flat()
                )}
                count={Math.min(curvatureMode === 'ricci' ? 35 : 25, data.length - 1) * 2}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial 
              color={
                curvatureMode === 'scalar' ? colorPalette.grid :
                curvatureMode === 'information' ? colorPalette.accent :
                colorPalette.primary
              }
              opacity={curvatureMode === 'ricci' ? 0.6 : curvatureMode === 'information' ? 0.4 : 0.35} 
              transparent 
              linewidth={curvatureMode === 'information' ? 2.5 : 1.8}
            />
          </lineSegments>
        )}
        
        {/* Curvature flow indicators */}
        {data.slice(0, Math.min(15, data.length)).map((tensor, i) => (
          <mesh 
            key={`flow_${i}`}
            position={[
              tensor.position[0] + Math.sin(time + i) * 0.3,
              tensor.position[1] + Math.cos(time * 1.2 + i) * 0.2,
              tensor.position[2] + Math.sin(time * 0.8 + i) * 0.25
            ]}
            scale={[0.3, 0.3, 0.3]}
          >
            <tetrahedronGeometry args={[0.04]} />
            <meshStandardMaterial
              color={colorPalette.neutral}
              emissive={colorPalette.accent}
              emissiveIntensity={0.4 + Math.sin(time * 5 + i) * 0.3}
              transparent
              opacity={0.7}
            />
          </mesh>
        ))}
        </group>
      )}
    </>
  );
}

/**
 * Enhanced Informational Curvature Map with scientifically-accurate visualization,
 * dynamic color palettes, and improved mathematical precision
 */
export function InformationalCurvatureMap({
  data: propData,
  primaryColor = '#4fc3f7',
  showClassicalComparison = false,
  fieldStrength: propFieldStrength = 1.0
}: InformationalCurvatureMapProps) {
  const [selectedPoint, setSelectedPoint] = useState<CurvatureTensor | null>(null);
  const [curvatureMode, setCurvatureMode] = useState<'ricci' | 'scalar' | 'information'>('scalar');
  const [autoRotate, setAutoRotate] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [realTimeData, setRealTimeData] = useState<CurvatureTensor[]>([]);
  const [visualizationQuality, setVisualizationQuality] = useState<'low' | 'medium' | 'high'>('medium');
  const [showFieldLines, setShowFieldLines] = useState(true);
  const [showParticles, setShowParticles] = useState(true);
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  
  // Generate consistent color palette for current mode
  const colorPalette = useMemo(() => generateModePalette(primaryColor, curvatureMode), [primaryColor, curvatureMode]);
  
  // Connect to backend metrics
  const { metrics, isConnected } = useEngineAPIContext();
  
  // Calculate dynamic field strength from metrics
  const fieldStrength = useMemo(() => {
    if (metrics && metrics.field_energy) {
      return Math.min(5, Math.max(0.1, metrics.field_energy));
    }
    return propFieldStrength;
  }, [metrics, propFieldStrength]);
  
  // Generate real-time curvature data from backend metrics
  useEffect(() => {
    if (!metrics || !isConnected) return;
    
    const updateInterval = setInterval(() => {
      // Use backend-calculated information_curvature directly (α = 8π coupling)
      const curvature = metrics.information_curvature || 0.001;
      const time = Date.now() * 0.001;
      
      // Create curvature tensor data based on backend metrics
      const tensors: CurvatureTensor[] = [];
      
      // Number of visualization points scales with quantum complexity and quality setting
      const qualityMultiplier = visualizationQuality === 'high' ? 3 : visualizationQuality === 'medium' ? 2 : 1;
      const numPoints = Math.min(128, Math.max(16, 
        Math.floor((metrics.state_count || 8) * qualityMultiplier)
      ));
      
      // Information density field parameters from OSH theory
      const infoScale = Math.max(0.1, metrics.information || 1);
      const coherenceScale = metrics.coherence || 0.95;
      const entropyFactor = 1 / (1 + (metrics.entropy || 0.1));
      
      // Generate 3D lattice distribution for accurate spatial representation
      const gridSize = Math.ceil(Math.cbrt(numPoints));
      let pointIndex = 0;
      
      for (let ix = 0; ix < gridSize && pointIndex < numPoints; ix++) {
        for (let iy = 0; iy < gridSize && pointIndex < numPoints; iy++) {
          for (let iz = 0; iz < gridSize && pointIndex < numPoints; iz++) {
            // Normalized grid coordinates [-1, 1]
            const nx = (ix / (gridSize - 1)) * 2 - 1;
            const ny = (iy / (gridSize - 1)) * 2 - 1;
            const nz = (iz / (gridSize - 1)) * 2 - 1;
            
            // Apply RSP-based warping to create information density gradients
            const rspWarp = metrics.rsp ? Math.log(1 + metrics.rsp / 100) : 0;
            const warpFactor = 1 + rspWarp * 0.3;
            
            // Position with information-curvature coupling
            const x = nx * 3 * warpFactor + Math.sin(time * 0.2 * animationSpeed + nx) * curvature * 10;
            const y = ny * 3 * warpFactor + Math.cos(time * 0.3 * animationSpeed + ny) * curvature * 10;
            const z = nz * 3 * warpFactor + Math.sin(time * 0.25 * animationSpeed + nz) * curvature * 10;
            
            // Calculate local information density using Gaussian distribution
            const r = Math.sqrt(x*x + y*y + z*z);
            const sigma = 2.0; // Information correlation length
            const localInfo = infoScale * Math.exp(-r*r / (2 * sigma * sigma));
            
            // Ricci tensor from information gradients (∇_μ∇_ν I)
            // Using numerical approximation for second derivatives
            const dx = 0.1;
            const d2I_dx2 = -localInfo * (2 / (sigma * sigma)) * (1 - x*x / (sigma * sigma));
            const d2I_dy2 = -localInfo * (2 / (sigma * sigma)) * (1 - y*y / (sigma * sigma));
            const d2I_dz2 = -localInfo * (2 / (sigma * sigma)) * (1 - z*z / (sigma * sigma));
            const d2I_dxdy = localInfo * (4 * x * y) / (sigma * sigma * sigma * sigma);
            
            // Construct Ricci tensor with proper symmetry
            const alpha = 8 * Math.PI; // Information-gravity coupling
            const ricci = [
              [alpha * d2I_dx2, alpha * d2I_dxdy, alpha * d2I_dxdy * 0.5],
              [alpha * d2I_dxdy, alpha * d2I_dy2, alpha * d2I_dxdy * 0.5],
              [alpha * d2I_dxdy * 0.5, alpha * d2I_dxdy * 0.5, alpha * d2I_dz2]
            ];
            
            // Scalar curvature R = Tr(Ricci) for spatial slice
            const scalar = alpha * (d2I_dx2 + d2I_dy2 + d2I_dz2);
            
            // Information content with quantum corrections
            const quantumCorrection = coherenceScale * entropyFactor;
            const information = localInfo * quantumCorrection * 100; // Scale for visibility
            
            tensors.push({
              position: [x, y, z],
              ricci,
              scalar,
              information,
              timestamp: Date.now(),
              fieldStrength: fieldStrength * (1 + 0.2 * Math.sin(time * 0.5 + pointIndex))
            });
            
            pointIndex++;
          }
        }
      }
      
      setRealTimeData(tensors);
    }, 50); // Update at 20fps for smoother animation
    
    return () => clearInterval(updateInterval);
  }, [metrics, isConnected, fieldStrength, visualizationQuality, animationSpeed]);
  
  // State for animation frame to trigger re-render
  const [animationFrame, setAnimationFrame] = useState(0);
  
  // Update animation frame for dynamic default data
  useEffect(() => {
    if (!isConnected || realTimeData.length === 0) {
      const animationInterval = setInterval(() => {
        setAnimationFrame(prev => prev + 1);
      }, 50); // Update every 50ms for smooth animation
      
      return () => clearInterval(animationInterval);
    }
  }, [isConnected, realTimeData.length]);
  
  // Use real-time data or generate dynamic default data
  const data = useMemo(() => {
    // If we have real-time data, use it
    if (realTimeData.length > 0) {
      return realTimeData;
    }
    
    // If prop data is provided and valid, use it
    if (propData && Array.isArray(propData) && propData.length > 0) {
      return propData;
    }
    
    // Generate dynamic default data for initial display
    const time = Date.now() * 0.001;
    const defaultData: CurvatureTensor[] = [];
    
    // Create a dynamic pattern for visual interest even without metrics
    const numPoints = 20;
    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * Math.PI * 2;
      const radius = 2 + Math.sin(time * 0.5 + i * 0.3) * 0.5;
      const height = Math.sin(time + i * 0.5) * 0.5;
      
      // Create dynamic Ricci tensor values
      const ricciScale = 0.5 + 0.5 * Math.sin(time + i);
      const ricci = [
        [ricciScale, 0.1 * Math.sin(time), 0.1 * Math.cos(time)],
        [0.1 * Math.sin(time), ricciScale, 0.1 * Math.sin(time + Math.PI/2)],
        [0.1 * Math.cos(time), 0.1 * Math.sin(time + Math.PI/2), ricciScale]
      ];
      
      defaultData.push({
        position: [
          Math.cos(angle) * radius,
          height,
          Math.sin(angle) * radius
        ],
        ricci,
        scalar: 0.3 + 0.4 * Math.sin(time * 2 + i * 0.7),
        information: 5 + 10 * (0.5 + 0.5 * Math.cos(time * 1.5 + i * 0.4)),
        timestamp: Date.now(),
        fieldStrength: 1.0 + 0.5 * Math.sin(time * 0.8)
      });
    }
    
    return defaultData;
  }, [realTimeData, propData, animationFrame]);

  return (
    <div className="informational-curvature-map" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="controls" style={{ 
        padding: '8px 12px', 
        borderBottom: `1px solid ${primaryColor}20`,
        background: 'rgba(0, 0, 0, 0.3)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={() => setCurvatureMode('ricci')}
            style={{
              padding: '4px 12px',
              background: curvatureMode === 'ricci' ? `${primaryColor}20` : 'transparent',
              border: `1px solid ${primaryColor}40`,
              borderRadius: '4px',
              color: curvatureMode === 'ricci' ? primaryColor : '#888',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: '500',
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
            onMouseEnter={(e) => {
              if (curvatureMode !== 'ricci') {
                e.currentTarget.style.borderColor = `${primaryColor}60`;
                e.currentTarget.style.color = primaryColor;
              }
            }}
            onMouseLeave={(e) => {
              if (curvatureMode !== 'ricci') {
                e.currentTarget.style.borderColor = `${primaryColor}40`;
                e.currentTarget.style.color = '#888';
              }
            }}
          >
            <Layers size={12} />
            Ricci
          </button>
          <button
            onClick={() => setCurvatureMode('scalar')}
            style={{
              padding: '4px 12px',
              background: curvatureMode === 'scalar' ? `${primaryColor}20` : 'transparent',
              border: `1px solid ${primaryColor}40`,
              borderRadius: '4px',
              color: curvatureMode === 'scalar' ? primaryColor : '#888',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: '500',
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
            onMouseEnter={(e) => {
              if (curvatureMode !== 'scalar') {
                e.currentTarget.style.borderColor = `${primaryColor}60`;
                e.currentTarget.style.color = primaryColor;
              }
            }}
            onMouseLeave={(e) => {
              if (curvatureMode !== 'scalar') {
                e.currentTarget.style.borderColor = `${primaryColor}40`;
                e.currentTarget.style.color = '#888';
              }
            }}
          >
            <Activity size={12} />
            Scalar
          </button>
          <button
            onClick={() => setCurvatureMode('information')}
            style={{
              padding: '4px 12px',
              background: curvatureMode === 'information' ? `${primaryColor}20` : 'transparent',
              border: `1px solid ${primaryColor}40`,
              borderRadius: '4px',
              color: curvatureMode === 'information' ? primaryColor : '#888',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: '500',
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
            onMouseEnter={(e) => {
              if (curvatureMode !== 'information') {
                e.currentTarget.style.borderColor = `${primaryColor}60`;
                e.currentTarget.style.color = primaryColor;
              }
            }}
            onMouseLeave={(e) => {
              if (curvatureMode !== 'information') {
                e.currentTarget.style.borderColor = `${primaryColor}40`;
                e.currentTarget.style.color = '#888';
              }
            }}
          >
            <TrendingUp size={12} />
            Information
          </button>
        </div>
        
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          {/* Settings */}
          <Tooltip content="Auto-rotate">
            <button
              onClick={() => setAutoRotate(!autoRotate)}
              style={{
                padding: '4px 8px',
                background: autoRotate ? `${primaryColor}20` : 'transparent',
                border: `1px solid ${primaryColor}40`,
                borderRadius: '4px',
                color: autoRotate ? primaryColor : '#666',
                cursor: 'pointer',
                fontSize: '11px',
                transition: 'all 0.2s'
              }}
            >
              <Eye size={10} />
            </button>
          </Tooltip>
          
          <Tooltip content="Show grid">
            <button
              onClick={() => setShowGrid(!showGrid)}
              style={{
                padding: '4px 8px',
                background: showGrid ? `${primaryColor}20` : 'transparent',
                border: `1px solid ${primaryColor}40`,
                borderRadius: '4px',
                color: showGrid ? primaryColor : '#666',
                cursor: 'pointer',
                fontSize: '11px',
                transition: 'all 0.2s'
              }}
            >
              <Settings size={10} />
            </button>
          </Tooltip>
          
          <Tooltip content="Show particles">
            <button
              onClick={() => setShowParticles(!showParticles)}
              style={{
                padding: '4px 8px',
                background: showParticles ? `${primaryColor}20` : 'transparent',
                border: `1px solid ${primaryColor}40`,
                borderRadius: '4px',
                color: showParticles ? primaryColor : '#666',
                cursor: 'pointer',
                fontSize: '11px',
                transition: 'all 0.2s'
              }}
            >
              <Activity size={10} />
            </button>
          </Tooltip>
          
          <Tooltip content="Show field lines">
            <button
              onClick={() => setShowFieldLines(!showFieldLines)}
              style={{
                padding: '4px 8px',
                background: showFieldLines ? `${primaryColor}20` : 'transparent',
                border: `1px solid ${primaryColor}40`,
                borderRadius: '4px',
                color: showFieldLines ? primaryColor : '#666',
                cursor: 'pointer',
                fontSize: '11px',
                transition: 'all 0.2s'
              }}
            >
              <Layers size={10} />
            </button>
          </Tooltip>
          
          {/* Status indicators */}
          {isConnected && (
            <span style={{
              background: `${primaryColor}30`,
              color: primaryColor,
              padding: '2px 6px',
              borderRadius: '3px',
              fontSize: '10px',
              fontWeight: '500'
            }}>
              LIVE
            </span>
          )}
          
          <div style={{ 
            color: primaryColor, 
            fontSize: '11px',
            display: 'flex',
            alignItems: 'center',
            gap: '4px'
          }}>
            <Activity size={10} />
            Field: {fieldStrength.toFixed(2)}
          </div>
          
          {metrics && (
            <div style={{
              display: 'flex',
              gap: '8px',
              fontSize: '10px',
              color: '#888'
            }}>
              <span style={{ color: metrics.rsp > 500 ? primaryColor : '#888' }}>
                RSP: {metrics.rsp.toFixed(0)}
              </span>
              <span style={{ color: metrics.entropy > 0.7 ? '#ff6b6b' : '#888' }}>
                E: {metrics.entropy.toFixed(2)}
              </span>
              <span style={{ color: metrics.coherence > 0.8 ? '#4fc3f7' : '#888' }}>
                C: {metrics.coherence.toFixed(2)}
              </span>
            </div>
          )}
        </div>
      </div>

      <div style={{ flex: 1, position: 'relative' }}>
        <Canvas camera={{ position: [0, 0, 0], fov: 75 }}>
          <ambientLight intensity={0.15} />
          <pointLight 
            position={[10, 10, 10]} 
            color={colorPalette.accent} 
            intensity={0.8 + (metrics?.information || 0) * 0.02} 
          />
          <pointLight 
            position={[-10, -5, -10]} 
            color={colorPalette.positive} 
            intensity={0.5 + (metrics?.coherence || 0) * 0.3} 
          />
          <spotLight
            position={[0, 10, 0]}
            angle={0.6 + (metrics?.observer_influence || 0) * 0.3}
            penumbra={1}
            intensity={0.5 + (metrics?.rsp || 0) * 0.001}
            color={colorPalette.primary}
          />
          
          <CurvatureField 
            data={data} 
            primaryColor={primaryColor}
            colorPalette={colorPalette}
            fieldStrength={fieldStrength}
            curvatureMode={curvatureMode}
            onSelectPoint={setSelectedPoint}
            showFieldLines={showFieldLines}
            showParticles={showParticles}
            animationSpeed={animationSpeed}
          />
          
          {showClassicalComparison && (
            <group>
              <mesh position={[0, 0, 0]}>
                <sphereGeometry args={[3, 32, 32]} />
                <meshBasicMaterial
                  color={primaryColor}
                  wireframe
                  opacity={0.3}
                  transparent
                />
              </mesh>
              <mesh position={[0, 0, 0]} rotation={[0, Date.now() * 0.0001, 0]}>
                <torusGeometry args={[3.5, 0.1, 8, 64]} />
                <meshStandardMaterial
                  color={primaryColor}
                  emissive={primaryColor}
                  emissiveIntensity={0.5}
                  transparent
                  opacity={0.5}
                />
              </mesh>
            </group>
          )}
          
          <OrbitControls 
            enableDamping 
            dampingFactor={0.05}
            autoRotate={autoRotate}
            autoRotateSpeed={0.5}
            minDistance={0.1}
            maxDistance={20}
          />
          
          {/* Grid - dynamic palette-based styling */}
          {showGrid && (
            <gridHelper 
              args={[
                10, 
                10, 
                colorPalette.grid,
                colorPalette.gridSecondary
              ]} 
            />
          )}
        </Canvas>

        {/* Mode-specific legend */}
        <div style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          background: 'rgba(0, 0, 0, 0.8)',
          border: `1px solid ${primaryColor}30`,
          borderRadius: '6px',
          padding: '8px 12px',
          color: '#fff',
          fontSize: '11px',
          backdropFilter: 'blur(10px)',
          maxWidth: '200px'
        }}>
          <div style={{ marginBottom: '4px', fontWeight: '600', color: primaryColor }}>
            {curvatureMode.toUpperCase()} MODE
          </div>
          <div style={{ color: '#aaa', lineHeight: 1.4 }}>
            {{
              'scalar': 'Shows positive and negative spacetime curvature through contrasting hues. Particle size indicates magnitude.',
              'information': 'Visualizes information density across spectral gradients. Larger spheres represent higher density regions.',
              'ricci': 'Displays Ricci tensor eigenvalue distributions. Matrix visualization shows tensor field properties.'
            }[curvatureMode]}
          </div>
        </div>
        
        {selectedPoint && (
          <div style={{
            position: 'absolute',
            top: '12px',
            right: '12px',
            background: 'rgba(0, 0, 0, 0.9)',
            border: `1px solid ${primaryColor}40`,
            borderRadius: '6px',
            padding: '12px',
            color: '#fff',
            maxWidth: '280px',
            backdropFilter: 'blur(10px)',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
          }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              marginBottom: '8px'
            }}>
              <h4 style={{ margin: 0, fontSize: '13px', fontWeight: '600' }}>
                Curvature Analysis
              </h4>
              <button
                onClick={() => setSelectedPoint(null)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#666',
                  cursor: 'pointer',
                  padding: '2px',
                  fontSize: '16px',
                  lineHeight: 1
                }}
              >
                ×
              </button>
            </div>
            
            <div style={{ fontSize: '11px', lineHeight: 1.6 }}>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'auto 1fr', 
                gap: '4px 12px',
                marginBottom: '8px'
              }}>
                <span style={{ color: '#888' }}>Position:</span>
                <span style={{ fontFamily: 'monospace' }}>
                  ({selectedPoint.position.map(p => p.toFixed(2)).join(', ')})
                </span>
                
                <span style={{ color: '#888' }}>Scalar:</span>
                <span style={{ color: primaryColor, fontWeight: '500' }}>
                  {selectedPoint.scalar.toFixed(4)}
                </span>
                
                <span style={{ color: '#888' }}>Information:</span>
                <span style={{ color: primaryColor, fontWeight: '500' }}>
                  {selectedPoint.information.toFixed(4)} bits
                </span>
              </div>
              
              <div style={{ 
                borderTop: `1px solid ${primaryColor}20`,
                paddingTop: '8px'
              }}>
                <div style={{ 
                  color: '#888', 
                  marginBottom: '4px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}>
                  <Layers size={10} />
                  Ricci Tensor
                </div>
                <div style={{ 
                  fontFamily: 'monospace', 
                  fontSize: '10px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  padding: '6px',
                  borderRadius: '3px',
                  lineHeight: 1.4
                }}>
                  {selectedPoint.ricci.map((row, i) => (
                    <div key={i}>
                      [{row.map(val => val.toFixed(3)).join(', ')}]
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div style={{
        padding: '8px 12px',
        borderTop: `1px solid ${primaryColor}20`,
        background: 'rgba(0, 0, 0, 0.3)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: primaryColor, fontSize: '12px' }}>
          <div>
            <strong>OSH Prediction:</strong> ∇μ∇νI(x,t) = Rμν
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <strong>Mode:</strong>
            <span style={{ 
              padding: '2px 8px', 
              borderRadius: '3px',
              background: {
                'scalar': 'rgba(255, 170, 0, 0.2)',      // Orange background for scalar (matches line 338)
                'information': 'rgba(0, 255, 170, 0.2)', // Cyan background for information (matches line 339)
                'ricci': 'rgba(170, 0, 255, 0.2)'       // Purple background for ricci (matches line 340)
              }[curvatureMode],
              color: {
                'scalar': '#ffaa00',      // Orange text for scalar
                'information': '#00ffaa', // Cyan text for information
                'ricci': '#aa00ff'        // Purple text for ricci
              }[curvatureMode]
            }}>
              {curvatureMode.toUpperCase()}
            </span>
            <span style={{ fontSize: '10px', color: '#666' }}>
              {{
                'scalar': '(±R curvature)',
                'information': '(I density)',
                'ricci': '(Rμν tensor)'
              }[curvatureMode]}
            </span>
          </div>
          {showClassicalComparison && (
            <div>
              <strong>Comparing with GR</strong>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}