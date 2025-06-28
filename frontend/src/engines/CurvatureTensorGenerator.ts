/**
 * Curvature Tensor Generator
 * Generates real-time information curvature data from memory fields
 */

import { Complex } from '../utils/complex';
import { MemoryField, MemoryFragment } from './MemoryFieldEngine';
import { ExtendedMemoryField, ExtendedMemoryFragment, toExtendedMemoryField, toExtendedMemoryFragment, BaseEngine } from '../types/engine-types';

export interface CurvatureTensor {
  position: [number, number, number];
  ricci: number[][];
  scalar: number;
  information: number;
  timestamp?: number;
  fieldStrength?: number;
}

export class CurvatureTensorGenerator implements BaseEngine {
  private gridResolution: number = 20;
  private curvatureCache: Map<string, CurvatureTensor> = new Map();
  private lastUpdateTime: number = 0;
  private cacheTimeout: number = 100; // Cache timeout in ms
  
  /**
   * Generate curvature tensors from memory field
   */
  generateFromMemoryField(memoryField: MemoryField): CurvatureTensor[] {
    // Convert to extended type
    const extendedField = toExtendedMemoryField(memoryField);
    const tensors: CurvatureTensor[] = [];
    
    try {
      // Validate memory field
      if (!memoryField || !Array.isArray(memoryField.fragments)) {
        console.warn('Invalid memory field provided to curvature generator');
        return tensors;
      }
      
      // Create spatial grid
      const bounds = this.calculateFieldBounds(memoryField.fragments);
      const step = (bounds.max - bounds.min) / this.gridResolution;
      
      // Validate step size
      if (!isFinite(step) || step <= 0) {
        console.warn('Invalid grid step calculated, using default');
        return this.generateDefaultTensors();
      }
      
      // Dynamic resolution based on field properties
      const infoScale = Math.min(1.5, (extendedField.informationFlow || 1) / 10);
      const dynamicResolution = Math.floor(5 + infoScale * 10);
      const reducedResolution = Math.min(this.gridResolution, dynamicResolution);
      
      // Focus grid generation on high-activity regions
      const activeRegions = this.findActiveRegions(memoryField.fragments);
      
      // Generate tensors with focus on active regions
      activeRegions.forEach((region, regionIndex) => {
        const regionStep = region.radius / 5;
        
        for (let r = 0; r < 5; r++) {
          const radius = r * regionStep;
          const angleStep = Math.PI / (3 + r); // More points at larger radii
          
          for (let theta = 0; theta < Math.PI * 2; theta += angleStep) {
            const position: [number, number, number] = [
              region.center[0] + radius * Math.cos(theta),
              region.center[1] + Math.sin(regionIndex + Date.now() * 0.001) * (extendedField.coherence || 0.5),
              region.center[2] + radius * Math.sin(theta)
            ];
            
            try {
              const tensor = this.calculateCurvatureAt(position, memoryField);
              if (tensor) {
                // Scale significance threshold based on field properties
                const threshold = 0.001 * (1 / ((extendedField.strain || 0.1) + 0.1));
                if (Math.abs(tensor.scalar) > threshold || tensor.information > 0.1) {
                  tensors.push(tensor);
                }
              }
            } catch (error) {
              console.warn('Error calculating tensor at position:', position, error);
            }
          }
        }
      });
      
      // Add boundary tensors for visual continuity
      const boundaryCount = Math.min(10, Math.floor(extendedField.recursiveDepth || 5));
      for (let i = 0; i < boundaryCount; i++) {
        const angle = (i / boundaryCount) * Math.PI * 2;
        const boundaryRadius = bounds.max * 1.2;
        
        const position: [number, number, number] = [
          Math.cos(angle) * boundaryRadius,
          Math.sin(Date.now() * 0.001 + i) * (extendedField.coherence || 0.5),
          Math.sin(angle) * boundaryRadius
        ];
        
        try {
          const tensor = this.calculateCurvatureAt(position, memoryField);
          if (tensor) {
            tensor.scalar *= 0.5; // Reduce boundary influence
            tensors.push(tensor);
          }
        } catch (error) {
          console.warn('Error calculating boundary tensor:', error);
        }
      }
    } catch (error) {
      console.error('Error generating curvature tensors:', error);
      return this.generateDefaultTensors();
    }
    
    return tensors;
  }
  
  /**
   * Generate default tensors for fallback
   */
  private generateDefaultTensors(): CurvatureTensor[] {
    const defaultTensors: CurvatureTensor[] = [];
    
    // Create a simple default tensor at origin
    defaultTensors.push({
      position: [0, 0, 0],
      ricci: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      scalar: 0.1,
      information: 1.0,
      timestamp: Date.now(),
      fieldStrength: 1.0
    });
    
    return defaultTensors;
  }
  
  /**
   * Calculate field bounds
   */
  private calculateFieldBounds(fragments: MemoryFragment[]): { min: number; max: number } {
    if (fragments.length === 0) return { min: -5, max: 5 };
    
    let min = Infinity;
    let max = -Infinity;
    
    fragments.forEach(fragment => {
      fragment.position.forEach(coord => {
        min = Math.min(min, coord);
        max = Math.max(max, coord);
      });
    });
    
    // Add padding
    const padding = (max - min) * 0.1;
    return { min: min - padding, max: max + padding };
  }
  
  /**
   * Calculate curvature tensor at a specific position
   */
  private calculateCurvatureAt(
    position: [number, number, number],
    memoryField: MemoryField
  ): CurvatureTensor {
    // Check cache with timeout
    const cacheKey = position.join(',');
    const now = Date.now();
    
    if (this.curvatureCache.has(cacheKey) && (now - this.lastUpdateTime) < this.cacheTimeout) {
      const cached = this.curvatureCache.get(cacheKey)!;
      // Add slight variation to cached values for dynamism
      const timeVariation = Math.sin(now * 0.001) * 0.1;
      cached.scalar *= (1 + timeVariation);
      cached.information *= (1 + timeVariation * 0.5);
      return cached;
    }
    
    this.lastUpdateTime = now;
    
    // Calculate metric tensor from nearby memory fragments
    const metric = this.calculateMetricTensor(position, memoryField.fragments);
    
    // Calculate Christoffel symbols
    const christoffel = this.calculateChristoffelSymbols(position, memoryField.fragments);
    
    // Calculate Ricci tensor
    const ricci = this.calculateRicciTensor(metric, christoffel);
    
    // Calculate scalar curvature
    const scalar = this.calculateScalarCurvature(metric, ricci);
    
    // Calculate information density
    const information = this.calculateInformationDensity(position, memoryField.fragments);
    
    const tensor: CurvatureTensor = {
      position,
      ricci,
      scalar,
      information,
      timestamp: Date.now(),
      fieldStrength: (toExtendedMemoryField(memoryField).informationFlow || 1.0)
    };
    
    // Cache result
    this.curvatureCache.set(cacheKey, tensor);
    
    return tensor;
  }
  
  /**
   * Calculate metric tensor induced by memory fragments
   */
  private calculateMetricTensor(
    position: [number, number, number],
    fragments: MemoryFragment[]
  ): number[][] {
    const metric = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ];
    
    // Each fragment warps spacetime based on its coherence and coupling strength
    fragments.forEach(fragment => {
      try {
        // Validate fragment
        if (!fragment || !fragment.position || !Array.isArray(fragment.position)) {
          return; // Skip invalid fragment
        }
        
        const distance = this.calculateDistance(position, fragment.position);
        if (distance < 5) { // Influence radius
          // Use fragment coherence and coupling strength for influence
          const coherence = Math.max(0, Math.min(1, fragment.coherence || 0));
          const extFragment = toExtendedMemoryFragment(fragment);
          const couplingStrength = Math.max(0, Math.min(1, extFragment.couplingStrength || 0.5));
          const influence = coherence * couplingStrength * Math.exp(-distance / 2);
          
          // Add perturbation to metric based on OSH theory
          // Information curvature: ∇μ∇νI(x,t) = Rμν
          for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
              const delta = (position[i] - fragment.position[i]) * 
                           (position[j] - fragment.position[j]) / 
                           (distance * distance + 0.01);
              
              // Include quantum phase effects with validation
              const phase = isFinite(extFragment.phase) ? extFragment.phase : 0;
              const phaseContribution = Math.cos(phase) * 0.1;
              
              const perturbation = influence * (1 + phaseContribution) * delta;
              
              // Only add if result is valid
              if (isFinite(perturbation)) {
                metric[i][j] += perturbation;
              }
            }
          }
        }
      } catch (error) {
        console.warn('Error processing fragment in metric calculation:', error);
      }
    });
    
    return metric;
  }
  
  /**
   * Calculate Christoffel symbols (simplified)
   */
  private calculateChristoffelSymbols(
    position: [number, number, number],
    fragments: MemoryFragment[]
  ): number[][][] {
    const christoffel: number[][][] = [];
    
    // Initialize
    for (let i = 0; i < 3; i++) {
      christoffel[i] = [];
      for (let j = 0; j < 3; j++) {
        christoffel[i][j] = [0, 0, 0];
      }
    }
    
    // Simplified calculation based on fragment distribution
    const h = 0.01; // Small displacement for derivatives
    
    for (let k = 0; k < 3; k++) {
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          // Numerical derivative approximation
          const posPlus = [...position];
          const posMinus = [...position];
          posPlus[k] += h;
          posMinus[k] -= h;
          
          const metricPlus = this.calculateMetricTensor(posPlus as [number, number, number], fragments);
          const metricMinus = this.calculateMetricTensor(posMinus as [number, number, number], fragments);
          
          christoffel[k][i][j] = (metricPlus[i][j] - metricMinus[i][j]) / (2 * h);
        }
      }
    }
    
    return christoffel;
  }
  
  /**
   * Calculate Ricci tensor
   */
  private calculateRicciTensor(
    metric: number[][],
    christoffel: number[][][]
  ): number[][] {
    const ricci: number[][] = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
    ];
    
    // Simplified Ricci tensor calculation
    // R_ij = ∂_k Γ^k_ij - ∂_j Γ^k_ik + Γ^k_kl Γ^l_ij - Γ^k_jl Γ^l_ik
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let sum = 0;
        
        // Contract over indices
        for (let k = 0; k < 3; k++) {
          for (let l = 0; l < 3; l++) {
            sum += christoffel[k][k][l] * christoffel[l][i][j] -
                   christoffel[k][j][l] * christoffel[l][i][k];
          }
        }
        
        ricci[i][j] = sum;
      }
    }
    
    return ricci;
  }
  
  /**
   * Calculate scalar curvature
   */
  private calculateScalarCurvature(metric: number[][], ricci: number[][]): number {
    // R = g^ij R_ij
    const metricInverse = this.invertMatrix3x3(metric);
    let scalar = 0;
    
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        scalar += metricInverse[i][j] * ricci[i][j];
      }
    }
    
    return scalar;
  }
  
  /**
   * Calculate information density at position
   */
  private calculateInformationDensity(
    position: [number, number, number],
    fragments: MemoryFragment[]
  ): number {
    let density = 0;
    
    // Validate fragments array
    if (!Array.isArray(fragments)) {
      return 0;
    }
    
    fragments.forEach(fragment => {
      try {
        // Validate fragment structure
        if (!fragment || !fragment.position) {
          return; // Skip invalid fragment
        }
        
        const extFragment = toExtendedMemoryFragment(fragment);
        const distance = this.calculateDistance(position, fragment.position);
        if (distance < 5) {
          // Information density based on OSH theory
          // I(x,t) = -Σ p(x,t) log p(x,t)
          const amplitude = extFragment.amplitude || fragment.coherence || 0.5;
          
          // Handle amplitude as number or complex
          let probability = 0;
          if (typeof amplitude === 'number') {
            probability = amplitude * amplitude;
          } else if (amplitude && typeof amplitude === 'object' && 'real' in amplitude && 'imag' in amplitude) {
            const complexAmp = amplitude as { real: number; imag: number };
            probability = complexAmp.real * complexAmp.real + complexAmp.imag * complexAmp.imag;
          } else {
            probability = 0.25; // Default value
          }
          
          // Information falls off with distance and is modulated by coherence
          const falloff = Math.exp(-distance * distance / 4);
          
          // Fragment information content with validation
          const entropy = this.calculateFragmentEntropy(fragment);
          const coherence = Math.max(0, Math.min(1, fragment.coherence || 0));
          const information = coherence * (1 - entropy);
          
          // Only add if result is valid
          if (isFinite(information) && isFinite(falloff)) {
            density += information * falloff;
          }
        }
      } catch (error) {
        console.warn('Error processing fragment in information density calculation:', error);
      }
    });
    
    return Math.max(0, density); // Ensure non-negative
  }
  
  /**
   * Calculate fragment entropy
   */
  private calculateFragmentEntropy(fragment: MemoryFragment): number {
    try {
      // Validate fragment
      if (!fragment) {
        return 1; // Return maximum entropy for invalid fragment
      }
      
      const extFragment = toExtendedMemoryFragment(fragment);
      const amplitude = extFragment.amplitude || fragment.coherence || 0.5;
      
      // Calculate probability based on amplitude type
      let prob = 0;
      if (typeof amplitude === 'number') {
        prob = amplitude * amplitude;
      } else if (amplitude && typeof amplitude === 'object' && 'real' in amplitude && 'imag' in amplitude) {
        const complexAmp = amplitude as { real: number; imag: number };
        if (isFinite(complexAmp.real) && isFinite(complexAmp.imag)) {
          prob = complexAmp.real * complexAmp.real + complexAmp.imag * complexAmp.imag;
        } else {
          prob = 0.25; // Default probability
        }
      } else {
        prob = 0.25; // Default probability
      }
      
      // Clamp probability to valid range
      const clampedProb = Math.max(0, Math.min(1, prob));
      
      if (clampedProb > 0 && clampedProb < 1) {
        // Shannon entropy: -p log p - (1-p) log (1-p)
        const entropy = -clampedProb * Math.log2(clampedProb) - 
                       (1 - clampedProb) * Math.log2(1 - clampedProb);
        return isFinite(entropy) ? entropy : 0;
      }
      
      // Return 0 entropy for deterministic states (p=0 or p=1)
      return 0;
    } catch (error) {
      console.warn('Error calculating fragment entropy:', error);
      return 1; // Return maximum entropy on error
    }
  }
  
  /**
   * Calculate distance between positions
   */
  private calculateDistance(p1: [number, number, number], p2: [number, number, number]): number {
    return Math.sqrt(
      Math.pow(p1[0] - p2[0], 2) +
      Math.pow(p1[1] - p2[1], 2) +
      Math.pow(p1[2] - p2[2], 2)
    );
  }
  
  /**
   * Invert 3x3 matrix
   */
  private invertMatrix3x3(m: number[][]): number[][] {
    const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    
    if (Math.abs(det) < 1e-10) {
      return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; // Return identity if singular
    }
    
    const invDet = 1 / det;
    
    return [
      [
        invDet * (m[1][1] * m[2][2] - m[1][2] * m[2][1]),
        invDet * (m[0][2] * m[2][1] - m[0][1] * m[2][2]),
        invDet * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
      ],
      [
        invDet * (m[1][2] * m[2][0] - m[1][0] * m[2][2]),
        invDet * (m[0][0] * m[2][2] - m[0][2] * m[2][0]),
        invDet * (m[0][2] * m[1][0] - m[0][0] * m[1][2])
      ],
      [
        invDet * (m[1][0] * m[2][1] - m[1][1] * m[2][0]),
        invDet * (m[0][1] * m[2][0] - m[0][0] * m[2][1]),
        invDet * (m[0][0] * m[1][1] - m[0][1] * m[1][0])
      ]
    ];
  }
  
  /**
   * Find active regions in the memory field
   */
  private findActiveRegions(fragments: MemoryFragment[]): Array<{center: [number, number, number], radius: number}> {
    const regions: Array<{center: [number, number, number], radius: number}> = [];
    
    if (fragments.length === 0) {
      // Default region at origin
      regions.push({ center: [0, 0, 0], radius: 3 });
      return regions;
    }
    
    // Cluster fragments into regions
    const clusterRadius = 2;
    const processed = new Set<number>();
    
    fragments.forEach((fragment, i) => {
      if (processed.has(i)) return;
      
      const cluster: MemoryFragment[] = [fragment];
      processed.add(i);
      
      // Find nearby fragments
      fragments.forEach((other, j) => {
        if (i !== j && !processed.has(j)) {
          const dist = this.calculateDistance(fragment.position, other.position);
          if (dist < clusterRadius) {
            cluster.push(other);
            processed.add(j);
          }
        }
      });
      
      // Calculate cluster center and radius
      if (cluster.length > 0) {
        const center: [number, number, number] = [0, 0, 0];
        cluster.forEach(f => {
          center[0] += f.position[0] / cluster.length;
          center[1] += f.position[1] / cluster.length;
          center[2] += f.position[2] / cluster.length;
        });
        
        let maxDist = 0;
        cluster.forEach(f => {
          const dist = this.calculateDistance(center, f.position);
          maxDist = Math.max(maxDist, dist);
        });
        
        regions.push({ center, radius: Math.max(1, maxDist * 1.5) });
      }
    });
    
    // Ensure we have at least one region
    if (regions.length === 0) {
      regions.push({ center: [0, 0, 0], radius: 3 });
    }
    
    return regions;
  }
  
  /**
   * Clear cache (call periodically to prevent memory issues)
   */
  clearCache(): void {
    this.curvatureCache.clear();
    this.lastUpdateTime = 0;
  }

  /**
   * Update method to implement BaseEngine interface
   */
  update(deltaTime: number, context?: any): void {
    // Clear cache periodically
    const now = Date.now();
    if (now - this.lastUpdateTime > this.cacheTimeout) {
      this.clearCache();
      this.lastUpdateTime = now;
    }
  }

  /**
   * Reset the generator
   */
  reset(): void {
    this.clearCache();
    this.gridResolution = 20;
  }

  /**
   * Get current state
   */
  getState(): any {
    return {
      cacheSize: this.curvatureCache.size,
      gridResolution: this.gridResolution,
      lastUpdateTime: this.lastUpdateTime
    };
  }
}