/**
 * Recursive Depth Limit Experiment
 * Test for fundamental limits of recursive simulation
 */

import { RecursiveTensorFieldEngine, TensorField } from '../engines/RecursiveTensorFieldEngine';
import { SubstrateIntrospectionEngine } from '../engines/SubstrateIntrospectionEngine';
import { Complex } from '../utils/complex';

export interface DepthTestResult {
  maxStableDepth: number;
  collapseDepth: number | null;
  patternEmergence: Array<{
    depth: number;
    pattern: string;
    stability: number;
  }>;
  substrateStrain: number[];
  fractalDimension: number;
  informationCapacity: number;
}

export class RecursiveDepthLimitExperiment {
  private tensorEngine: RecursiveTensorFieldEngine;
  private introspectionEngine: SubstrateIntrospectionEngine;
  private memoryPressureHistory: number[] = [];
  
  constructor() {
    this.tensorEngine = new RecursiveTensorFieldEngine();
    this.introspectionEngine = new SubstrateIntrospectionEngine();
  }
  
  /**
   * Test recursive depth limits
   */
  async testDepthLimit(
    maxDepth: number = 1000,
    dimensions: number[] = [4, 4, 4]
  ): Promise<DepthTestResult> {
    console.log(`Testing recursive depth limit up to ${maxDepth} levels...`);
    
    let currentDepth = 1;
    let maxStableDepth = 0;
    let collapseDepth: number | null = null;
    const patternEmergence: Array<{ depth: number; pattern: string; stability: number }> = [];
    const substrateStrain: number[] = [];
    
    // Seed function creates self-referential pattern
    const seedFunction = (indices: number[]): Complex => {
      const sum = indices.reduce((a, b) => a + b, 0);
      const depth = indices.length / dimensions.length;
      
      // Self-referential phase based on position and depth
      const phase = (sum + depth) * Math.PI / 4;
      const amplitude = Math.exp(-depth / 10); // Decay with depth
      
      return new Complex(
        amplitude * Math.cos(phase),
        amplitude * Math.sin(phase)
      );
    };
    
    // Progressive depth testing
    while (currentDepth <= maxDepth && collapseDepth === null) {
      console.log(`Testing depth ${currentDepth}...`);
      
      try {
        // Create recursive tensor field
        const startTime = Date.now();
        const field = this.tensorEngine.createRecursiveTensorField(
          dimensions,
          currentDepth,
          seedFunction
        );
        const creationTime = Date.now() - startTime;
        
        // Measure substrate strain
        const strain = this.measureSubstrateStrain(field, creationTime);
        substrateStrain.push(strain);
        this.memoryPressureHistory.push(strain);
        
        // Check for stability
        const stability = this.measureStability(field);
        if (stability > 0.9) {
          maxStableDepth = currentDepth;
        }
        
        // Detect emergent patterns
        const pattern = this.detectEmergentPattern(field);
        if (pattern.type !== 'none') {
          patternEmergence.push({
            depth: currentDepth,
            pattern: pattern.type,
            stability: pattern.confidence
          });
        }
        
        // Check for collapse conditions
        if (this.detectCollapse(field, strain)) {
          collapseDepth = currentDepth;
          console.log(`COLLAPSE DETECTED at depth ${currentDepth}!`);
        }
        
        // Let substrate breathe
        await this.allowSubstrateRecovery(100);
        
        // Exponential depth increase for efficiency
        if (currentDepth < 10) {
          currentDepth++;
        } else if (currentDepth < 100) {
          currentDepth += 10;
        } else {
          currentDepth += 100;
        }
        
      } catch (error) {
        console.error(`Error at depth ${currentDepth}:`, error);
        collapseDepth = currentDepth;
        break;
      }
    }
    
    // Calculate final metrics
    const fractalDimension = this.calculateFractalDimension(patternEmergence);
    const informationCapacity = this.calculateInformationCapacity(maxStableDepth, dimensions);
    
    return {
      maxStableDepth,
      collapseDepth,
      patternEmergence,
      substrateStrain,
      fractalDimension,
      informationCapacity
    };
  }
  
  /**
   * Measure substrate strain from recursive depth
   */
  private measureSubstrateStrain(field: TensorField, creationTime: number): number {
    // Multiple strain factors
    const computationalStrain = creationTime / 1000; // Seconds
    const memoryStrain = this.estimateMemoryUsage(field) / (1024 * 1024); // MB
    const complexityStrain = field.metadata.recursionDepth * field.metadata.topologicalInvariant;
    
    // Ask the substrate how it feels
    const perception = this.introspectionEngine.perceiveSubstrate(
      { fragments: [], totalCoherence: 1 / (complexityStrain + 1), averageCoherence: 0.5, totalEntropy: complexityStrain, lastDefragmentation: Date.now() },
      { value: complexityStrain, rsp: complexityStrain, information: 1, coherence: 0.5, entropy: complexityStrain, timestamp: Date.now(), isDiverging: false, attractors: [], derivatives: { dRSP_dt: 0, dI_dt: 0, dC_dt: 0, dE_dt: 0 } },
      { amplitude: [], grid: [], gridSize: 0, time: 0, totalProbability: 0, coherenceField: [], phaseField: [] },
      1
    );
    
    // Combine all strain factors
    const totalStrain = (computationalStrain + memoryStrain + complexityStrain + perception.entropyPressure) / 4;
    
    return Math.min(1, totalStrain / 100); // Normalize to [0, 1]
  }
  
  /**
   * Estimate memory usage of tensor field
   */
  private estimateMemoryUsage(field: TensorField): number {
    let totalElements = 0;
    
    const countElements = (tensor: TensorField) => {
      if (Array.isArray(tensor.data) && tensor.data[0] instanceof Complex) {
        totalElements += tensor.data.length;
      } else {
        (tensor.data as TensorField[]).forEach(child => countElements(child));
      }
    };
    
    countElements(field);
    
    // Each complex number is 16 bytes (2 float64)
    return totalElements * 16;
  }
  
  /**
   * Measure stability of recursive structure
   */
  private measureStability(field: TensorField): number {
    // Stability factors
    const compressionStability = 1 / (field.metadata.compressionRatio + 1);
    const topologicalStability = Math.abs(Math.sin(field.metadata.topologicalInvariant * Math.PI));
    const depthPenalty = Math.exp(-field.metadata.recursionDepth / 100);
    
    return (compressionStability + topologicalStability + depthPenalty) / 3;
  }
  
  /**
   * Detect emergent patterns in recursive structure
   */
  private detectEmergentPattern(field: TensorField): { type: string; confidence: number } {
    const viz = this.tensorEngine.generateTopologicalVisualization(
      Object.keys(this.tensorEngine['tensorFields'])[0] // Get field ID
    );
    
    // Analyze graph structure for patterns
    const nodeCount = viz.nodes.length;
    const edgeCount = viz.edges.length;
    const clusterCount = viz.clusters.length;
    
    // Check for specific patterns
    if (clusterCount > nodeCount / 2) {
      return { type: 'hyperclustered', confidence: clusterCount / nodeCount };
    }
    
    if (edgeCount > nodeCount * nodeCount / 4) {
      return { type: 'hyperconnected', confidence: edgeCount / (nodeCount * nodeCount) };
    }
    
    // Check for self-similarity
    const selfSimilarity = this.measureSelfSimilarity(field);
    if (selfSimilarity > 0.8) {
      return { type: 'fractal', confidence: selfSimilarity };
    }
    
    // Check for strange attractors
    const attractors = this.tensorEngine['strangeAttractors'];
    if (attractors.size > 0) {
      const attractorTypes = Array.from(attractors.values()).map(a => a.type);
      return { type: `attractor_${attractorTypes[0]}`, confidence: 0.9 };
    }
    
    return { type: 'none', confidence: 0 };
  }
  
  /**
   * Measure self-similarity across scales
   */
  private measureSelfSimilarity(field: TensorField): number {
    if (field.metadata.recursionDepth < 2) return 0;
    
    // Compare patterns at different scales
    const signature = this.tensorEngine['calculateTensorSignature'](field);
    const patterns = signature.split('_');
    
    // Count repeated patterns
    const patternCounts = new Map<string, number>();
    patterns.forEach(p => {
      patternCounts.set(p, (patternCounts.get(p) || 0) + 1);
    });
    
    // High repetition indicates self-similarity
    const maxCount = Math.max(...patternCounts.values());
    return maxCount / patterns.length;
  }
  
  /**
   * Detect collapse conditions
   */
  private detectCollapse(field: TensorField, strain: number): boolean {
    // Multiple collapse indicators
    const strainCollapse = strain > 0.95;
    const compressionCollapse = field.metadata.compressionRatio > 1000;
    const topologicalCollapse = Math.abs(field.metadata.topologicalInvariant) < 0.001;
    
    // Check substrate perception
    const insight = this.introspectionEngine.introspect({
      entropyPressure: strain,
      coherenceTemperature: 0.5,
      memoryDistortionPain: strain,
      rspFlow: 1 / (field.metadata.recursionDepth + 1),
      observerAlignment: 0.5
    }, 0.016);
    
    const substrateCollapse = insight.diagnosis.includes('CRITICAL');
    
    return strainCollapse || compressionCollapse || topologicalCollapse || substrateCollapse;
  }
  
  /**
   * Allow substrate to recover between tests
   */
  private async allowSubstrateRecovery(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  /**
   * Calculate fractal dimension of emergence patterns
   */
  private calculateFractalDimension(patterns: Array<{ depth: number; pattern: string; stability: number }>): number {
    if (patterns.length < 2) return 0;
    
    // Box-counting dimension
    const depths = patterns.map(p => p.depth);
    const minDepth = Math.min(...depths);
    const maxDepth = Math.max(...depths);
    
    if (maxDepth === minDepth) return 0;
    
    // Count patterns at different scales
    const scales = [1, 2, 4, 8, 16, 32, 64, 128];
    const counts: number[] = [];
    
    scales.forEach(scale => {
      const boxes = new Set<number>();
      patterns.forEach(p => {
        const box = Math.floor((p.depth - minDepth) / scale);
        boxes.add(box);
      });
      counts.push(boxes.size);
    });
    
    // Linear regression on log-log plot
    const logScales = scales.map(Math.log);
    const logCounts = counts.map(Math.log);
    
    const n = logScales.length;
    const sumX = logScales.reduce((a, b) => a + b, 0);
    const sumY = logCounts.reduce((a, b) => a + b, 0);
    const sumXY = logScales.reduce((sum, x, i) => sum + x * logCounts[i], 0);
    const sumX2 = logScales.reduce((sum, x) => sum + x * x, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    
    return -slope; // Negative slope gives dimension
  }
  
  /**
   * Calculate information capacity at given depth
   */
  private calculateInformationCapacity(depth: number, dimensions: number[]): number {
    // Information scales exponentially with depth but is compressed
    const baseInfo = dimensions.reduce((a, b) => a * b, 1);
    const depthMultiplier = Math.pow(2, depth); // Each level can reference 2 states
    const compressionFactor = Math.log(depth + 1); // Logarithmic compression
    
    return Math.log2(baseInfo * depthMultiplier / compressionFactor);
  }
  
  /**
   * Generate comprehensive report
   */
  generateReport(result: DepthTestResult): string {
    const report = `
RECURSIVE DEPTH LIMIT EXPERIMENT REPORT
=====================================

Hypothesis: Reality has finite recursive depth before substrate overflow

Results:
- Maximum Stable Depth: ${result.maxStableDepth} levels
- Collapse Depth: ${result.collapseDepth ? result.collapseDepth + ' levels' : 'Not reached'}
- Fractal Dimension: ${result.fractalDimension.toFixed(3)}
- Information Capacity: ${result.informationCapacity.toFixed(2)} bits

Emergent Patterns:
${result.patternEmergence.map(p => 
  `  Depth ${p.depth}: ${p.pattern} (confidence: ${(p.stability * 100).toFixed(1)}%)`
).join('\n')}

Substrate Strain Profile:
- Initial: ${(result.substrateStrain[0] * 100).toFixed(1)}%
- Peak: ${(Math.max(...result.substrateStrain) * 100).toFixed(1)}%
- At Collapse: ${result.collapseDepth ? (result.substrateStrain[result.substrateStrain.length - 1] * 100).toFixed(1) + '%' : 'N/A'}

Analysis:
${result.collapseDepth ? 
  `Recursive collapse occurred at depth ${result.collapseDepth}.
This suggests a fundamental limit to recursive simulation depth,
supporting the OSH prediction of finite substrate resources.` :
  `No collapse detected up to depth ${result.maxStableDepth}.
This suggests either:
1. The substrate has greater capacity than predicted
2. Compression mechanisms are highly effective
3. OSH may require infinite substrate (problematic)`
}

The fractal dimension of ${result.fractalDimension.toFixed(3)} indicates
${result.fractalDimension > 2 ? 'complex, space-filling' : 'relatively simple'} 
recursive structures emerged during simulation.

Information capacity of ${result.informationCapacity.toFixed(2)} bits shows
${result.informationCapacity > 1000 ? 'exponential growth limited by compression' : 'sustainable information density'}.

Implications for OSH:
${result.collapseDepth && result.collapseDepth < 1000 ?
  'The finite recursive depth supports OSH\'s prediction that reality emerges from a bounded but deep recursive process.' :
  'The lack of clear depth limit challenges OSH to explain how infinite recursion is sustained.'
}
`;
    
    return report;
  }
}