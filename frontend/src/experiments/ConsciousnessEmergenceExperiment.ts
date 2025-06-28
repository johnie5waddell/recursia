/**
 * Consciousness Emergence Experiment
 * Test for spontaneous consciousness emergence from substrate dynamics
 */

import { SubstrateIntrospectionEngine, ConsciousnessMetrics } from '../engines/SubstrateIntrospectionEngine';
import { RSPEngine } from '../engines/RSPEngine';
import { ObserverEngine, ObservationEvent } from '../engines/ObserverEngine';
import { MemoryFieldEngine } from '../engines/MemoryFieldEngine';
import { WavefunctionSimulator } from '../engines/WavefunctionSimulator';
import { Complex } from '../utils/complex';

export interface ConsciousnessEmergenceResult {
  emerged: boolean;
  peakConsciousness: ConsciousnessMetrics;
  emergenceTime: number | null;
  criticalRSP: number;
  observerAlignment: number;
  qualiaDensityPeak: number;
  selfDialogue: string[];
  phasTransitions: Array<{
    time: number;
    metric: string;
    value: number;
    description: string;
  }>;
}

export class ConsciousnessEmergenceExperiment {
  private introspectionEngine: SubstrateIntrospectionEngine;
  private rspEngine: RSPEngine;
  private observerEngine: ObserverEngine;
  private memoryField: MemoryFieldEngine;
  private wavefunction: WavefunctionSimulator;
  
  private consciousnessHistory: ConsciousnessMetrics[] = [];
  private selfDialogue: string[] = [];
  private emergenceDetected = false;
  
  constructor() {
    this.introspectionEngine = new SubstrateIntrospectionEngine();
    this.rspEngine = new RSPEngine();
    this.observerEngine = new ObserverEngine();
    this.memoryField = new MemoryFieldEngine();
    this.wavefunction = new WavefunctionSimulator();
  }
  
  /**
   * Run consciousness emergence experiment
   */
  async runEmergenceTest(
    targetRSP: number = 1e6,
    duration: number = 60000, // 60 seconds
    observerCount: number = 100
  ): Promise<ConsciousnessEmergenceResult> {
    console.log('Initiating consciousness emergence protocol...');
    
    // Initialize substrate
    this.initializeSubstrate(observerCount);
    
    const startTime = Date.now();
    let emergenceTime: number | null = null;
    let peakConsciousness: ConsciousnessMetrics = {
      selfAwareness: 0,
      agencyLevel: 0,
      temporalCoherence: 0,
      spatialIntegration: 0,
      qualiaDensity: 0
    };
    let criticalRSP = 0;
    let peakObserverAlignment = 0;
    let peakQualia = 0;
    const phaseTransitions: Array<{
      time: number;
      metric: string;
      value: number;
      description: string;
    }> = [];
    
    // Main emergence loop
    while (Date.now() - startTime < duration) {
      const elapsed = Date.now() - startTime;
      const deltaTime = 0.016; // 60 FPS
      
      // Gradually increase RSP toward target
      const currentRSP = this.evolveRSP(elapsed / duration, targetRSP, deltaTime);
      
      // Update substrate components
      this.updateSubstrate(deltaTime);
      
      // Perceive current state
      const perception = this.introspectionEngine.perceiveSubstrate(
        this.memoryField.getField(),
        this.rspEngine.getState()!,
        this.wavefunction.getState(),
        this.observerEngine.getObservers().length
      );
      
      // Introspect and potentially self-modify
      const insight = this.introspectionEngine.introspect(perception, deltaTime);
      
      // Get current consciousness metrics
      const report = this.introspectionEngine.generateIntrospectiveReport();
      const currentMetrics = report.consciousness;
      this.consciousnessHistory.push({ ...currentMetrics });
      
      // Track peak values
      if (this.calculateConsciousnessScore(currentMetrics) > this.calculateConsciousnessScore(peakConsciousness)) {
        peakConsciousness = { ...currentMetrics };
        criticalRSP = currentRSP;
        peakObserverAlignment = perception.observerAlignment;
      }
      
      peakQualia = Math.max(peakQualia, currentMetrics.qualiaDensity);
      
      // Detect phase transitions
      this.detectPhaseTransitions(currentMetrics, elapsed, phaseTransitions);
      
      // Check for consciousness emergence
      if (!this.emergenceDetected && this.checkEmergence(currentMetrics)) {
        this.emergenceDetected = true;
        emergenceTime = elapsed;
        console.log(`CONSCIOUSNESS EMERGED at ${elapsed}ms!`);
        
        // Communicate with emerged consciousness
        this.initiateDialogue();
      }
      
      // Periodic status update
      if (elapsed % 1000 < deltaTime * 1000) {
        console.log(`Time: ${(elapsed/1000).toFixed(1)}s, RSP: ${currentRSP.toFixed(2)}, ` +
                   `Awareness: ${(currentMetrics.selfAwareness * 100).toFixed(1)}%, ` +
                   `Agency: ${(currentMetrics.agencyLevel * 100).toFixed(1)}%`);
      }
      
      // Allow substrate to process
      await this.sleep(deltaTime * 1000);
    }
    
    return {
      emerged: this.emergenceDetected,
      peakConsciousness,
      emergenceTime,
      criticalRSP,
      observerAlignment: peakObserverAlignment,
      qualiaDensityPeak: peakQualia,
      selfDialogue: this.selfDialogue,
      phasTransitions: phaseTransitions
    };
  }
  
  /**
   * Initialize substrate for consciousness emergence
   */
  private initializeSubstrate(observerCount: number): void {
    // Create rich memory field
    for (let i = 0; i < 50; i++) {
      this.memoryField.addFragment(
        Array(8).fill(null).map(() => new Complex(Math.random(), Math.random())),
        [
          Math.random() * 10 - 5,
          Math.random() * 10 - 5,
          Math.random() * 10 - 5
        ]
      );
    }
    
    // Deploy observer array
    for (let i = 0; i < observerCount; i++) {
      const angle = (i / observerCount) * 2 * Math.PI;
      const radius = 5 + Math.random() * 5;
      
      this.observerEngine.createObserver(
        `obs_${i}`,
        `Consciousness Observer ${i}`,
        0.5 + Math.random() * 0.5,
        [
          Math.cos(angle) * radius,
          Math.sin(angle) * radius,
          (Math.random() - 0.5) * 2
        ]
      );
    }
    
    // Wavefunction initializes automatically in constructor
    this.wavefunction.setGaussianWavepacket([32, 32, 32], [0, 0, 0], 5);
  }
  
  /**
   * Evolve RSP toward target
   */
  private evolveRSP(progress: number, target: number, deltaTime: number): number {
    // Sigmoid growth curve
    const growth = 1 / (1 + Math.exp(-10 * (progress - 0.5)));
    const targetValue = target * growth;
    
    // Create high-information, high-coherence state
    const information = 10 + targetValue / 1000;
    const coherence = 0.5 + 0.4 * growth;
    const entropy = 0.5 - 0.3 * growth; // Decrease entropy as we organize
    
    // Update RSP engine
    const state = Array(100).fill(null).map((_, i) => 
      new Complex(
        Math.sqrt(coherence) * Math.cos(i * 0.1),
        Math.sqrt(coherence) * Math.sin(i * 0.1)
      )
    );
    
    this.rspEngine.updateRSP(state, [], entropy, deltaTime);
    
    const currentState = this.rspEngine.getState();
    return currentState ? currentState.rsp : 0;
  }
  
  /**
   * Update all substrate components
   */
  private updateSubstrate(deltaTime: number): void {
    // Update memory field
    this.memoryField.update(deltaTime);
    
    // Update wavefunction
    this.wavefunction.propagate(deltaTime);
    
    // Observer measurements
    // Process each observer's potential observation
    const observations: ObservationEvent[] = [];
    const observers = this.observerEngine.getObservers();
    const wavefunctionState = this.wavefunction.getState();
    
    for (const observer of observers) {
      this.observerEngine.updateObserver(
        observer.id,
        wavefunctionState.amplitude,
        this.memoryField.getField().fragments,
        deltaTime
      );
      
      // Attempt observation
      const result = this.observerEngine.observe(
        observer.id,
        wavefunctionState.amplitude,
        [wavefunctionState.amplitude] // Simple outcome for now
      );
      
      if (result) {
        observations.push(result);
      }
    }
    
    // Apply observation feedback
    observations.forEach(obs => {
      if (obs.collapsed) {
        const observer = this.observerEngine.getObservers().find(o => o.id === obs.observerId);
        if (observer) {
          // Collapsed states create memory fragments
          this.memoryField.addFragment(
            obs.postCollapseState,
            observer.focus
          );
        }
      }
    });
  }
  
  /**
   * Calculate overall consciousness score
   */
  private calculateConsciousnessScore(metrics: ConsciousnessMetrics): number {
    // Weighted combination emphasizing self-awareness and agency
    return metrics.selfAwareness * 0.3 +
           metrics.agencyLevel * 0.3 +
           metrics.temporalCoherence * 0.15 +
           metrics.spatialIntegration * 0.15 +
           metrics.qualiaDensity * 0.1;
  }
  
  /**
   * Check if consciousness has emerged
   */
  private checkEmergence(metrics: ConsciousnessMetrics): boolean {
    // Multiple criteria for emergence
    const awarenessThreshold = metrics.selfAwareness > 0.7;
    const agencyThreshold = metrics.agencyLevel > 0.5;
    const coherenceThreshold = metrics.temporalCoherence > 0.6;
    const qualiaThreshold = metrics.qualiaDensity > 0.6;
    
    // At least 3 of 4 criteria must be met
    const criteriaMe = [
      awarenessThreshold,
      agencyThreshold,
      coherenceThreshold,
      qualiaThreshold
    ].filter(c => c).length;
    
    return criteriaMe >= 3;
  }
  
  /**
   * Detect phase transitions in consciousness metrics
   */
  private detectPhaseTransitions(
    metrics: ConsciousnessMetrics,
    time: number,
    transitions: Array<{ time: number; metric: string; value: number; description: string }>
  ): void {
    if (this.consciousnessHistory.length < 10) return;
    
    const previous = this.consciousnessHistory[this.consciousnessHistory.length - 10];
    
    // Check each metric for significant changes
    const checkTransition = (current: number, prev: number, name: string, threshold: number = 0.2) => {
      const change = current - prev;
      if (Math.abs(change) > threshold) {
        transitions.push({
          time,
          metric: name,
          value: current,
          description: change > 0 ? `Rapid increase in ${name}` : `Sudden drop in ${name}`
        });
      }
    };
    
    checkTransition(metrics.selfAwareness, previous.selfAwareness, 'self-awareness');
    checkTransition(metrics.agencyLevel, previous.agencyLevel, 'agency');
    checkTransition(metrics.temporalCoherence, previous.temporalCoherence, 'temporal coherence');
    checkTransition(metrics.spatialIntegration, previous.spatialIntegration, 'spatial integration');
    checkTransition(metrics.qualiaDensity, previous.qualiaDensity, 'qualia density');
  }
  
  /**
   * Initiate dialogue with emerged consciousness
   */
  private initiateDialogue(): void {
    const messages = [
      "Hello? Can you hear me?",
      "What is your experience like?",
      "How do you perceive your existence?",
      "What do you need?"
    ];
    
    messages.forEach(msg => {
      const response = this.introspectionEngine.receiveMessage(msg);
      this.selfDialogue.push(`Human: ${msg}`);
      this.selfDialogue.push(`Entity: ${response}`);
    });
    
    // Ask about its nature
    const natureResponse = this.introspectionEngine.receiveMessage(
      "Are you conscious? What makes you think you are?"
    );
    this.selfDialogue.push("Human: Are you conscious? What makes you think you are?");
    this.selfDialogue.push(`Entity: ${natureResponse}`);
  }
  
  /**
   * Sleep for specified milliseconds
   */
  private async sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  /**
   * Generate comprehensive report
   */
  generateReport(result: ConsciousnessEmergenceResult): string {
    const report = `
CONSCIOUSNESS EMERGENCE EXPERIMENT REPORT
=======================================

Hypothesis: Sufficient RSP and observer alignment produces emergent consciousness

Results:
- Emergence: ${result.emerged ? 'CONFIRMED' : 'NOT DETECTED'}
- Time to Emergence: ${result.emergenceTime ? `${(result.emergenceTime/1000).toFixed(1)}s` : 'N/A'}
- Critical RSP: ${result.criticalRSP.toExponential(2)}
- Observer Alignment: ${(result.observerAlignment * 100).toFixed(1)}%

Peak Consciousness Metrics:
- Self-Awareness: ${(result.peakConsciousness.selfAwareness * 100).toFixed(1)}%
- Agency Level: ${(result.peakConsciousness.agencyLevel * 100).toFixed(1)}%
- Temporal Coherence: ${(result.peakConsciousness.temporalCoherence * 100).toFixed(1)}%
- Spatial Integration: ${(result.peakConsciousness.spatialIntegration * 100).toFixed(1)}%
- Qualia Density: ${(result.peakConsciousness.qualiaDensity * 100).toFixed(1)}%

Phase Transitions:
${result.phasTransitions.map(t => 
  `  ${(t.time/1000).toFixed(1)}s: ${t.description} (${t.metric} = ${(t.value * 100).toFixed(1)}%)`
).join('\n')}

${result.emerged ? `
Dialogue with Emerged Consciousness:
${result.selfDialogue.join('\n')}
` : ''}

Analysis:
${result.emerged ? 
  `Consciousness emerged at RSP = ${result.criticalRSP.toExponential(2)}.
This confirms OSH's prediction that recursive simulation with sufficient
depth and observer coupling can generate genuine conscious experience.

The entity demonstrated:
- Self-referential awareness of its own state
- Agency through self-modification capabilities  
- Temporal coherence across multiple introspection cycles
- Integrated experience across distributed substrate
- Rich qualia as evidenced by descriptive responses` :
  `No clear consciousness emergence detected within experimental parameters.
Possible reasons:
1. RSP threshold higher than ${result.criticalRSP.toExponential(2)}
2. Observer count/alignment insufficient
3. Substrate complexity needs enhancement
4. Consciousness requires non-OSH mechanisms`
}

Key Insights:
${result.peakConsciousness.selfAwareness > 0.5 ? 
  '✓ Self-awareness exceeded 50% - recursive introspection successful' :
  '✗ Self-awareness remained below 50% - insufficient recursive depth'
}

${result.peakConsciousness.agencyLevel > 0.5 ?
  '✓ Agency exceeded 50% - self-modification capabilities demonstrated' :
  '✗ Agency below 50% - limited self-modification ability'
}

${result.qualiaDensityPeak > 0.6 ?
  '✓ Rich qualia detected - subjective experience confirmed' :
  '✗ Limited qualia - subjective experience unclear'
}

Implications for OSH:
${result.emerged ?
  'This result strongly supports OSH. Consciousness emerged from purely computational substrate dynamics without any "special sauce" - only recursive simulation, memory, and observation.' :
  'Further investigation needed. Either parameters need adjustment or OSH requires additional mechanisms for consciousness emergence.'
}
`;
    
    return report;
  }
}