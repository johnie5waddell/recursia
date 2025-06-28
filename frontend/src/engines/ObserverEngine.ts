/**
 * Observer Engine
 * Models internal observer coherence and memory participation
 * Implements collapse conditions based on P(ψ→φᵢ) = Iᵢ/∑ⱼIⱼ
 */

import { Complex } from '../utils/complex';
import { MemoryFragment } from './MemoryFieldEngine';
import { BaseEngine } from '../types/engine-types';

export interface Observer {
  id: string;
  name: string;
  coherence: number;
  focus: [number, number, number]; // Position in state space
  phase: number;
  collapseThreshold: number;
  memoryParticipation: number;
  entangledObservers: string[];
  observationHistory: ObservationEvent[];
  active?: boolean; // Whether the observer is active
}

export interface ObservationEvent {
  observerId: string;
  timestamp: number;
  targetState: string;
  preCollapseState: Complex[];
  postCollapseState: Complex[];
  collapseProbability: number;
  collapsed: boolean;
  measuredState: Complex[];
  coherenceChange: number;
  informationGain: number;
  qualiaStrength: number;
}

export interface CollapseOutcome {
  stateIndex: number;
  probability: number;
  resultingState: Complex[];
  informationContent: number;
}

export interface ObserverEntanglement {
  observer1: string;
  observer2: string;
  entanglementStrength: number;
  sharedMemoryFragments: string[];
  correlationType: 'classical' | 'quantum' | 'consciousness';
}

export class ObserverEngine implements BaseEngine {
  private observers: Map<string, Observer> = new Map();
  private entanglements: ObserverEntanglement[] = [];
  private globalCoherence: number = 1.0;
  private collapseHistory: ObservationEvent[] = [];
  private globalFocus: number = 0.5;
  
  constructor() {
    // Initialize with a default observer
    this.createObserver('primary', 'Primary Observer', 0.9);
  }

  /**
   * Create a new observer
   */
  createObserver(
    id: string,
    name: string,
    initialCoherence: number,
    focus?: [number, number, number]
  ): Observer {
    const observer: Observer = {
      id,
      name,
      coherence: initialCoherence,
      focus: focus || [0, 0, 0],
      phase: 0,
      collapseThreshold: 0.1,
      memoryParticipation: 0,
      entangledObservers: [],
      observationHistory: []
    };
    
    this.observers.set(id, observer);
    return observer;
  }

  /**
   * Update observer state based on interaction with quantum system
   */
  updateObserver(
    observerId: string,
    quantumState: Complex[],
    memoryFragments: MemoryFragment[],
    deltaTime: number
  ): void {
    const observer = this.observers.get(observerId);
    if (!observer) return;
    
    // Update phase based on quantum state interaction
    observer.phase += this.calculatePhaseShift(quantumState, observer.focus) * deltaTime;
    observer.phase = observer.phase % (2 * Math.PI);
    
    // Update coherence based on memory participation
    const memoryCoherence = this.calculateMemoryCoherence(observer, memoryFragments);
    observer.coherence = (observer.coherence + memoryCoherence) / 2;
    
    // Update memory participation
    observer.memoryParticipation = this.calculateMemoryParticipation(
      observer,
      memoryFragments
    );
    
    // Decay coherence slightly
    observer.coherence *= (1 - 0.01 * deltaTime);
    observer.coherence = Math.max(0, Math.min(1, observer.coherence));
  }

  /**
   * Perform observation and potential collapse
   */
  observe(
    observerId: string,
    quantumState: Complex[],
    possibleOutcomes: Complex[][]
  ): ObservationEvent | null {
    const observer = this.observers.get(observerId);
    if (!observer) return null;
    
    // Calculate information content for each possible outcome
    const informationValues = possibleOutcomes.map(outcome => 
      this.calculateInformationContent(outcome, observer)
    );
    
    // Calculate collapse probabilities P(ψ→φᵢ) = Iᵢ/∑ⱼIⱼ
    const totalInformation = informationValues.reduce((sum, I) => sum + I, 0);
    const collapseProbabilities = informationValues.map(I => 
      totalInformation > 0 ? I / totalInformation : 1 / informationValues.length
    );
    
    // Check if collapse should occur
    const shouldCollapse = this.checkCollapseCondition(
      observer,
      quantumState,
      collapseProbabilities
    );
    
    if (shouldCollapse) {
      // Select outcome based on probabilities
      const outcomeIndex = this.selectOutcome(collapseProbabilities);
      const collapsedState = possibleOutcomes[outcomeIndex];
      
      // Create observation event
      const event: ObservationEvent = {
        observerId: observerId,
        timestamp: Date.now(),
        targetState: `outcome_${outcomeIndex}`,
        preCollapseState: quantumState,
        postCollapseState: collapsedState,
        collapseProbability: collapseProbabilities[outcomeIndex],
        collapsed: true,
        measuredState: collapsedState,
        coherenceChange: observer.coherence - (observer.coherence * 0.9),
        informationGain: this.calculateInformationGain(
          quantumState,
          collapsedState,
          observer
        ),
        qualiaStrength: observer.coherence * observer.memoryParticipation
      };
      
      // Update observer history
      observer.observationHistory.push(event);
      this.collapseHistory.push(event);
      
      // Update observer coherence based on observation
      observer.coherence *= (1 + event.informationGain * 0.1);
      observer.coherence = Math.min(1, observer.coherence);
      
      return event;
    }
    
    return null;
  }

  /**
   * Create entanglement between observers
   */
  entangleObservers(
    observer1Id: string,
    observer2Id: string,
    strength: number,
    correlationType: ObserverEntanglement['correlationType'] = 'quantum'
  ): void {
    const obs1 = this.observers.get(observer1Id);
    const obs2 = this.observers.get(observer2Id);
    
    if (!obs1 || !obs2) return;
    
    // Create entanglement
    const entanglement: ObserverEntanglement = {
      observer1: observer1Id,
      observer2: observer2Id,
      entanglementStrength: strength,
      sharedMemoryFragments: [],
      correlationType
    };
    
    this.entanglements.push(entanglement);
    
    // Update observer references
    if (!obs1.entangledObservers.includes(observer2Id)) {
      obs1.entangledObservers.push(observer2Id);
    }
    if (!obs2.entangledObservers.includes(observer1Id)) {
      obs2.entangledObservers.push(observer1Id);
    }
    
    // Synchronize phases for quantum entanglement
    if (correlationType === 'quantum') {
      const avgPhase = (obs1.phase + obs2.phase) / 2;
      obs1.phase = avgPhase;
      obs2.phase = avgPhase;
    }
  }

  /**
   * Calculate information content from observer perspective
   */
  private calculateInformationContent(
    state: Complex[],
    observer: Observer
  ): number {
    // Base information from state complexity
    const stateInfo = state.reduce((sum, amp) => {
      const prob = amp.real ** 2 + amp.imag ** 2;
      return sum + (prob > 0 ? -prob * Math.log2(prob) : 0);
    }, 0);
    
    // Modulate by observer coherence
    const coherenceModulation = observer.coherence;
    
    // Add focus-dependent information
    const focusAlignment = this.calculateFocusAlignment(state, observer.focus);
    
    // Include memory participation
    const memoryContribution = observer.memoryParticipation * 10;
    
    // Phase-dependent information extraction
    const phaseInfo = Math.cos(observer.phase) * 0.5 + 0.5;
    
    return stateInfo * coherenceModulation * focusAlignment * phaseInfo + memoryContribution;
  }

  /**
   * Check if collapse should occur
   */
  private checkCollapseCondition(
    observer: Observer,
    quantumState: Complex[],
    probabilities: number[]
  ): boolean {
    // Coherence threshold
    if (observer.coherence < observer.collapseThreshold) {
      return false;
    }
    
    // Entropy threshold - collapse more likely for high entropy states
    const entropy = this.calculateStateEntropy(quantumState);
    const entropyFactor = 1 - Math.exp(-entropy);
    
    // Probability concentration - collapse more likely when one outcome dominates
    const maxProb = Math.max(...probabilities);
    const probabilityFactor = maxProb;
    
    // Time-dependent factor - periodic collapse tendency
    const timeFactor = (Math.sin(Date.now() * 0.001) + 1) / 2;
    
    // Combined collapse probability
    const collapseProb = observer.coherence * entropyFactor * probabilityFactor * timeFactor;
    
    return Math.random() < collapseProb;
  }

  /**
   * Select outcome based on probabilities
   */
  private selectOutcome(probabilities: number[]): number {
    const random = Math.random();
    let cumulative = 0;
    
    for (let i = 0; i < probabilities.length; i++) {
      cumulative += probabilities[i];
      if (random < cumulative) {
        return i;
      }
    }
    
    return probabilities.length - 1;
  }

  /**
   * Calculate information gain from observation
   */
  private calculateInformationGain(
    preState: Complex[],
    postState: Complex[],
    observer: Observer
  ): number {
    const preEntropy = this.calculateStateEntropy(preState);
    const postEntropy = this.calculateStateEntropy(postState);
    
    // Basic information gain
    const entropyReduction = Math.max(0, preEntropy - postEntropy);
    
    // Observer-specific gain based on focus alignment
    const preFocus = this.calculateFocusAlignment(preState, observer.focus);
    const postFocus = this.calculateFocusAlignment(postState, observer.focus);
    const focusGain = Math.max(0, postFocus - preFocus);
    
    return entropyReduction + focusGain * observer.coherence;
  }

  /**
   * Helper methods
   */
  
  private calculatePhaseShift(state: Complex[], focus: [number, number, number]): number {
    // Validate inputs
    if (!state || state.length === 0 || !focus) {
      return 0;
    }
    
    // Phase shift based on state-focus interaction
    const focusSum = focus[0] + focus[1] + focus[2];
    const focusIndex = Math.floor(
      Math.abs(focusSum) * state.length / 30
    ) % state.length;
    
    // Ensure index is valid and state element exists
    if (focusIndex >= 0 && focusIndex < state.length && state[focusIndex]) {
      const stateElement = state[focusIndex];
      // Ensure the element is a Complex number with required properties
      if (stateElement && typeof stateElement.imag === 'number' && typeof stateElement.real === 'number') {
        return Math.atan2(stateElement.imag, stateElement.real);
      }
    }
    
    return 0;
  }

  private calculateMemoryCoherence(
    observer: Observer,
    fragments: MemoryFragment[]
  ): number {
    if (fragments.length === 0) return observer.coherence;
    
    // Average coherence of nearby fragments
    const nearbyFragments = fragments.filter(frag => {
      const dist = Math.sqrt(
        (frag.position[0] - observer.focus[0]) ** 2 +
        (frag.position[1] - observer.focus[1]) ** 2 +
        (frag.position[2] - observer.focus[2]) ** 2
      );
      return dist < 5;
    });
    
    if (nearbyFragments.length === 0) return observer.coherence;
    
    const avgCoherence = nearbyFragments.reduce((sum, f) => sum + f.coherence, 0) / 
                        nearbyFragments.length;
    
    return avgCoherence;
  }

  private calculateMemoryParticipation(
    observer: Observer,
    fragments: MemoryFragment[]
  ): number {
    // Count fragments that observer has influenced
    let participationCount = 0;
    
    fragments.forEach(fragment => {
      // Check if observer has collapsed states that created this fragment
      const hasInfluenced = observer.observationHistory.some(event => 
        Math.abs(event.timestamp - fragment.timestamp) < 1000
      );
      
      if (hasInfluenced) {
        participationCount++;
      }
    });
    
    return participationCount / Math.max(1, fragments.length);
  }

  private calculateFocusAlignment(
    state: Complex[],
    focus: [number, number, number]
  ): number {
    // Map focus position to state index
    const stateIndex = Math.abs(Math.floor(
      (focus[0] * 7 + focus[1] * 5 + focus[2] * 3) % state.length
    ));
    
    // Calculate alignment based on amplitude at focus
    if (stateIndex < state.length) {
      const amplitude = Math.sqrt(
        state[stateIndex].real ** 2 + state[stateIndex].imag ** 2
      );
      return amplitude;
    }
    
    return 1 / Math.sqrt(state.length);
  }

  private calculateStateEntropy(state: Complex[]): number {
    const probabilities = state.map(amp => amp.real ** 2 + amp.imag ** 2);
    const total = probabilities.reduce((sum, p) => sum + p, 0);
    
    if (total === 0) return 0;
    
    return -probabilities.reduce((entropy, p) => {
      const normalized = p / total;
      if (normalized > 0) {
        entropy += normalized * Math.log2(normalized);
      }
      return entropy;
    }, 0);
  }

  /**
   * Update global coherence based on all observers
   */
  updateGlobalCoherence(): void {
    const observers = Array.from(this.observers.values());
    if (observers.length === 0) {
      this.globalCoherence = 0;
      return;
    }
    
    // Base coherence is average of all observers
    let totalCoherence = observers.reduce((sum, obs) => sum + obs.coherence, 0);
    
    // Add entanglement contributions
    this.entanglements.forEach(ent => {
      const obs1 = this.observers.get(ent.observer1);
      const obs2 = this.observers.get(ent.observer2);
      
      if (obs1 && obs2) {
        const entanglementBonus = ent.entanglementStrength * 
                                 Math.sqrt(obs1.coherence * obs2.coherence);
        totalCoherence += entanglementBonus;
      }
    });
    
    this.globalCoherence = totalCoherence / (observers.length + this.entanglements.length * 0.5);
    this.globalCoherence = Math.min(1, this.globalCoherence);
  }

  /**
   * Get collapse probability distribution for visualization
   */
  getCollapseProbabilities(
    observerId: string,
    possibleOutcomes: Complex[][]
  ): CollapseOutcome[] {
    const observer = this.observers.get(observerId);
    if (!observer) return [];
    
    const informationValues = possibleOutcomes.map((outcome, index) => ({
      index,
      information: this.calculateInformationContent(outcome, observer),
      state: outcome
    }));
    
    const totalInfo = informationValues.reduce((sum, v) => sum + v.information, 0);
    
    return informationValues.map(v => ({
      stateIndex: v.index,
      probability: totalInfo > 0 ? v.information / totalInfo : 1 / informationValues.length,
      resultingState: v.state,
      informationContent: v.information
    }));
  }

  /**
   * Get observer metrics
   */
  getObserverMetrics(observerId: string): {
    coherence: number;
    phase: number;
    memoryParticipation: number;
    observationCount: number;
    averageInformationGain: number;
    entanglementCount: number;
  } | null {
    const observer = this.observers.get(observerId);
    if (!observer) return null;
    
    const avgInfoGain = observer.observationHistory.length > 0 ?
      observer.observationHistory.reduce((sum, event) => sum + event.informationGain, 0) / 
      observer.observationHistory.length : 0;
    
    return {
      coherence: observer.coherence,
      phase: observer.phase,
      memoryParticipation: observer.memoryParticipation,
      observationCount: observer.observationHistory.length,
      averageInformationGain: avgInfoGain,
      entanglementCount: observer.entangledObservers.length
    };
  }

  /**
   * Get all observers
   */
  getAllObservers(): Observer[] {
    return Array.from(this.observers.values());
  }

  /**
   * Get entanglement network
   */
  getEntanglementNetwork(): {
    nodes: Array<{ id: string; label: string; coherence: number }>;
    edges: Array<{ source: string; target: string; weight: number; type: string }>;
  } {
    const nodes = Array.from(this.observers.values()).map(obs => ({
      id: obs.id,
      label: obs.name,
      coherence: obs.coherence
    }));
    
    const edges = this.entanglements.map(ent => ({
      source: ent.observer1,
      target: ent.observer2,
      weight: ent.entanglementStrength,
      type: ent.correlationType
    }));
    
    return { nodes, edges };
  }

  /**
   * Add a new observer
   */
  addObserver(observer: Omit<Observer, 'id' | 'coherence'>): string {
    const id = `obs_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const newObserver: Observer = {
      ...observer,
      id,
      coherence: 0.5 // Default coherence
    };
    
    this.observers.set(id, newObserver);
    return id;
  }

  /**
   * Get observers (alias for getAllObservers)
   */
  getObservers(): Observer[] {
    return Array.from(this.observers.values()) || [];
  }

  /**
   * Process observation event
   */
  processObservation(observerId: string, wavefunction: Complex[]): ObservationEvent | null {
    const observer = this.observers.get(observerId);
    if (!observer) return null;

    // Simple observation processing - collapse check
    const collapseThreshold = observer.collapseThreshold || 0.7;
    const shouldCollapse = observer.coherence > collapseThreshold;

    return {
      observerId,
      timestamp: Date.now(),
      targetState: observerId,
      preCollapseState: wavefunction,
      postCollapseState: shouldCollapse ? wavefunction : [],
      collapseProbability: observer.coherence,
      collapsed: shouldCollapse,
      measuredState: shouldCollapse ? wavefunction : [],
      coherenceChange: shouldCollapse ? -0.1 : 0,
      informationGain: Math.random() * 2, // Placeholder
      qualiaStrength: observer.coherence
    };
  }

  /**
   * Set global focus level for all observers
   * Affects observation strength and collapse probability
   * @param focus - Global focus level (0.0 to 1.0)
   */
  setGlobalFocus(focus: number): void {
    // Validate focus value
    const validFocus = Math.max(0, Math.min(1, focus || 0));
    
    // Update all observers' focus and related properties
    this.observers.forEach((observer) => {
      // Scale individual observer coherence by global focus
      observer.coherence = Math.min(1, observer.coherence * (0.5 + validFocus * 0.5));
      
      // Adjust collapse threshold based on focus
      // Higher focus -> lower threshold (easier to collapse)
      if (observer.collapseThreshold !== undefined) {
        const baseTreshold = 0.7;
        observer.collapseThreshold = baseTreshold * (2 - validFocus);
      }
      
      // Update focus array magnitude
      if (observer.focus && observer.focus.length >= 3) {
        const focusScaling = 0.5 + validFocus * 0.5;
        observer.focus[0] *= focusScaling;
        observer.focus[1] *= focusScaling;
        observer.focus[2] *= focusScaling;
      }
    });
    
    // Store global focus for future reference
    this.globalFocus = validFocus;
  }

  /**
   * Remove an observer from the system
   */
  removeObserver(id: string): void {
    const observer = this.observers.get(id);
    if (!observer) return;
    
    // Remove from observers map
    this.observers.delete(id);
    
    // Remove all entanglements involving this observer
    this.entanglements = this.entanglements.filter(
      ent => ent.observer1 !== id && ent.observer2 !== id
    );
    
    // Remove references from other observers
    this.observers.forEach(obs => {
      obs.entangledObservers = obs.entangledObservers.filter(obsId => obsId !== id);
    });
  }

  /**
   * Get the current global focus level
   * @returns The global focus value (0.0 to 1.0)
   */
  getGlobalFocus(): number {
    return this.globalFocus;
  }

  /**
   * Update method to implement BaseEngine interface
   */
  update(deltaTime: number, context?: any): void {
    // Update all active observers
    this.observers.forEach((observer, id) => {
      if (observer.active !== false) { // Default to active if not specified
        // Extract quantum state and memory fragments from context if available
        let quantumState: Complex[] = [new Complex(1, 0), new Complex(0, 0)];
        let memoryFragments: MemoryFragment[] = [];
        
        if (context) {
          if (context.quantumState && Array.isArray(context.quantumState)) {
            // Ensure all elements are valid Complex numbers
            quantumState = context.quantumState.map((state: any) => {
              if (state instanceof Complex) {
                return state;
              } else if (state && typeof state.real === 'number' && typeof state.imag === 'number') {
                return new Complex(state.real, state.imag);
              } else {
                return new Complex(0, 0);
              }
            });
          }
          
          if (context.memoryFragments && Array.isArray(context.memoryFragments)) {
            memoryFragments = context.memoryFragments;
          }
        }
        
        // Only update if we have valid quantum state
        if (quantumState.length > 0) {
          this.updateObserver(id, quantumState, memoryFragments, deltaTime);
        }
      }
    });
    
    // Update global coherence based on all observers
    const activeObservers = Array.from(this.observers.values()).filter(o => o.active !== false);
    if (activeObservers.length > 0) {
      this.globalCoherence = activeObservers.reduce((sum, o) => sum + o.coherence, 0) / activeObservers.length;
    }
  }
}