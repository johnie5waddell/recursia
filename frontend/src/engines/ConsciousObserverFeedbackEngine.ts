/**
 * Conscious Observer Feedback Loops (COFL) Engine
 * 
 * Direct integration of human consciousness into quantum error correction through
 * trained observer-operators who maintain intentional focus on qubit stability.
 * Uses EEG/fMRI feedback to optimize the observer's effect on quantum measurements.
 * 
 * Based on OSH core hypothesis: consciousness can stabilize quantum systems through
 * recursive observation and focused intention.
 */

import { Complex } from '../utils/complex';

export interface EEGData {
  timestamp: number;
  delta_power: number;    // 0.5-4 Hz
  theta_power: number;    // 4-8 Hz  
  alpha_power: number;    // 8-13 Hz
  beta_power: number;     // 13-30 Hz
  gamma_power: number;    // 30-100 Hz
  high_gamma_power: number; // 100-200 Hz
  coherence_index: number; // Inter-hemispheric coherence
  focus_intensity: number; // Calculated focus strength
  attention_stability: number; // Attention consistency over time
}

export interface fMRIData {
  timestamp: number;
  prefrontal_activation: number;    // PFC activity (attention/focus)
  anterior_cingulate_activation: number; // ACC (error monitoring)
  parietal_activation: number;      // Parietal cortex (spatial attention)
  thalamic_activation: number;      // Thalamus (consciousness gateway)
  default_mode_deactivation: number; // DMN suppression (focused state)
  neural_coherence: number;         // Cross-regional coherence
  consciousness_signature: number;  // Integrated information measure
}

export interface ConsciousObserver {
  id: string;
  name: string;
  training_level: 'novice' | 'intermediate' | 'advanced' | 'master';
  meditation_experience_years: number;
  focus_specialization: 'sustained_attention' | 'selective_attention' | 'meta_cognitive' | 'transcendental';
  current_session_duration: number; // minutes
  fatigue_level: number; // 0-1
  baseline_eeg: EEGData;
  baseline_fmri: fMRIData;
  current_eeg: EEGData;
  current_fmri: fMRIData;
  quantum_coupling_strength: number;
  stabilization_effectiveness: number;
  historical_performance: number[];
  biofeedback_training_score: number;
}

export interface QuantumTarget {
  qubit_id: string;
  position: [number, number, number];
  current_coherence: number;
  target_coherence: number;
  error_rate: number;
  decoherence_time: number;
  entanglement_partners: string[];
  measurement_basis: 'computational' | 'hadamard' | 'circular';
  state_vector: Complex[];
  observation_history: ObservationRecord[];
}

export interface ObservationRecord {
  timestamp: number;
  observer_id: string;
  pre_observation_state: Complex[];
  post_observation_state: Complex[];
  collapse_probability: number;
  measurement_outcome: number[];
  observer_intention: 'stabilize' | 'measure' | 'entangle' | 'isolate';
  consciousness_effect_strength: number;
  error_rate_change: number;
}

export interface ConsciousnessQuantumCoupling {
  observer_id: string;
  target_qubit_id: string;
  coupling_strength: number;
  coupling_type: 'direct_focus' | 'meditative_resonance' | 'intention_field' | 'quantum_entanglement';
  effective_range: number; // meters
  coherence_enhancement: number;
  error_suppression: number;
  measurement_influence: number;
  entanglement_protection: number;
}

export interface BiofeedbackProtocol {
  id: string;
  name: string;
  target_brainwave: 'gamma' | 'theta' | 'alpha' | 'coherence';
  training_phases: BiofeedbackPhase[];
  optimal_eeg_pattern: EEGData;
  optimal_fmri_pattern: fMRIData;
  success_criteria: {
    min_focus_duration: number; // seconds
    min_coherence_index: number;
    max_attention_drift: number;
    target_gamma_power: number;
    target_theta_power: number;
  };
  quantum_effectiveness_threshold: number;
}

export interface BiofeedbackPhase {
  phase_name: string;
  duration_minutes: number;
  instructions: string;
  target_metrics: Partial<EEGData>;
  feedback_modality: 'visual' | 'auditory' | 'haptic' | 'multimodal';
  difficulty_level: number; // 1-10
  quantum_integration: boolean;
}

export class ConsciousObserverFeedbackEngine {
  private observers: Map<string, ConsciousObserver> = new Map();
  private quantumTargets: Map<string, QuantumTarget> = new Map();
  private consciousnessCouplings: Map<string, ConsciousnessQuantumCoupling> = new Map();
  private biofeedbackProtocols: Map<string, BiofeedbackProtocol> = new Map();
  private observationRecords: ObservationRecord[] = [];
  
  // Real-time data streams
  private eegDataStream: Map<string, EEGData[]> = new Map();
  private fmriDataStream: Map<string, fMRIData[]> = new Map();
  
  // Performance tracking
  private consciousness_effect_history: number[] = [];
  private quantum_error_reduction_history: number[] = [];
  private observer_performance_metrics: Map<string, number[]> = new Map();
  
  // System parameters
  private readonly CONSCIOUSNESS_COUPLING_THRESHOLD = 0.1;
  private readonly MAXIMUM_OBSERVATION_EFFECT = 0.95;
  private readonly QUANTUM_MEASUREMENT_PRECISION = 1e-6;
  private readonly BIOFEEDBACK_UPDATE_RATE = 10; // Hz
  private readonly CONSCIOUSNESS_FIELD_RANGE = 1e-3; // 1mm
  
  constructor() {
    this.initializeSystem();
  }

  /**
   * Initialize COFL system with default protocols and baselines
   */
  private initializeSystem(): void {
    this.createBiofeedbackProtocols();
    console.log('COFL System initialized with advanced consciousness-quantum interfaces');
  }

  /**
   * Create comprehensive biofeedback training protocols
   */
  private createBiofeedbackProtocols(): void {
    // Protocol 1: Gamma Wave Enhancement for Quantum Focus
    const gammaProtocol: BiofeedbackProtocol = {
      id: 'gamma_quantum_focus',
      name: 'Gamma Wave Quantum Focus Training',
      target_brainwave: 'gamma',
      training_phases: [
        {
          phase_name: 'Baseline Establishment',
          duration_minutes: 5,
          instructions: 'Sit quietly with eyes closed, establish natural breathing rhythm',
          target_metrics: { gamma_power: 0.3, alpha_power: 0.4 },
          feedback_modality: 'visual',
          difficulty_level: 1,
          quantum_integration: false
        },
        {
          phase_name: 'Gamma Enhancement',
          duration_minutes: 15,
          instructions: 'Focus on increasing high-frequency brain activity through sustained attention',
          target_metrics: { gamma_power: 0.7, coherence_index: 0.6, focus_intensity: 0.8 },
          feedback_modality: 'multimodal',
          difficulty_level: 5,
          quantum_integration: false
        },
        {
          phase_name: 'Quantum System Integration',
          duration_minutes: 20,
          instructions: 'Direct focused attention toward quantum target while maintaining gamma coherence',
          target_metrics: { gamma_power: 0.8, coherence_index: 0.7, focus_intensity: 0.9 },
          feedback_modality: 'visual',
          difficulty_level: 8,
          quantum_integration: true
        }
      ],
      optimal_eeg_pattern: {
        timestamp: 0,
        delta_power: 0.1,
        theta_power: 0.2,
        alpha_power: 0.3,
        beta_power: 0.4,
        gamma_power: 0.8,
        high_gamma_power: 0.6,
        coherence_index: 0.7,
        focus_intensity: 0.9,
        attention_stability: 0.85
      },
      optimal_fmri_pattern: {
        timestamp: 0,
        prefrontal_activation: 0.8,
        anterior_cingulate_activation: 0.7,
        parietal_activation: 0.6,
        thalamic_activation: 0.9,
        default_mode_deactivation: 0.8,
        neural_coherence: 0.75,
        consciousness_signature: 0.85
      },
      success_criteria: {
        min_focus_duration: 300, // 5 minutes
        min_coherence_index: 0.6,
        max_attention_drift: 0.2,
        target_gamma_power: 0.7,
        target_theta_power: 0.2
      },
      quantum_effectiveness_threshold: 0.15
    };

    // Protocol 2: Theta-Gamma Coupling for Deep Quantum Resonance
    const thetaGammaProtocol: BiofeedbackProtocol = {
      id: 'theta_gamma_coupling',
      name: 'Theta-Gamma Coupling for Quantum Resonance',
      target_brainwave: 'coherence',
      training_phases: [
        {
          phase_name: 'Theta Induction',
          duration_minutes: 10,
          instructions: 'Enter meditative state with slow, deep breathing to induce theta waves',
          target_metrics: { theta_power: 0.6, alpha_power: 0.3 },
          feedback_modality: 'auditory',
          difficulty_level: 3,
          quantum_integration: false
        },
        {
          phase_name: 'Gamma Overlay',
          duration_minutes: 10,
          instructions: 'Maintain theta state while adding focused attention (gamma)',
          target_metrics: { theta_power: 0.6, gamma_power: 0.5, coherence_index: 0.7 },
          feedback_modality: 'visual',
          difficulty_level: 7,
          quantum_integration: false
        },
        {
          phase_name: 'Quantum Field Resonance',
          duration_minutes: 15,
          instructions: 'Synchronize theta-gamma coupling with quantum field oscillations',
          target_metrics: { theta_power: 0.7, gamma_power: 0.6, coherence_index: 0.8 },
          feedback_modality: 'multimodal',
          difficulty_level: 9,
          quantum_integration: true
        }
      ],
      optimal_eeg_pattern: {
        timestamp: 0,
        delta_power: 0.2,
        theta_power: 0.7,
        alpha_power: 0.3,
        beta_power: 0.2,
        gamma_power: 0.6,
        high_gamma_power: 0.4,
        coherence_index: 0.8,
        focus_intensity: 0.7,
        attention_stability: 0.9
      },
      optimal_fmri_pattern: {
        timestamp: 0,
        prefrontal_activation: 0.6,
        anterior_cingulate_activation: 0.8,
        parietal_activation: 0.5,
        thalamic_activation: 0.9,
        default_mode_deactivation: 0.9,
        neural_coherence: 0.85,
        consciousness_signature: 0.9
      },
      success_criteria: {
        min_focus_duration: 600, // 10 minutes
        min_coherence_index: 0.7,
        max_attention_drift: 0.15,
        target_gamma_power: 0.6,
        target_theta_power: 0.7
      },
      quantum_effectiveness_threshold: 0.25
    };

    this.biofeedbackProtocols.set(gammaProtocol.id, gammaProtocol);
    this.biofeedbackProtocols.set(thetaGammaProtocol.id, thetaGammaProtocol);
  }

  /**
   * Register new conscious observer with baseline measurements
   */
  registerObserver(
    observerId: string,
    name: string,
    trainingLevel: 'novice' | 'intermediate' | 'advanced' | 'master' = 'novice',
    meditationExperience: number = 0,
    focusSpecialization: 'sustained_attention' | 'selective_attention' | 'meta_cognitive' | 'transcendental' = 'sustained_attention'
  ): void {
    const baselineEEG: EEGData = {
      timestamp: Date.now(),
      delta_power: 0.3,
      theta_power: 0.25,
      alpha_power: 0.4,
      beta_power: 0.35,
      gamma_power: 0.2,
      high_gamma_power: 0.1,
      coherence_index: 0.4,
      focus_intensity: 0.3,
      attention_stability: 0.5
    };

    const baselineFMRI: fMRIData = {
      timestamp: Date.now(),
      prefrontal_activation: 0.4,
      anterior_cingulate_activation: 0.3,
      parietal_activation: 0.35,
      thalamic_activation: 0.5,
      default_mode_deactivation: 0.2,
      neural_coherence: 0.4,
      consciousness_signature: 0.3
    };

    const observer: ConsciousObserver = {
      id: observerId,
      name,
      training_level: trainingLevel,
      meditation_experience_years: meditationExperience,
      focus_specialization: focusSpecialization,
      current_session_duration: 0,
      fatigue_level: 0,
      baseline_eeg: baselineEEG,
      baseline_fmri: baselineFMRI,
      current_eeg: { ...baselineEEG },
      current_fmri: { ...baselineFMRI },
      quantum_coupling_strength: 0,
      stabilization_effectiveness: 0,
      historical_performance: [],
      biofeedback_training_score: 0
    };

    this.observers.set(observerId, observer);
    this.eegDataStream.set(observerId, []);
    this.fmriDataStream.set(observerId, []);
    this.observer_performance_metrics.set(observerId, []);

    console.log(`Observer ${name} registered with ${trainingLevel} level training`);
  }

  /**
   * Register quantum target for conscious observation
   */
  registerQuantumTarget(
    qubitId: string,
    position: [number, number, number],
    initialCoherence: number = 0.9
  ): void {
    const target: QuantumTarget = {
      qubit_id: qubitId,
      position,
      current_coherence: initialCoherence,
      target_coherence: 0.99,
      error_rate: 0.001,
      decoherence_time: 100, // microseconds
      entanglement_partners: [],
      measurement_basis: 'computational',
      state_vector: [new Complex(1, 0), new Complex(0, 0)], // |0⟩ state
      observation_history: []
    };

    this.quantumTargets.set(qubitId, target);
    console.log(`Quantum target ${qubitId} registered for conscious observation`);
  }

  /**
   * Update real-time EEG data from observer
   */
  updateObserverEEG(observerId: string, eegData: EEGData): void {
    const observer = this.observers.get(observerId);
    if (!observer) {
      console.warn(`Observer ${observerId} not found`);
      return;
    }

    // Update current EEG state
    observer.current_eeg = { ...eegData, timestamp: Date.now() };
    
    // Add to data stream (keep last 1000 samples)
    const stream = this.eegDataStream.get(observerId) || [];
    stream.push(observer.current_eeg);
    if (stream.length > 1000) stream.shift();
    this.eegDataStream.set(observerId, stream);

    // Update observer metrics based on EEG
    this.updateObserverMetrics(observer);
    
    this.observers.set(observerId, observer);
  }

  /**
   * Update real-time fMRI data from observer
   */
  updateObserverFMRI(observerId: string, fmriData: fMRIData): void {
    const observer = this.observers.get(observerId);
    if (!observer) return;

    observer.current_fmri = { ...fmriData, timestamp: Date.now() };
    
    const stream = this.fmriDataStream.get(observerId) || [];
    stream.push(observer.current_fmri);
    if (stream.length > 1000) stream.shift();
    this.fmriDataStream.set(observerId, stream);

    this.updateObserverMetrics(observer);
    this.observers.set(observerId, observer);
  }

  /**
   * Update observer performance metrics based on neuroimaging data
   */
  private updateObserverMetrics(observer: ConsciousObserver): void {
    const eeg = observer.current_eeg;
    const fmri = observer.current_fmri;

    // Calculate quantum coupling strength based on neuroimaging
    const gamma_factor = Math.min(1, eeg.gamma_power / 0.8);
    const coherence_factor = Math.min(1, eeg.coherence_index / 0.7);
    const focus_factor = Math.min(1, eeg.focus_intensity / 0.9);
    const consciousness_factor = Math.min(1, fmri.consciousness_signature / 0.8);
    const attention_factor = Math.min(1, fmri.prefrontal_activation / 0.8);

    observer.quantum_coupling_strength = 
      (gamma_factor * 0.3 + 
       coherence_factor * 0.25 + 
       focus_factor * 0.2 + 
       consciousness_factor * 0.15 + 
       attention_factor * 0.1);

    // Calculate stabilization effectiveness
    const training_multiplier = {
      'novice': 0.5,
      'intermediate': 0.75,
      'advanced': 0.9,
      'master': 1.0
    }[observer.training_level];

    const experience_bonus = Math.min(0.2, observer.meditation_experience_years * 0.02);
    const fatigue_penalty = observer.fatigue_level * 0.3;

    observer.stabilization_effectiveness = 
      (observer.quantum_coupling_strength * training_multiplier + experience_bonus - fatigue_penalty);

    // Update fatigue based on session duration
    observer.fatigue_level = Math.min(1, observer.current_session_duration / 120); // 2 hour max

    // Update biofeedback training score
    const protocol = this.biofeedbackProtocols.get('gamma_quantum_focus');
    if (protocol) {
      observer.biofeedback_training_score = this.evaluateBiofeedbackPerformance(observer, protocol);
    }
  }

  /**
   * Evaluate observer performance against biofeedback protocol criteria
   */
  private evaluateBiofeedbackPerformance(
    observer: ConsciousObserver, 
    protocol: BiofeedbackProtocol
  ): number {
    const eeg = observer.current_eeg;
    const criteria = protocol.success_criteria;

    let score = 0;
    let maxScore = 5;

    // Focus duration
    if (observer.current_session_duration * 60 >= criteria.min_focus_duration) {
      score += 1;
    }

    // Coherence index
    if (eeg.coherence_index >= criteria.min_coherence_index) {
      score += 1;
    }

    // Attention stability
    if ((1 - eeg.attention_stability) <= criteria.max_attention_drift) {
      score += 1;
    }

    // Target gamma power
    if (eeg.gamma_power >= criteria.target_gamma_power) {
      score += 1;
    }

    // Target theta power (for specific protocols)
    if (eeg.theta_power >= criteria.target_theta_power) {
      score += 1;
    }

    return score / maxScore;
  }

  /**
   * Establish consciousness-quantum coupling between observer and target
   */
  establishQuantumCoupling(
    observerId: string,
    targetQubitId: string,
    couplingType: 'direct_focus' | 'meditative_resonance' | 'intention_field' | 'quantum_entanglement' = 'direct_focus'
  ): string {
    const observer = this.observers.get(observerId);
    const target = this.quantumTargets.get(targetQubitId);
    
    if (!observer || !target) {
      throw new Error('Observer or target not found');
    }

    const couplingId = `coupling_${observerId}_${targetQubitId}_${Date.now()}`;
    
    // Calculate coupling strength based on observer state and distance
    const distance = 0.001; // Assume 1mm for simulation
    const proximityFactor = Math.exp(-distance / this.CONSCIOUSNESS_FIELD_RANGE);
    
    const coupling: ConsciousnessQuantumCoupling = {
      observer_id: observerId,
      target_qubit_id: targetQubitId,
      coupling_strength: observer.quantum_coupling_strength * proximityFactor,
      coupling_type: couplingType,
      effective_range: this.CONSCIOUSNESS_FIELD_RANGE,
      coherence_enhancement: 0,
      error_suppression: 0,
      measurement_influence: 0,
      entanglement_protection: 0
    };

    // Calculate specific effects based on coupling type
    switch (couplingType) {
      case 'direct_focus':
        coupling.coherence_enhancement = coupling.coupling_strength * 0.1;
        coupling.error_suppression = coupling.coupling_strength * 0.05;
        coupling.measurement_influence = coupling.coupling_strength * 0.2;
        break;
      
      case 'meditative_resonance':
        coupling.coherence_enhancement = coupling.coupling_strength * 0.15;
        coupling.error_suppression = coupling.coupling_strength * 0.1;
        coupling.entanglement_protection = coupling.coupling_strength * 0.1;
        break;
      
      case 'intention_field':
        coupling.error_suppression = coupling.coupling_strength * 0.2;
        coupling.measurement_influence = coupling.coupling_strength * 0.15;
        break;
      
      case 'quantum_entanglement':
        coupling.coherence_enhancement = coupling.coupling_strength * 0.2;
        coupling.entanglement_protection = coupling.coupling_strength * 0.25;
        break;
    }

    this.consciousnessCouplings.set(couplingId, coupling);
    console.log(`Quantum coupling established: ${couplingType} between ${observerId} and ${targetQubitId}`);
    
    return couplingId;
  }

  /**
   * Perform conscious observation of quantum target
   */
  async performConsciousObservation(
    observerId: string,
    targetQubitId: string,
    intention: 'stabilize' | 'measure' | 'entangle' | 'isolate' = 'stabilize',
    duration: number = 1000 // milliseconds
  ): Promise<ObservationRecord> {
    const observer = this.observers.get(observerId);
    const target = this.quantumTargets.get(targetQubitId);
    
    if (!observer || !target) {
      throw new Error('Observer or target not found');
    }

    // Find active coupling
    const coupling = Array.from(this.consciousnessCouplings.values()).find(
      c => c.observer_id === observerId && c.target_qubit_id === targetQubitId
    );

    if (!coupling) {
      throw new Error('No active coupling between observer and target');
    }

    const startTime = Date.now();
    const preObservationState = [...target.state_vector];
    
    // Simulate consciousness effect on quantum state
    const consciousnessEffect = this.calculateConsciousnessEffect(observer, target, coupling, intention);
    
    // Apply consciousness effect to quantum state
    const postObservationState = this.applyConsciousnessEffect(
      target.state_vector, 
      consciousnessEffect, 
      intention
    );
    
    // Update target state
    target.state_vector = postObservationState;
    target.current_coherence += coupling.coherence_enhancement;
    target.current_coherence = Math.min(0.99, target.current_coherence);
    target.error_rate *= (1 - coupling.error_suppression);
    
    // Calculate measurement outcome
    const measurementOutcome = this.performQuantumMeasurement(postObservationState, target.measurement_basis);
    
    // Create observation record
    const record: ObservationRecord = {
      timestamp: startTime,
      observer_id: observerId,
      pre_observation_state: preObservationState,
      post_observation_state: postObservationState,
      collapse_probability: this.calculateCollapseProbability(preObservationState, postObservationState),
      measurement_outcome: measurementOutcome,
      observer_intention: intention,
      consciousness_effect_strength: consciousnessEffect,
      error_rate_change: coupling.error_suppression
    };

    // Update histories
    target.observation_history.push(record);
    this.observationRecords.push(record);
    observer.historical_performance.push(consciousnessEffect);
    
    // Update performance metrics
    const performance = this.observer_performance_metrics.get(observerId) || [];
    performance.push(consciousnessEffect);
    this.observer_performance_metrics.set(observerId, performance);
    
    // Update consciousness effect history
    this.consciousness_effect_history.push(consciousnessEffect);
    this.quantum_error_reduction_history.push(coupling.error_suppression);

    // Update session duration
    observer.current_session_duration += duration / 60000; // convert to minutes
    
    this.observers.set(observerId, observer);
    this.quantumTargets.set(targetQubitId, target);
    
    console.log(`Conscious observation completed: ${intention} by ${observerId} on ${targetQubitId}`);
    return record;
  }

  /**
   * Calculate consciousness effect strength on quantum system
   */
  private calculateConsciousnessEffect(
    observer: ConsciousObserver,
    target: QuantumTarget,
    coupling: ConsciousnessQuantumCoupling,
    intention: 'stabilize' | 'measure' | 'entangle' | 'isolate'
  ): number {
    const baseEffect = observer.stabilization_effectiveness * coupling.coupling_strength;
    
    // Intention-specific modifiers
    const intentionModifiers = {
      'stabilize': 1.2,
      'measure': 0.8,
      'entangle': 1.0,
      'isolate': 0.9
    };
    
    const intentionEffect = baseEffect * intentionModifiers[intention];
    
    // Training level bonus
    const trainingBonus = {
      'novice': 0,
      'intermediate': 0.1,
      'advanced': 0.25,
      'master': 0.5
    }[observer.training_level];
    
    // Specialization bonus for sustained attention during stabilization
    const specializationBonus = 
      (observer.focus_specialization === 'sustained_attention' && intention === 'stabilize') ? 0.15 : 0;
    
    // Coherence state bonus (better coupling when observer is in optimal state)
    const coherenceBonus = Math.max(0, observer.current_eeg.coherence_index - 0.5) * 0.2;
    
    const totalEffect = intentionEffect + trainingBonus + specializationBonus + coherenceBonus;
    
    return Math.min(this.MAXIMUM_OBSERVATION_EFFECT, totalEffect);
  }

  /**
   * Apply consciousness effect to quantum state vector
   */
  private applyConsciousnessEffect(
    stateVector: Complex[],
    effectStrength: number,
    intention: 'stabilize' | 'measure' | 'entangle' | 'isolate'
  ): Complex[] {
    const newState = stateVector.map(amplitude => new Complex(amplitude.real, amplitude.imag));
    
    switch (intention) {
      case 'stabilize':
        // Reduce decoherence by increasing state purity
        const norm = Math.sqrt(newState.reduce((sum, amp) => sum + (amp.magnitude() ** 2), 0));
        for (let i = 0; i < newState.length; i++) {
          const stabilizationFactor = 1 + effectStrength * 0.1;
          newState[i] = newState[i].multiply(new Complex(stabilizationFactor / norm, 0));
        }
        break;
        
      case 'measure':
        // Enhance measurement basis alignment
        if (newState.length >= 2) {
          const measurementBias = effectStrength * 0.1;
          newState[0] = newState[0].multiply(new Complex(1 + measurementBias, 0));
          newState[1] = newState[1].multiply(new Complex(1 - measurementBias, 0));
        }
        break;
        
      case 'entangle':
        // Enhance superposition coherence
        if (newState.length >= 2) {
          const coherenceFactor = 1 + effectStrength * 0.05;
          newState[0] = newState[0].multiply(new Complex(coherenceFactor, 0));
          newState[1] = newState[1].multiply(new Complex(coherenceFactor, 0));
        }
        break;
        
      case 'isolate':
        // Reduce entanglement by localizing state
        const isolationFactor = 1 - effectStrength * 0.05;
        for (let i = 1; i < newState.length; i++) {
          newState[i] = newState[i].multiply(new Complex(isolationFactor, 0));
        }
        break;
    }
    
    // Renormalize state
    const finalNorm = Math.sqrt(newState.reduce((sum, amp) => sum + (amp.magnitude() ** 2), 0));
    return newState.map(amp => amp.multiply(new Complex(1 / finalNorm, 0)));
  }

  /**
   * Perform quantum measurement in specified basis
   */
  private performQuantumMeasurement(
    stateVector: Complex[],
    basis: 'computational' | 'hadamard' | 'circular'
  ): number[] {
    // Calculate measurement probabilities
    const probabilities = stateVector.map(amp => (amp.magnitude() ** 2));
    
    // Simulate measurement outcome based on probabilities
    const random = Math.random();
    let cumulative = 0;
    
    for (let i = 0; i < probabilities.length; i++) {
      cumulative += probabilities[i];
      if (random < cumulative) {
        const outcome = new Array(probabilities.length).fill(0);
        outcome[i] = 1;
        return outcome;
      }
    }
    
    // Fallback (should rarely happen)
    const outcome = new Array(probabilities.length).fill(0);
    outcome[0] = 1;
    return outcome;
  }

  /**
   * Calculate collapse probability between two states
   */
  private calculateCollapseProbability(stateBefore: Complex[], stateAfter: Complex[]): number {
    if (stateBefore.length !== stateAfter.length) return 1;
    
    // Calculate fidelity between states
    let fidelity = 0;
    for (let i = 0; i < stateBefore.length; i++) {
      const overlap = stateBefore[i].conjugate().multiply(stateAfter[i]);
      fidelity += overlap.real; // Real part of overlap
    }
    
    return 1 - Math.abs(fidelity);
  }

  /**
   * Run biofeedback training session for observer
   */
  async runBiofeedbackTraining(
    observerId: string,
    protocolId: string,
    realTimeEEGCallback?: (feedback: string) => void
  ): Promise<{
    success: boolean;
    finalScore: number;
    phaseResults: { phase: string; score: number; duration: number }[];
    quantumCouplingImprovement: number;
  }> {
    const observer = this.observers.get(observerId);
    const protocol = this.biofeedbackProtocols.get(protocolId);
    
    if (!observer || !protocol) {
      throw new Error('Observer or protocol not found');
    }

    console.log(`Starting biofeedback training: ${protocol.name} for ${observer.name}`);
    
    const phaseResults: { phase: string; score: number; duration: number }[] = [];
    let totalScore = 0;
    const initialCouplingStrength = observer.quantum_coupling_strength;
    
    for (const phase of protocol.training_phases) {
      console.log(`Phase: ${phase.phase_name} - ${phase.instructions}`);
      
      const phaseStartTime = Date.now();
      
      // Simulate training phase progression
      const phaseDuration = phase.duration_minutes * 60 * 1000; // convert to ms
      const updateInterval = 100; // 10 Hz updates
      const totalUpdates = phaseDuration / updateInterval;
      
      let phaseScore = 0;
      
      for (let update = 0; update < totalUpdates; update++) {
        // Simulate EEG progression toward target
        const progress = update / totalUpdates;
        const targetMetrics = phase.target_metrics;
        
        // Generate synthetic EEG data moving toward targets
        const simulatedEEG: EEGData = {
          timestamp: Date.now(),
          delta_power: observer.baseline_eeg.delta_power,
          theta_power: this.interpolateToTarget(observer.baseline_eeg.theta_power, targetMetrics.theta_power || 0.3, progress),
          alpha_power: this.interpolateToTarget(observer.baseline_eeg.alpha_power, targetMetrics.alpha_power || 0.4, progress),
          beta_power: observer.baseline_eeg.beta_power,
          gamma_power: this.interpolateToTarget(observer.baseline_eeg.gamma_power, targetMetrics.gamma_power || 0.6, progress),
          high_gamma_power: observer.baseline_eeg.high_gamma_power,
          coherence_index: this.interpolateToTarget(observer.baseline_eeg.coherence_index, targetMetrics.coherence_index || 0.7, progress),
          focus_intensity: this.interpolateToTarget(observer.baseline_eeg.focus_intensity, targetMetrics.focus_intensity || 0.8, progress),
          attention_stability: this.interpolateToTarget(observer.baseline_eeg.attention_stability, 0.9, progress)
        };
        
        this.updateObserverEEG(observerId, simulatedEEG);
        
        // Calculate real-time score
        const currentScore = this.evaluateBiofeedbackPerformance(observer, protocol);
        phaseScore = Math.max(phaseScore, currentScore);
        
        // Provide real-time feedback
        if (realTimeEEGCallback && update % 10 === 0) {
          const feedback = this.generateBiofeedbackMessage(simulatedEEG, targetMetrics, phase.feedback_modality);
          realTimeEEGCallback(feedback);
        }
        
        await new Promise(resolve => setTimeout(resolve, updateInterval));
      }
      
      const phaseEndTime = Date.now();
      const actualDuration = (phaseEndTime - phaseStartTime) / 1000 / 60; // minutes
      
      phaseResults.push({
        phase: phase.phase_name,
        score: phaseScore,
        duration: actualDuration
      });
      
      totalScore += phaseScore;
      console.log(`Phase ${phase.phase_name} completed with score: ${phaseScore.toFixed(2)}`);
    }
    
    const finalScore = totalScore / protocol.training_phases.length;
    const success = finalScore >= 0.7; // 70% success threshold
    
    // Update observer training score
    observer.biofeedback_training_score = finalScore;
    const quantumCouplingImprovement = observer.quantum_coupling_strength - initialCouplingStrength;
    
    this.observers.set(observerId, observer);
    
    console.log(`Biofeedback training completed. Final score: ${finalScore.toFixed(2)}, Success: ${success}`);
    
    return {
      success,
      finalScore,
      phaseResults,
      quantumCouplingImprovement
    };
  }

  /**
   * Interpolate current value toward target based on progress
   */
  private interpolateToTarget(current: number, target: number, progress: number): number {
    // Add some noise and non-linear progression
    const noise = (Math.random() - 0.5) * 0.1;
    const nonLinearProgress = Math.pow(progress, 1.5); // Slower start, faster end
    return current + (target - current) * nonLinearProgress + noise;
  }

  /**
   * Generate biofeedback message based on current vs target metrics
   */
  private generateBiofeedbackMessage(
    current: EEGData,
    target: Partial<EEGData>,
    modality: 'visual' | 'auditory' | 'haptic' | 'multimodal'
  ): string {
    let message = '';
    
    if (target.gamma_power && current.gamma_power < target.gamma_power) {
      message += 'Increase focus intensity. ';
    }
    
    if (target.coherence_index && current.coherence_index < target.coherence_index) {
      message += 'Synchronize brain hemispheres. ';
    }
    
    if (target.theta_power && current.theta_power < target.theta_power) {
      message += 'Deepen meditative state. ';
    }
    
    if (target.focus_intensity && current.focus_intensity < target.focus_intensity) {
      message += 'Strengthen attention. ';
    }
    
    if (message === '') {
      message = 'Excellent! Maintain current state.';
    }
    
    return `[${modality.toUpperCase()}] ${message.trim()}`;
  }

  /**
   * Get comprehensive system status and performance metrics
   */
  getSystemStatus(): {
    totalObservers: number;
    activeObservers: number;
    totalQuantumTargets: number;
    activeCouplings: number;
    averageConsciousnessEffect: number;
    averageQuantumErrorReduction: number;
    topPerformingObserver: string | null;
    systemEffectiveness: number;
  } {
    const activeObservers = Array.from(this.observers.values()).filter(
      obs => obs.current_session_duration > 0 && obs.quantum_coupling_strength > this.CONSCIOUSNESS_COUPLING_THRESHOLD
    );
    
    const avgEffect = this.consciousness_effect_history.length > 0 ?
      this.consciousness_effect_history.reduce((a, b) => a + b) / this.consciousness_effect_history.length : 0;
    
    const avgErrorReduction = this.quantum_error_reduction_history.length > 0 ?
      this.quantum_error_reduction_history.reduce((a, b) => a + b) / this.quantum_error_reduction_history.length : 0;
    
    // Find top performing observer
    let topObserver: string | null = null;
    let topPerformance = 0;
    
    for (const [id, performance] of this.observer_performance_metrics) {
      if (performance.length > 0) {
        const avgPerformance = performance.reduce((a, b) => a + b) / performance.length;
        if (avgPerformance > topPerformance) {
          topPerformance = avgPerformance;
          topObserver = id;
        }
      }
    }
    
    const systemEffectiveness = (avgEffect + avgErrorReduction) / 2;
    
    return {
      totalObservers: this.observers.size,
      activeObservers: activeObservers.length,
      totalQuantumTargets: this.quantumTargets.size,
      activeCouplings: this.consciousnessCouplings.size,
      averageConsciousnessEffect: avgEffect,
      averageQuantumErrorReduction: avgErrorReduction,
      topPerformingObserver: topObserver,
      systemEffectiveness
    };
  }

  /**
   * Simulate consciousness-enhanced quantum algorithm execution
   */
  async simulateConsciousnessEnhancedQuantumExecution(
    algorithmName: string,
    observerIds: string[],
    duration: number = 10000 // milliseconds
  ): Promise<{
    success: boolean;
    finalErrorRate: number;
    consciousnessContribution: number;
    quantumCoherenceImprovement: number;
    oshEvidence: 'supports' | 'challenges' | 'neutral';
    evidenceStrength: number;
  }> {
    console.log(`Simulating consciousness-enhanced execution of ${algorithmName}`);
    
    const startTime = Date.now();
    const initialErrorRates = Array.from(this.quantumTargets.values()).map(t => t.error_rate);
    const initialCoherence = Array.from(this.quantumTargets.values()).map(t => t.current_coherence);
    
    // Establish couplings between observers and quantum targets
    const couplings: string[] = [];
    const targets = Array.from(this.quantumTargets.keys());
    
    for (let i = 0; i < observerIds.length && i < targets.length; i++) {
      const couplingId = this.establishQuantumCoupling(observerIds[i], targets[i], 'meditative_resonance');
      couplings.push(couplingId);
    }
    
    // Run simulation with conscious observation
    const steps = Math.ceil(duration / 100); // 10 Hz updates
    let totalConsciousnessEffect = 0;
    
    for (let step = 0; step < steps; step++) {
      for (let i = 0; i < observerIds.length && i < targets.length; i++) {
        const record = await this.performConsciousObservation(
          observerIds[i], 
          targets[i], 
          'stabilize', 
          100
        );
        totalConsciousnessEffect += record.consciousness_effect_strength;
      }
      
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // Calculate results
    const finalErrorRates = Array.from(this.quantumTargets.values()).map(t => t.error_rate);
    const finalCoherence = Array.from(this.quantumTargets.values()).map(t => t.current_coherence);
    
    const avgInitialErrorRate = initialErrorRates.reduce((a, b) => a + b) / initialErrorRates.length;
    const avgFinalErrorRate = finalErrorRates.reduce((a, b) => a + b) / finalErrorRates.length;
    const avgInitialCoherence = initialCoherence.reduce((a, b) => a + b) / initialCoherence.length;
    const avgFinalCoherence = finalCoherence.reduce((a, b) => a + b) / finalCoherence.length;
    
    const consciousnessContribution = totalConsciousnessEffect / (steps * observerIds.length);
    const errorReduction = (avgInitialErrorRate - avgFinalErrorRate) / avgInitialErrorRate;
    const coherenceImprovement = (avgFinalCoherence - avgInitialCoherence) / avgInitialCoherence;
    
    // Determine OSH evidence
    let oshEvidence: 'supports' | 'challenges' | 'neutral' = 'neutral';
    let evidenceStrength = 0.5;
    
    if (errorReduction > 0.1 && coherenceImprovement > 0.05) {
      oshEvidence = 'supports';
      evidenceStrength = Math.min(0.95, 0.5 + (errorReduction + coherenceImprovement) * 2);
    } else if (errorReduction < -0.05 || coherenceImprovement < -0.05) {
      oshEvidence = 'challenges';
      evidenceStrength = Math.min(0.95, 0.5 + Math.abs(errorReduction) + Math.abs(coherenceImprovement));
    }
    
    const success = avgFinalErrorRate < 0.0002; // 0.02% target
    
    console.log(`Consciousness-enhanced simulation completed. Error reduction: ${(errorReduction * 100).toFixed(2)}%`);
    
    return {
      success,
      finalErrorRate: avgFinalErrorRate,
      consciousnessContribution,
      quantumCoherenceImprovement: coherenceImprovement,
      oshEvidence,
      evidenceStrength
    };
  }

  /**
   * Get current metrics
   */
  getMetrics(): Record<string, number> {
    // Calculate average observer coherence
    let observerCoherence = 0;
    let observerCount = 0;
    for (const observer of this.observers.values()) {
      if (observer.quantum_coupling_strength > 0) {
        observerCoherence += observer.quantum_coupling_strength;
        observerCount++;
      }
    }
    if (observerCount > 0) {
      observerCoherence /= observerCount;
    }

    // Calculate average error reduction
    let errorReduction = 0;
    if (this.quantum_error_reduction_history.length > 0) {
      errorReduction = this.quantum_error_reduction_history.reduce((a, b) => a + b) / this.quantum_error_reduction_history.length;
    }

    // Calculate average final error rate
    let finalErrorRate = 0.001;
    let targetCount = 0;
    for (const target of this.quantumTargets.values()) {
      finalErrorRate += target.error_rate;
      targetCount++;
    }
    if (targetCount > 0) {
      finalErrorRate /= targetCount;
    }

    return {
      observerCoherence,
      errorReduction,
      finalErrorRate
    };
  }

  /**
   * Stop the engine
   */
  async stop(): Promise<void> {
    console.log('Stopping ConsciousObserverFeedbackEngine...');
    // Clear any active couplings
    this.consciousnessCouplings.clear();
    // Reset session durations
    for (const observer of this.observers.values()) {
      observer.current_session_duration = 0;
      observer.fatigue_level = 0;
    }
    console.log('ConsciousObserverFeedbackEngine stopped');
  }
}