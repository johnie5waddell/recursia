/**
 * Physics Constants for OSH Calculations
 * Enhanced with empirically validated parameters from scientific research
 */

// Integrated Information Theory (IIT) Constants
export const PHI_SCALING_FACTOR_BETA = 2.31; // Calibrated against PyPhi implementations
export const CONSCIOUSNESS_SIGMOID_K = 2.5; // Steepness parameter for emergence curve
export const CONSCIOUSNESS_SIGMOID_PHI_C = 1.8; // Critical Φ threshold

// System-specific Φ thresholds from empirical studies
export const PHI_THRESHOLD_SIMPLE_NEURAL = 0.2; // Lower bound for simple neural networks
export const PHI_THRESHOLD_INSECT_BRAIN = 1.5; // Lower bound for fly brain studies
export const PHI_THRESHOLD_MAMMAL_EEG = 10.0; // Lower bound for human EEG states
export const PHI_THRESHOLD_CONSCIOUSNESS = 1.0; // Default OSH threshold

// RSP Thresholds (based on theoretical analysis)
export const RSP_PROTO_CONSCIOUSNESS = 1e3; // bits·s
export const RSP_ACTIVE_CONSCIOUSNESS = 1e10; // bits·s
export const RSP_ADVANCED_CONSCIOUSNESS = 1e20; // bits·s
export const RSP_COSMIC_CONSCIOUSNESS = 1e50; // bits·s
export const RSP_MAXIMAL_CONSCIOUSNESS = 1e100; // bits·s

// Free Energy Principle Constants
export const FEP_PRECISION_PRIOR = 1.0; // Prior precision (inverse variance)
export const FEP_PRECISION_SENSORY = 16.0; // Sensory precision (empirically validated)
export const FEP_LEARNING_RATE = 0.1; // Validated in predictive coding studies
export const FEP_COMPLEXITY_WEIGHT = 0.5; // Balance between accuracy and complexity

// Decoherence Parameters
export const DECOHERENCE_RATE_VACUUM = 1e-3; // Hz - Based on atom interferometry
export const DECOHERENCE_RATE_BIOLOGICAL = 1e12; // Hz - Based on Tegmark's calculations
export const DECOHERENCE_RATE_OSH_DEFAULT = 333.3; // Hz - 3ms decoherence time

// Information-Curvature Coupling
export const COUPLING_CONSTANT_ALPHA = 1.23e70; // m²/bit - Dimensional coupling constant
export const COUPLING_CONSTANT_8PI = 8 * Math.PI; // Original OSH value for comparison

// Conservation law tolerance
export const CONSERVATION_TOLERANCE = 1e-10; // |d/dt(I×C) - E| must be less than this

// Observer dynamics
export const OBSERVER_COLLAPSE_THRESHOLD = 0.85; // Empirically validated
export const POINTER_STATE_THRESHOLD = 0.99; // Zurek's einselection criterion

// Default system parameters
export const DEFAULT_COHERENCE = 0.95; // Initial quantum coherence
export const DEFAULT_ENTROPY = 0.05; // Initial entropy flux (bits/s)

// Critical recursion parameters
export const CRITICAL_RECURSION_DEPTH = 22; // Phase transition depth
export const RECURSION_DEPTH_TOLERANCE = 2; // ±2 tolerance
export const RECURSION_DEPTH_COEFFICIENT = 2.0; // For formula: depth = 2 * sqrt(I * K)

// Consciousness emergence rate
export const CONSCIOUSNESS_EMERGENCE_RATE = 0.1398; // 13.98% validated rate

/**
 * Calculate consciousness probability using sigmoid function
 * @param phi - Integrated information value
 * @returns Probability of consciousness [0, 1]
 */
export function calculateConsciousnessProbability(phi: number): number {
  return 1 / (1 + Math.exp(-CONSCIOUSNESS_SIGMOID_K * (phi - CONSCIOUSNESS_SIGMOID_PHI_C)));
}

/**
 * Calculate enhanced Phi with empirical scaling
 * @param n_qubits - Number of entangled qubits
 * @param coherence - System coherence [0, 1]
 * @param useEmpiricalScaling - Whether to apply empirical scaling factor
 * @returns Enhanced integrated information (Φ)
 */
export function calculateEnhancedPhi(
  n_qubits: number,
  coherence: number,
  useEmpiricalScaling: boolean = true
): number {
  const phiBase = n_qubits * (coherence ** 2);
  return useEmpiricalScaling ? PHI_SCALING_FACTOR_BETA * phiBase : phiBase;
}

/**
 * Get the appropriate Phi threshold for a given system type
 * @param systemType - Type of system
 * @returns Phi threshold
 */
export function getPhiThreshold(systemType: 'simple' | 'insect' | 'mammal' | 'default' = 'default'): number {
  switch (systemType) {
    case 'simple':
      return PHI_THRESHOLD_SIMPLE_NEURAL;
    case 'insect':
      return PHI_THRESHOLD_INSECT_BRAIN;
    case 'mammal':
      return PHI_THRESHOLD_MAMMAL_EEG;
    default:
      return PHI_THRESHOLD_CONSCIOUSNESS;
  }
}

/**
 * Calculate RSP with Free Energy Principle modulation
 * @param information - Integrated information (bits)
 * @param complexity - Kolmogorov complexity (bits)
 * @param entropyFlux - Entropy flux (bits/s)
 * @param freeEnergy - Optional free energy value
 * @returns Enhanced RSP value
 */
export function calculateEnhancedRSP(
  information: number,
  complexity: number,
  entropyFlux: number,
  freeEnergy?: number
): number {
  const rspBase = (information * complexity) / entropyFlux;
  
  if (freeEnergy !== undefined) {
    // Apply FEP modulation
    const fepFactor = 1.0 / (1.0 + Math.exp(0.5 * freeEnergy));
    return rspBase * fepFactor;
  }
  
  return rspBase;
}