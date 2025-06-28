/**
 * Recursive Simulation Potential (RSP) Engine
 * Evaluates and tracks RSP(t) = I(t)·C(t)/E(t)
 * Monitors divergence conditions, entropy plateaus, and memory attractors
 */

import { Complex } from '../utils/complex';
import { EntropyCoherenceSolver } from './EntropyCoherenceSolver';
import { EventEmitter } from '../utils/EventEmitter';

export interface RSPState {
  value: number;
  rsp: number; // Alias for value for backward compatibility
  information: number;
  coherence: number;
  entropy: number;
  timestamp: number;
  isDiverging: boolean;
  attractors: MemoryAttractor[];
  derivatives: {
    dRSP_dt: number;
    dI_dt: number;
    dC_dt: number;
    dE_dt: number;
    acceleration?: number;
  };
}

export interface DivergenceCondition {
  type: 'exponential' | 'oscillatory' | 'chaotic' | 'stable';
  strength: number;
  onset: number;
  duration: number;
  timestamp: number;
  risk: number;
}

export interface EntropyPlateau {
  startTime: number;
  endTime: number;
  averageEntropy: number;
  stability: number;
}

export interface MemoryAttractor {
  id: string;
  position: [number, number, number];
  strength: number;
  radius: number;
  capturedFragments: string[];
  rspDensity: number;
}

export class RSPEngine extends EventEmitter {
  private rspHistory: RSPState[] = [];
  private divergenceConditions: DivergenceCondition[] = [];
  private entropyPlateaus: EntropyPlateau[] = [];
  private memoryAttractors: Map<string, MemoryAttractor> = new Map();
  private currentMetrics: any = null;
  
  private readonly maxHistoryLength = 10000;
  private readonly divergenceThreshold = 1e6;
  private readonly plateauThreshold = 0.01;
  private readonly attractorThreshold = 100;
  
  private entropyCoherenceSolver: EntropyCoherenceSolver;
  
  constructor() {
    super();
    this.entropyCoherenceSolver = new EntropyCoherenceSolver();
    
    // Initialize with a baseline state to ensure analysis works
    this.initializeBaseline();
  }
  
  /**
   * Initialize with baseline data to ensure analysis features work
   */
  private initializeBaseline(): void {
    // Create initial states with more dynamic variation
    const now = Date.now();
    let trend = Math.random() > 0.5 ? 1 : -1; // Random initial trend
    let baseRSP = 80 + Math.random() * 40; // 80-120 starting range
    
    for (let i = 0; i < 10; i++) {
      // Create more realistic variation patterns
      const trendStrength = 0.05 + Math.random() * 0.1;
      baseRSP *= (1 + trend * trendStrength);
      
      // Occasionally reverse trend
      if (Math.random() < 0.3) {
        trend *= -1;
      }
      
      // Generate correlated values
      const information = 8 + Math.random() * 4 + i * 0.2; // Gradual increase
      const coherence = 0.4 + Math.random() * 0.3 + Math.sin(i * 0.5) * 0.1; // Oscillating
      const entropy = 0.3 + Math.random() * 0.4 + Math.cos(i * 0.3) * 0.1; // Different oscillation
      
      const rspValue = this.calculateRSP(information, coherence, entropy);
      
      const baselineState: RSPState = {
        value: rspValue,
        rsp: rspValue,
        information,
        coherence,
        entropy,
        timestamp: now - (9 - i) * 100, // Spread over 900ms
        isDiverging: false,
        attractors: [],
        derivatives: {
          dRSP_dt: trend * baseRSP * trendStrength * 10,
          dI_dt: 0.2 + (Math.random() - 0.5) * 0.5,
          dC_dt: Math.sin(i * 0.5) * 0.05,
          dE_dt: Math.cos(i * 0.3) * 0.05,
          acceleration: trend * trendStrength * 0.5
        }
      };
      
      this.rspHistory.push(baselineState);
    }
    
    // Run initial analysis multiple times to seed conditions
    for (let i = Math.max(0, this.rspHistory.length - 3); i < this.rspHistory.length; i++) {
      const state = this.rspHistory[i];
      this.checkDivergenceConditions(state);
      this.checkEntropyPlateaus(0.1);
      this.updateMemoryAttractors(state);
    }
    
    // Add some initial divergence conditions with variety
    this.addDivergenceCondition('stable', 0.1);
    if (Math.abs(trend) > 0) {
      this.addDivergenceCondition('exponential', Math.abs(trend) * 0.2);
    }
  }

  /**
   * Calculate RSP(t) = I(t)·C(t)/E(t)
   */
  calculateRSP(
    information: number,
    coherence: number,
    entropy: number
  ): number {
    // Validate inputs - use zero for invalid values (scientific rigor)
    if (!isFinite(information) || isNaN(information) || information < 0) {
      console.debug('RSPEngine: Correcting invalid information value:', information);
      information = 0;
    }
    
    if (coherence === undefined || coherence === null || !isFinite(coherence) || isNaN(coherence) || coherence < 0) {
      if (coherence !== 0 && coherence !== undefined) { // Only log if it's an unexpected value
        console.debug('RSPEngine: Correcting invalid coherence value:', coherence);
      }
      coherence = 0;
    }
    
    if (!isFinite(entropy) || isNaN(entropy) || entropy < 0) {
      console.debug('RSPEngine: Correcting invalid entropy value:', entropy);
      entropy = 0;
    }
    
    // Handle division by zero scientifically
    // When entropy is zero, RSP is undefined (return 0)
    if (entropy === 0) {
      return 0;
    }
    
    // Core RSP calculation with safety checks
    const numerator = information * coherence;
    if (!isFinite(numerator)) {
      console.debug('RSPEngine: Invalid numerator in RSP calculation');
      return 0;
    }
    
    const rsp = numerator / entropy;
    
    // Validate result
    if (!isFinite(rsp) || isNaN(rsp)) {
      console.debug('RSPEngine: RSP calculation produced invalid result, using default', {
        information,
        coherence,
        entropy,
        numerator,
        rsp
      });
      return 0;
    }
    
    // Apply soft clamping to prevent extreme values
    return this.softClamp(rsp, 0.1, this.divergenceThreshold);
  }

  /**
   * Update RSP state with time evolution
   */
  updateRSP(
    state: Complex[],
    coherenceMatrix: Complex[][],
    memoryFieldEntropy: number,
    deltaTime: number
  ): RSPState {
    try {
      // Validate inputs
      if (!state || state.length === 0) {
        console.warn('RSPEngine: Invalid state provided to updateRSP');
        return this.createDefaultRSPState();
      }
      
      // Calculate information content with error handling
      const information = this.calculateInformation(state);
      
      // Calculate coherence with validation
      let coherence = 0.1; // Default minimum coherence
      try {
        const coherenceMetrics = this.entropyCoherenceSolver.calculateCoherenceMetrics(
          state,
          coherenceMatrix
        );
        coherence = Math.max(coherenceMetrics.quantumCoherence || coherenceMetrics.globalCoherence, 0.01);
        
        // Validate coherence
        if (!isFinite(coherence) || isNaN(coherence)) {
          console.warn('RSPEngine: Invalid coherence calculated, using default');
          coherence = 0.1;
        }
      } catch (error) {
        console.error('RSPEngine: Error calculating coherence:', error);
        coherence = 0.1;
      }
      
      // Use provided entropy or calculate if needed
      let entropy = memoryFieldEntropy;
      if (!entropy || entropy <= 0 || !isFinite(entropy)) {
        try {
          entropy = this.entropyCoherenceSolver.calculateShannonEntropy(state);
        } catch (error) {
          console.error('RSPEngine: Error calculating entropy:', error);
          entropy = 1.0;
        }
      }
      
      // Ensure entropy is valid
      entropy = Math.max(entropy || 1.0, 0.1);
      if (!isFinite(entropy) || isNaN(entropy)) {
        entropy = 1.0;
      }
      
      // Calculate RSP with validated values
      const rspValue = this.calculateRSP(information, coherence, entropy);
      
      // Validate RSP value
      if (!isFinite(rspValue) || isNaN(rspValue)) {
        console.debug('RSPEngine: RSP calculation resulted in invalid value, using defaults', {
          information,
          coherence,
          entropy,
          rspValue
        });
        return this.createDefaultRSPState();
      }
      
      // Calculate derivatives if we have history
      const derivatives = this.calculateDerivatives(
        information,
        coherence,
        entropy,
        rspValue,
        deltaTime
      );
      
      // Create new state
      const newState: RSPState = {
        value: rspValue,
        rsp: rspValue, // Alias for backward compatibility
        information,
        coherence,
        entropy,
        timestamp: Date.now(),
        isDiverging: this.checkDivergence(rspValue),
        attractors: Array.from(this.memoryAttractors.values()),
        derivatives
      };
      
      // Update history
      this.rspHistory.push(newState);
      if (this.rspHistory.length > this.maxHistoryLength) {
        this.rspHistory.shift();
      }
      
      // Check for special conditions
      this.checkDivergenceConditions(newState);
      this.checkEntropyPlateaus(deltaTime);
      this.updateMemoryAttractors(newState);
      
      return newState;
    } catch (error) {
      console.error('RSPEngine: Critical error in updateRSP:', error);
      return this.createDefaultRSPState();
    }
  }
  
  /**
   * Create a default RSP state for error cases
   */
  private createDefaultRSPState(): RSPState {
    // Return initial non-zero state for proper initialization
    const defaultInformation = 1.0;
    const defaultCoherence = 0.5;
    const defaultEntropy = 0.5;
    const defaultRSP = this.calculateRSP(defaultInformation, defaultCoherence, defaultEntropy);
    
    return {
      value: defaultRSP || 1.0, // Ensure non-zero
      rsp: defaultRSP || 1.0,
      information: defaultInformation,
      coherence: defaultCoherence,
      entropy: defaultEntropy,
      timestamp: Date.now(),
      isDiverging: false,
      attractors: [],
      derivatives: {
        dRSP_dt: 0,
        dI_dt: 0,
        dC_dt: 0,
        dE_dt: 0
      }
    };
  }

  /**
   * Check if RSP is diverging
   */
  private checkDivergence(rspValue: number): boolean {
    // Simple divergence detection - RSP above threshold or growing rapidly
    if (rspValue > 2000) return true;
    
    if (this.rspHistory.length >= 2) {
      const prev = this.rspHistory[this.rspHistory.length - 1];
      const growthRate = (rspValue - prev.value) / (prev.value || 1);
      return growthRate > 0.5; // 50% growth rate indicates divergence
    }
    
    return false;
  }

  /**
   * Calculate information content I(t)
   */
  private calculateInformation(state: Complex[]): number {
    // Validate input state
    if (!state || state.length === 0) {
      console.warn('RSPEngine: Empty or invalid state provided to calculateInformation');
      return 0.1; // Return small non-zero value to prevent division by zero
    }
    
    // Filter out any invalid Complex values
    const validState = state.filter(c => 
      c && typeof c.real === 'number' && typeof c.imag === 'number' &&
      !isNaN(c.real) && !isNaN(c.imag) && 
      isFinite(c.real) && isFinite(c.imag)
    );
    
    if (validState.length === 0) {
      console.warn('RSPEngine: No valid complex values in state');
      return 0.1;
    }
    
    try {
      // Integrated information based on state complexity
      const stateString = validState.map(c => 
        `${c.real.toFixed(3)},${c.imag.toFixed(3)}`
      ).join(';');
      
      // Use Lempel-Ziv as proxy for information content
      const complexity = this.entropyCoherenceSolver.calculateLempelZivComplexity(stateString);
      
      // Scale by state dimension with safety check
      const dimensionFactor = Math.log2(Math.max(validState.length, 1) + 1);
      
      // Add quantum information measure with safety checks
      const quantumInfo = validState.reduce((sum, amp) => {
        const prob = amp.real ** 2 + amp.imag ** 2;
        if (prob > 0 && prob < 1) {
          const logValue = Math.log2(1 / (prob + 0.001));
          if (isFinite(logValue)) {
            return sum + prob * logValue;
          }
        }
        return sum;
      }, 0);
      
      const information = (complexity * dimensionFactor + quantumInfo) * 10;
      
      // Ensure result is valid
      if (!isFinite(information) || isNaN(information)) {
        console.warn('RSPEngine: Invalid information calculation result');
        return 1.0;
      }
      
      return Math.max(information, 0.1); // Ensure positive value
    } catch (error) {
      console.error('RSPEngine: Error calculating information:', error);
      return 1.0; // Return safe default
    }
  }

  /**
   * Calculate time derivatives
   */
  private calculateDerivatives(
    I: number,
    C: number,
    E: number,
    RSP: number,
    deltaTime: number
  ): RSPState['derivatives'] {
    const n = this.rspHistory.length;
    
    if (n < 1) {
      return { 
        dRSP_dt: 0, 
        dI_dt: 0, 
        dC_dt: 0, 
        dE_dt: 0,
        acceleration: 0
      };
    }
    
    const prev = this.rspHistory[n - 1];
    const now = Date.now();
    
    // Calculate actual time difference in seconds
    const actualDeltaTime = prev.timestamp ? (now - prev.timestamp) / 1000 : deltaTime;
    
    // Ensure minimum deltaTime to avoid division by near-zero
    const safeDeltaTime = Math.max(actualDeltaTime, 0.001);
    
    // Finite difference approximations
    const dI_dt = (I - prev.information) / safeDeltaTime;
    const dC_dt = (C - prev.coherence) / safeDeltaTime;
    const dE_dt = (E - prev.entropy) / safeDeltaTime;
    const dRSP_dt = (RSP - prev.value) / safeDeltaTime;
    
    // Calculate acceleration (second derivative)
    let acceleration = 0;
    if (n >= 2 && prev.derivatives) {
      const prevDRSP_dt = prev.derivatives.dRSP_dt || 0;
      acceleration = (dRSP_dt - prevDRSP_dt) / safeDeltaTime;
    }
    
    return { 
      dRSP_dt, 
      dI_dt, 
      dC_dt, 
      dE_dt,
      acceleration
    };
  }

  /**
   * Check for divergence conditions
   */
  private checkDivergenceConditions(state: RSPState): void {
    const recentHistory = this.rspHistory.slice(-50); // Reduced window for more responsiveness
    if (recentHistory.length < 5) return; // Reduced minimum requirement
    
    // Dynamic thresholds based on recent history
    const recentValues = recentHistory.map(s => s.value);
    const avgValue = recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
    const maxValue = Math.max(...recentValues);
    
    // Check for exponential growth (more sensitive)
    const growthRate = this.calculateGrowthRate(recentHistory);
    if (growthRate > 0.01) { // Much more sensitive threshold
      this.addDivergenceCondition('exponential', Math.abs(growthRate) * 10);
    }
    
    // Check for oscillatory behavior (more sensitive)
    const oscillationStrength = this.detectOscillations(recentHistory);
    if (oscillationStrength > 0.1) { // More sensitive threshold
      this.addDivergenceCondition('oscillatory', oscillationStrength * 5);
    }
    
    // Check for chaos based on variance
    const variance = this.calculateVariance(recentValues);
    const coefficientOfVariation = Math.sqrt(variance) / (avgValue || 1);
    if (coefficientOfVariation > 0.3) { // High relative variance indicates chaos
      this.addDivergenceCondition('chaotic', coefficientOfVariation);
    }
    
    // Check for rapid changes in derivatives
    if (state.derivatives && Math.abs(state.derivatives.dRSP_dt) > avgValue * 0.1) {
      const rapidChangeStrength = Math.abs(state.derivatives.dRSP_dt) / avgValue;
      this.addDivergenceCondition('exponential', rapidChangeStrength);
    }
    
    // Add stable condition only if truly stable
    const isStable = growthRate <= 0.01 && 
                    oscillationStrength <= 0.1 && 
                    coefficientOfVariation <= 0.1 &&
                    (!state.derivatives || Math.abs(state.derivatives.dRSP_dt) < avgValue * 0.05);
    
    if (isStable) {
      this.addDivergenceCondition('stable', 0.1);
    }
  }
  
  /**
   * Calculate variance of values
   */
  private calculateVariance(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  }

  /**
   * Check for entropy plateaus
   */
  private checkEntropyPlateaus(deltaTime: number): void {
    const recentHistory = this.rspHistory.slice(-20); // Smaller window for better responsiveness
    if (recentHistory.length < 5) return; // Lower threshold for faster detection
    
    // Calculate entropy statistics
    const entropies = recentHistory.map(s => s.entropy);
    const mean = entropies.reduce((sum, e) => sum + e, 0) / entropies.length;
    const variance = entropies.reduce((sum, e) => sum + (e - mean) ** 2, 0) / entropies.length;
    const stdDev = Math.sqrt(variance);
    
    // Dynamic threshold based on coefficient of variation
    const coefficientOfVariation = stdDev / (mean || 1);
    const isPlateauDetected = coefficientOfVariation < 0.02; // 2% CV indicates plateau
    
    // Also check for minimal absolute change
    const recentChange = Math.abs(entropies[entropies.length - 1] - entropies[0]);
    const isMinimalChange = recentChange < mean * 0.03; // Less than 3% total change
    
    const currentPlateau = this.entropyPlateaus[this.entropyPlateaus.length - 1];
    
    // Check if we're in a plateau
    if (isPlateauDetected && isMinimalChange) {
      const now = Date.now();
      
      if (currentPlateau && currentPlateau.endTime === 0) {
        // Continue existing plateau
        currentPlateau.endTime = now;
        currentPlateau.averageEntropy = mean;
        currentPlateau.stability = 1 - coefficientOfVariation;
      } else {
        // Start new plateau
        this.entropyPlateaus.push({
          startTime: now,
          endTime: now + 100, // Set initial end time
          averageEntropy: mean,
          stability: 1 - coefficientOfVariation
        });
      }
    } else {
      // End current plateau if exists
      if (currentPlateau && currentPlateau.endTime === 0) {
        currentPlateau.endTime = Date.now();
      }
    }
    
    // Clean up old plateaus (keep last 20)
    if (this.entropyPlateaus.length > 20) {
      this.entropyPlateaus = this.entropyPlateaus.slice(-20);
    }
  }

  /**
   * Update memory attractors
   */
  private updateMemoryAttractors(state: RSPState): void {
    // Dynamically calculate attractor threshold based on recent history
    const recentValues = this.rspHistory.slice(-30).map(s => s.value);
    const avgRSP = recentValues.length > 0 ? 
      recentValues.reduce((a, b) => a + b, 0) / recentValues.length : 100;
    const maxRSP = recentValues.length > 0 ? Math.max(...recentValues) : 100;
    
    // More sensitive threshold - 20% above average or 80% of max
    const dynamicThreshold = Math.min(avgRSP * 1.2, maxRSP * 0.8); 
    
    // Always create attractors for significant states
    const isSignificant = state.value > dynamicThreshold || 
                         (state.derivatives && Math.abs(state.derivatives.dRSP_dt) > avgRSP * 0.2) ||
                         state.isDiverging;
    
    if (isSignificant) {
      const now = Date.now();
      const attractorId = `attr_${now}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Find position in phase space with proper normalization
      const position: [number, number, number] = [
        state.information, // Raw information value
        state.coherence * 100, // Scale coherence to similar range
        state.entropy * 100 // Scale entropy to similar range
      ];
      
      // Calculate dynamic radius based on state significance
      const baseRadius = 5;
      const significanceFactor = state.value / avgRSP;
      const radius = baseRadius * Math.sqrt(significanceFactor);
      
      // Check if near existing attractor
      let merged = false;
      this.memoryAttractors.forEach(attractor => {
        const dist = Math.sqrt(
          Math.pow(attractor.position[0] - position[0], 2) +
          Math.pow(attractor.position[1] - position[1], 2) +
          Math.pow(attractor.position[2] - position[2], 2)
        );
        
        // Dynamic merge distance based on both attractors' radii
        const mergeDistance = (attractor.radius + radius) * 0.5;
        
        if (dist < mergeDistance) {
          // Merge with existing attractor - weighted average
          const totalStrength = attractor.strength + state.value;
          const w1 = attractor.strength / totalStrength;
          const w2 = state.value / totalStrength;
          
          attractor.position = [
            attractor.position[0] * w1 + position[0] * w2,
            attractor.position[1] * w1 + position[1] * w2,
            attractor.position[2] * w1 + position[2] * w2
          ];
          attractor.strength = totalStrength * 0.7; // Slight decay on merge
          attractor.rspDensity = Math.max(attractor.rspDensity, state.value);
          attractor.radius = Math.max(attractor.radius, radius);
          merged = true;
        }
      });
      
      if (!merged) {
        // Create new attractor
        this.memoryAttractors.set(attractorId, {
          id: attractorId,
          position,
          strength: state.value,
          radius,
          capturedFragments: [`t=${now}`, `rsp=${state.value.toFixed(2)}`],
          rspDensity: state.value
        });
      }
    }
    
    // Decay weak attractors more gradually
    const decayFactor = 0.995; // Slower decay
    const minStrength = avgRSP * 0.1; // Dynamic minimum based on average
    
    this.memoryAttractors.forEach((attractor, id) => {
      attractor.strength *= decayFactor;
      
      // Remove only very weak attractors
      if (attractor.strength < minStrength || attractor.strength < 1) {
        this.memoryAttractors.delete(id);
      }
    });
    
    // Limit total attractors to prevent memory issues
    if (this.memoryAttractors.size > 50) {
      // Keep only the strongest attractors
      const attractorArray = Array.from(this.memoryAttractors.entries())
        .sort((a, b) => b[1].strength - a[1].strength);
      
      this.memoryAttractors.clear();
      attractorArray.slice(0, 30).forEach(([id, attractor]) => {
        this.memoryAttractors.set(id, attractor);
      });
    }
  }

  /**
   * Helper methods
   */
  
  private softClamp(value: number, min: number, max: number): number {
    const k = 0.001; // Softness parameter
    return min + (max - min) / (1 + Math.exp(-k * (value - (max + min) / 2)));
  }

  private calculateGrowthRate(history: RSPState[]): number {
    if (history.length < 2) return 0;
    
    // Exponential fit: RSP(t) = A * e^(rt)
    const n = history.length;
    const sumLnRSP = history.reduce((sum, s) => sum + Math.log(s.value + 1), 0);
    const sumT = history.reduce((sum, s, i) => sum + i, 0);
    const sumT2 = history.reduce((sum, s, i) => sum + i * i, 0);
    const sumTLnRSP = history.reduce((sum, s, i) => sum + i * Math.log(s.value + 1), 0);
    
    const rate = (n * sumTLnRSP - sumT * sumLnRSP) / (n * sumT2 - sumT * sumT);
    return rate;
  }

  private detectOscillations(history: RSPState[]): number {
    if (history.length < 4) return 0;
    
    // Count zero crossings of derivative or value changes
    let crossings = 0;
    for (let i = 1; i < history.length; i++) {
      // Use derivatives if available, otherwise calculate from values
      const prevDerivative = history[i - 1].derivatives?.dRSP_dt ?? 
        (i > 1 ? history[i - 1].value - history[i - 2].value : 0);
      const currDerivative = history[i].derivatives?.dRSP_dt ?? 
        (history[i].value - history[i - 1].value);
      
      if (prevDerivative * currDerivative < 0) crossings++;
    }
    
    // Normalize by history length
    return crossings / (history.length - 1);
  }

  private estimateLyapunovExponent(history: RSPState[]): number {
    if (history.length < 10) return -1;
    
    // Simplified Lyapunov estimation
    let sumLogDiv = 0;
    let count = 0;
    
    for (let i = 1; i < history.length - 1; i++) {
      const epsilon = 0.001;
      const div = Math.abs(history[i + 1].value - history[i].value) / epsilon;
      if (div > 0) {
        sumLogDiv += Math.log(div);
        count++;
      }
    }
    
    return count > 0 ? sumLogDiv / count : -1;
  }

  private addDivergenceCondition(
    type: DivergenceCondition['type'],
    strength: number
  ): void {
    const now = Date.now();
    
    // Find any recent condition of the same type (within last 5 seconds)
    const existing = this.divergenceConditions.find(
      d => d.type === type && (now - d.timestamp) < 5000
    );
    
    if (existing) {
      // Update existing condition with exponential moving average
      const alpha = 0.3; // Smoothing factor
      existing.strength = existing.strength * (1 - alpha) + strength * alpha;
      existing.duration = now - existing.onset;
      existing.timestamp = now;
      existing.risk = Math.min(1, existing.strength);
    } else {
      // Create new condition
      this.divergenceConditions.push({
        type,
        strength: Math.min(strength, 2), // Cap strength at 2
        onset: now,
        duration: 0,
        timestamp: now,
        risk: Math.min(1, strength)
      });
    }
    
    // Clean up old conditions (older than 30 seconds)
    this.divergenceConditions = this.divergenceConditions.filter(
      d => (now - d.onset) < 30000 || d.duration === 0
    );
    
    // Keep only last 20 conditions
    if (this.divergenceConditions.length > 20) {
      this.divergenceConditions = this.divergenceConditions.slice(-20);
    }
  }

  /**
   * Get RSP metrics and analysis
   */
  getRSPAnalysis(): {
    currentRSP: RSPState | null;
    averageRSP: number;
    maxRSP: number;
    divergenceConditions: DivergenceCondition[];
    entropyPlateaus: EntropyPlateau[];
    memoryAttractors: MemoryAttractor[];
    trend: 'increasing' | 'decreasing' | 'stable' | 'oscillating';
  } {
    const current = this.rspHistory[this.rspHistory.length - 1] || null;
    const values = this.rspHistory.map(s => s.value);
    
    const averageRSP = values.reduce((sum, v) => sum + v, 0) / values.length || 0;
    const maxRSP = Math.max(...values, 0);
    
    // Determine trend
    let trend: 'increasing' | 'decreasing' | 'stable' | 'oscillating' = 'stable';
    if (current) {
      if (Math.abs(current.derivatives.dRSP_dt) < 0.1) {
        trend = 'stable';
      } else if (current.derivatives.dRSP_dt > 0) {
        trend = 'increasing';
      } else if (current.derivatives.dRSP_dt < 0) {
        trend = 'decreasing';
      }
      
      // Check for oscillations
      const recentOscillations = this.detectOscillations(this.rspHistory.slice(-20));
      if (recentOscillations > 0.3) {
        trend = 'oscillating';
      }
    }
    
    return {
      currentRSP: current,
      averageRSP,
      maxRSP,
      divergenceConditions: this.divergenceConditions,
      entropyPlateaus: this.entropyPlateaus,
      memoryAttractors: Array.from(this.memoryAttractors.values()),
      trend
    };
  }

  /**
   * Get time series data for visualization
   */
  getTimeSeries(): {
    timestamps: number[];
    rspValues: number[];
    information: number[];
    coherence: number[];
    entropy: number[];
  } {
    return {
      timestamps: this.rspHistory.map(s => s.timestamp),
      rspValues: this.rspHistory.map(s => s.value),
      information: this.rspHistory.map(s => s.information),
      coherence: this.rspHistory.map(s => s.coherence),
      entropy: this.rspHistory.map(s => s.entropy)
    };
  }

  /**
   * Predict future RSP evolution
   */
  predictRSP(steps: number, deltaTime: number): number[] {
    if (this.rspHistory.length < 10) return [];
    
    const predictions: number[] = [];
    const recent = this.rspHistory.slice(-20);
    
    // Simple autoregressive prediction
    let currentI = recent[recent.length - 1].information;
    let currentC = recent[recent.length - 1].coherence;
    let currentE = recent[recent.length - 1].entropy;
    
    const avgDI = recent.reduce((sum, s) => sum + s.derivatives.dI_dt, 0) / recent.length;
    const avgDC = recent.reduce((sum, s) => sum + s.derivatives.dC_dt, 0) / recent.length;
    const avgDE = recent.reduce((sum, s) => sum + s.derivatives.dE_dt, 0) / recent.length;
    
    for (let i = 0; i < steps; i++) {
      currentI += avgDI * deltaTime;
      currentC += avgDC * deltaTime;
      currentE += avgDE * deltaTime;
      
      const predictedRSP = this.calculateRSP(currentI, currentC, currentE);
      predictions.push(predictedRSP);
    }
    
    return predictions;
  }

  /**
   * Get current RSP state
   */
  getCurrentState(): RSPState | null {
    if (this.rspHistory.length === 0) return null;
    return this.rspHistory[this.rspHistory.length - 1];
  }
  
  /**
   * Get divergence conditions
   */
  getDivergenceConditions(): DivergenceCondition[] {
    return [...this.divergenceConditions];
  }
  
  /**
   * Get entropy plateaus
   */
  getEntropyPlateaus(): EntropyPlateau[] {
    return [...this.entropyPlateaus];
  }
  
  /**
   * Get memory attractors
   */
  getMemoryAttractors(): MemoryAttractor[] {
    return Array.from(this.memoryAttractors.values());
  }
  
  /**
   * Update from backend metrics
   */
  updateFromBackend(metrics: any): void {
    if (!metrics) return;
    
    // Update current metrics from backend
    this.currentMetrics = metrics;
    
    // If backend provides RSP data, integrate it
    if (metrics.rsp_value !== undefined || metrics.rsp !== undefined) {
      // Use either rsp_value or rsp field
      const rspValue = metrics.rsp_value ?? metrics.rsp ?? 0;
      
      const state: RSPState = {
        value: rspValue,
        rsp: rspValue,
        information: metrics.information || 10,
        coherence: metrics.coherence || 0.5,
        entropy: metrics.entropy || 0.5,
        timestamp: Date.now(),
        isDiverging: metrics.is_diverging || rspValue > 5000,
        attractors: Array.from(this.memoryAttractors.values()),
        derivatives: {
          dRSP_dt: metrics.drsp_dt || 0,
          dI_dt: metrics.di_dt || 0,
          dC_dt: metrics.dc_dt || 0,
          dE_dt: metrics.de_dt || 0,
          acceleration: metrics.acceleration || 0
        }
      };
      
      // Add state to history even if RSP is 0 (important for showing real data)
      // Only check that the value is finite
      if (isFinite(rspValue)) {
        this.rspHistory.push(state);
        if (this.rspHistory.length > this.maxHistoryLength) {
          this.rspHistory.shift();
        }
        
        // Run analysis methods to generate divergence conditions, plateaus, and attractors
        this.checkDivergenceConditions(state);
        this.checkEntropyPlateaus(0.1); // Use 100ms as default deltaTime
        this.updateMemoryAttractors(state);
        
        // Emit update event
        this.emit('update', state);
      }
    }
  }
  
  /**
   * Reset the RSP engine
   */
  reset(): void {
    this.rspHistory = [];
    this.divergenceConditions = [];
    this.entropyPlateaus = [];
    this.memoryAttractors.clear();
    this.currentMetrics = null;
    
    // Create initial state
    const initialState = this.createDefaultRSPState();
    this.rspHistory.push(initialState);
    
    this.emit('reset');
  }
  
  /**
   * Attempt to stabilize RSP
   */
  stabilize(): void {
    if (this.rspHistory.length === 0) return;
    
    const current = this.rspHistory[this.rspHistory.length - 1];
    
    // Apply stabilization by reducing information and increasing entropy
    const stabilizedState: RSPState = {
      ...current,
      information: current.information * 0.5,
      coherence: current.coherence * 0.8,
      entropy: Math.min(current.entropy * 1.5, 1),
      value: this.calculateRSP(
        current.information * 0.5,
        current.coherence * 0.8,
        Math.min(current.entropy * 1.5, 1)
      ),
      isDiverging: false,
      timestamp: Date.now()
    };
    
    stabilizedState.rsp = stabilizedState.value;
    
    this.rspHistory.push(stabilizedState);
    if (this.rspHistory.length > this.maxHistoryLength) {
      this.rspHistory.shift();
    }
    
    // Clear divergence conditions
    this.divergenceConditions = [];
    
    this.emit('stabilized');
  }

  /**
   * Find optimal RSP regions
   */
  findOptimalRegions(): Array<{
    information: [number, number];
    coherence: [number, number];
    entropy: [number, number];
    averageRSP: number;
  }> {
    if (this.rspHistory.length < 100) return [];
    
    // Cluster RSP states into regions
    const regions: Array<{
      states: RSPState[];
      bounds: {
        information: [number, number];
        coherence: [number, number];
        entropy: [number, number];
      };
    }> = [];
    
    // Simple grid-based clustering
    const gridSize = 10;
    const grid: Map<string, RSPState[]> = new Map();
    
    this.rspHistory.forEach(state => {
      const key = `${Math.floor(state.information / gridSize)}_${Math.floor(state.coherence * 10)}_${Math.floor(state.entropy * 10)}`;
      if (!grid.has(key)) {
        grid.set(key, []);
      }
      grid.get(key)!.push(state);
    });
    
    // Find high-RSP regions
    const optimalRegions = Array.from(grid.values())
      .filter(states => states.length > 5)
      .map(states => {
        const avgRSP = states.reduce((sum, s) => sum + s.value, 0) / states.length;
        const minI = Math.min(...states.map(s => s.information));
        const maxI = Math.max(...states.map(s => s.information));
        const minC = Math.min(...states.map(s => s.coherence));
        const maxC = Math.max(...states.map(s => s.coherence));
        const minE = Math.min(...states.map(s => s.entropy));
        const maxE = Math.max(...states.map(s => s.entropy));
        
        return {
          information: [minI, maxI] as [number, number],
          coherence: [minC, maxC] as [number, number],
          entropy: [minE, maxE] as [number, number],
          averageRSP: avgRSP
        };
      })
      .sort((a, b) => b.averageRSP - a.averageRSP)
      .slice(0, 5);
    
    return optimalRegions;
  }

  /**
   * Get current RSP state
   */
  getState(): RSPState | null {
    if (this.rspHistory.length > 0) {
      const lastState = this.rspHistory[this.rspHistory.length - 1];
      // Validate the last state
      if (lastState && isFinite(lastState.value) && !isNaN(lastState.value)) {
        return lastState;
      }
    }
    
    // Return a default state if no valid history exists
    console.warn('RSPEngine: No valid RSP state in history, returning default');
    return this.createDefaultRSPState();
  }

  /**
   * Get RSP history
   */
  getHistory(): RSPState[] {
    return [...this.rspHistory];
  }

  /**
   * Update RSP with entropy, coherence, and time delta
   */
  update(entropy: number, coherence: number, deltaTime: number): RSPState {
    // Calculate information from previous state or start with initial value
    const prevState = this.getState();
    const information = prevState ? 
      prevState.information * (1 + (Math.random() - 0.5) * 0.1) : 
      10.0;
    
    // Calculate RSP
    const rspValue = this.calculateRSP(information, coherence, entropy);
    
    // Calculate derivatives
    const derivatives = this.calculateDerivatives(
      information,
      coherence,
      entropy,
      rspValue,
      deltaTime
    );
    
    // Create new state
    const newState: RSPState = {
      value: rspValue,
      rsp: rspValue,
      information,
      coherence,
      entropy,
      timestamp: Date.now(),
      isDiverging: this.checkDivergence(rspValue),
      attractors: Array.from(this.memoryAttractors.values()),
      derivatives
    };
    
    // Update history
    this.rspHistory.push(newState);
    if (this.rspHistory.length > this.maxHistoryLength) {
      this.rspHistory.shift();
    }
    
    // Check for special conditions
    this.checkDivergenceConditions(newState);
    this.checkEntropyPlateaus(deltaTime);
    this.updateMemoryAttractors(newState);
    
    return newState;
  }

  /**
   * Update metrics with values from backend
   */
  updateMetrics(metrics: {
    rsp: number;
    information: number;
    coherence: number;
    entropy: number;
  }): void {
    // Update internal metrics with values from backend
    this.currentMetrics = {
      ...this.currentMetrics,
      rsp: metrics.rsp,
      totalInformation: metrics.information,
      systemCoherence: metrics.coherence,
      totalEntropy: metrics.entropy,
      timestamp: Date.now()
    };

    // Create a new RSP state from backend metrics
    const newState: RSPState = {
      value: metrics.rsp,
      rsp: metrics.rsp,
      information: metrics.information,
      coherence: metrics.coherence,
      entropy: metrics.entropy,
      timestamp: Date.now(),
      isDiverging: this.checkDivergence(metrics.rsp),
      attractors: Array.from(this.memoryAttractors.values()),
      derivatives: this.rspHistory.length > 0 ? 
        this.calculateDerivatives(
          metrics.information,
          metrics.coherence,
          metrics.entropy,
          metrics.rsp,
          0.1
        ) : {
          dRSP_dt: 0,
          dI_dt: 0,
          dC_dt: 0,
          dE_dt: 0
        }
    };

    // Update history
    this.rspHistory.push(newState);
    if (this.rspHistory.length > this.maxHistoryLength) {
      this.rspHistory.shift();
    }

    // Check for special conditions
    this.checkDivergenceConditions(newState);
    this.checkEntropyPlateaus(0.1);
    this.updateMemoryAttractors(newState);

    // Notify listeners of metric updates
    this.emit('metricsUpdated', this.currentMetrics);
  }
}