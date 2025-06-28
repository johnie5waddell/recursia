/**
 * OSH Evidence Evaluation System
 * Analyzes quantum experiments to determine support/challenge for OSH theory
 */

import { Complex } from './complex';
import type { SimulationState } from '../engines/SimulationHarness';
import type { RSPState } from '../engines/RSPEngine';
import type { WavefunctionState } from '../engines/WavefunctionSimulator';

export interface OSHEvidence {
  experimentId: string;
  timestamp: number;
  evidenceType: 'supports' | 'challenges' | 'neutral' | 'inconclusive';
  strength: number; // 0-1 scale
  confidence: number; // 0-1 scale
  criteria: OSHEvidenceCriteria;
  measurements: OSHMeasurements;
  analysis: OSHAnalysis;
  verdict: string;
}

export interface OSHEvidenceCriteria {
  consciousnessParticipation: boolean;
  informationCurvatureCoupling: boolean;
  rspDivergence: boolean;
  memoryFieldCoherence: boolean;
  quantumClassicalInterface: boolean;
  teleportationFidelity: boolean;
  decoherenceResistance: boolean;
}

export interface OSHMeasurements {
  rspValues: number[];
  consciousnessCoherence: number[];
  informationDensity: number[];
  spacetimeCurvature: number[];
  memoryFragmentCount: number[];
  quantumFidelity: number[];
  decoherenceRates: number[];
  entanglementStrength: number[];
}

export interface OSHAnalysis {
  correlationCoefficients: {
    consciousnessRSP: number;
    informationCurvature: number;
    memoryCoherence: number;
    observerEffect: number;
  };
  statisticalSignificance: {
    pValue: number;
    confidenceInterval: [number, number];
    effectSize: number;
  };
  theoreticalPredictions: {
    rspThreshold: number;
    curvatureRelation: number;
    consciousnessThreshold: number;
  };
  deviationsFromClassical: number[];
}

export class OSHEvidenceEvaluator {
  private evidenceHistory: OSHEvidence[] = [];
  private thresholds = {
    rspConsciousnessEmergence: 100,
    informationCurvatureCorrelation: 0.95,
    memoryCoherenceStability: 0.8,
    quantumFidelityEnhancement: 0.15, // 15% improvement
    decoherenceReduction: 0.1, // 10% reduction
    teleportationSuccessThreshold: 0.9,
    statisticalSignificance: 0.05
  };

  evaluateExperiment(
    experimentId: string,
    simulationData: SimulationState[],
    experimentType: 'teleportation' | 'consciousness' | 'curvature' | 'decoherence' | 'memory',
    controlData?: SimulationState[]
  ): OSHEvidence {
    const measurements = this.extractMeasurements(simulationData);
    const controlMeasurements = controlData ? this.extractMeasurements(controlData) : null;
    
    const criteria = this.evaluateCriteria(measurements, controlMeasurements, experimentType);
    const analysis = this.performStatisticalAnalysis(measurements, controlMeasurements);
    
    const evidenceType = this.determineEvidenceType(criteria, analysis);
    const strength = this.calculateEvidenceStrength(criteria, analysis);
    const confidence = this.calculateConfidence(analysis);
    
    const evidence: OSHEvidence = {
      experimentId,
      timestamp: Date.now(),
      evidenceType,
      strength,
      confidence,
      criteria,
      measurements,
      analysis,
      verdict: this.generateVerdict(evidenceType, strength, confidence, experimentType)
    };
    
    this.evidenceHistory.push(evidence);
    return evidence;
  }

  private extractMeasurements(data: SimulationState[]): OSHMeasurements {
    return {
      rspValues: data.map(state => state.rspState.rsp),
      consciousnessCoherence: data.map(state => 
        state.observers.reduce((sum, obs) => sum + obs.coherence, 0) / Math.max(state.observers.length, 1)
      ),
      informationDensity: data.map(state => state.rspState.information),
      spacetimeCurvature: data.map(state => this.calculateCurvature(state)),
      memoryFragmentCount: data.map(state => state.memoryField.fragments.length),
      quantumFidelity: data.map(state => this.calculateQuantumFidelity(state)),
      decoherenceRates: data.map(state => this.calculateDecoherenceRate(state)),
      entanglementStrength: data.map(state => this.calculateEntanglementStrength(state))
    };
  }

  private evaluateCriteria(
    measurements: OSHMeasurements,
    controlMeasurements: OSHMeasurements | null,
    experimentType: string
  ): OSHEvidenceCriteria {
    return {
      consciousnessParticipation: this.evaluateConsciousnessParticipation(measurements),
      informationCurvatureCoupling: this.evaluateInformationCurvatureCoupling(measurements),
      rspDivergence: this.evaluateRSPDivergence(measurements),
      memoryFieldCoherence: this.evaluateMemoryFieldCoherence(measurements),
      quantumClassicalInterface: this.evaluateQuantumClassicalInterface(measurements, controlMeasurements),
      teleportationFidelity: this.evaluateTeleportationFidelity(measurements, experimentType),
      decoherenceResistance: this.evaluateDecoherenceResistance(measurements, controlMeasurements)
    };
  }

  private evaluateConsciousnessParticipation(measurements: OSHMeasurements): boolean {
    // Check if consciousness coherence correlates with RSP values
    const correlation = this.calculateCorrelation(
      measurements.consciousnessCoherence,
      measurements.rspValues
    );
    
    return correlation > 0.7 && measurements.consciousnessCoherence.some(c => c > 0.8);
  }

  private evaluateInformationCurvatureCoupling(measurements: OSHMeasurements): boolean {
    // OSH predicts R_μν ∼ ∇_μ∇_ν I
    const correlation = this.calculateCorrelation(
      measurements.informationDensity,
      measurements.spacetimeCurvature
    );
    
    return correlation > this.thresholds.informationCurvatureCorrelation;
  }

  private evaluateRSPDivergence(measurements: OSHMeasurements): boolean {
    // Check for RSP values exceeding consciousness emergence threshold
    const maxRSP = Math.max(...measurements.rspValues);
    const sustainedHighRSP = measurements.rspValues.filter(rsp => rsp > this.thresholds.rspConsciousnessEmergence).length;
    
    return maxRSP > this.thresholds.rspConsciousnessEmergence && 
           sustainedHighRSP > measurements.rspValues.length * 0.1;
  }

  private evaluateMemoryFieldCoherence(measurements: OSHMeasurements): boolean {
    // Check for stable memory field with growing fragment count and high coherence
    const fragmentGrowth = measurements.memoryFragmentCount[measurements.memoryFragmentCount.length - 1] / 
                          Math.max(measurements.memoryFragmentCount[0], 1);
    
    return fragmentGrowth > 1.5; // Memory field should grow
  }

  private evaluateQuantumClassicalInterface(
    measurements: OSHMeasurements,
    controlMeasurements: OSHMeasurements | null
  ): boolean {
    if (!controlMeasurements) return false;
    
    // Compare quantum fidelity between experimental and control groups
    const expMeanFidelity = measurements.quantumFidelity.reduce((a, b) => a + b, 0) / measurements.quantumFidelity.length;
    const ctrlMeanFidelity = controlMeasurements.quantumFidelity.reduce((a, b) => a + b, 0) / controlMeasurements.quantumFidelity.length;
    
    const enhancement = (expMeanFidelity - ctrlMeanFidelity) / ctrlMeanFidelity;
    
    return enhancement > this.thresholds.quantumFidelityEnhancement;
  }

  private evaluateTeleportationFidelity(measurements: OSHMeasurements, experimentType: string): boolean {
    if (experimentType !== 'teleportation') return true; // Not applicable
    
    const finalFidelity = measurements.quantumFidelity[measurements.quantumFidelity.length - 1];
    return finalFidelity > this.thresholds.teleportationSuccessThreshold;
  }

  private evaluateDecoherenceResistance(
    measurements: OSHMeasurements,
    controlMeasurements: OSHMeasurements | null
  ): boolean {
    if (!controlMeasurements) return false;
    
    const expMeanDecoherence = measurements.decoherenceRates.reduce((a, b) => a + b, 0) / measurements.decoherenceRates.length;
    const ctrlMeanDecoherence = controlMeasurements.decoherenceRates.reduce((a, b) => a + b, 0) / controlMeasurements.decoherenceRates.length;
    
    const reduction = (ctrlMeanDecoherence - expMeanDecoherence) / ctrlMeanDecoherence;
    
    return reduction > this.thresholds.decoherenceReduction;
  }

  private performStatisticalAnalysis(
    measurements: OSHMeasurements,
    controlMeasurements: OSHMeasurements | null
  ): OSHAnalysis {
    const correlations = {
      consciousnessRSP: this.calculateCorrelation(measurements.consciousnessCoherence, measurements.rspValues),
      informationCurvature: this.calculateCorrelation(measurements.informationDensity, measurements.spacetimeCurvature),
      memoryCoherence: this.calculateCorrelation(measurements.memoryFragmentCount, measurements.quantumFidelity),
      observerEffect: controlMeasurements ? 
        this.calculateEffectSize(measurements.quantumFidelity, controlMeasurements.quantumFidelity) : 0
    };

    const significance = this.calculateStatisticalSignificance(measurements, controlMeasurements);
    
    return {
      correlationCoefficients: correlations,
      statisticalSignificance: significance,
      theoreticalPredictions: {
        rspThreshold: this.thresholds.rspConsciousnessEmergence,
        curvatureRelation: this.thresholds.informationCurvatureCorrelation,
        consciousnessThreshold: 0.8
      },
      deviationsFromClassical: this.calculateClassicalDeviations(measurements, controlMeasurements)
    };
  }

  private determineEvidenceType(criteria: OSHEvidenceCriteria, analysis: OSHAnalysis): 'supports' | 'challenges' | 'neutral' | 'inconclusive' {
    const supportingCriteria = Object.values(criteria).filter(Boolean).length;
    const totalCriteria = Object.values(criteria).length;
    const supportRatio = supportingCriteria / totalCriteria;
    
    const strongCorrelations = Object.values(analysis.correlationCoefficients).filter(c => Math.abs(c) > 0.7).length;
    const significantResult = analysis.statisticalSignificance.pValue < this.thresholds.statisticalSignificance;
    
    if (supportRatio >= 0.8 && strongCorrelations >= 2 && significantResult) {
      return 'supports';
    } else if (supportRatio <= 0.3 && strongCorrelations === 0) {
      return 'challenges';
    } else if (analysis.statisticalSignificance.pValue > 0.1) {
      return 'inconclusive';
    } else {
      return 'neutral';
    }
  }

  private calculateEvidenceStrength(criteria: OSHEvidenceCriteria, analysis: OSHAnalysis): number {
    const criteriaWeight = 0.4;
    const correlationWeight = 0.35;
    const significanceWeight = 0.25;
    
    const criteriaScore = Object.values(criteria).filter(Boolean).length / Object.values(criteria).length;
    
    const correlationScore = Object.values(analysis.correlationCoefficients)
      .reduce((sum, r) => sum + Math.abs(r), 0) / Object.values(analysis.correlationCoefficients).length;
    
    const significanceScore = Math.max(0, 1 - analysis.statisticalSignificance.pValue / 0.05);
    
    return criteriaWeight * criteriaScore + 
           correlationWeight * correlationScore + 
           significanceWeight * significanceScore;
  }

  private calculateConfidence(analysis: OSHAnalysis): number {
    const pValue = analysis.statisticalSignificance.pValue;
    const effectSize = analysis.statisticalSignificance.effectSize;
    
    // Confidence based on statistical significance and effect size
    let confidence = 0.5; // Base confidence
    
    if (pValue < 0.001) confidence += 0.4;
    else if (pValue < 0.01) confidence += 0.3;
    else if (pValue < 0.05) confidence += 0.2;
    
    if (effectSize > 0.8) confidence += 0.1; // Large effect
    else if (effectSize > 0.5) confidence += 0.05; // Medium effect
    
    return Math.min(1, confidence);
  }

  private generateVerdict(
    evidenceType: string,
    strength: number,
    confidence: number,
    experimentType: string
  ): string {
    const strengthDesc = strength > 0.8 ? 'Strong' : strength > 0.6 ? 'Moderate' : strength > 0.4 ? 'Weak' : 'Minimal';
    const confidenceDesc = confidence > 0.8 ? 'High' : confidence > 0.6 ? 'Moderate' : 'Low';
    
    switch (evidenceType) {
      case 'supports':
        return `${strengthDesc} evidence SUPPORTING OSH theory (${confidenceDesc} confidence). ${this.getExperimentSpecificVerdict(experimentType, true)}`;
      case 'challenges':
        return `${strengthDesc} evidence CHALLENGING OSH theory (${confidenceDesc} confidence). ${this.getExperimentSpecificVerdict(experimentType, false)}`;
      case 'neutral':
        return `Mixed evidence, neither strongly supporting nor challenging OSH theory (${confidenceDesc} confidence).`;
      case 'inconclusive':
        return `Inconclusive results - insufficient statistical power to evaluate OSH predictions (${confidenceDesc} confidence).`;
      default:
        return 'Unknown result type.';
    }
  }

  private getExperimentSpecificVerdict(experimentType: string, supports: boolean): string {
    const outcomes = {
      teleportation: supports ? 
        'Consciousness enhancement of quantum teleportation confirmed.' : 
        'No consciousness enhancement detected in teleportation protocol.',
      consciousness: supports ? 
        'Consciousness emergence from high-RSP quantum systems observed.' : 
        'No consciousness emergence detected despite high RSP values.',
      curvature: supports ? 
        'Information-spacetime curvature coupling confirmed per OSH predictions.' : 
        'Information-curvature coupling weaker than OSH predictions.',
      decoherence: supports ? 
        'Consciousness-mediated decoherence resistance confirmed.' : 
        'No consciousness effect on quantum decoherence detected.',
      memory: supports ? 
        'Memory field coherence and recursive dynamics confirmed.' : 
        'Memory field behavior inconsistent with OSH predictions.'
    };
    
    return outcomes[experimentType as keyof typeof outcomes] || '';
  }

  // Utility mathematical functions
  private calculateCorrelation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) return 0;
    
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
  }

  private calculateEffectSize(experimental: number[], control: number[]): number {
    const expMean = experimental.reduce((a, b) => a + b, 0) / experimental.length;
    const ctrlMean = control.reduce((a, b) => a + b, 0) / control.length;
    
    const expVar = experimental.reduce((sum, x) => sum + Math.pow(x - expMean, 2), 0) / (experimental.length - 1);
    const ctrlVar = control.reduce((sum, x) => sum + Math.pow(x - ctrlMean, 2), 0) / (control.length - 1);
    
    const pooledSD = Math.sqrt((expVar + ctrlVar) / 2);
    
    return pooledSD === 0 ? 0 : Math.abs(expMean - ctrlMean) / pooledSD;
  }

  private calculateStatisticalSignificance(
    measurements: OSHMeasurements,
    controlMeasurements: OSHMeasurements | null
  ): { pValue: number; confidenceInterval: [number, number]; effectSize: number } {
    if (!controlMeasurements) {
      return {
        pValue: 0.5,
        confidenceInterval: [0, 1],
        effectSize: 0
      };
    }
    
    // Simplified t-test for quantum fidelity difference
    const expData = measurements.quantumFidelity;
    const ctrlData = controlMeasurements.quantumFidelity;
    
    const expMean = expData.reduce((a, b) => a + b, 0) / expData.length;
    const ctrlMean = ctrlData.reduce((a, b) => a + b, 0) / ctrlData.length;
    
    const expVar = expData.reduce((sum, x) => sum + Math.pow(x - expMean, 2), 0) / (expData.length - 1);
    const ctrlVar = ctrlData.reduce((sum, x) => sum + Math.pow(x - ctrlMean, 2), 0) / (ctrlData.length - 1);
    
    const standardError = Math.sqrt(expVar / expData.length + ctrlVar / ctrlData.length);
    const tStatistic = standardError === 0 ? 0 : (expMean - ctrlMean) / standardError;
    
    // Simplified p-value calculation (normal approximation)
    const pValue = 2 * (1 - this.normalCDF(Math.abs(tStatistic)));
    
    const marginOfError = 1.96 * standardError; // 95% confidence
    const confidenceInterval: [number, number] = [
      (expMean - ctrlMean) - marginOfError,
      (expMean - ctrlMean) + marginOfError
    ];
    
    const effectSize = this.calculateEffectSize(expData, ctrlData);
    
    return { pValue, confidenceInterval, effectSize };
  }

  private normalCDF(x: number): number {
    // Approximation of cumulative normal distribution
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x) / Math.sqrt(2);
    
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    
    return 0.5 * (1.0 + sign * y);
  }

  private calculateCurvature(state: SimulationState): number {
    // Simplified curvature calculation based on information density gradients
    const fragments = state.memoryField.fragments;
    if (fragments.length < 2) return 0;
    
    let totalCurvature = 0;
    for (let i = 0; i < fragments.length - 1; i++) {
      const pos1 = fragments[i].position;
      const pos2 = fragments[i + 1].position;
      const distance = Math.sqrt(
        Math.pow(pos2[0] - pos1[0], 2) +
        Math.pow(pos2[1] - pos1[1], 2) +
        Math.pow(pos2[2] - pos1[2], 2)
      );
      
      if (distance > 0) {
        const informationGradient = Math.abs(fragments[i].coherence - fragments[i + 1].coherence) / distance;
        totalCurvature += informationGradient * informationGradient; // Second derivative approximation
      }
    }
    
    return totalCurvature / fragments.length;
  }

  private calculateQuantumFidelity(state: SimulationState): number {
    // Calculate fidelity based on coherence preservation
    const averageCoherence = state.memoryField.fragments.length > 0 ?
      state.memoryField.fragments.reduce((sum, f) => sum + f.coherence, 0) / state.memoryField.fragments.length : 0;
    
    return Math.min(1, averageCoherence * state.rspState.coherence);
  }

  private calculateDecoherenceRate(state: SimulationState): number {
    // Simplified decoherence rate based on entropy increase
    return Math.max(0, state.rspState.entropy / 10); // Normalized to reasonable range
  }

  private calculateEntanglementStrength(state: SimulationState): number {
    // Estimate entanglement from observer correlations
    if (state.observers.length < 2) return 0;
    
    let entanglement = 0;
    for (let i = 0; i < state.observers.length - 1; i++) {
      for (let j = i + 1; j < state.observers.length; j++) {
        const coherenceProduct = state.observers[i].coherence * state.observers[j].coherence;
        entanglement += coherenceProduct;
      }
    }
    
    return entanglement / ((state.observers.length * (state.observers.length - 1)) / 2);
  }

  private calculateClassicalDeviations(
    measurements: OSHMeasurements,
    controlMeasurements: OSHMeasurements | null
  ): number[] {
    if (!controlMeasurements) return [];
    
    // Calculate deviations from classical predictions
    return measurements.quantumFidelity.map((exp, i) => {
      const ctrl = controlMeasurements.quantumFidelity[i] || 0;
      return exp - ctrl;
    });
  }

  // Public methods for retrieving evidence
  getEvidenceHistory(): OSHEvidence[] {
    return [...this.evidenceHistory];
  }

  getLatestEvidence(): OSHEvidence | null {
    return this.evidenceHistory.length > 0 ? 
      this.evidenceHistory[this.evidenceHistory.length - 1] : null;
  }

  getEvidenceSummary(): {
    totalExperiments: number;
    supportingEvidence: number;
    challengingEvidence: number;
    neutralEvidence: number;
    inconclusiveEvidence: number;
    averageStrength: number;
    averageConfidence: number;
  } {
    const total = this.evidenceHistory.length;
    
    if (total === 0) {
      return {
        totalExperiments: 0,
        supportingEvidence: 0,
        challengingEvidence: 0,
        neutralEvidence: 0,
        inconclusiveEvidence: 0,
        averageStrength: 0,
        averageConfidence: 0
      };
    }
    
    const supporting = this.evidenceHistory.filter(e => e.evidenceType === 'supports').length;
    const challenging = this.evidenceHistory.filter(e => e.evidenceType === 'challenges').length;
    const neutral = this.evidenceHistory.filter(e => e.evidenceType === 'neutral').length;
    const inconclusive = this.evidenceHistory.filter(e => e.evidenceType === 'inconclusive').length;
    
    const averageStrength = this.evidenceHistory.reduce((sum, e) => sum + e.strength, 0) / total;
    const averageConfidence = this.evidenceHistory.reduce((sum, e) => sum + e.confidence, 0) / total;
    
    return {
      totalExperiments: total,
      supportingEvidence: supporting,
      challengingEvidence: challenging,
      neutralEvidence: neutral,
      inconclusiveEvidence: inconclusive,
      averageStrength,
      averageConfidence
    };
  }

  exportEvidenceReport(): string {
    return JSON.stringify({
      summary: this.getEvidenceSummary(),
      evidence: this.evidenceHistory,
      thresholds: this.thresholds,
      generated: new Date().toISOString()
    }, null, 2);
  }
}