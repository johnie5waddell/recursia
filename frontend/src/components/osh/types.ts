// OSH Calculation Service Types

export interface OSHCalculationService {
  calculateRSP(data: RSPCalculation): Promise<CalculationResult>;
  calculateRSPBound(area: number, minEntropyFlux?: number): Promise<CalculationResult>;
  analyzeCMBComplexity(data: number[], samplingRate: number): Promise<CalculationResult>;
  searchGWEchoes(
    strainData: number[],
    samplingRate: number,
    mergerTime?: number,
    expectedEchoDelay?: number
  ): Promise<CalculationResult>;
  analyzeEEGCosmicResonance(
    eegData: number[][],
    cosmicData: number[],
    samplingRate: number
  ): Promise<CalculationResult>;
  mapConsciousnessDynamics(
    scale: string,
    information: number,
    complexity: number,
    entropyFlux: number
  ): Promise<CalculationResult>;
}

export interface RSPCalculation {
  integratedInformation: number;
  kolmogorovComplexity: number;
  entropyFlux: number;
  systemName?: string;
}

export interface RSPResult {
  rspValue: number;
  classification: string;
  dimensionalAnalysis: Record<string, string>;
  limitBehavior?: Record<string, any>;
  timestamp: string;
}

export interface CalculationResult {
  calculationType: string;
  success: boolean;
  result: Record<string, any>;
  errors: string[];
  warnings: string[];
  metadata: Record<string, any>;
  timestamp: string;
}

export interface CMBAnalysisResult {
  lempelZiv: {
    complexity: number;
    normalizedComplexity: number;
    compressionRatio: number;
    sequenceLength: number;
    vocabularySize: number;
    patterns: Array<{
      pattern: string;
      count: number;
      length: number;
    }>;
  };
  statistics: {
    mean: number;
    stdDev: number;
    skewness: number;
    kurtosis: number;
    minTemp: number;
    maxTemp: number;
  };
  powerSpectrum: {
    l: number[];
    cl: number[];
    peaks: Array<{
      l: number;
      cl: number;
      significance: number;
    }>;
  };
  recursivePatterns: Array<{
    scale: number;
    pattern: string;
    confidence: number;
    interpretation: string;
  }>;
  oshSignatures: {
    informationDensity: number;
    memoryStrainSignature: boolean;
    recursiveDepth: number;
    coherenceLevel: number;
  };
}