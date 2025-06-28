// Type definitions for Gravitational Wave Echo Search

export interface GWData {
  time: number[];          // Time array in seconds
  strain: number[];        // Strain data array
  sampleRate: number;      // Sampling rate in Hz
  metadata?: {
    detector?: string;     // Detector name (e.g., 'H1', 'L1', 'V1')
    gpsTime?: number;      // GPS time of the event
    source?: string;       // Source identifier
    [key: string]: any;    // Additional metadata
  };
}

export interface EchoCandidate {
  time: number;            // Time of echo detection (seconds)
  timeDelay: number;       // Delay from primary signal (seconds)
  amplitude: number;       // Echo amplitude
  confidence: number;      // Confidence score (0-1)
  snr: number;            // Signal-to-noise ratio in dB
  frequency?: number;      // Dominant frequency (Hz)
  duration?: number;       // Echo duration (seconds)
  metadata?: {
    echoOrder?: number;    // 1st order, 2nd order echo, etc.
    correlationCoeff?: number;
    matchedFilterScore?: number;
    [key: string]: any;
  };
}

export interface OSHMetrics {
  informationLeakage: number;      // Fraction of information leaked through echoes
  memoryFieldCoherence: number;    // Coherence measure of memory field (0-1)
  recursiveDepth: number;          // Estimated recursive depth
  observerInfluence: number;       // Observer effect strength (0-1)
  totalEchoes: number;             // Total number of detected echoes
  averageEchoStrength: number;     // Average echo energy
}

export interface SearchParameters {
  minTimeDelay: number;            // Minimum echo delay to search (seconds)
  maxTimeDelay: number;            // Maximum echo delay to search (seconds)
  confidenceThreshold: number;     // Minimum confidence for detection (0-1)
  windowSize: number;              // Analysis window size (seconds)
  overlapFraction: number;         // Window overlap fraction (0-1)
  detectionMethod: 'autocorrelation' | 'matched_filter' | 'hybrid';
  noiseReduction: boolean;         // Apply noise reduction preprocessing
  adaptiveThreshold: boolean;      // Use adaptive thresholding
}

export class GWEchoDetector {
  private data: GWData;
  private params: SearchParameters;
  private workingStrain: number[];
  
  constructor(data: GWData, params: SearchParameters) {
    this.data = data;
    this.params = params;
    this.workingStrain = [...data.strain];
    
    if (params.noiseReduction) {
      this.applyNoiseReduction();
    }
  }
  
  async findEchoes(): Promise<EchoCandidate[]> {
    const candidates: EchoCandidate[] = [];
    
    switch (this.params.detectionMethod) {
      case 'autocorrelation':
        return this.autocorrelationMethod();
      case 'matched_filter':
        return this.matchedFilterMethod();
      case 'hybrid':
        return this.hybridMethod();
    }
  }
  
  private applyNoiseReduction(): void {
    // Implement spectral subtraction or wavelet denoising
    // This is a simplified high-pass filter
    const cutoffFreq = 20; // Hz
    const nyquist = this.data.sampleRate / 2;
    const normalizedCutoff = cutoffFreq / nyquist;
    
    // Simple moving average subtraction as placeholder
    const windowSize = Math.floor(this.data.sampleRate / cutoffFreq);
    for (let i = windowSize; i < this.workingStrain.length - windowSize; i++) {
      let sum = 0;
      for (let j = -windowSize; j <= windowSize; j++) {
        sum += this.workingStrain[i + j];
      }
      const avg = sum / (2 * windowSize + 1);
      this.workingStrain[i] -= avg;
    }
  }
  
  private async autocorrelationMethod(): Promise<EchoCandidate[]> {
    const candidates: EchoCandidate[] = [];
    const { minTimeDelay, maxTimeDelay, windowSize, confidenceThreshold } = this.params;
    
    const minSamples = Math.floor(minTimeDelay * this.data.sampleRate);
    const maxSamples = Math.floor(maxTimeDelay * this.data.sampleRate);
    const windowSamples = Math.floor(windowSize * this.data.sampleRate);
    
    // Compute autocorrelation for different lags
    for (let lag = minSamples; lag <= maxSamples; lag += Math.floor(minSamples / 10)) {
      const correlation = this.computeAutocorrelation(lag, windowSamples);
      const peaks = this.findPeaks(correlation);
      
      for (const peak of peaks) {
        const confidence = this.calculateConfidence(peak, correlation);
        if (confidence >= confidenceThreshold) {
          const time = this.data.time[peak.index];
          const amplitude = peak.value;
          const snr = this.calculateSNR(peak, correlation);
          
          candidates.push({
            time,
            timeDelay: lag / this.data.sampleRate,
            amplitude,
            confidence,
            snr,
            metadata: {
              correlationCoeff: peak.value,
              echoOrder: Math.floor(Math.log2(lag / minSamples)) + 1,
            }
          });
        }
      }
    }
    
    return this.mergeDuplicates(candidates);
  }
  
  private async matchedFilterMethod(): Promise<EchoCandidate[]> {
    const candidates: EchoCandidate[] = [];
    const { windowSize, confidenceThreshold } = this.params;
    const windowSamples = Math.floor(windowSize * this.data.sampleRate);
    
    // Extract template from highest amplitude region
    const template = this.extractTemplate(windowSamples);
    
    // Apply matched filter across the data
    const filtered = this.applyMatchedFilter(template);
    const peaks = this.findPeaks(filtered);
    
    for (const peak of peaks) {
      const confidence = this.calculateConfidence(peak, filtered);
      if (confidence >= confidenceThreshold) {
        const time = this.data.time[peak.index];
        const primaryIndex = this.findPrimarySignal(peak.index);
        const timeDelay = Math.abs(time - this.data.time[primaryIndex]);
        
        if (timeDelay >= this.params.minTimeDelay && timeDelay <= this.params.maxTimeDelay) {
          candidates.push({
            time,
            timeDelay,
            amplitude: this.workingStrain[peak.index],
            confidence,
            snr: this.calculateSNR(peak, filtered),
            metadata: {
              matchedFilterScore: peak.value,
            }
          });
        }
      }
    }
    
    return this.mergeDuplicates(candidates);
  }
  
  private async hybridMethod(): Promise<EchoCandidate[]> {
    // Combine results from both methods
    const autoCandidates = await this.autocorrelationMethod();
    const matchedCandidates = await this.matchedFilterMethod();
    
    // Merge and validate candidates
    const allCandidates = [...autoCandidates, ...matchedCandidates];
    return this.validateAndRank(allCandidates);
  }
  
  private computeAutocorrelation(lag: number, windowSize: number): number[] {
    const result: number[] = new Array(this.workingStrain.length).fill(0);
    
    for (let i = 0; i < this.workingStrain.length - lag - windowSize; i++) {
      let sum = 0;
      let norm1 = 0;
      let norm2 = 0;
      
      for (let j = 0; j < windowSize; j++) {
        const val1 = this.workingStrain[i + j];
        const val2 = this.workingStrain[i + j + lag];
        sum += val1 * val2;
        norm1 += val1 * val1;
        norm2 += val2 * val2;
      }
      
      result[i + lag] = sum / Math.sqrt(norm1 * norm2);
    }
    
    return result;
  }
  
  private extractTemplate(windowSize: number): number[] {
    // Find the region with highest energy
    let maxEnergy = 0;
    let maxIndex = 0;
    
    for (let i = 0; i < this.workingStrain.length - windowSize; i++) {
      let energy = 0;
      for (let j = 0; j < windowSize; j++) {
        energy += this.workingStrain[i + j] ** 2;
      }
      if (energy > maxEnergy) {
        maxEnergy = energy;
        maxIndex = i;
      }
    }
    
    return this.workingStrain.slice(maxIndex, maxIndex + windowSize);
  }
  
  private applyMatchedFilter(template: number[]): number[] {
    const result: number[] = new Array(this.workingStrain.length).fill(0);
    const templateNorm = Math.sqrt(template.reduce((sum, val) => sum + val * val, 0));
    
    for (let i = 0; i < this.workingStrain.length - template.length; i++) {
      let sum = 0;
      let dataNorm = 0;
      
      for (let j = 0; j < template.length; j++) {
        sum += this.workingStrain[i + j] * template[j];
        dataNorm += this.workingStrain[i + j] ** 2;
      }
      
      result[i] = sum / (Math.sqrt(dataNorm) * templateNorm);
    }
    
    return result;
  }
  
  private findPeaks(data: number[]): Array<{index: number, value: number}> {
    const peaks: Array<{index: number, value: number}> = [];
    const threshold = this.params.adaptiveThreshold ? 
      this.calculateAdaptiveThreshold(data) : 
      Math.max(...data) * 0.3;
    
    for (let i = 1; i < data.length - 1; i++) {
      if (data[i] > threshold && 
          data[i] > data[i - 1] && 
          data[i] > data[i + 1]) {
        peaks.push({ index: i, value: data[i] });
      }
    }
    
    return peaks;
  }
  
  private calculateAdaptiveThreshold(data: number[]): number {
    // Use median absolute deviation
    const sorted = [...data].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const mad = sorted.map(x => Math.abs(x - median))
      .sort((a, b) => a - b)[Math.floor(sorted.length / 2)];
    
    return median + 3 * mad * 1.4826; // 3-sigma threshold
  }
  
  private calculateConfidence(peak: {index: number, value: number}, data: number[]): number {
    // Calculate local SNR and normalize
    const noise = this.estimateLocalNoise(peak.index, data);
    const snr = peak.value / noise;
    return Math.tanh(snr / 10); // Sigmoid-like mapping to [0,1]
  }
  
  private calculateSNR(peak: {index: number, value: number}, data: number[]): number {
    const noise = this.estimateLocalNoise(peak.index, data);
    return 20 * Math.log10(peak.value / noise);
  }
  
  private estimateLocalNoise(index: number, data: number[]): number {
    const windowSize = Math.floor(this.data.sampleRate * 0.1); // 100ms window
    const start = Math.max(0, index - windowSize);
    const end = Math.min(data.length, index + windowSize);
    
    const localData = data.slice(start, end);
    const mean = localData.reduce((a, b) => a + b, 0) / localData.length;
    const variance = localData.reduce((sum, val) => sum + (val - mean) ** 2, 0) / localData.length;
    
    return Math.sqrt(variance);
  }
  
  private findPrimarySignal(echoIndex: number): number {
    // Find the nearest high-amplitude event before this echo
    let maxAmplitude = 0;
    let primaryIndex = 0;
    
    const searchStart = Math.max(0, echoIndex - Math.floor(this.params.maxTimeDelay * this.data.sampleRate));
    
    for (let i = searchStart; i < echoIndex; i++) {
      if (Math.abs(this.workingStrain[i]) > maxAmplitude) {
        maxAmplitude = Math.abs(this.workingStrain[i]);
        primaryIndex = i;
      }
    }
    
    return primaryIndex;
  }
  
  private mergeDuplicates(candidates: EchoCandidate[]): EchoCandidate[] {
    // Merge candidates that are within 10ms of each other
    const merged: EchoCandidate[] = [];
    const used = new Set<number>();
    
    for (let i = 0; i < candidates.length; i++) {
      if (used.has(i)) continue;
      
      const group = [candidates[i]];
      
      for (let j = i + 1; j < candidates.length; j++) {
        if (Math.abs(candidates[i].time - candidates[j].time) < 0.01) {
          group.push(candidates[j]);
          used.add(j);
        }
      }
      
      // Take the candidate with highest confidence from the group
      merged.push(group.reduce((best, curr) => 
        curr.confidence > best.confidence ? curr : best
      ));
    }
    
    return merged.sort((a, b) => a.time - b.time);
  }
  
  private validateAndRank(candidates: EchoCandidate[]): EchoCandidate[] {
    // Remove duplicates and rank by combined score
    const unique = this.mergeDuplicates(candidates);
    
    return unique.map(candidate => {
      // Combine confidence from multiple detection methods
      const allConfidences = candidates
        .filter(c => Math.abs(c.time - candidate.time) < 0.01)
        .map(c => c.confidence);
      
      const avgConfidence = allConfidences.reduce((a, b) => a + b, 0) / allConfidences.length;
      const maxConfidence = Math.max(...allConfidences);
      
      return {
        ...candidate,
        confidence: 0.7 * maxConfidence + 0.3 * avgConfidence,
      };
    }).sort((a, b) => b.confidence - a.confidence);
  }
}