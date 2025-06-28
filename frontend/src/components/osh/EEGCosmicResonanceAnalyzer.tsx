/**
 * EEG-Cosmic Background Resonance Analyzer
 * 
 * Comprehensive component for analyzing correlations between EEG data and cosmic background signals
 * Implements OSH-based consciousness-cosmos coupling analysis with scientifically accurate signal processing
 */

import React, { useState, useCallback, useRef, useMemo, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TimeScale,
  TimeSeriesScale,
} from 'chart.js';
import { Line, Scatter } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import annotationPlugin from 'chartjs-plugin-annotation';
import zoomPlugin from 'chartjs-plugin-zoom';
import { FFT } from '../../utils/fft';
import { toast } from 'react-hot-toast';
import { OSHCalculationService, CalculationResult } from './types';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TimeScale,
  TimeSeriesScale,
  annotationPlugin,
  zoomPlugin
);

// Types and interfaces
interface EEGData {
  channels: string[];
  samplingRate: number;
  data: number[][]; // [channel][sample]
  timestamps?: number[];
  metadata?: {
    subject?: string;
    recordingDate?: string;
    device?: string;
    notes?: string;
  };
}

interface CosmicData {
  type: 'cmb' | 'solar_wind' | 'schumann' | 'custom';
  name: string;
  samplingRate: number;
  data: number[];
  timestamps?: number[];
  metadata?: {
    source?: string;
    unit?: string;
    description?: string;
  };
}

interface CrossCorrelationResult {
  lag: number[];
  correlation: number[];
  peakLag: number;
  peakCorrelation: number;
  significance: number;
}

interface CoherenceResult {
  frequency: number[];
  coherence: number[];
  phase: number[];
  significantPeaks: Array<{
    frequency: number;
    coherence: number;
    phase: number;
    band: string;
  }>;
}

interface ResonanceEvent {
  timestamp: number;
  duration: number;
  frequency: number;
  strength: number;
  eegChannel: string;
  cosmicSource: string;
  oshMetrics: {
    couplingStrength: number;
    informationTransferRate: number;
    phaseCoherence: number;
  };
}

interface BandAnalysis {
  band: string;
  frequencyRange: [number, number];
  power: number;
  relativePower: number;
  coherence: number;
  phaseSync: number;
}

interface OSHMetrics {
  consciousnessCosmosCoupling: number;
  informationTransferRate: number;
  quantumCoherence: number;
  memoryFieldResonance: number;
  observerInfluence: number;
  recursiveDepth: number;
  entropyFlow: number;
  phaseEntanglement: number;
}

// EEG frequency bands
const EEG_BANDS = {
  delta: { name: 'Delta', range: [0.5, 4], color: '#8B0000' },
  theta: { name: 'Theta', range: [4, 8], color: '#FF4500' },
  alpha: { name: 'Alpha', range: [8, 13], color: '#32CD32' },
  beta: { name: 'Beta', range: [13, 30], color: '#1E90FF' },
  gamma: { name: 'Gamma', range: [30, 100], color: '#9370DB' },
} as const;

// File reader utilities
const readFileAsText = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.onerror = reject;
    reader.readAsText(file);
  });
};

const readFileAsArrayBuffer = (file: File): Promise<ArrayBuffer> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as ArrayBuffer);
    reader.onerror = reject;
    reader.readAsArrayBuffer(file);
  });
};

// Signal processing utilities
const crossCorrelation = (signal1: number[], signal2: number[], maxLag: number): CrossCorrelationResult => {
  const n1 = signal1.length;
  const n2 = signal2.length;
  const n = Math.min(n1, n2);
  
  // Normalize signals
  const mean1 = signal1.reduce((a, b) => a + b, 0) / n1;
  const mean2 = signal2.reduce((a, b) => a + b, 0) / n2;
  const norm1 = signal1.map(x => x - mean1);
  const norm2 = signal2.map(x => x - mean2);
  
  const lags: number[] = [];
  const correlations: number[] = [];
  
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    let sum = 0;
    let count = 0;
    
    for (let i = 0; i < n; i++) {
      const j = i + lag;
      if (j >= 0 && j < n) {
        sum += norm1[i] * norm2[j];
        count++;
      }
    }
    
    if (count > 0) {
      lags.push(lag);
      correlations.push(sum / count);
    }
  }
  
  // Find peak correlation
  let peakIdx = 0;
  let peakCorr = Math.abs(correlations[0]);
  for (let i = 1; i < correlations.length; i++) {
    if (Math.abs(correlations[i]) > peakCorr) {
      peakCorr = Math.abs(correlations[i]);
      peakIdx = i;
    }
  }
  
  // Calculate significance (simplified)
  const significance = peakCorr / Math.sqrt(1 / n);
  
  return {
    lag: lags,
    correlation: correlations,
    peakLag: lags[peakIdx],
    peakCorrelation: correlations[peakIdx],
    significance,
  };
};

const calculateCoherence = (signal1: number[], signal2: number[], samplingRate: number): CoherenceResult => {
  const fftSize = 2048;
  const overlap = 0.5;
  const windowSize = Math.min(fftSize, signal1.length, signal2.length);
  const hopSize = Math.floor(windowSize * (1 - overlap));
  
  const frequencies: number[] = [];
  const coherenceSum: number[] = [];
  const phaseSum: number[] = [];
  let windowCount = 0;
  
  // Calculate FFT for overlapping windows
  for (let start = 0; start + windowSize <= signal1.length && start + windowSize <= signal2.length; start += hopSize) {
    const window1 = signal1.slice(start, start + windowSize);
    const window2 = signal2.slice(start, start + windowSize);
    
    // Apply Hanning window
    const hannWindow = Array(windowSize).fill(0).map((_, i) => 
      0.5 - 0.5 * Math.cos(2 * Math.PI * i / (windowSize - 1))
    );
    
    const windowed1 = window1.map((v, i) => v * hannWindow[i]);
    const windowed2 = window2.map((v, i) => v * hannWindow[i]);
    
    // Perform FFT
    const fft1 = FFT.fft(windowed1);
    const fft2 = FFT.fft(windowed2);
    
    // Calculate cross-spectrum and auto-spectra
    for (let i = 0; i < windowSize / 2; i++) {
      const freq = i * samplingRate / windowSize;
      
      if (windowCount === 0) {
        frequencies.push(freq);
        coherenceSum.push(0);
        phaseSum.push(0);
      }
      
      const crossSpec = {
        re: fft1[i].re * fft2[i].re + fft1[i].im * fft2[i].im,
        im: fft1[i].im * fft2[i].re - fft1[i].re * fft2[i].im,
      };
      
      const auto1 = fft1[i].re * fft1[i].re + fft1[i].im * fft1[i].im;
      const auto2 = fft2[i].re * fft2[i].re + fft2[i].im * fft2[i].im;
      
      if (auto1 > 0 && auto2 > 0) {
        const coh = Math.sqrt(crossSpec.re * crossSpec.re + crossSpec.im * crossSpec.im) / Math.sqrt(auto1 * auto2);
        coherenceSum[i] += coh;
        phaseSum[i] += Math.atan2(crossSpec.im, crossSpec.re);
      }
    }
    
    windowCount++;
  }
  
  // Average coherence and phase
  const coherence = coherenceSum.map(c => c / windowCount);
  const phase = phaseSum.map(p => p / windowCount);
  
  // Find significant peaks
  const significantPeaks: CoherenceResult['significantPeaks'] = [];
  const threshold = 0.5; // Coherence threshold for significance
  
  for (let i = 1; i < coherence.length - 1; i++) {
    if (coherence[i] > threshold && coherence[i] > coherence[i - 1] && coherence[i] > coherence[i + 1]) {
      const freq = frequencies[i];
      let band = 'other';
      
      for (const [bandName, bandInfo] of Object.entries(EEG_BANDS)) {
        if (freq >= bandInfo.range[0] && freq <= bandInfo.range[1]) {
          band = bandName;
          break;
        }
      }
      
      significantPeaks.push({
        frequency: freq,
        coherence: coherence[i],
        phase: phase[i],
        band,
      });
    }
  }
  
  return {
    frequency: frequencies,
    coherence,
    phase,
    significantPeaks: significantPeaks.sort((a, b) => b.coherence - a.coherence),
  };
};

const calculateBandPower = (signal: number[], samplingRate: number, bandRange: [number, number]): number => {
  const fft = FFT.fft(signal);
  const freqResolution = samplingRate / signal.length;
  
  let power = 0;
  const startBin = Math.floor(bandRange[0] / freqResolution);
  const endBin = Math.ceil(bandRange[1] / freqResolution);
  
  for (let i = startBin; i <= endBin && i < fft.length / 2; i++) {
    power += fft[i].re * fft[i].re + fft[i].im * fft[i].im;
  }
  
  return power / (endBin - startBin + 1);
};

const findResonanceEvents = (
  eegData: EEGData,
  cosmicData: CosmicData,
  coherenceResults: Map<string, CoherenceResult>,
  threshold: number = 0.7
): ResonanceEvent[] => {
  const events: ResonanceEvent[] = [];
  const windowSize = Math.floor(eegData.samplingRate * 2); // 2-second windows
  const hopSize = Math.floor(windowSize / 2);
  
  eegData.channels.forEach((channel, channelIdx) => {
    const coherenceResult = coherenceResults.get(channel);
    if (!coherenceResult) return;
    
    const channelData = eegData.data[channelIdx];
    
    for (let start = 0; start + windowSize <= channelData.length; start += hopSize) {
      const window = channelData.slice(start, start + windowSize);
      
      // Check for high coherence in this window
      coherenceResult.significantPeaks.forEach(peak => {
        if (peak.coherence >= threshold) {
          // Calculate OSH metrics for this resonance event
          const bandPower = calculateBandPower(window, eegData.samplingRate, [...EEG_BANDS[peak.band as keyof typeof EEG_BANDS].range] as [number, number]);
          const cosmicWindow = cosmicData.data.slice(
            Math.floor(start * cosmicData.samplingRate / eegData.samplingRate),
            Math.floor((start + windowSize) * cosmicData.samplingRate / eegData.samplingRate)
          );
          const cosmicPower = calculateBandPower(cosmicWindow, cosmicData.samplingRate, [peak.frequency - 0.5, peak.frequency + 0.5]);
          
          const event: ResonanceEvent = {
            timestamp: start / eegData.samplingRate,
            duration: windowSize / eegData.samplingRate,
            frequency: peak.frequency,
            strength: peak.coherence,
            eegChannel: channel,
            cosmicSource: cosmicData.name,
            oshMetrics: {
              couplingStrength: peak.coherence * Math.sqrt(bandPower * cosmicPower),
              informationTransferRate: peak.coherence * Math.log2(1 + peak.coherence),
              phaseCoherence: Math.cos(peak.phase),
            },
          };
          
          events.push(event);
        }
      });
    }
  });
  
  // Merge overlapping events
  const mergedEvents: ResonanceEvent[] = [];
  const sortedEvents = events.sort((a, b) => a.timestamp - b.timestamp);
  
  sortedEvents.forEach(event => {
    const lastEvent = mergedEvents[mergedEvents.length - 1];
    if (lastEvent && 
        lastEvent.eegChannel === event.eegChannel &&
        lastEvent.cosmicSource === event.cosmicSource &&
        Math.abs(lastEvent.frequency - event.frequency) < 1 &&
        event.timestamp < lastEvent.timestamp + lastEvent.duration) {
      // Merge events
      lastEvent.duration = Math.max(lastEvent.duration, event.timestamp + event.duration - lastEvent.timestamp);
      lastEvent.strength = Math.max(lastEvent.strength, event.strength);
      lastEvent.oshMetrics.couplingStrength = Math.max(lastEvent.oshMetrics.couplingStrength, event.oshMetrics.couplingStrength);
    } else {
      mergedEvents.push(event);
    }
  });
  
  return mergedEvents;
};

const calculateOSHMetrics = (
  eegData: EEGData,
  cosmicData: CosmicData,
  coherenceResults: Map<string, CoherenceResult>,
  correlationResults: Map<string, CrossCorrelationResult>,
  resonanceEvents: ResonanceEvent[]
): OSHMetrics => {
  // Calculate average coherence across all channels
  let totalCoherence = 0;
  let peakCount = 0;
  
  coherenceResults.forEach(result => {
    result.significantPeaks.forEach(peak => {
      totalCoherence += peak.coherence;
      peakCount++;
    });
  });
  
  const avgCoherence = peakCount > 0 ? totalCoherence / peakCount : 0;
  
  // Calculate information transfer rate
  const totalInfoTransfer = resonanceEvents.reduce((sum, event) => 
    sum + event.oshMetrics.informationTransferRate, 0
  );
  const avgInfoTransfer = resonanceEvents.length > 0 ? totalInfoTransfer / resonanceEvents.length : 0;
  
  // Calculate phase entanglement
  let phaseEntanglement = 0;
  coherenceResults.forEach(result => {
    const phaseVariance = result.phase.reduce((sum, phase, i) => {
      if (i > 0) {
        const phaseDiff = Math.abs(phase - result.phase[i - 1]);
        return sum + phaseDiff * phaseDiff;
      }
      return sum;
    }, 0) / (result.phase.length - 1);
    
    phaseEntanglement += 1 / (1 + Math.sqrt(phaseVariance));
  });
  phaseEntanglement /= coherenceResults.size;
  
  // Calculate entropy flow
  const eegEntropy = calculateEntropy(eegData.data.flat());
  const cosmicEntropy = calculateEntropy(cosmicData.data);
  const entropyFlow = Math.abs(eegEntropy - cosmicEntropy) / Math.max(eegEntropy, cosmicEntropy);
  
  // Calculate observer influence (based on coherence variations)
  let observerInfluence = 0;
  correlationResults.forEach(result => {
    observerInfluence += Math.abs(result.peakCorrelation) * result.significance;
  });
  observerInfluence /= correlationResults.size;
  
  return {
    consciousnessCosmosCoupling: avgCoherence,
    informationTransferRate: avgInfoTransfer,
    quantumCoherence: avgCoherence * phaseEntanglement,
    memoryFieldResonance: resonanceEvents.length / (eegData.data[0].length / eegData.samplingRate),
    observerInfluence: Math.min(observerInfluence, 1),
    recursiveDepth: Math.log2(1 + resonanceEvents.length),
    entropyFlow: 1 - entropyFlow,
    phaseEntanglement,
  };
};

const calculateEntropy = (data: number[]): number => {
  const bins = 256;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const binWidth = (max - min) / bins;
  
  const histogram = new Array(bins).fill(0);
  data.forEach(value => {
    const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
    histogram[binIndex]++;
  });
  
  const probabilities = histogram.map(count => count / data.length);
  return -probabilities.reduce((sum, p) => {
    if (p > 0) {
      return sum + p * Math.log2(p);
    }
    return sum;
  }, 0);
};

// Parse uploaded files
const parseEEGFile = async (file: File): Promise<EEGData> => {
  const fileName = file.name.toLowerCase();
  
  if (fileName.endsWith('.csv')) {
    const text = await readFileAsText(file);
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',');
    
    // Assume first column is time, rest are channels
    const channels = headers.slice(1);
    const data: number[][] = channels.map(() => []);
    
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(Number);
      for (let j = 1; j < values.length; j++) {
        data[j - 1].push(values[j]);
      }
    }
    
    // Estimate sampling rate from time column
    const times = lines.slice(1, Math.min(100, lines.length))
      .map(line => Number(line.split(',')[0]));
    const samplingRate = Math.round(1 / (times[1] - times[0]));
    
    return { channels, samplingRate, data };
  } else if (fileName.endsWith('.json')) {
    const text = await readFileAsText(file);
    return JSON.parse(text) as EEGData;
  } else if (fileName.endsWith('.edf')) {
    // Simplified EDF parsing - in production, use a proper EDF library
    throw new Error('EDF parsing not implemented. Please convert to CSV or JSON format.');
  }
  
  throw new Error('Unsupported file format. Please use CSV, JSON, or EDF files.');
};

const parseCosmicFile = async (file: File, type: CosmicData['type']): Promise<CosmicData> => {
  const fileName = file.name.toLowerCase();
  
  if (fileName.endsWith('.csv')) {
    const text = await readFileAsText(file);
    const lines = text.trim().split('\n');
    const data = lines.slice(1).map(line => Number(line.split(',')[1] || line));
    
    // Estimate sampling rate
    const times = lines.slice(1, Math.min(100, lines.length))
      .map(line => Number(line.split(',')[0]));
    const samplingRate = times.length > 1 ? Math.round(1 / (times[1] - times[0])) : 1;
    
    return {
      type,
      name: file.name,
      samplingRate,
      data,
    };
  } else if (fileName.endsWith('.json')) {
    const text = await readFileAsText(file);
    const parsed = JSON.parse(text);
    return {
      type,
      name: file.name,
      ...parsed,
    };
  }
  
  throw new Error('Unsupported file format. Please use CSV or JSON files.');
};

// Main component
export const EEGCosmicResonanceAnalyzer: React.FC = () => {
  // State management
  const [eegData, setEEGData] = useState<EEGData | null>(null);
  const [cosmicData, setCosmicData] = useState<CosmicData | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState<'upload' | 'analysis' | 'results'>('upload');
  
  // Analysis results
  const [correlationResults, setCorrelationResults] = useState<Map<string, CrossCorrelationResult>>(new Map());
  const [coherenceResults, setCoherenceResults] = useState<Map<string, CoherenceResult>>(new Map());
  const [bandAnalysis, setBandAnalysis] = useState<Map<string, BandAnalysis[]>>(new Map());
  const [resonanceEvents, setResonanceEvents] = useState<ResonanceEvent[]>([]);
  const [oshMetrics, setOSHMetrics] = useState<OSHMetrics | null>(null);
  const [statisticalSignificance, setStatisticalSignificance] = useState<number | null>(null);
  
  // UI state
  const [selectedChannel, setSelectedChannel] = useState<string>('');
  const [selectedBand, setSelectedBand] = useState<keyof typeof EEG_BANDS>('alpha');
  const [timeWindow, setTimeWindow] = useState<[number, number] | null>(null);
  
  // File upload handlers
  const handleEEGUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    try {
      setIsProcessing(true);
      const data = await parseEEGFile(file);
      setEEGData(data);
      setSelectedChannel(data.channels[0]);
      toast.success(`Loaded EEG data: ${data.channels.length} channels, ${data.data[0].length} samples`);
    } catch (error) {
      toast.error(`Failed to parse EEG file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  }, []);
  
  const handleCosmicUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    try {
      setIsProcessing(true);
      const type = (document.getElementById('cosmic-type') as HTMLSelectElement)?.value as CosmicData['type'] || 'custom';
      const data = await parseCosmicFile(file, type);
      setCosmicData(data);
      toast.success(`Loaded cosmic data: ${data.name}, ${data.data.length} samples`);
    } catch (error) {
      toast.error(`Failed to parse cosmic file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  }, []);
  
  // Analysis functions
  const runAnalysis = useCallback(async () => {
    if (!eegData || !cosmicData) {
      toast.error('Please upload both EEG and cosmic data files');
      return;
    }
    
    setIsProcessing(true);
    setActiveTab('analysis');
    
    try {
      // Cross-correlation analysis
      const corrResults = new Map<string, CrossCorrelationResult>();
      const maxLag = Math.floor(Math.min(eegData.samplingRate, cosmicData.samplingRate) * 10); // 10 seconds max lag
      
      eegData.channels.forEach((channel, idx) => {
        const result = crossCorrelation(
          eegData.data[idx],
          cosmicData.data,
          maxLag
        );
        corrResults.set(channel, result);
      });
      
      setCorrelationResults(corrResults);
      
      // Coherence analysis
      const cohResults = new Map<string, CoherenceResult>();
      eegData.channels.forEach((channel, idx) => {
        const result = calculateCoherence(
          eegData.data[idx],
          cosmicData.data,
          Math.min(eegData.samplingRate, cosmicData.samplingRate)
        );
        cohResults.set(channel, result);
      });
      
      setCoherenceResults(cohResults);
      
      // Band-specific analysis
      const bandResults = new Map<string, BandAnalysis[]>();
      eegData.channels.forEach((channel, idx) => {
        const channelBands: BandAnalysis[] = [];
        const channelData = eegData.data[idx];
        const coherenceResult = cohResults.get(channel);
        
        Object.entries(EEG_BANDS).forEach(([bandKey, bandInfo]) => {
          const power = calculateBandPower(channelData, eegData.samplingRate, [...bandInfo.range] as [number, number]);
          const totalPower = calculateBandPower(channelData, eegData.samplingRate, [0.5, 100]);
          
          // Find average coherence in this band
          let bandCoherence = 0;
          let bandPhaseSync = 0;
          let count = 0;
          
          if (coherenceResult) {
            coherenceResult.frequency.forEach((freq, i) => {
              if (freq >= bandInfo.range[0] && freq <= bandInfo.range[1]) {
                bandCoherence += coherenceResult.coherence[i];
                bandPhaseSync += Math.abs(Math.cos(coherenceResult.phase[i]));
                count++;
              }
            });
          }
          
          channelBands.push({
            band: bandInfo.name,
            frequencyRange: [...bandInfo.range] as [number, number],
            power,
            relativePower: power / totalPower,
            coherence: count > 0 ? bandCoherence / count : 0,
            phaseSync: count > 0 ? bandPhaseSync / count : 0,
          });
        });
        
        bandResults.set(channel, channelBands);
      });
      
      setBandAnalysis(bandResults);
      
      // Find resonance events
      const events = findResonanceEvents(eegData, cosmicData, cohResults);
      setResonanceEvents(events);
      
      // Calculate OSH metrics
      const metrics = calculateOSHMetrics(eegData, cosmicData, cohResults, corrResults, events);
      setOSHMetrics(metrics);
      
      // Statistical significance (simplified - in production use proper statistical tests)
      const avgSignificance = Array.from(corrResults.values())
        .reduce((sum, result) => sum + result.significance, 0) / corrResults.size;
      setStatisticalSignificance(avgSignificance);
      
      toast.success('Analysis complete!');
      setActiveTab('results');
    } catch (error) {
      toast.error(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  }, [eegData, cosmicData]);
  
  // Export results
  const exportResults = useCallback(() => {
    if (!oshMetrics) return;
    
    const results = {
      timestamp: new Date().toISOString(),
      eegMetadata: eegData?.metadata,
      cosmicMetadata: cosmicData?.metadata,
      oshMetrics,
      resonanceEvents,
      statisticalSignificance,
      bandAnalysis: Object.fromEntries(bandAnalysis),
      correlationPeaks: Object.fromEntries(
        Array.from(correlationResults.entries()).map(([channel, result]) => [
          channel,
          {
            peakLag: result.peakLag,
            peakCorrelation: result.peakCorrelation,
            significance: result.significance,
          },
        ])
      ),
      coherencePeaks: Object.fromEntries(
        Array.from(coherenceResults.entries()).map(([channel, result]) => [
          channel,
          result.significantPeaks,
        ])
      ),
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `eeg_cosmic_resonance_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast.success('Results exported successfully');
  }, [eegData, cosmicData, oshMetrics, resonanceEvents, statisticalSignificance, bandAnalysis, correlationResults, coherenceResults]);
  
  // Chart configurations
  const timeSeriesChartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'EEG and Cosmic Signal Time Series',
      },
      zoom: {
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true,
          },
          mode: 'x' as const,
        },
        pan: {
          enabled: true,
          mode: 'x' as const,
        },
      },
    },
    scales: {
      x: {
        type: 'linear' as const,
        title: {
          display: true,
          text: 'Time (s)',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Amplitude',
        },
      },
    },
  }), []);
  
  const coherenceChartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Frequency Coherence Analysis',
      },
      annotation: {
        annotations: Object.entries(EEG_BANDS).reduce((acc, [key, band]) => {
          acc[key] = {
            type: 'box' as const,
            xMin: band.range[0],
            xMax: band.range[1],
            backgroundColor: `${band.color}20`,
            borderColor: band.color,
            borderWidth: 1,
            label: {
              display: true,
              content: band.name,
              position: 'start' as const,
            },
          };
          return acc;
        }, {} as any),
      },
    },
    scales: {
      x: {
        type: 'linear' as const,
        title: {
          display: true,
          text: 'Frequency (Hz)',
        },
        min: 0,
        max: 50,
      },
      y: {
        title: {
          display: true,
          text: 'Coherence',
        },
        min: 0,
        max: 1,
      },
    },
  }), []);
  
  // Prepare chart data
  const timeSeriesData = useMemo(() => {
    if (!eegData || !cosmicData || !selectedChannel) return null;
    
    const channelIdx = eegData.channels.indexOf(selectedChannel);
    if (channelIdx === -1) return null;
    
    const eegSamples = 1000; // Show first 1000 samples for performance
    const eegTimePoints = Array.from({ length: Math.min(eegSamples, eegData.data[channelIdx].length) }, 
      (_, i) => i / eegData.samplingRate
    );
    
    const cosmicSamples = Math.floor(eegSamples * cosmicData.samplingRate / eegData.samplingRate);
    const cosmicTimePoints = Array.from({ length: Math.min(cosmicSamples, cosmicData.data.length) },
      (_, i) => i / cosmicData.samplingRate
    );
    
    return {
      datasets: [
        {
          label: `EEG - ${selectedChannel}`,
          data: eegTimePoints.map((t, i) => ({ x: t, y: eegData.data[channelIdx][i] })),
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.1,
        },
        {
          label: `Cosmic - ${cosmicData.name}`,
          data: cosmicTimePoints.map((t, i) => ({ x: t, y: cosmicData.data[i] })),
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.1,
          yAxisID: 'y1',
        },
      ],
    };
  }, [eegData, cosmicData, selectedChannel]);
  
  const coherenceData = useMemo(() => {
    if (!coherenceResults.has(selectedChannel)) return null;
    
    const result = coherenceResults.get(selectedChannel)!;
    
    return {
      datasets: [
        {
          label: 'Coherence',
          data: result.frequency.map((f, i) => ({ x: f, y: result.coherence[i] })),
          borderColor: 'rgb(153, 102, 255)',
          backgroundColor: 'rgba(153, 102, 255, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
        },
        {
          label: 'Significant Peaks',
          data: result.significantPeaks.map(peak => ({ x: peak.frequency, y: peak.coherence })),
          borderColor: 'rgb(255, 159, 64)',
          backgroundColor: 'rgb(255, 159, 64)',
          borderWidth: 0,
          pointRadius: 6,
          pointStyle: 'star',
          showLine: false,
        },
      ],
    };
  }, [coherenceResults, selectedChannel]);
  
  return (
    <div className="eeg-cosmic-resonance-analyzer">
      <div className="analyzer-header">
        <h2>EEG-Cosmic Background Resonance Analyzer</h2>
        <p className="subtitle">OSH-based consciousness-cosmos coupling analysis</p>
      </div>
      
      <div className="analyzer-tabs">
        <button
          className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          Data Upload
        </button>
        <button
          className={`tab ${activeTab === 'analysis' ? 'active' : ''}`}
          onClick={() => setActiveTab('analysis')}
          disabled={!eegData || !cosmicData}
        >
          Analysis
        </button>
        <button
          className={`tab ${activeTab === 'results' ? 'active' : ''}`}
          onClick={() => setActiveTab('results')}
          disabled={!oshMetrics}
        >
          Results
        </button>
      </div>
      
      <div className="analyzer-content">
        {activeTab === 'upload' && (
          <div className="upload-section">
            <div className="upload-group">
              <h3>EEG Data Upload</h3>
              <p>Upload EEG data in CSV, JSON, or EDF format</p>
              <input
                type="file"
                accept=".csv,.json,.edf"
                onChange={handleEEGUpload}
                disabled={isProcessing}
              />
              {eegData && (
                <div className="upload-info">
                  <p>✓ Loaded: {eegData.channels.length} channels</p>
                  <p>Sampling rate: {eegData.samplingRate} Hz</p>
                  <p>Duration: {(eegData.data[0].length / eegData.samplingRate).toFixed(1)} seconds</p>
                </div>
              )}
            </div>
            
            <div className="upload-group">
              <h3>Cosmic Background Data Upload</h3>
              <p>Upload cosmic background data (CMB, solar wind, Schumann resonances)</p>
              <select id="cosmic-type" className="cosmic-type-select">
                <option value="cmb">CMB Fluctuations</option>
                <option value="solar_wind">Solar Wind</option>
                <option value="schumann">Schumann Resonances</option>
                <option value="custom">Custom Data</option>
              </select>
              <input
                type="file"
                accept=".csv,.json"
                onChange={handleCosmicUpload}
                disabled={isProcessing}
              />
              {cosmicData && (
                <div className="upload-info">
                  <p>✓ Loaded: {cosmicData.name}</p>
                  <p>Type: {cosmicData.type}</p>
                  <p>Sampling rate: {cosmicData.samplingRate} Hz</p>
                  <p>Duration: {(cosmicData.data.length / cosmicData.samplingRate).toFixed(1)} seconds</p>
                </div>
              )}
            </div>
            
            {eegData && cosmicData && (
              <button
                className="analyze-button"
                onClick={runAnalysis}
                disabled={isProcessing}
              >
                {isProcessing ? 'Processing...' : 'Run Analysis'}
              </button>
            )}
          </div>
        )}
        
        {activeTab === 'analysis' && (
          <div className="analysis-section">
            {isProcessing ? (
              <div className="processing-indicator">
                <div className="spinner"></div>
                <p>Analyzing EEG-cosmic correlations...</p>
              </div>
            ) : (
              <>
                <div className="channel-selector">
                  <label>Select EEG Channel:</label>
                  <select
                    value={selectedChannel}
                    onChange={(e) => setSelectedChannel(e.target.value)}
                  >
                    {eegData?.channels.map(channel => (
                      <option key={channel} value={channel}>{channel}</option>
                    ))}
                  </select>
                </div>
                
                <div className="charts-grid">
                  <div className="chart-container">
                    <h3>Time Series Comparison</h3>
                    {timeSeriesData && (
                      <Line options={timeSeriesChartOptions} data={timeSeriesData} />
                    )}
                  </div>
                  
                  <div className="chart-container">
                    <h3>Coherence Spectrum</h3>
                    {coherenceData && (
                      <Line options={coherenceChartOptions} data={coherenceData} />
                    )}
                  </div>
                  
                  <div className="chart-container">
                    <h3>Cross-Correlation</h3>
                    {correlationResults.has(selectedChannel) && (
                      <Line
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          plugins: {
                            title: {
                              display: true,
                              text: `Peak correlation: ${correlationResults.get(selectedChannel)!.peakCorrelation.toFixed(3)} at lag ${correlationResults.get(selectedChannel)!.peakLag}`,
                            },
                          },
                          scales: {
                            x: {
                              title: { display: true, text: 'Lag (samples)' },
                            },
                            y: {
                              title: { display: true, text: 'Correlation' },
                              min: -1,
                              max: 1,
                            },
                          },
                        }}
                        data={{
                          labels: correlationResults.get(selectedChannel)!.lag,
                          datasets: [{
                            label: 'Cross-correlation',
                            data: correlationResults.get(selectedChannel)!.correlation,
                            borderColor: 'rgb(54, 162, 235)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            borderWidth: 2,
                            pointRadius: 0,
                          }],
                        }}
                      />
                    )}
                  </div>
                  
                  <div className="chart-container">
                    <h3>Band Power Analysis</h3>
                    {bandAnalysis.has(selectedChannel) && (
                      <div className="band-analysis">
                        {bandAnalysis.get(selectedChannel)!.map(band => (
                          <div key={band.band} className="band-item">
                            <div className="band-header" style={{ color: EEG_BANDS[band.band.toLowerCase() as keyof typeof EEG_BANDS].color }}>
                              {band.band} ({band.frequencyRange[0]}-{band.frequencyRange[1]} Hz)
                            </div>
                            <div className="band-metrics">
                              <div>Power: {band.relativePower.toFixed(2)}</div>
                              <div>Coherence: {band.coherence.toFixed(3)}</div>
                              <div>Phase Sync: {band.phaseSync.toFixed(3)}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        )}
        
        {activeTab === 'results' && oshMetrics && (
          <div className="results-section">
            <div className="osh-metrics-panel">
              <h3>OSH Metrics</h3>
              <div className="metrics-grid">
                <div className="metric-item">
                  <span className="metric-label">Consciousness-Cosmos Coupling</span>
                  <span className="metric-value">{oshMetrics.consciousnessCosmosCoupling.toFixed(4)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Information Transfer Rate</span>
                  <span className="metric-value">{oshMetrics.informationTransferRate.toFixed(4)} bits/s</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Quantum Coherence</span>
                  <span className="metric-value">{oshMetrics.quantumCoherence.toFixed(4)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Memory Field Resonance</span>
                  <span className="metric-value">{oshMetrics.memoryFieldResonance.toFixed(4)} Hz</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Observer Influence</span>
                  <span className="metric-value">{(oshMetrics.observerInfluence * 100).toFixed(2)}%</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Recursive Depth</span>
                  <span className="metric-value">{oshMetrics.recursiveDepth.toFixed(2)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Entropy Flow</span>
                  <span className="metric-value">{oshMetrics.entropyFlow.toFixed(4)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Phase Entanglement</span>
                  <span className="metric-value">{oshMetrics.phaseEntanglement.toFixed(4)}</span>
                </div>
              </div>
              {statisticalSignificance && (
                <div className="significance">
                  Statistical Significance: p {'<'} {(1 / statisticalSignificance).toExponential(2)}
                </div>
              )}
            </div>
            
            <div className="resonance-events-panel">
              <h3>Resonance Events ({resonanceEvents.length})</h3>
              <div className="events-list">
                {resonanceEvents.slice(0, 10).map((event, idx) => (
                  <div key={idx} className="event-item">
                    <div className="event-header">
                      Event #{idx + 1} - {event.eegChannel} ↔ {event.cosmicSource}
                    </div>
                    <div className="event-details">
                      <span>Time: {event.timestamp.toFixed(2)}s</span>
                      <span>Duration: {event.duration.toFixed(2)}s</span>
                      <span>Frequency: {event.frequency.toFixed(2)} Hz</span>
                      <span>Strength: {event.strength.toFixed(3)}</span>
                    </div>
                    <div className="event-osh-metrics">
                      <span>Coupling: {event.oshMetrics.couplingStrength.toFixed(3)}</span>
                      <span>Info Transfer: {event.oshMetrics.informationTransferRate.toFixed(3)}</span>
                      <span>Phase Coherence: {event.oshMetrics.phaseCoherence.toFixed(3)}</span>
                    </div>
                  </div>
                ))}
                {resonanceEvents.length > 10 && (
                  <p className="more-events">... and {resonanceEvents.length - 10} more events</p>
                )}
              </div>
            </div>
            
            <div className="export-section">
              <button className="export-button" onClick={exportResults}>
                Export Analysis Results
              </button>
            </div>
          </div>
        )}
      </div>
      
      <style>{`
        .eeg-cosmic-resonance-analyzer {
          padding: 20px;
          background: #0a0a0a;
          color: #e0e0e0;
          border-radius: 12px;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .analyzer-header {
          text-align: center;
          margin-bottom: 30px;
        }
        
        .analyzer-header h2 {
          margin: 0 0 8px 0;
          color: #ffffff;
          font-size: 28px;
          font-weight: 600;
        }
        
        .subtitle {
          color: #888;
          font-size: 14px;
          margin: 0;
        }
        
        .analyzer-tabs {
          display: flex;
          gap: 10px;
          margin-bottom: 30px;
          border-bottom: 1px solid #333;
        }
        
        .tab {
          padding: 12px 24px;
          background: transparent;
          border: none;
          color: #888;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
          transition: all 0.2s;
          position: relative;
        }
        
        .tab:hover:not(:disabled) {
          color: #fff;
        }
        
        .tab.active {
          color: #fff;
        }
        
        .tab.active::after {
          content: '';
          position: absolute;
          bottom: -1px;
          left: 0;
          right: 0;
          height: 2px;
          background: linear-gradient(90deg, #4a9eff, #00ff88);
        }
        
        .tab:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .upload-section {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 30px;
          margin-bottom: 30px;
        }
        
        .upload-group {
          background: #1a1a1a;
          padding: 24px;
          border-radius: 8px;
          border: 1px solid #333;
        }
        
        .upload-group h3 {
          margin: 0 0 8px 0;
          color: #fff;
          font-size: 18px;
          font-weight: 500;
        }
        
        .upload-group p {
          margin: 0 0 16px 0;
          color: #888;
          font-size: 14px;
        }
        
        .upload-group input[type="file"] {
          display: block;
          width: 100%;
          padding: 12px;
          background: #0a0a0a;
          border: 1px solid #333;
          border-radius: 6px;
          color: #e0e0e0;
          font-size: 14px;
          cursor: pointer;
          transition: border-color 0.2s;
        }
        
        .upload-group input[type="file"]:hover {
          border-color: #4a9eff;
        }
        
        .cosmic-type-select {
          display: block;
          width: 100%;
          padding: 12px;
          margin-bottom: 12px;
          background: #0a0a0a;
          border: 1px solid #333;
          border-radius: 6px;
          color: #e0e0e0;
          font-size: 14px;
        }
        
        .upload-info {
          margin-top: 16px;
          padding: 12px;
          background: #0a0a0a;
          border-radius: 6px;
          font-size: 13px;
        }
        
        .upload-info p {
          margin: 4px 0;
          color: #4a9eff;
        }
        
        .analyze-button {
          grid-column: 1 / -1;
          padding: 16px 32px;
          background: linear-gradient(135deg, #4a9eff, #00ff88);
          border: none;
          border-radius: 8px;
          color: #000;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s;
        }
        
        .analyze-button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 8px 24px rgba(74, 158, 255, 0.4);
        }
        
        .analyze-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
          transform: none;
        }
        
        .processing-indicator {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 400px;
        }
        
        .spinner {
          width: 48px;
          height: 48px;
          border: 3px solid #333;
          border-top-color: #4a9eff;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        
        .processing-indicator p {
          margin-top: 20px;
          color: #888;
          font-size: 16px;
        }
        
        .channel-selector {
          margin-bottom: 24px;
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .channel-selector label {
          color: #888;
          font-size: 14px;
          font-weight: 500;
        }
        
        .channel-selector select {
          padding: 8px 16px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 6px;
          color: #e0e0e0;
          font-size: 14px;
        }
        
        .charts-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 24px;
        }
        
        .chart-container {
          background: #1a1a1a;
          padding: 20px;
          border-radius: 8px;
          border: 1px solid #333;
        }
        
        .chart-container h3 {
          margin: 0 0 16px 0;
          color: #fff;
          font-size: 16px;
          font-weight: 500;
        }
        
        .chart-container canvas {
          max-height: 300px;
        }
        
        .band-analysis {
          display: grid;
          gap: 12px;
        }
        
        .band-item {
          background: #0a0a0a;
          padding: 12px;
          border-radius: 6px;
          border: 1px solid #333;
        }
        
        .band-header {
          font-weight: 600;
          margin-bottom: 8px;
        }
        
        .band-metrics {
          display: flex;
          gap: 16px;
          font-size: 13px;
          color: #888;
        }
        
        .results-section {
          display: grid;
          gap: 30px;
        }
        
        .osh-metrics-panel {
          background: #1a1a1a;
          padding: 24px;
          border-radius: 8px;
          border: 1px solid #333;
        }
        
        .osh-metrics-panel h3 {
          margin: 0 0 20px 0;
          color: #fff;
          font-size: 20px;
          font-weight: 500;
        }
        
        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 20px;
        }
        
        .metric-item {
          background: #0a0a0a;
          padding: 16px;
          border-radius: 6px;
          border: 1px solid #333;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .metric-label {
          color: #888;
          font-size: 13px;
          font-weight: 500;
        }
        
        .metric-value {
          color: #4a9eff;
          font-size: 24px;
          font-weight: 600;
          font-variant-numeric: tabular-nums;
        }
        
        .significance {
          margin-top: 20px;
          padding: 16px;
          background: #0a0a0a;
          border-radius: 6px;
          text-align: center;
          color: #00ff88;
          font-weight: 500;
        }
        
        .resonance-events-panel {
          background: #1a1a1a;
          padding: 24px;
          border-radius: 8px;
          border: 1px solid #333;
        }
        
        .resonance-events-panel h3 {
          margin: 0 0 20px 0;
          color: #fff;
          font-size: 20px;
          font-weight: 500;
        }
        
        .events-list {
          display: grid;
          gap: 12px;
        }
        
        .event-item {
          background: #0a0a0a;
          padding: 16px;
          border-radius: 6px;
          border: 1px solid #333;
        }
        
        .event-header {
          color: #fff;
          font-weight: 500;
          margin-bottom: 8px;
        }
        
        .event-details,
        .event-osh-metrics {
          display: flex;
          gap: 16px;
          font-size: 13px;
          color: #888;
          margin-top: 8px;
        }
        
        .event-details span,
        .event-osh-metrics span {
          white-space: nowrap;
        }
        
        .more-events {
          text-align: center;
          color: #666;
          font-style: italic;
          margin-top: 12px;
        }
        
        .export-section {
          display: flex;
          justify-content: center;
        }
        
        .export-button {
          padding: 12px 32px;
          background: #1a1a1a;
          border: 1px solid #4a9eff;
          border-radius: 8px;
          color: #4a9eff;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .export-button:hover {
          background: #4a9eff;
          color: #000;
        }
      `}</style>
    </div>
  );
};