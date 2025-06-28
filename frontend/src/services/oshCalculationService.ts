import { 
  OSHCalculationService as IOSHCalculationService,
  RSPCalculation,
  CalculationResult,
  CMBAnalysisResult
} from '../components/osh/types';
import { API_ENDPOINTS } from '../config/api';

export class OSHCalculationService implements IOSHCalculationService {
  async calculateRSP(data: RSPCalculation): Promise<CalculationResult> {
    const response = await fetch(API_ENDPOINTS.osh.rsp, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        integrated_information: data.integratedInformation,
        kolmogorov_complexity: data.kolmogorovComplexity,
        entropy_flux: data.entropyFlux,
        system_name: data.systemName,
      }),
    });
    return response.json();
  }

  async calculateRSPBound(area: number, minEntropyFlux?: number): Promise<CalculationResult> {
    const params = new URLSearchParams({ area: area.toString() });
    if (minEntropyFlux) params.append('min_entropy_flux', minEntropyFlux.toString());
    
    const response = await fetch(`${API_ENDPOINTS.osh.rspBound}?${params}`, {
      method: 'POST',
    });
    return response.json();
  }

  async analyzeCMBComplexity(data: number[], samplingRate: number): Promise<CalculationResult> {
    const response = await fetch(API_ENDPOINTS.osh.cmbComplexity, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        cmb_data: data,
        sampling_rate: samplingRate,
        search_recursive_patterns: true,
      }),
    });
    return response.json();
  }

  async searchGWEchoes(
    strainData: number[],
    samplingRate: number,
    mergerTime?: number,
    expectedEchoDelay: number = 15.0
  ): Promise<CalculationResult> {
    const response = await fetch(API_ENDPOINTS.osh.gwEchoes, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        strain_data: strainData,
        sampling_rate: samplingRate,
        merger_time: mergerTime,
        expected_echo_delay: expectedEchoDelay,
      }),
    });
    return response.json();
  }

  async analyzeEEGCosmicResonance(
    eegData: number[][],
    cosmicData: number[],
    samplingRate: number
  ): Promise<CalculationResult> {
    const response = await fetch(API_ENDPOINTS.osh.eegCosmic, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        eeg_data: eegData,
        cosmic_data: cosmicData,
        sampling_rate: samplingRate,
      }),
    });
    return response.json();
  }

  async mapConsciousnessDynamics(
    scale: string,
    information: number,
    complexity: number,
    entropyFlux: number
  ): Promise<CalculationResult> {
    const response = await fetch(API_ENDPOINTS.osh.consciousnessMap, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        system_scale: scale,
        information_content: information,
        complexity: complexity,
        entropy_flux: entropyFlux,
      }),
    });
    return response.json();
  }

  // Enhanced CMB analysis with full complexity calculation
  async analyzeCMBComplexityFull(
    data: number[], 
    metadata: {
      samplingRate?: number;
      resolution?: number;
      source?: string;
    } = {}
  ): Promise<CalculationResult & { result: CMBAnalysisResult }> {
    const response = await fetch(`${API_ENDPOINTS.osh.cmbComplexity.replace('/cmb-complexity', '/cmb-analysis-full')}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        cmb_data: data,
        sampling_rate: metadata.samplingRate || 1.0,
        resolution: metadata.resolution,
        source: metadata.source,
        analysis_options: {
          calculate_lempel_ziv: true,
          calculate_statistics: true,
          calculate_power_spectrum: true,
          search_recursive_patterns: true,
          calculate_osh_signatures: true,
        },
      }),
    });
    return response.json();
  }
}