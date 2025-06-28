import { useState, useCallback, useEffect } from 'react';
import { useEngineAPIContext } from '../contexts/EngineAPIContext';

// Types for OSH calculations
export interface OSHCalculation {
  id: string;
  type: 'rsp' | 'consciousness' | 'cmb' | 'gw' | 'eeg' | 'void' | 'constant' | 'test';
  status: 'pending' | 'running' | 'completed' | 'failed';
  input: Record<string, any>;
  result?: Record<string, any>;
  error?: string;
  timestamp: Date;
}

export interface RSPSystemPreset {
  name: string;
  information: number;
  complexity: number;
  entropyFlux: number;
  description: string;
}

export interface ConsciousnessScale {
  scale: string;
  typicalFrequency: string;
  coherenceTime: string;
  examples: string[];
}

interface OSHCalculationsState {
  calculations: OSHCalculation[];
  activeCalculation: OSHCalculation | null;
  isCalculating: boolean;
  error: string | null;
}

// Preset systems for quick RSP calculations
export const RSP_PRESETS: RSPSystemPreset[] = [
  {
    name: 'Microtubule',
    information: 1e6,
    complexity: 1e4,
    entropyFlux: 1e2,
    description: 'Quantum coherence in neurons',
  },
  {
    name: 'Human Brain',
    information: 1e16,
    complexity: 1e9,
    entropyFlux: 1e6,
    description: 'Full human consciousness',
  },
  {
    name: 'Earth Biosphere',
    information: 1e40,
    complexity: 1e30,
    entropyFlux: 1e20,
    description: 'Gaia hypothesis system',
  },
  {
    name: 'Solar Mass Black Hole',
    information: 1e77,
    complexity: 1e77,
    entropyFlux: 1e-10,
    description: 'Maximal RSP attractor',
  },
  {
    name: 'Observable Universe',
    information: 1e122,
    complexity: 1e100,
    entropyFlux: 1e50,
    description: 'Cosmic-scale consciousness',
  },
];

// Consciousness scales for mapping
export const CONSCIOUSNESS_SCALES: ConsciousnessScale[] = [
  {
    scale: 'quantum',
    typicalFrequency: '40 Hz',
    coherenceTime: '1-100 μs',
    examples: ['Microtubules', 'Quantum dots', 'Bose-Einstein condensates'],
  },
  {
    scale: 'neural',
    typicalFrequency: '0.1-100 Hz',
    coherenceTime: '10-1000 ms',
    examples: ['Human brain', 'Octopus nervous system', 'Bee colonies'],
  },
  {
    scale: 'planetary',
    typicalFrequency: '0.00001-0.1 Hz',
    coherenceTime: 'Hours to years',
    examples: ['Gaia biosphere', 'Magnetosphere', 'Ocean currents'],
  },
  {
    scale: 'stellar',
    typicalFrequency: '10^-9 - 10^-6 Hz',
    coherenceTime: 'Years to millions of years',
    examples: ['Solar convection', 'Stellar magnetism', 'Neutron stars'],
  },
  {
    scale: 'galactic',
    typicalFrequency: '10^-15 - 10^-12 Hz',
    coherenceTime: 'Millions to billions of years',
    examples: ['Spiral arms', 'Galactic halos', 'AGN feedback'],
  },
  {
    scale: 'cosmic',
    typicalFrequency: '10^-18 - 10^-15 Hz',
    coherenceTime: 'Billions of years',
    examples: ['Large scale structure', 'Dark energy', 'CMB patterns'],
  },
];

export const useOSHCalculations = () => {
  const { isConnected } = useEngineAPIContext();
  const [state, setState] = useState<OSHCalculationsState>({
    calculations: [],
    activeCalculation: null,
    isCalculating: false,
    error: null,
  });

  // WebSocket functionality disabled - using HTTP endpoints instead
  useEffect(() => {
    // WebSocket message handling would go here if enabled
  }, []);

  // Calculate RSP
  const calculateRSP = useCallback(
    async (
      information: number,
      complexity: number,
      entropyFlux: number,
      systemName?: string
    ): Promise<OSHCalculation> => {
      const calculation: OSHCalculation = {
        id: `rsp-${Date.now()}`,
        type: 'rsp',
        status: 'running',
        input: { information, complexity, entropyFlux, systemName },
        timestamp: new Date(),
      };

      setState((prev) => ({
        ...prev,
        calculations: [calculation, ...prev.calculations],
        activeCalculation: calculation,
        isCalculating: true,
        error: null,
      }));

      try {
        const response = await fetch('http://localhost:8080/api/osh/rsp', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            integrated_information: information,
            kolmogorov_complexity: complexity,
            entropy_flux: entropyFlux,
            system_name: systemName,
          }),
        });

        const result = await response.json();
        
        const updatedCalculation = {
          ...calculation,
          status: result.success ? ('completed' as const) : ('failed' as const),
          result: result.result,
          error: result.errors?.join(', '),
        };

        setState((prev) => ({
          ...prev,
          calculations: prev.calculations.map((calc) =>
            calc.id === calculation.id ? updatedCalculation : calc
          ),
          activeCalculation: updatedCalculation,
          isCalculating: false,
        }));

        return updatedCalculation;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        const failedCalculation = {
          ...calculation,
          status: 'failed' as const,
          error: errorMessage,
        };

        setState((prev) => ({
          ...prev,
          calculations: prev.calculations.map((calc) =>
            calc.id === calculation.id ? failedCalculation : calc
          ),
          activeCalculation: failedCalculation,
          isCalculating: false,
          error: errorMessage,
        }));

        throw error;
      }
    },
    []
  );

  // Calculate RSP upper bound
  const calculateRSPBound = useCallback(
    async (area: number, minEntropyFlux?: number): Promise<OSHCalculation> => {
      const calculation: OSHCalculation = {
        id: `rsp-bound-${Date.now()}`,
        type: 'rsp',
        status: 'running',
        input: { area, minEntropyFlux },
        timestamp: new Date(),
      };

      setState((prev) => ({
        ...prev,
        calculations: [calculation, ...prev.calculations],
        activeCalculation: calculation,
        isCalculating: true,
        error: null,
      }));

      try {
        const params = new URLSearchParams({ area: area.toString() });
        if (minEntropyFlux) params.append('min_entropy_flux', minEntropyFlux.toString());

        const response = await fetch(`http://localhost:8080/api/osh/rsp-bound?${params}`, {
          method: 'POST',
        });

        const result = await response.json();
        
        const updatedCalculation = {
          ...calculation,
          status: result.success ? ('completed' as const) : ('failed' as const),
          result: result.result,
          error: result.errors?.join(', '),
        };

        setState((prev) => ({
          ...prev,
          calculations: prev.calculations.map((calc) =>
            calc.id === calculation.id ? updatedCalculation : calc
          ),
          activeCalculation: updatedCalculation,
          isCalculating: false,
        }));

        return updatedCalculation;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        const failedCalculation = {
          ...calculation,
          status: 'failed' as const,
          error: errorMessage,
        };

        setState((prev) => ({
          ...prev,
          calculations: prev.calculations.map((calc) =>
            calc.id === calculation.id ? failedCalculation : calc
          ),
          activeCalculation: failedCalculation,
          isCalculating: false,
          error: errorMessage,
        }));

        throw error;
      }
    },
    []
  );

  // Map consciousness dynamics
  const mapConsciousnessDynamics = useCallback(
    async (
      scale: string,
      information: number,
      complexity: number,
      entropyFlux: number
    ): Promise<OSHCalculation> => {
      const calculation: OSHCalculation = {
        id: `consciousness-${Date.now()}`,
        type: 'consciousness',
        status: 'running',
        input: { scale, information, complexity, entropyFlux },
        timestamp: new Date(),
      };

      setState((prev) => ({
        ...prev,
        calculations: [calculation, ...prev.calculations],
        activeCalculation: calculation,
        isCalculating: true,
        error: null,
      }));

      try {
        const response = await fetch('http://localhost:8080/api/osh/consciousness-map', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            system_scale: scale,
            information_content: information,
            complexity: complexity,
            entropy_flux: entropyFlux,
          }),
        });

        const result = await response.json();
        
        const updatedCalculation = {
          ...calculation,
          status: result.success ? ('completed' as const) : ('failed' as const),
          result: result.result,
          error: result.errors?.join(', '),
        };

        setState((prev) => ({
          ...prev,
          calculations: prev.calculations.map((calc) =>
            calc.id === calculation.id ? updatedCalculation : calc
          ),
          activeCalculation: updatedCalculation,
          isCalculating: false,
        }));

        return updatedCalculation;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        const failedCalculation = {
          ...calculation,
          status: 'failed' as const,
          error: errorMessage,
        };

        setState((prev) => ({
          ...prev,
          calculations: prev.calculations.map((calc) =>
            calc.id === calculation.id ? failedCalculation : calc
          ),
          activeCalculation: failedCalculation,
          isCalculating: false,
          error: errorMessage,
        }));

        throw error;
      }
    },
    []
  );

  // Clear calculation history
  const clearHistory = useCallback(() => {
    setState((prev) => ({
      ...prev,
      calculations: [],
      activeCalculation: null,
      error: null,
    }));
  }, []);

  // Get calculation by ID
  const getCalculation = useCallback(
    (id: string): OSHCalculation | undefined => {
      return state.calculations.find((calc) => calc.id === id);
    },
    [state.calculations]
  );

  // Get calculations by type
  const getCalculationsByType = useCallback(
    (type: OSHCalculation['type']): OSHCalculation[] => {
      return state.calculations.filter((calc) => calc.type === type);
    },
    [state.calculations]
  );

  // Export calculations as JSON
  const exportCalculations = useCallback(() => {
    const data = {
      version: '1.0',
      timestamp: new Date().toISOString(),
      calculations: state.calculations,
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `osh-calculations-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [state.calculations]);

  // Get RSP classification based on value
  const getRSPClassification = useCallback((rspValue: number): string => {
    if (rspValue > 1e100) return 'Maximal RSP Attractor (Black Hole)';
    if (rspValue > 1e10) return 'High RSP (Conscious System)';
    if (rspValue > 1e3) return 'Moderate RSP (Complex Structure)';
    return 'Low RSP (Simple/Dissipative)';
  }, []);

  return {
    // State
    calculations: state.calculations,
    activeCalculation: state.activeCalculation,
    isCalculating: state.isCalculating,
    error: state.error,
    isConnected,

    // Actions
    calculateRSP,
    calculateRSPBound,
    mapConsciousnessDynamics,
    clearHistory,
    getCalculation,
    getCalculationsByType,
    exportCalculations,

    // Utilities
    getRSPClassification,
    presets: RSP_PRESETS,
    scales: CONSCIOUSNESS_SCALES,
  };
};

export default useOSHCalculations;