import React, { useState, useCallback, useMemo, useRef } from 'react';
import {
  CloudUploadIcon,
  PlayIcon,
  RefreshCwIcon,
  DownloadIcon,
  ZoomInIcon,
  ZoomOutIcon,
  ActivityIcon,
  ChevronDownIcon,
  InfoIcon,
  AlertTriangleIcon,
  CheckCircleIcon,
} from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  TimeScale,
  Filler,
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import zoomPlugin from 'chartjs-plugin-zoom';
import annotationPlugin from 'chartjs-plugin-annotation';
import Papa from 'papaparse';
import { saveAs } from 'file-saver';
import { GWEchoDetector, EchoCandidate, GWData, OSHMetrics } from './types/gwEchoTypes';
import { SpectrogramVisualizer } from './SpectrogramVisualizer';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  Legend,
  TimeScale,
  Filler,
  zoomPlugin,
  annotationPlugin
);

interface GravitationalWaveEchoSearchProps {
  onEchoDetected?: (echoes: EchoCandidate[]) => void;
  onOSHMetricsCalculated?: (metrics: OSHMetrics) => void;
}

export const GravitationalWaveEchoSearch: React.FC<GravitationalWaveEchoSearchProps> = ({
  onEchoDetected,
  onOSHMetricsCalculated,
}) => {
  // State management
  const [gwData, setGwData] = useState<GWData | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [echoCandidates, setEchoCandidates] = useState<EchoCandidate[]>([]);
  const [oshMetrics, setOshMetrics] = useState<OSHMetrics | null>(null);
  const [selectedEcho, setSelectedEcho] = useState<EchoCandidate | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState({
    parameters: true,
    visualization: true,
    results: true,
  });
  
  // Search parameters
  const [searchParams, setSearchParams] = useState({
    minTimeDelay: 0.01, // seconds
    maxTimeDelay: 1.0,  // seconds
    confidenceThreshold: 0.7,
    windowSize: 0.1,    // seconds
    overlapFraction: 0.5,
    detectionMethod: 'autocorrelation' as 'autocorrelation' | 'matched_filter' | 'hybrid',
    noiseReduction: true,
    adaptiveThreshold: false,
  });

  // Visualization settings
  const [vizSettings, setVizSettings] = useState({
    showSpectrogram: true,
    showEchoes: true,
    showConfidenceBands: true,
    timeRange: { start: 0, end: 1 },
    amplitudeScale: 'linear' as 'linear' | 'log',
  });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chartRef = useRef<any>(null);

  // File upload handler
  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadError(null);
    setIsProcessing(true);

    try {
      let data: GWData;
      
      if (file.name.endsWith('.csv')) {
        data = await parseCSV(file);
      } else if (file.name.endsWith('.json')) {
        data = await parseJSON(file);
      } else if (file.name.endsWith('.h5') || file.name.endsWith('.hdf5')) {
        data = await parseHDF5(file);
      } else {
        throw new Error('Unsupported file format. Please use CSV, JSON, or HDF5.');
      }

      setGwData(data);
      // Auto-adjust time range
      setVizSettings(prev => ({
        ...prev,
        timeRange: { start: 0, end: data.time[data.time.length - 1] }
      }));
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Failed to parse file');
    } finally {
      setIsProcessing(false);
    }
  }, []);

  // Parse CSV file
  const parseCSV = async (file: File): Promise<GWData> => {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        complete: (results) => {
          try {
            const time = results.data.map((row: any) => row.time || row.t || row.Time);
            const strain = results.data.map((row: any) => row.strain || row.h || row.Strain);
            
            if (time.some((t: any) => isNaN(t)) || strain.some((s: any) => isNaN(s))) {
              throw new Error('Invalid data format in CSV');
            }

            resolve({
              time: time as number[],
              strain: strain as number[],
              sampleRate: 1 / (time[1] - time[0]),
              metadata: {
                detector: results.meta.fields?.includes('detector') ? 
                  results.data[0].detector : 'Unknown',
                gpsTime: results.data[0].gps_time || Date.now() / 1000,
              }
            });
          } catch (error) {
            reject(error);
          }
        },
        error: (error) => reject(error),
      });
    });
  };

  // Parse JSON file
  const parseJSON = async (file: File): Promise<GWData> => {
    const text = await file.text();
    const data = JSON.parse(text);
    
    if (!data.time || !data.strain) {
      throw new Error('JSON must contain "time" and "strain" arrays');
    }

    return {
      time: data.time,
      strain: data.strain,
      sampleRate: data.sampleRate || 1 / (data.time[1] - data.time[0]),
      metadata: data.metadata || {},
    };
  };

  // Parse HDF5 file (placeholder - requires additional library)
  const parseHDF5 = async (file: File): Promise<GWData> => {
    // In a real implementation, you would use a library like h5wasm
    throw new Error('HDF5 support coming soon. Please use CSV or JSON format.');
  };

  // Echo detection algorithm
  const detectEchoes = useCallback(async () => {
    if (!gwData) return;

    setIsProcessing(true);
    setEchoCandidates([]);

    try {
      const detector = new GWEchoDetector(gwData, searchParams);
      const candidates = await detector.findEchoes();
      
      setEchoCandidates(candidates);
      onEchoDetected?.(candidates);

      // Calculate OSH metrics
      if (candidates.length > 0) {
        const metrics = calculateOSHMetrics(candidates, gwData);
        setOshMetrics(metrics);
        onOSHMetricsCalculated?.(metrics);
      }
    } catch (error) {
      console.error('Echo detection failed:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [gwData, searchParams, onEchoDetected, onOSHMetricsCalculated]);

  // Calculate OSH metrics from echo candidates
  const calculateOSHMetrics = (echoes: EchoCandidate[], data: GWData): OSHMetrics => {
    // Information leakage estimation
    const totalEchoEnergy = echoes.reduce((sum, echo) => sum + echo.amplitude ** 2, 0);
    const totalSignalEnergy = data.strain.reduce((sum, s) => sum + s ** 2, 0);
    const informationLeakage = totalEchoEnergy / totalSignalEnergy;

    // Memory field coherence from echo spacing regularity
    const echoDelays = echoes.map(e => e.timeDelay).sort((a, b) => a - b);
    const delayDifferences = echoDelays.slice(1).map((d, i) => d - echoDelays[i]);
    const avgDiff = delayDifferences.reduce((a, b) => a + b, 0) / delayDifferences.length;
    const variance = delayDifferences.reduce((sum, d) => sum + (d - avgDiff) ** 2, 0) / delayDifferences.length;
    const memoryFieldCoherence = 1 / (1 + Math.sqrt(variance));

    // Recursive depth estimation
    const maxEchoOrder = Math.max(...echoes.map(e => e.metadata?.echoOrder || 1));
    const recursiveDepth = Math.log2(maxEchoOrder + 1);

    // Observer influence metric
    const avgConfidence = echoes.reduce((sum, e) => sum + e.confidence, 0) / echoes.length;
    const observerInfluence = avgConfidence * memoryFieldCoherence;

    return {
      informationLeakage,
      memoryFieldCoherence,
      recursiveDepth,
      observerInfluence,
      totalEchoes: echoes.length,
      averageEchoStrength: totalEchoEnergy / echoes.length,
    };
  };

  // Export detected echoes
  const exportEchoes = useCallback(() => {
    if (echoCandidates.length === 0) return;

    const exportData = {
      metadata: {
        detectionDate: new Date().toISOString(),
        searchParameters: searchParams,
        dataSource: gwData?.metadata,
      },
      echoes: echoCandidates.map(echo => ({
        ...echo,
        oshMetrics: oshMetrics,
      })),
      summary: {
        totalEchoes: echoCandidates.length,
        averageConfidence: echoCandidates.reduce((sum, e) => sum + e.confidence, 0) / echoCandidates.length,
        timeRange: {
          min: Math.min(...echoCandidates.map(e => e.time)),
          max: Math.max(...echoCandidates.map(e => e.time)),
        },
      },
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    saveAs(blob, `gw_echoes_${Date.now()}.json`);
  }, [echoCandidates, searchParams, gwData, oshMetrics]);

  // Chart data preparation
  const chartData = useMemo(() => {
    if (!gwData) return null;

    const { time, strain } = gwData;
    const { timeRange } = vizSettings;
    
    // Filter data to visible range
    const startIdx = time.findIndex(t => t >= timeRange.start);
    const endIdx = time.findIndex(t => t > timeRange.end);
    const visibleTime = time.slice(startIdx, endIdx);
    const visibleStrain = strain.slice(startIdx, endIdx);

    // Prepare echo annotations
    const annotations = vizSettings.showEchoes ? echoCandidates
      .filter(echo => echo.time >= timeRange.start && echo.time <= timeRange.end)
      .reduce((acc, echo, idx) => {
        acc[`echo_${idx}`] = {
          type: 'line',
          xMin: echo.time,
          xMax: echo.time,
          borderColor: `rgba(255, 99, 132, ${echo.confidence})`,
          borderWidth: 2,
          label: {
            enabled: true,
            content: `Echo ${idx + 1} (${(echo.confidence * 100).toFixed(0)}%)`,
            position: 'start',
          },
        };
        return acc;
      }, {} as any) : {};

    return {
      labels: visibleTime,
      datasets: [
        {
          label: 'GW Strain',
          data: visibleStrain,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0,
          fill: true,
        },
      ],
    };
  }, [gwData, vizSettings, echoCandidates]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      title: {
        display: true,
        text: 'Gravitational Wave Strain Data',
      },
      legend: {
        display: true,
        position: 'top' as const,
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
      annotation: {
        annotations: chartData?.datasets[0] ? 
          Object.entries(chartData.datasets[0]).reduce((acc, [key, value]) => {
            if (key.startsWith('echo_')) acc[key] = value;
            return acc;
          }, {} as any) : {},
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
        type: vizSettings.amplitudeScale as 'linear' | 'logarithmic',
        title: {
          display: true,
          text: 'Strain',
        },
      },
    },
  }), [chartData, vizSettings]);

  return (
    <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-900">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
          Gravitational Wave Echo Search
        </h2>
        <div className="flex items-center space-x-2">
          <ActivityIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          <span className="text-sm text-gray-600 dark:text-gray-400">
            OSH Echo Detection System
          </span>
        </div>
      </div>
      
      {/* File Upload Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.json,.h5,.hdf5"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isProcessing}
              className="w-full flex items-center justify-center px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
            >
              <CloudUploadIcon className="w-5 h-5 mr-2" />
              Upload GW Strain Data
            </button>
            {uploadError && (
              <div className="mt-3 p-3 bg-red-100 dark:bg-red-900/20 border border-red-400 dark:border-red-600 rounded-lg">
                <div className="flex items-center">
                  <AlertTriangleIcon className="w-5 h-5 text-red-600 dark:text-red-400 mr-2" />
                  <span className="text-sm text-red-800 dark:text-red-200">{uploadError}</span>
                </div>
              </div>
            )}
          </div>
          
          <div>
            {gwData && (
              <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                  Data Information
                </h4>
                <div className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                  <p>Samples: {gwData.strain.length.toLocaleString()}</p>
                  <p>Duration: {gwData.time[gwData.time.length - 1].toFixed(2)}s</p>
                  <p>Sample Rate: {gwData.sampleRate.toFixed(0)} Hz</p>
                  {gwData.metadata?.detector && (
                    <p>Detector: {gwData.metadata.detector}</p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Search Parameters */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md">
        <button
          onClick={() => setExpandedSections(prev => ({ ...prev, parameters: !prev.parameters }))}
          className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Echo Search Parameters
          </h3>
          <ChevronDownIcon 
            className={`w-5 h-5 text-gray-500 transition-transform ${
              expandedSections.parameters ? 'rotate-180' : ''
            }`}
          />
        </button>
        
        {expandedSections.parameters && (
          <div className="px-6 pb-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Time Delay Range (s)
                </label>
                <div className="flex space-x-2">
                  <input
                    type="number"
                    value={searchParams.minTimeDelay}
                    onChange={(e) => setSearchParams(prev => ({
                      ...prev,
                      minTimeDelay: parseFloat(e.target.value) || 0
                    }))}
                    className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    min="0"
                    step="0.001"
                    placeholder="Min"
                  />
                  <input
                    type="number"
                    value={searchParams.maxTimeDelay}
                    onChange={(e) => setSearchParams(prev => ({
                      ...prev,
                      maxTimeDelay: parseFloat(e.target.value) || 1
                    }))}
                    className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    min="0.001"
                    step="0.01"
                    placeholder="Max"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Confidence Threshold
                </label>
                <input
                  type="range"
                  value={searchParams.confidenceThreshold}
                  onChange={(e) => setSearchParams(prev => ({
                    ...prev,
                    confidenceThreshold: parseFloat(e.target.value)
                  }))}
                  min="0"
                  max="1"
                  step="0.01"
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                  <span>0%</span>
                  <span>{(searchParams.confidenceThreshold * 100).toFixed(0)}%</span>
                  <span>100%</span>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Detection Method
                </label>
                <select
                  value={searchParams.detectionMethod}
                  onChange={(e) => setSearchParams(prev => ({
                    ...prev,
                    detectionMethod: e.target.value as any
                  }))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="autocorrelation">Autocorrelation</option>
                  <option value="matched_filter">Matched Filter</option>
                  <option value="hybrid">Hybrid</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Window Size (s)
                </label>
                <input
                  type="number"
                  value={searchParams.windowSize}
                  onChange={(e) => setSearchParams(prev => ({
                    ...prev,
                    windowSize: parseFloat(e.target.value) || 0.1
                  }))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  min="0.001"
                  step="0.01"
                />
              </div>
              
              <div className="flex items-center space-x-6">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={searchParams.noiseReduction}
                    onChange={(e) => setSearchParams(prev => ({
                      ...prev,
                      noiseReduction: e.target.checked
                    }))}
                    className="mr-2"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    Noise Reduction
                  </span>
                </label>
                
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={searchParams.adaptiveThreshold}
                    onChange={(e) => setSearchParams(prev => ({
                      ...prev,
                      adaptiveThreshold: e.target.checked
                    }))}
                    className="mr-2"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    Adaptive Threshold
                  </span>
                </label>
              </div>
            </div>
            
            <div className="mt-6 flex space-x-3">
              <button
                onClick={detectEchoes}
                disabled={!gwData || isProcessing}
                className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 transition-colors"
              >
                <PlayIcon className="w-5 h-5 mr-2" />
                Start Echo Search
              </button>
              <button
                onClick={() => {
                  setEchoCandidates([]);
                  setOshMetrics(null);
                }}
                disabled={isProcessing}
                className="flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              >
                <RefreshCwIcon className="w-5 h-5 mr-2" />
                Clear Results
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Main Visualization */}
      {gwData && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <div className="relative" style={{ height: '400px' }}>
            {chartData && (
              <Line
                ref={chartRef}
                data={chartData}
                options={chartOptions}
              />
            )}
            {isProcessing && (
              <div className="absolute inset-0 bg-white/80 dark:bg-gray-800/80 flex items-center justify-center">
                <div className="text-center">
                  <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                  <p className="text-gray-700 dark:text-gray-300">Searching for echoes...</p>
                </div>
              </div>
            )}
          </div>
          
          {/* Visualization Controls */}
          <div className="mt-4 flex flex-wrap gap-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={vizSettings.showEchoes}
                onChange={(e) => setVizSettings(prev => ({
                  ...prev,
                  showEchoes: e.target.checked
                }))}
                className="mr-2"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">Show Echoes</span>
            </label>
            
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={vizSettings.amplitudeScale === 'log'}
                onChange={(e) => setVizSettings(prev => ({
                  ...prev,
                  amplitudeScale: e.target.checked ? 'log' : 'linear'
                }))}
                className="mr-2"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">Log Scale</span>
            </label>
            
            <button
              onClick={() => {
                if (chartRef.current) {
                  chartRef.current.resetZoom();
                }
              }}
              className="flex items-center px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <ZoomOutIcon className="w-4 h-4 mr-1" />
              Reset Zoom
            </button>
          </div>
        </div>
      )}

      {/* Spectrogram Visualization */}
      {gwData && vizSettings.showSpectrogram && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Spectrogram Analysis
          </h3>
          <SpectrogramVisualizer
            data={gwData}
            echoCandidates={echoCandidates}
            timeRange={vizSettings.timeRange}
          />
        </div>
      )}

      {/* Echo Candidates Table */}
      {echoCandidates.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Detected Echo Candidates ({echoCandidates.length})
            </h3>
            <button
              onClick={exportEchoes}
              className="flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <DownloadIcon className="w-4 h-4 mr-2" />
              Export Results
            </button>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Echo #
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Time (s)
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Delay (ms)
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Amplitude
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Confidence
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    SNR
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {echoCandidates.map((echo, idx) => (
                  <tr
                    key={idx}
                    className={`border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer ${
                      selectedEcho === echo ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                    }`}
                    onClick={() => setSelectedEcho(echo)}
                  >
                    <td className="px-4 py-3 text-sm text-gray-900 dark:text-white">
                      {idx + 1}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900 dark:text-white">
                      {echo.time.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900 dark:text-white">
                      {(echo.timeDelay * 1000).toFixed(1)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900 dark:text-white">
                      {echo.amplitude.toExponential(2)}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        echo.confidence > 0.8 
                          ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
                          : echo.confidence > 0.6
                          ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
                          : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                      }`}>
                        {(echo.confidence * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900 dark:text-white">
                      {echo.snr.toFixed(1)} dB
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          // Zoom to echo location
                          if (chartRef.current && gwData) {
                            const newRange = {
                              start: Math.max(0, echo.time - 0.05),
                              end: Math.min(gwData.time[gwData.time.length - 1], echo.time + 0.05)
                            };
                            setVizSettings(prev => ({ ...prev, timeRange: newRange }));
                          }
                        }}
                        className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                      >
                        <ZoomInIcon className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* OSH Metrics Dashboard */}
      {oshMetrics && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            OSH Metrics Analysis
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                Information Leakage
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {(oshMetrics.informationLeakage * 100).toFixed(2)}%
              </p>
              <div className="mt-2 w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                <div
                  className="bg-blue-600 dark:bg-blue-400 h-2 rounded-full"
                  style={{ width: `${oshMetrics.informationLeakage * 100}%` }}
                />
              </div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                Memory Field Coherence
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {oshMetrics.memoryFieldCoherence.toFixed(3)}
              </p>
              <div className="mt-2 w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                <div
                  className="bg-purple-600 dark:bg-purple-400 h-2 rounded-full"
                  style={{ width: `${oshMetrics.memoryFieldCoherence * 100}%` }}
                />
              </div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                Recursive Depth
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {oshMetrics.recursiveDepth.toFixed(1)}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Estimated layers
              </p>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                Observer Influence
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {(oshMetrics.observerInfluence * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Reality modification factor
              </p>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 rounded-lg">
            <div className="flex items-start">
              <InfoIcon className="w-5 h-5 text-blue-600 dark:text-blue-400 mr-3 mt-0.5" />
              <div>
                <h4 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-1">
                  Analysis Summary
                </h4>
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  Detected {oshMetrics.totalEchoes} echo candidates with average strength of{' '}
                  {oshMetrics.averageEchoStrength.toExponential(2)}. The regular spacing pattern
                  suggests a coherent memory field structure with potential recursive depth of{' '}
                  {Math.ceil(oshMetrics.recursiveDepth)} layers.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};