import React, { useEffect, useRef, useMemo, useState } from 'react';
import { GWData, EchoCandidate } from './types/gwEchoTypes';

interface SpectrogramVisualizerProps {
  data: GWData;
  echoCandidates?: EchoCandidate[];
  timeRange?: { start: number; end: number };
  height?: number;
  colormap?: 'viridis' | 'plasma' | 'inferno' | 'magma' | 'jet';
  dynamicRange?: number; // dB
}

export const SpectrogramVisualizer: React.FC<SpectrogramVisualizerProps> = ({
  data,
  echoCandidates = [],
  timeRange,
  height = 300,
  colormap = 'viridis',
  dynamicRange = 60,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frequencyRange, setFrequencyRange] = useState({ min: 0, max: data.sampleRate / 2 });
  const [selectedColormap, setSelectedColormap] = useState(colormap);
  
  // Compute spectrogram data
  const spectrogramData = useMemo(() => {
    const windowSize = 256; // FFT window size
    const overlap = 0.75; // 75% overlap
    const stepSize = Math.floor(windowSize * (1 - overlap));
    const nFreqs = windowSize / 2 + 1;
    
    // Apply time range filter
    const startIdx = timeRange ? 
      Math.floor(timeRange.start * data.sampleRate) : 0;
    const endIdx = timeRange ? 
      Math.floor(timeRange.end * data.sampleRate) : data.strain.length;
    
    const filteredStrain = data.strain.slice(startIdx, endIdx);
    const nWindows = Math.floor((filteredStrain.length - windowSize) / stepSize) + 1;
    
    // Initialize spectrogram matrix
    const spectrogram: number[][] = Array(nFreqs).fill(null).map(() => Array(nWindows).fill(0));
    
    // Hanning window
    const window = Array(windowSize).fill(0).map((_, i) => 
      0.5 - 0.5 * Math.cos(2 * Math.PI * i / (windowSize - 1))
    );
    
    // Compute STFT
    for (let i = 0; i < nWindows; i++) {
      const start = i * stepSize;
      const segment = filteredStrain.slice(start, start + windowSize);
      
      // Apply window
      const windowed = segment.map((val, j) => val * window[j]);
      
      // FFT (simplified - in production use proper FFT library)
      const fft = computeFFT(windowed);
      
      // Store magnitude spectrum
      for (let j = 0; j < nFreqs; j++) {
        spectrogram[j][i] = 20 * Math.log10(Math.abs(fft[j]) + 1e-10); // Convert to dB
      }
    }
    
    return {
      data: spectrogram,
      timeAxis: Array(nWindows).fill(0).map((_, i) => 
        (startIdx + i * stepSize) / data.sampleRate
      ),
      freqAxis: Array(nFreqs).fill(0).map((_, i) => 
        i * data.sampleRate / windowSize
      ),
    };
  }, [data, timeRange]);
  
  // Render spectrogram
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const { data: spectData, timeAxis, freqAxis } = spectrogramData;
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Find min/max for normalization
    let minVal = Infinity;
    let maxVal = -Infinity;
    spectData.forEach(row => {
      row.forEach(val => {
        if (val > maxVal) maxVal = val;
        if (val < minVal) minVal = val;
      });
    });
    
    // Apply dynamic range
    minVal = maxVal - dynamicRange;
    
    // Draw spectrogram
    const pixelWidth = width / timeAxis.length;
    const pixelHeight = height / freqAxis.length;
    
    for (let i = 0; i < spectData.length; i++) {
      const freq = freqAxis[i];
      if (freq < frequencyRange.min || freq > frequencyRange.max) continue;
      
      const y = height - (i / spectData.length) * height;
      
      for (let j = 0; j < spectData[i].length; j++) {
        const x = j * pixelWidth;
        const value = spectData[i][j];
        const normalized = Math.max(0, Math.min(1, (value - minVal) / (maxVal - minVal)));
        
        ctx.fillStyle = getColor(normalized, selectedColormap);
        ctx.fillRect(x, y - pixelHeight, pixelWidth + 1, pixelHeight + 1);
      }
    }
    
    // Draw echo markers
    if (echoCandidates.length > 0) {
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      
      echoCandidates.forEach(echo => {
        const x = ((echo.time - timeAxis[0]) / (timeAxis[timeAxis.length - 1] - timeAxis[0])) * width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      });
      
      ctx.setLineDash([]);
    }
    
    // Draw axes
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height);
    ctx.lineTo(width, height);
    ctx.lineTo(width, 0);
    ctx.stroke();
    
  }, [spectrogramData, echoCandidates, frequencyRange, selectedColormap, dynamicRange]);
  
  return (
    <div className="space-y-4">
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={800}
          height={height}
          className="w-full border border-gray-300 dark:border-gray-600 rounded"
          style={{ height: `${height}px` }}
        />
        
        {/* Y-axis label */}
        <div className="absolute left-0 top-1/2 -translate-y-1/2 -rotate-90 text-sm text-gray-600 dark:text-gray-400">
          Frequency (Hz)
        </div>
        
        {/* X-axis label */}
        <div className="text-center text-sm text-gray-600 dark:text-gray-400 mt-2">
          Time (s)
        </div>
      </div>
      
      <div className="flex flex-wrap gap-4">
        <div className="flex-1 min-w-[200px]">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Frequency Range (Hz)
          </label>
          <div className="flex items-center space-x-2">
            <input
              type="number"
              value={frequencyRange.min}
              onChange={(e) => setFrequencyRange(prev => ({ 
                ...prev, 
                min: parseFloat(e.target.value) || 0 
              }))}
              className="w-24 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              min="0"
              max={data.sampleRate / 2}
            />
            <span className="text-gray-500 dark:text-gray-400">to</span>
            <input
              type="number"
              value={frequencyRange.max}
              onChange={(e) => setFrequencyRange(prev => ({ 
                ...prev, 
                max: parseFloat(e.target.value) || data.sampleRate / 2 
              }))}
              className="w-24 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              min="0"
              max={data.sampleRate / 2}
            />
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Colormap
          </label>
          <select
            value={selectedColormap}
            onChange={(e) => setSelectedColormap(e.target.value as any)}
            className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            <option value="viridis">Viridis</option>
            <option value="plasma">Plasma</option>
            <option value="inferno">Inferno</option>
            <option value="magma">Magma</option>
            <option value="jet">Jet</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Dynamic Range: {dynamicRange} dB
          </label>
          <input
            type="range"
            value={dynamicRange}
            onChange={(e) => {/* Add handler if you want to make this adjustable */}}
            min="20"
            max="100"
            step="5"
            className="w-32"
            disabled
          />
        </div>
      </div>
      
      {echoCandidates.length > 0 && (
        <div className="text-sm text-gray-600 dark:text-gray-400">
          <span className="inline-flex items-center">
            <span className="w-8 h-0.5 bg-white border-2 border-white mr-2" style={{ borderStyle: 'dashed' }}></span>
            Echo candidates marked with dashed lines
          </span>
        </div>
      )}
    </div>
  );
};

// Simplified FFT implementation (replace with proper library in production)
function computeFFT(data: number[]): number[] {
  const N = data.length;
  const result: number[] = Array(N).fill(0);
  
  for (let k = 0; k < N; k++) {
    let real = 0;
    let imag = 0;
    for (let n = 0; n < N; n++) {
      const angle = -2 * Math.PI * k * n / N;
      real += data[n] * Math.cos(angle);
      imag += data[n] * Math.sin(angle);
    }
    result[k] = Math.sqrt(real * real + imag * imag);
  }
  
  return result;
}

// Color mapping functions
function getColor(value: number, colormap: string): string {
  const v = Math.max(0, Math.min(1, value));
  
  switch (colormap) {
    case 'viridis':
      return viridisColormap(v);
    case 'plasma':
      return plasmaColormap(v);
    case 'inferno':
      return infernoColormap(v);
    case 'magma':
      return magmaColormap(v);
    case 'jet':
      return jetColormap(v);
    default:
      return viridisColormap(v);
  }
}

function viridisColormap(t: number): string {
  const r = Math.floor(255 * (0.267 + 0.004 * t + 0.329 * t * t - 0.349 * t * t * t + 0.784 * t * t * t * t));
  const g = Math.floor(255 * (0.001 + 1.388 * t - 0.397 * t * t - 0.653 * t * t * t + 0.644 * t * t * t * t));
  const b = Math.floor(255 * (0.329 + 0.479 * t - 2.281 * t * t + 4.613 * t * t * t - 2.783 * t * t * t * t));
  return `rgb(${r},${g},${b})`;
}

function plasmaColormap(t: number): string {
  const r = Math.floor(255 * (0.050 + 2.731 * t - 5.274 * t * t + 5.927 * t * t * t - 2.440 * t * t * t * t));
  const g = Math.floor(255 * (0.029 + 0.338 * t - 0.861 * t * t + 1.755 * t * t * t - 0.910 * t * t * t * t));
  const b = Math.floor(255 * (0.528 + 1.786 * t - 5.866 * t * t + 7.827 * t * t * t - 3.298 * t * t * t * t));
  return `rgb(${r},${g},${b})`;
}

function infernoColormap(t: number): string {
  const r = Math.floor(255 * (0.002 + 1.899 * t - 0.522 * t * t + 0.690 * t * t * t - 0.071 * t * t * t * t));
  const g = Math.floor(255 * (0.001 + 0.289 * t + 1.358 * t * t - 1.892 * t * t * t + 0.946 * t * t * t * t));
  const b = Math.floor(255 * (0.015 + 1.652 * t - 3.421 * t * t + 5.066 * t * t * t - 2.316 * t * t * t * t));
  return `rgb(${r},${g},${b})`;
}

function magmaColormap(t: number): string {
  const r = Math.floor(255 * (0.002 + 1.463 * t + 0.422 * t * t - 0.534 * t * t * t + 0.649 * t * t * t * t));
  const g = Math.floor(255 * (0.001 + 0.280 * t + 0.847 * t * t - 0.317 * t * t * t + 0.189 * t * t * t * t));
  const b = Math.floor(255 * (0.014 + 2.158 * t - 5.474 * t * t + 7.877 * t * t * t - 3.608 * t * t * t * t));
  return `rgb(${r},${g},${b})`;
}

function jetColormap(t: number): string {
  let r = 0, g = 0, b = 0;
  
  if (t < 0.125) {
    r = 0;
    g = 0;
    b = 0.5 + 4 * t;
  } else if (t < 0.375) {
    r = 0;
    g = 4 * (t - 0.125);
    b = 1;
  } else if (t < 0.625) {
    r = 4 * (t - 0.375);
    g = 1;
    b = 1 - 4 * (t - 0.375);
  } else if (t < 0.875) {
    r = 1;
    g = 1 - 4 * (t - 0.625);
    b = 0;
  } else {
    r = 1 - 4 * (t - 0.875);
    g = 0;
    b = 0;
  }
  
  return `rgb(${Math.floor(255 * r)},${Math.floor(255 * g)},${Math.floor(255 * b)})`;
}