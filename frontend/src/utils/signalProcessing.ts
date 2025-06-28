// Signal processing utilities for gravitational wave echo detection

export interface Complex {
  real: number;
  imag: number;
}

/**
 * Compute Fast Fourier Transform
 * @param signal Input signal
 * @returns Complex frequency spectrum
 */
export function performFFT(signal: number[]): Complex[] {
  const N = signal.length;
  
  // Pad to nearest power of 2
  const paddedLength = Math.pow(2, Math.ceil(Math.log2(N)));
  const padded = new Array(paddedLength).fill(0);
  signal.forEach((val, i) => { padded[i] = val; });
  
  return fftRecursive(padded);
}

function fftRecursive(signal: Complex[]): Complex[] {
  const N = signal.length;
  
  // Convert numbers to complex if needed
  if (typeof signal[0] === 'number') {
    signal = signal.map(x => ({ real: x as any, imag: 0 }));
  }
  
  if (N <= 1) return signal;
  
  // Divide
  const even = new Array(N / 2);
  const odd = new Array(N / 2);
  
  for (let i = 0; i < N / 2; i++) {
    even[i] = signal[2 * i];
    odd[i] = signal[2 * i + 1];
  }
  
  // Conquer
  const evenFFT = fftRecursive(even);
  const oddFFT = fftRecursive(odd);
  
  // Combine
  const result = new Array(N);
  for (let k = 0; k < N / 2; k++) {
    const angle = -2 * Math.PI * k / N;
    const twiddle: Complex = {
      real: Math.cos(angle),
      imag: Math.sin(angle)
    };
    
    const t = complexMultiply(twiddle, oddFFT[k]);
    result[k] = complexAdd(evenFFT[k], t);
    result[k + N / 2] = complexSubtract(evenFFT[k], t);
  }
  
  return result;
}

/**
 * Compute autocorrelation of a signal
 * @param signal Input signal
 * @param maxLag Maximum lag to compute
 * @returns Autocorrelation values
 */
export function calculateAutocorrelation(signal: number[], maxLag?: number): number[] {
  const N = signal.length;
  maxLag = maxLag || N - 1;
  const result = new Array(maxLag + 1).fill(0);
  
  // Compute mean
  const mean = signal.reduce((a, b) => a + b, 0) / N;
  
  // Compute variance
  const variance = signal.reduce((sum, x) => sum + (x - mean) ** 2, 0) / N;
  
  // Compute autocorrelation
  for (let lag = 0; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < N - lag; i++) {
      sum += (signal[i] - mean) * (signal[i + lag] - mean);
    }
    result[lag] = sum / ((N - lag) * variance);
  }
  
  return result;
}

/**
 * Apply matched filter to signal
 * @param signal Input signal
 * @param template Template to match
 * @returns Filtered signal
 */
export function matchedFilter(signal: number[], template: number[]): number[] {
  const signalFFT = performFFT(signal);
  const templateFFT = performFFT(template.reverse()); // Time-reverse template
  
  // Multiply in frequency domain
  const productFFT = signalFFT.map((s, i) => 
    complexMultiply(s, templateFFT[i] || { real: 0, imag: 0 })
  );
  
  // Inverse FFT
  const filtered = ifft(productFFT);
  
  // Return magnitude
  return filtered.map(c => Math.sqrt(c.real ** 2 + c.imag ** 2));
}

/**
 * Inverse Fast Fourier Transform
 * @param spectrum Frequency spectrum
 * @returns Time-domain signal
 */
export function ifft(spectrum: Complex[]): Complex[] {
  const N = spectrum.length;
  
  // Conjugate the spectrum
  const conjugated = spectrum.map(c => ({ real: c.real, imag: -c.imag }));
  
  // Forward FFT
  const result = fftRecursive(conjugated);
  
  // Conjugate and scale
  return result.map(c => ({
    real: c.real / N,
    imag: -c.imag / N
  }));
}

/**
 * Apply Hanning window to signal
 * @param signal Input signal
 * @returns Windowed signal
 */
export function hanningWindow(signal: number[]): number[] {
  const N = signal.length;
  return signal.map((val, i) => 
    val * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)))
  );
}

/**
 * Apply Hamming window to signal
 * @param signal Input signal
 * @returns Windowed signal
 */
export function hammingWindow(signal: number[]): number[] {
  const N = signal.length;
  return signal.map((val, i) => 
    val * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (N - 1)))
  );
}

/**
 * Compute power spectral density
 * @param signal Input signal
 * @param sampleRate Sample rate in Hz
 * @returns PSD and frequency array
 */
export function powerSpectralDensity(signal: number[], sampleRate: number): {
  psd: number[];
  frequencies: number[];
} {
  const windowed = hanningWindow(signal);
  const fft = performFFT(windowed);
  const N = signal.length;
  
  // Compute PSD (one-sided)
  const psd = new Array(Math.floor(N / 2) + 1);
  const frequencies = new Array(Math.floor(N / 2) + 1);
  
  for (let i = 0; i <= N / 2; i++) {
    const magnitude = Math.sqrt(fft[i].real ** 2 + fft[i].imag ** 2);
    psd[i] = (2 * magnitude ** 2) / (sampleRate * N);
    frequencies[i] = i * sampleRate / N;
  }
  
  return { psd, frequencies };
}

/**
 * Bandpass filter using FFT
 * @param signal Input signal
 * @param sampleRate Sample rate
 * @param lowFreq Low cutoff frequency
 * @param highFreq High cutoff frequency
 * @returns Filtered signal
 */
export function bandpassFilter(
  signal: number[], 
  sampleRate: number, 
  lowFreq: number, 
  highFreq: number
): number[] {
  const fft = performFFT(signal);
  const N = signal.length;
  const freqResolution = sampleRate / N;
  
  // Apply frequency mask
  const filtered = fft.map((val, i) => {
    const freq = i * freqResolution;
    if (freq >= lowFreq && freq <= highFreq) {
      return val;
    } else if (freq >= sampleRate - highFreq && freq <= sampleRate - lowFreq) {
      // Mirror for negative frequencies
      return val;
    } else {
      return { real: 0, imag: 0 };
    }
  });
  
  // Inverse FFT
  const result = ifft(filtered);
  return result.map(c => c.real);
}

/**
 * Estimate noise floor using median absolute deviation
 * @param signal Input signal
 * @returns Noise level estimate
 */
export function estimateNoiseFloor(signal: number[]): number {
  const sorted = [...signal].sort((a, b) => Math.abs(a) - Math.abs(b));
  const median = sorted[Math.floor(sorted.length / 2)];
  
  const deviations = signal.map(x => Math.abs(x - median));
  const mad = deviations.sort((a, b) => a - b)[Math.floor(deviations.length / 2)];
  
  // Convert MAD to standard deviation estimate
  return mad * 1.4826;
}

/**
 * Find peaks in signal above threshold
 * @param signal Input signal
 * @param threshold Detection threshold
 * @param minDistance Minimum distance between peaks (samples)
 * @returns Peak indices and values
 */
export function findPeaks(
  signal: number[], 
  threshold?: number, 
  minDistance: number = 1
): Array<{ index: number; value: number }> {
  threshold = threshold || estimateNoiseFloor(signal) * 3;
  const peaks: Array<{ index: number; value: number }> = [];
  
  for (let i = 1; i < signal.length - 1; i++) {
    if (signal[i] > threshold && 
        signal[i] > signal[i - 1] && 
        signal[i] > signal[i + 1]) {
      
      // Check minimum distance to previous peak
      if (peaks.length === 0 || i - peaks[peaks.length - 1].index >= minDistance) {
        peaks.push({ index: i, value: signal[i] });
      } else if (signal[i] > peaks[peaks.length - 1].value) {
        // Replace with higher peak
        peaks[peaks.length - 1] = { index: i, value: signal[i] };
      }
    }
  }
  
  return peaks;
}

/**
 * Whiten signal by normalizing spectrum
 * @param signal Input signal
 * @param sampleRate Sample rate
 * @returns Whitened signal
 */
export function whiten(signal: number[], sampleRate: number): number[] {
  const { psd } = powerSpectralDensity(signal, sampleRate);
  const fft = performFFT(signal);
  
  // Normalize by sqrt of PSD
  const whitened = fft.map((val, i) => {
    const normFactor = Math.sqrt(psd[Math.min(i, psd.length - 1)] + 1e-10);
    return {
      real: val.real / normFactor,
      imag: val.imag / normFactor
    };
  });
  
  const result = ifft(whitened);
  return result.map(c => c.real);
}

// Helper functions for complex arithmetic
function complexAdd(a: Complex, b: Complex): Complex {
  return { real: a.real + b.real, imag: a.imag + b.imag };
}

function complexSubtract(a: Complex, b: Complex): Complex {
  return { real: a.real - b.real, imag: a.imag - b.imag };
}

function complexMultiply(a: Complex, b: Complex): Complex {
  return {
    real: a.real * b.real - a.imag * b.imag,
    imag: a.real * b.imag + a.imag * b.real
  };
}

/**
 * Compute spectrogram using STFT
 * @param signal Input signal
 * @param windowSize Window size in samples
 * @param overlap Overlap fraction (0-1)
 * @param sampleRate Sample rate
 * @returns Spectrogram matrix and axes
 */
export function computeSpectrogram(
  signal: number[],
  windowSize: number = 256,
  overlap: number = 0.75,
  sampleRate: number
): {
  spectrogram: number[][];
  timeAxis: number[];
  freqAxis: number[];
} {
  const stepSize = Math.floor(windowSize * (1 - overlap));
  const nWindows = Math.floor((signal.length - windowSize) / stepSize) + 1;
  const nFreqs = Math.floor(windowSize / 2) + 1;
  
  const spectrogram: number[][] = Array(nFreqs).fill(null).map(() => Array(nWindows).fill(0));
  const timeAxis: number[] = [];
  const freqAxis = Array(nFreqs).fill(0).map((_, i) => i * sampleRate / windowSize);
  
  for (let i = 0; i < nWindows; i++) {
    const start = i * stepSize;
    const segment = signal.slice(start, start + windowSize);
    
    if (segment.length < windowSize) {
      // Pad with zeros if needed
      while (segment.length < windowSize) {
        segment.push(0);
      }
    }
    
    // Apply window
    const windowed = hanningWindow(segment);
    
    // Compute FFT
    const fft = performFFT(windowed);
    
    // Store magnitude spectrum
    for (let j = 0; j < nFreqs; j++) {
      const magnitude = Math.sqrt(fft[j].real ** 2 + fft[j].imag ** 2);
      spectrogram[j][i] = 20 * Math.log10(magnitude + 1e-10); // Convert to dB
    }
    
    timeAxis.push((start + windowSize / 2) / sampleRate);
  }
  
  return { spectrogram, timeAxis, freqAxis };
}