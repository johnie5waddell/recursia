/**
 * Fast Fourier Transform Implementation
 * Cooley-Tukey FFT algorithm for quantum wavefunction evolution
 */

import { Complex } from './complex';

export class FFT {
  /**
   * 1D FFT using Cooley-Tukey algorithm
   */
  static fft1D(input: Complex[]): Complex[] {
    const N = input.length;
    
    if (N <= 1) {
      return [...input];
    }
    
    // Check if N is power of 2
    if ((N & (N - 1)) !== 0) {
      throw new Error('FFT size must be a power of 2');
    }
    
    // Bit-reversal permutation
    const output = new Array(N);
    for (let i = 0; i < N; i++) {
      output[i] = input[this.bitReverse(i, Math.log2(N))];
    }
    
    // Cooley-Tukey FFT
    for (let length = 2; length <= N; length *= 2) {
      const halfLength = length / 2;
      const step = 2 * Math.PI / length;
      
      for (let start = 0; start < N; start += length) {
        for (let i = 0; i < halfLength; i++) {
          const omega = Complex.fromPolar(1, -step * i);
          const u = output[start + i];
          const v = omega.multiply(output[start + i + halfLength]);
          
          output[start + i] = u.add(v);
          output[start + i + halfLength] = u.subtract(v);
        }
      }
    }
    
    return output;
  }
  
  /**
   * 1D Inverse FFT
   */
  static ifft1D(input: Complex[]): Complex[] {
    const N = input.length;
    
    // Conjugate input
    const conjugated = input.map(c => c.conjugate());
    
    // Forward FFT
    const result = this.fft1D(conjugated);
    
    // Conjugate and scale
    return result.map(c => c.conjugate().scale(1 / N));
  }
  
  /**
   * 2D FFT
   */
  static fft2D(input: Complex[][]): Complex[][] {
    const rows = input.length;
    const cols = input[0].length;
    
    // FFT along rows
    let result = input.map(row => this.fft1D(row));
    
    // Transpose
    const transposed: Complex[][] = Array(cols).fill(null).map(() => Array(rows));
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        transposed[j][i] = result[i][j];
      }
    }
    
    // FFT along columns (now rows after transpose)
    result = transposed.map(row => this.fft1D(row));
    
    // Transpose back
    const final: Complex[][] = Array(rows).fill(null).map(() => Array(cols));
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        final[j][i] = result[i][j];
      }
    }
    
    return final;
  }
  
  /**
   * 2D Inverse FFT
   */
  static ifft2D(input: Complex[][]): Complex[][] {
    const rows = input.length;
    const cols = input[0].length;
    
    // Conjugate input
    const conjugated = input.map(row => row.map(c => c.conjugate()));
    
    // Forward 2D FFT
    const result = this.fft2D(conjugated);
    
    // Conjugate and scale
    return result.map(row => 
      row.map(c => c.conjugate().scale(1 / (rows * cols)))
    );
  }
  
  /**
   * 3D FFT
   */
  static fft3D(input: Complex[][][]): Complex[][][] {
    const sizeX = input.length;
    const sizeY = input[0].length;
    const sizeZ = input[0][0].length;
    
    // FFT along X dimension
    let result = input.map(plane => plane.map(row => this.fft1D(row)));
    
    // FFT along Y dimension
    result = result.map(plane => this.fft2D(plane));
    
    // FFT along Z dimension (need to reorganize data)
    const finalResult: Complex[][][] = Array(sizeX).fill(null).map(() =>
      Array(sizeY).fill(null).map(() => Array(sizeZ))
    );
    
    for (let i = 0; i < sizeX; i++) {
      for (let j = 0; j < sizeY; j++) {
        // Extract Z-direction slice
        const zSlice: Complex[] = [];
        for (let k = 0; k < sizeZ; k++) {
          zSlice.push(result[i][j][k]);
        }
        
        // Transform and store back
        const transformedZ = this.fft1D(zSlice);
        for (let k = 0; k < sizeZ; k++) {
          finalResult[i][j][k] = transformedZ[k];
        }
      }
    }
    
    return finalResult;
  }
  
  /**
   * 3D Inverse FFT
   */
  static ifft3D(input: Complex[][][]): Complex[][][] {
    const sizeX = input.length;
    const sizeY = input[0].length;
    const sizeZ = input[0][0].length;
    
    // Conjugate input
    const conjugated = input.map(plane => 
      plane.map(row => 
        row.map(c => c.conjugate())
      )
    );
    
    // Forward 3D FFT
    const result = this.fft3D(conjugated);
    
    // Conjugate and scale
    return result.map(plane =>
      plane.map(row =>
        row.map(c => c.conjugate().scale(1 / (sizeX * sizeY * sizeZ)))
      )
    );
  }
  
  /**
   * Bit reversal for FFT
   */
  private static bitReverse(num: number, bits: number): number {
    let result = 0;
    for (let i = 0; i < bits; i++) {
      result = (result << 1) | (num & 1);
      num >>= 1;
    }
    return result;
  }
  
  /**
   * Zero-pad to next power of 2
   */
  static padToPowerOfTwo(input: Complex[]): Complex[] {
    const n = input.length;
    const nextPower = 1 << Math.ceil(Math.log2(n));
    
    if (n === nextPower) {
      return [...input];
    }
    
    const padded = new Array(nextPower);
    for (let i = 0; i < n; i++) {
      padded[i] = input[i];
    }
    for (let i = n; i < nextPower; i++) {
      padded[i] = Complex.zero();
    }
    
    return padded;
  }
  
  /**
   * Frequency domain filtering
   */
  static lowPassFilter(input: Complex[], cutoffFreq: number): Complex[] {
    const N = input.length;
    const fftResult = this.fft1D(input);
    
    // Apply filter in frequency domain
    const filtered = fftResult.map((val, i) => {
      const freq = i < N/2 ? i : i - N;
      const normalizedFreq = Math.abs(freq) / (N/2);
      
      if (normalizedFreq <= cutoffFreq) {
        return val;
      } else {
        return Complex.zero();
      }
    });
    
    return this.ifft1D(filtered);
  }
  
  /**
   * Convolution using FFT
   */
  static convolve(signal: Complex[], kernel: Complex[]): Complex[] {
    const N = Math.max(signal.length, kernel.length);
    const nextPower = 1 << Math.ceil(Math.log2(2 * N - 1));
    
    // Pad both to same size
    const paddedSignal = this.padToPowerOfTwo(signal);
    while (paddedSignal.length < nextPower) {
      paddedSignal.push(Complex.zero());
    }
    
    const paddedKernel = this.padToPowerOfTwo(kernel);
    while (paddedKernel.length < nextPower) {
      paddedKernel.push(Complex.zero());
    }
    
    // Transform to frequency domain
    const signalFFT = this.fft1D(paddedSignal);
    const kernelFFT = this.fft1D(paddedKernel);
    
    // Multiply in frequency domain
    const productFFT = signalFFT.map((val, i) => val.multiply(kernelFFT[i]));
    
    // Transform back
    const result = this.ifft1D(productFFT);
    
    // Return only valid portion
    return result.slice(0, signal.length + kernel.length - 1);
  }
  
  /**
   * FFT for real-valued signals
   * Converts real array to complex, performs FFT, and returns complex result
   */
  static fft(realSignal: number[]): Complex[] {
    // Convert real signal to complex
    const complexSignal = realSignal.map(val => new Complex(val, 0));
    
    // Pad to power of 2 if needed
    const paddedSignal = this.padToPowerOfTwo(complexSignal);
    
    // Perform FFT
    return this.fft1D(paddedSignal);
  }
  
  /**
   * Inverse FFT for real-valued signals
   * Takes complex FFT result and returns real part of inverse transform
   */
  static ifft(complexSignal: Complex[]): number[] {
    const result = this.ifft1D(complexSignal);
    return result.map(c => c.re);
  }
}