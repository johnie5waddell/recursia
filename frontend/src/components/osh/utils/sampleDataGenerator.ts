// Sample gravitational wave data generator for testing
import { GWData } from '../types/gwEchoTypes';

export function generateSampleGWData(
  duration: number = 1.0,  // seconds
  sampleRate: number = 4096,  // Hz
  includeEchoes: boolean = true
): GWData {
  const numSamples = Math.floor(duration * sampleRate);
  const time = Array(numSamples).fill(0).map((_, i) => i / sampleRate);
  const strain = Array(numSamples).fill(0);
  
  // Generate base noise
  for (let i = 0; i < numSamples; i++) {
    strain[i] = (Math.random() - 0.5) * 1e-21;
  }
  
  // Add a simulated GW signal (chirp)
  const t0 = 0.3;  // Start time
  const f0 = 35;   // Initial frequency
  const chirpMass = 30;  // Solar masses
  const amplitude = 1e-20;
  
  for (let i = 0; i < numSamples; i++) {
    const t = time[i];
    if (t >= t0 && t < t0 + 0.2) {
      const tau = t - t0;
      const freq = f0 * Math.pow(1 - tau / 0.2, -3/8);
      const phase = 2 * Math.PI * freq * tau;
      strain[i] += amplitude * Math.sin(phase) * Math.exp(-tau / 0.1);
    }
  }
  
  // Add echoes if requested
  if (includeEchoes) {
    const echoDelays = [0.05, 0.11, 0.23];  // seconds
    const echoAmplitudes = [0.3, 0.15, 0.08];  // relative to primary
    
    for (let e = 0; e < echoDelays.length; e++) {
      const delay = echoDelays[e];
      const relAmplitude = echoAmplitudes[e];
      
      for (let i = 0; i < numSamples; i++) {
        const t = time[i];
        const echoTime = t0 + delay;
        if (t >= echoTime && t < echoTime + 0.2) {
          const tau = t - echoTime;
          const freq = f0 * Math.pow(1 - tau / 0.2, -3/8);
          const phase = 2 * Math.PI * freq * tau;
          strain[i] += amplitude * relAmplitude * Math.sin(phase) * Math.exp(-tau / 0.1);
        }
      }
    }
  }
  
  return {
    time,
    strain,
    sampleRate,
    metadata: {
      detector: 'Simulated',
      gpsTime: Date.now() / 1000,
      source: 'Test Generator',
      description: includeEchoes ? 'GW signal with echoes' : 'GW signal without echoes',
    }
  };
}

export function exportToCSV(data: GWData): string {
  let csv = 'time,strain\n';
  for (let i = 0; i < data.time.length; i++) {
    csv += `${data.time[i]},${data.strain[i]}\n`;
  }
  return csv;
}

export function exportToJSON(data: GWData): string {
  return JSON.stringify(data, null, 2);
}