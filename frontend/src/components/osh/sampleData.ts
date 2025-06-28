// Sample CMB data generator for testing

export interface CMBDataPoint {
  l: number;
  temperature: number;
  error?: number;
}

// Generate realistic CMB power spectrum data
export function generateSampleCMBData(maxL: number = 2500): CMBDataPoint[] {
  const data: CMBDataPoint[] = [];
  
  // CMB power spectrum approximation (simplified)
  for (let l = 2; l <= maxL; l++) {
    // Acoustic peaks approximation
    const acousticPeak1 = Math.exp(-Math.pow((l - 220) / 50, 2)) * 5800;
    const acousticPeak2 = Math.exp(-Math.pow((l - 540) / 80, 2)) * 2500;
    const acousticPeak3 = Math.exp(-Math.pow((l - 810) / 100, 2)) * 1200;
    
    // Damping tail
    const dampingFactor = Math.exp(-l / 1500);
    
    // Background spectrum
    const background = 1000 * dampingFactor / (l * (l + 1));
    
    // Combine components
    const cl = background + acousticPeak1 + acousticPeak2 + acousticPeak3;
    
    // Add some noise
    const noise = (Math.random() - 0.5) * 0.1 * cl;
    
    // Convert to temperature fluctuation
    const temperature = Math.sqrt(cl + noise) * (1 + (Math.random() - 0.5) * 0.05);
    
    // Error estimation (simplified)
    const error = temperature * 0.05 * (1 + l / 1000);
    
    data.push({
      l,
      temperature,
      error,
    });
  }
  
  return data;
}

// Generate sample CSV content
export function generateSampleCSV(): string {
  const data = generateSampleCMBData(1000);
  const header = 'l,temperature,error\n';
  const rows = data.map(d => `${d.l},${d.temperature},${d.error}`).join('\n');
  return header + rows;
}

// Generate sample JSON content
export function generateSampleJSON(): string {
  const data = generateSampleCMBData(1000);
  return JSON.stringify({
    source: 'Sample CMB Data',
    metadata: {
      generated: new Date().toISOString(),
      maxL: 1000,
      units: 'microKelvin',
    },
    data: data,
  }, null, 2);
}

// Pre-defined test datasets
export const testDatasets = {
  planckLike: {
    name: 'Planck-like Sample',
    description: 'Simulated data resembling Planck satellite observations',
    generator: () => generateSampleCMBData(2500),
  },
  wmap: {
    name: 'WMAP-like Sample',
    description: 'Lower resolution data similar to WMAP',
    generator: () => generateSampleCMBData(800),
  },
  highNoise: {
    name: 'High Noise Sample',
    description: 'Data with enhanced noise for testing robustness',
    generator: () => {
      const data = generateSampleCMBData(1500);
      return data.map(d => ({
        ...d,
        temperature: d.temperature * (1 + (Math.random() - 0.5) * 0.3),
        error: d.error ? d.error * 2 : undefined,
      }));
    },
  },
  recursivePattern: {
    name: 'Recursive Pattern Sample',
    description: 'Data with embedded recursive patterns for OSH testing',
    generator: () => {
      const data = generateSampleCMBData(1000);
      // Embed recursive patterns
      const pattern = [1, 1, 0, 1, 0, 0, 1, 1];
      return data.map((d, i) => ({
        ...d,
        temperature: d.temperature * (1 + 0.1 * pattern[i % pattern.length]),
      }));
    },
  },
};