/**
 * Sample data generators for EEG-Cosmic Resonance Analysis
 * Creates realistic test data for development and demonstration
 */

// Generate synthetic EEG data with realistic characteristics
export function generateSampleEEGData(
  duration: number = 60, // seconds
  samplingRate: number = 256, // Hz
  channels: string[] = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
): { channels: string[]; samplingRate: number; data: number[][] } {
  const numSamples = Math.floor(duration * samplingRate);
  const data: number[][] = [];
  
  channels.forEach((channel, channelIdx) => {
    const channelData: number[] = [];
    
    // Base EEG characteristics
    const baseAmplitude = 20 + Math.random() * 10; // μV
    
    // Generate realistic EEG signal with multiple frequency components
    for (let i = 0; i < numSamples; i++) {
      const t = i / samplingRate;
      let sample = 0;
      
      // Delta waves (0.5-4 Hz)
      sample += (Math.random() * 0.3 + 0.2) * baseAmplitude * 
                Math.sin(2 * Math.PI * (0.5 + Math.random() * 3.5) * t);
      
      // Theta waves (4-8 Hz)
      sample += (Math.random() * 0.4 + 0.3) * baseAmplitude * 
                Math.sin(2 * Math.PI * (4 + Math.random() * 4) * t);
      
      // Alpha waves (8-13 Hz) - stronger in posterior channels
      const alphaStrength = channel.startsWith('O') || channel.startsWith('P') ? 0.8 : 0.4;
      sample += alphaStrength * baseAmplitude * 
                Math.sin(2 * Math.PI * (8 + Math.random() * 5) * t);
      
      // Beta waves (13-30 Hz)
      sample += (Math.random() * 0.3 + 0.1) * baseAmplitude * 
                Math.sin(2 * Math.PI * (13 + Math.random() * 17) * t);
      
      // Gamma waves (30-100 Hz)
      sample += (Math.random() * 0.1 + 0.05) * baseAmplitude * 
                Math.sin(2 * Math.PI * (30 + Math.random() * 70) * t);
      
      // Add noise
      sample += (Math.random() - 0.5) * baseAmplitude * 0.2;
      
      // Add occasional artifacts (eye blinks, muscle activity)
      if (Math.random() < 0.001) {
        sample += (Math.random() - 0.5) * baseAmplitude * 5;
      }
      
      channelData.push(sample);
    }
    
    data.push(channelData);
  });
  
  return { channels, samplingRate, data };
}

// Generate synthetic cosmic background data
export function generateSampleCosmicData(
  type: 'cmb' | 'solar_wind' | 'schumann',
  duration: number = 60, // seconds
  samplingRate: number = 256 // Hz
): { type: string; name: string; samplingRate: number; data: number[] } {
  const numSamples = Math.floor(duration * samplingRate);
  const data: number[] = [];
  
  switch (type) {
    case 'cmb':
      // CMB temperature fluctuations (microkelvin variations)
      for (let i = 0; i < numSamples; i++) {
        const t = i / samplingRate;
        let sample = 2.725; // Base CMB temperature in Kelvin
        
        // Large-scale fluctuations
        sample += 0.00001 * Math.sin(2 * Math.PI * 0.001 * t);
        sample += 0.000005 * Math.sin(2 * Math.PI * 0.005 * t);
        
        // Small-scale fluctuations
        sample += (Math.random() - 0.5) * 0.000002;
        
        data.push(sample);
      }
      break;
      
    case 'solar_wind':
      // Solar wind parameters (e.g., magnetic field strength in nT)
      for (let i = 0; i < numSamples; i++) {
        const t = i / samplingRate;
        let sample = 5; // Base field strength
        
        // Solar wind variations
        sample += 2 * Math.sin(2 * Math.PI * 0.0001 * t); // Very slow variations
        sample += 0.5 * Math.sin(2 * Math.PI * 0.001 * t);
        sample += 0.2 * Math.sin(2 * Math.PI * 0.01 * t);
        
        // Turbulence
        sample += (Math.random() - 0.5) * 0.5;
        
        data.push(sample);
      }
      break;
      
    case 'schumann':
      // Schumann resonances (Earth's electromagnetic field)
      for (let i = 0; i < numSamples; i++) {
        const t = i / samplingRate;
        let sample = 0;
        
        // Primary Schumann frequencies
        sample += 1.0 * Math.sin(2 * Math.PI * 7.83 * t);   // First mode
        sample += 0.5 * Math.sin(2 * Math.PI * 14.3 * t);   // Second mode
        sample += 0.3 * Math.sin(2 * Math.PI * 20.8 * t);   // Third mode
        sample += 0.2 * Math.sin(2 * Math.PI * 27.3 * t);   // Fourth mode
        sample += 0.1 * Math.sin(2 * Math.PI * 33.8 * t);   // Fifth mode
        
        // Daily variations
        sample *= (1 + 0.2 * Math.sin(2 * Math.PI * t / 86400));
        
        // Noise
        sample += (Math.random() - 0.5) * 0.1;
        
        data.push(sample);
      }
      break;
  }
  
  return {
    type,
    name: `${type.toUpperCase()} Data Sample`,
    samplingRate,
    data
  };
}

// Generate correlated EEG and cosmic data for testing
export function generateCorrelatedData(
  duration: number = 60,
  samplingRate: number = 256,
  correlationStrength: number = 0.3
): {
  eegData: ReturnType<typeof generateSampleEEGData>;
  cosmicData: ReturnType<typeof generateSampleCosmicData>;
} {
  const eegData = generateSampleEEGData(duration, samplingRate);
  const cosmicData = generateSampleCosmicData('schumann', duration, samplingRate);
  
  // Add correlation between alpha waves and Schumann resonance
  const alphaFreq = 10; // Hz
  const schumannFreq = 7.83; // Hz
  
  for (let channelIdx = 0; channelIdx < eegData.channels.length; channelIdx++) {
    const channel = eegData.channels[channelIdx];
    
    // Stronger correlation for posterior channels
    const channelCorrelation = (channel.startsWith('O') || channel.startsWith('P')) 
      ? correlationStrength * 1.5 
      : correlationStrength;
    
    for (let i = 0; i < eegData.data[channelIdx].length; i++) {
      const t = i / samplingRate;
      
      // Create phase coupling between alpha and Schumann
      const schumannPhase = 2 * Math.PI * schumannFreq * t;
      const alphaModulation = Math.sin(schumannPhase) * channelCorrelation;
      
      // Modulate alpha amplitude based on Schumann phase
      eegData.data[channelIdx][i] += 
        alphaModulation * 10 * Math.sin(2 * Math.PI * alphaFreq * t);
    }
  }
  
  return { eegData, cosmicData };
}

// Export data to JSON format
export function exportToJSON(data: any, filename: string): void {
  const jsonStr = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Export data to CSV format
export function exportEEGToCSV(
  eegData: { channels: string[]; samplingRate: number; data: number[][] },
  filename: string
): void {
  const headers = ['Time', ...eegData.channels].join(',');
  const rows: string[] = [headers];
  
  const numSamples = eegData.data[0].length;
  for (let i = 0; i < numSamples; i++) {
    const time = i / eegData.samplingRate;
    const values = [time.toFixed(4)];
    
    for (let ch = 0; ch < eegData.channels.length; ch++) {
      values.push(eegData.data[ch][i].toFixed(6));
    }
    
    rows.push(values.join(','));
  }
  
  const csvContent = rows.join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Create sample files for download
export function createSampleFiles(): void {
  // Generate 10 seconds of correlated data
  const { eegData, cosmicData } = generateCorrelatedData(10, 256, 0.5);
  
  // Export EEG data
  exportToJSON(eegData, 'sample_eeg_data.json');
  exportEEGToCSV(eegData, 'sample_eeg_data.csv');
  
  // Export cosmic data
  exportToJSON(cosmicData, 'sample_cosmic_data.json');
  
  // Export cosmic data as CSV
  const cosmicCSV = [
    'Time,Value',
    ...cosmicData.data.map((value, i) => 
      `${(i / cosmicData.samplingRate).toFixed(4)},${value.toFixed(6)}`
    )
  ].join('\n');
  
  const blob = new Blob([cosmicCSV], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'sample_cosmic_data.csv';
  a.click();
  URL.revokeObjectURL(url);
}