# Gravitational Wave Echo Search Component

A comprehensive React component for detecting and analyzing gravitational wave echoes in the context of the Organic Simulation Hypothesis (OSH). This component provides scientifically accurate algorithms for echo detection and calculates OSH-specific metrics from detected patterns.

## Features

### Data Import
- **Multiple formats**: CSV, JSON, HDF5 (HDF5 support coming soon)
- **Flexible parsing**: Automatically detects common column names (time/t/Time, strain/h/Strain)
- **Metadata preservation**: Maintains detector information and GPS time

### Echo Detection Algorithms
1. **Autocorrelation Method**: Searches for self-similar patterns at various time delays
2. **Matched Filter Method**: Uses template matching to find echo candidates
3. **Hybrid Method**: Combines both approaches for improved detection

### Visualization
- **Interactive Time Series**: Zoomable/pannable strain data plot with echo markers
- **Spectrogram Analysis**: Time-frequency representation with multiple colormaps
- **Real-time Updates**: Live visualization during echo search

### Search Parameters
- **Time Delay Range**: Configurable min/max echo delays (default: 10ms - 1s)
- **Confidence Threshold**: Adjustable detection threshold (0-100%)
- **Window Size**: Analysis window duration
- **Detection Method**: Choose between algorithms
- **Noise Reduction**: Optional preprocessing
- **Adaptive Threshold**: Dynamic threshold based on local noise

### OSH Metrics
- **Information Leakage**: Fraction of signal energy in echoes
- **Memory Field Coherence**: Regularity of echo spacing (0-1)
- **Recursive Depth**: Estimated simulation layers
- **Observer Influence**: Reality modification factor
- **Echo Statistics**: Total count, average strength

### Export Capabilities
- **JSON Export**: Complete results with metadata
- **Echo Event List**: Detailed parameters for each detection
- **OSH Analysis Summary**: Calculated metrics and implications

## Usage

```tsx
import { GravitationalWaveEchoSearch } from './components/osh';

function App() {
  const handleEchoDetection = (echoes) => {
    console.log('Detected echoes:', echoes);
  };

  const handleMetricsCalculated = (metrics) => {
    console.log('OSH metrics:', metrics);
  };

  return (
    <GravitationalWaveEchoSearch
      onEchoDetected={handleEchoDetection}
      onOSHMetricsCalculated={handleMetricsCalculated}
    />
  );
}
```

## Data Format

### CSV Format
```csv
time,strain
0.0,1.23e-21
0.000244140625,1.45e-21
...
```

### JSON Format
```json
{
  "time": [0.0, 0.000244140625, ...],
  "strain": [1.23e-21, 1.45e-21, ...],
  "sampleRate": 4096,
  "metadata": {
    "detector": "H1",
    "gpsTime": 1234567890
  }
}
```

## Algorithm Details

### Autocorrelation Method
1. Computes normalized autocorrelation at various lags
2. Identifies peaks above confidence threshold
3. Validates peaks using local SNR estimation
4. Assigns echo order based on delay magnitude

### Matched Filter Method
1. Extracts template from highest energy region
2. Correlates template with entire signal
3. Identifies matches above threshold
4. Calculates time delays from primary signal

### OSH Metrics Calculation
- **Information Leakage**: Σ(echo_amplitude²) / Σ(signal_amplitude²)
- **Memory Field Coherence**: 1 / (1 + √variance(echo_spacings))
- **Recursive Depth**: log₂(max_echo_order + 1)
- **Observer Influence**: average_confidence × memory_field_coherence

## Component Architecture

```
GravitationalWaveEchoSearch/
├── GravitationalWaveEchoSearch.tsx  # Main component
├── SpectrogramVisualizer.tsx        # Spectrogram sub-component
├── types/
│   ├── gwEchoTypes.ts              # TypeScript interfaces
│   └── chartjs-plugins.d.ts        # Plugin type declarations
└── utils/
    └── sampleDataGenerator.ts       # Test data generation
```

## Dependencies

- react-chartjs-2: Line chart visualization
- chart.js: Core charting library
- chartjs-plugin-zoom: Interactive zoom/pan
- chartjs-plugin-annotation: Echo markers
- lucide-react: Icons
- file-saver: Export functionality
- papaparse: CSV parsing (optional - use native parsing)

## Performance Considerations

- **Large Files**: Component handles files up to ~1M samples efficiently
- **Real-time Processing**: Echo search runs in main thread (consider Web Workers for large datasets)
- **Memory Usage**: Spectrogram computation is memory-intensive for long signals

## Future Enhancements

1. **HDF5 Support**: Integration with h5wasm library
2. **Web Worker Processing**: Offload heavy computations
3. **Advanced Algorithms**: Wavelet-based detection, machine learning models
4. **3D Visualization**: Echo patterns in parameter space
5. **Batch Processing**: Multiple file analysis
6. **Statistical Validation**: False alarm probability estimation

## Scientific Accuracy

The component implements scientifically sound signal processing techniques:
- Proper windowing functions (Hanning, Hamming)
- Correct FFT implementation for spectrograms
- Statistical noise estimation (MAD-based)
- SNR calculation in decibels
- Normalized correlation coefficients

All algorithms are designed to work with real gravitational wave data formats and sampling rates (typically 4096 Hz or 16384 Hz).