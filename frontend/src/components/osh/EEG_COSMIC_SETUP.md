# EEG-Cosmic Resonance Analyzer Setup

## Required Dependencies

The EEG-Cosmic Resonance Analyzer component requires the following Chart.js plugins to be installed:

```bash
npm install chartjs-plugin-annotation chartjs-plugin-zoom chartjs-adapter-date-fns
```

Or with yarn:

```bash
yarn add chartjs-plugin-annotation chartjs-plugin-zoom chartjs-adapter-date-fns
```

## Usage

### Basic Usage

```tsx
import { EEGCosmicResonanceAnalyzer } from './components/osh/EEGCosmicResonanceAnalyzer';
import { Toaster } from 'react-hot-toast';

function App() {
  return (
    <div>
      <Toaster />
      <EEGCosmicResonanceAnalyzer />
    </div>
  );
}
```

### File Formats

The component accepts the following file formats:

#### EEG Data
- **CSV**: First column should be time, remaining columns are EEG channels
- **JSON**: Format shown below
- **EDF**: European Data Format (parsing not yet implemented, convert to CSV/JSON)

#### Cosmic Data
- **CSV**: Two columns - time and value
- **JSON**: Format shown below

### Example JSON Formats

#### EEG Data JSON
```json
{
  "channels": ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"],
  "samplingRate": 256,
  "data": [
    [/* channel 1 samples */],
    [/* channel 2 samples */],
    // ... more channels
  ],
  "metadata": {
    "subject": "Subject001",
    "recordingDate": "2024-01-15",
    "device": "BrainAmp DC",
    "notes": "Eyes closed resting state"
  }
}
```

#### Cosmic Data JSON
```json
{
  "type": "schumann",
  "name": "Schumann Resonance Recording",
  "samplingRate": 256,
  "data": [/* samples */],
  "metadata": {
    "source": "Global Coherence Monitoring System",
    "unit": "pT",
    "description": "Earth's electromagnetic field oscillations"
  }
}
```

### Generating Sample Data

You can generate sample data for testing:

```tsx
import { createSampleFiles } from './components/osh/utils/eegCosmicSampleData';

// This will download sample EEG and cosmic data files
createSampleFiles();
```

### OSH Metrics Explained

The component calculates the following OSH (Organic Simulation Hypothesis) metrics:

1. **Consciousness-Cosmos Coupling**: Average coherence between EEG and cosmic signals
2. **Information Transfer Rate**: Bits per second of information exchange
3. **Quantum Coherence**: Combined measure of coherence and phase synchronization
4. **Memory Field Resonance**: Frequency of resonance events per second
5. **Observer Influence**: Correlation strength weighted by significance
6. **Recursive Depth**: Logarithmic measure of resonance event complexity
7. **Entropy Flow**: Normalized entropy difference between signals
8. **Phase Entanglement**: Stability of phase relationships

### Analysis Features

- **Cross-correlation**: Finds time-lagged relationships between signals
- **Coherence Analysis**: Frequency-domain correlation measure
- **Band-specific Analysis**: Separate analysis for delta, theta, alpha, beta, and gamma bands
- **Resonance Event Detection**: Identifies periods of strong coupling
- **Statistical Significance Testing**: Validates correlation strength
- **Real-time Visualization**: Interactive charts with zoom and pan
- **Data Export**: Export results in JSON format

### Performance Considerations

- The component uses WebWorkers for heavy computations (when available)
- FFT operations are optimized for power-of-2 sample sizes
- Large files may take several seconds to process
- Recommended maximum file size: 100MB

### Scientific Background

This component is based on research exploring potential correlations between:
- Human EEG activity and cosmic background signals
- Brain rhythms and Schumann resonances
- Consciousness states and environmental electromagnetic fields

The analysis implements the OSH framework's prediction of consciousness-cosmos coupling through information-theoretic measures.