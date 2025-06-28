import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import { GravitationalWaveEchoVisualizer } from './GravitationalWaveEchoVisualizer';
import * as THREE from 'three';

// Mock Three.js
vi.mock('three', () => ({
  Vector3: vi.fn().mockImplementation((x, y, z) => ({ x, y, z })),
  Color: vi.fn().mockImplementation((color) => ({ color })),
  Clock: vi.fn().mockImplementation(() => ({
    getElapsedTime: vi.fn().mockReturnValue(1),
    getDelta: vi.fn().mockReturnValue(0.016)
  }))
}));

// Mock @react-three/fiber
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: any) => <div data-testid="canvas">{children}</div>,
  useFrame: vi.fn(),
  useThree: () => ({
    gl: { domElement: document.createElement('canvas') },
    scene: {},
    camera: {},
    size: { width: 800, height: 600 }
  })
}));

// Mock @react-three/drei
vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls" />,
  Box: () => <div data-testid="box" />,
  Sphere: () => <div data-testid="sphere" />,
  Text: ({ children }: any) => <div data-testid="text">{children}</div>,
  Line: () => <div data-testid="line" />,
  Billboard: ({ children }: any) => <div data-testid="billboard">{children}</div>
}));

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>
  },
  AnimatePresence: ({ children }: any) => <>{children}</>
}));

// Mock engine hook
vi.mock('../../hooks/useEngineAPI', () => ({
  useEngineAPI: () => ({
    engineState: {
      running: true,
      metrics: {
        totalCollapseMagnitude: 0.75,
        recursiveDepth: 5,
        complexityScore: 0.6,
        informationContent: 0.8
      }
    },
    quantumStates: [
      {
        qubitId: 'q1',
        collapsed: true,
        probability: 0.9,
        measurementOutcome: 1
      },
      {
        qubitId: 'q2',
        collapsed: false,
        probability: 0.6,
        measurementOutcome: null
      }
    ]
  })
}));

describe('GravitationalWaveEchoVisualizer', () => {
  const mockSimulationData = {
    echoes: [
      {
        id: 'echo-1',
        timestamp: Date.now() - 1000,
        magnitude: 0.8,
        frequency: 100,
        position: new THREE.Vector3(1, 0, 0),
        waveform: 'chirp' as const,
        informationContent: 0.7,
        memoryFieldCoupling: 0.6,
        coherence: 0.85,
        entropy: 0.3
      },
      {
        id: 'echo-2',
        timestamp: Date.now() - 2000,
        magnitude: 0.6,
        frequency: 150,
        position: new THREE.Vector3(-1, 0, 1),
        waveform: 'ringdown' as const,
        informationContent: 0.5,
        memoryFieldCoupling: 0.4,
        coherence: 0.7,
        entropy: 0.4
      }
    ],
    interferencePatterns: [
      {
        position: new THREE.Vector3(0, 0, 0),
        strength: 0.9,
        frequency: 125,
        phase: 0
      }
    ],
    memoryField: {
      density: new Float32Array([0.5, 0.6, 0.7, 0.8]),
      gradient: new THREE.Vector3(0.1, 0.2, 0.3),
      curvature: 0.05
    },
    primaryColor: '#0ea5e9'
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders without crashing', () => {
    render(<GravitationalWaveEchoVisualizer simulationData={mockSimulationData} />);
    expect(screen.getByTestId('canvas')).toBeInTheDocument();
  });

  it('displays controls panel', () => {
    render(<GravitationalWaveEchoVisualizer simulationData={mockSimulationData} />);
    expect(screen.getByText('Visualization Settings')).toBeInTheDocument();
    expect(screen.getByText('View Mode')).toBeInTheDocument();
    expect(screen.getByText('Visual Options')).toBeInTheDocument();
  });

  it('toggles visualization options', () => {
    render(<GravitationalWaveEchoVisualizer simulationData={mockSimulationData} />);
    
    const waveformsCheckbox = screen.getByLabelText('Show Waveforms');
    const interferenceCheckbox = screen.getByLabelText('Show Interference');
    
    expect(waveformsCheckbox).toBeChecked();
    expect(interferenceCheckbox).toBeChecked();
    
    fireEvent.click(waveformsCheckbox);
    expect(waveformsCheckbox).not.toBeChecked();
  });

  it('changes view mode', () => {
    render(<GravitationalWaveEchoVisualizer simulationData={mockSimulationData} />);
    
    const viewModeSelect = screen.getByLabelText('View Mode');
    fireEvent.change(viewModeSelect, { target: { value: 'top' } });
    
    expect(viewModeSelect).toHaveValue('top');
  });

  it('displays echo information when selected', async () => {
    render(<GravitationalWaveEchoVisualizer simulationData={mockSimulationData} />);
    
    // Wait for the component to render
    await waitFor(() => {
      expect(screen.getByText(/Echo #1/)).toBeInTheDocument();
    });
  });

  it('exports data when export button is clicked', () => {
    const mockCreateElement = vi.spyOn(document, 'createElement');
    const mockClick = vi.fn();
    const mockRemove = vi.fn();
    
    mockCreateElement.mockReturnValue({
      click: mockClick,
      remove: mockRemove,
      style: {},
      href: '',
      download: ''
    } as any);
    
    render(<GravitationalWaveEchoVisualizer simulationData={mockSimulationData} />);
    
    const exportButton = screen.getByText('Export Data');
    fireEvent.click(exportButton);
    
    expect(mockCreateElement).toHaveBeenCalledWith('a');
    expect(mockClick).toHaveBeenCalled();
  });

  it('integrates with real-time engine data', () => {
    render(<GravitationalWaveEchoVisualizer />);
    
    // Should create echoes from quantum states
    expect(screen.getByTestId('canvas')).toBeInTheDocument();
    
    // Check that metrics are being used
    const metricsDisplay = screen.getByText(/Information Density/);
    expect(metricsDisplay).toBeInTheDocument();
  });

  it('handles empty simulation data gracefully', () => {
    render(<GravitationalWaveEchoVisualizer simulationData={{ echoes: [] }} />);
    expect(screen.getByTestId('canvas')).toBeInTheDocument();
    expect(screen.queryByText(/Echo #/)).not.toBeInTheDocument();
  });

  it('updates quality setting', () => {
    render(<GravitationalWaveEchoVisualizer simulationData={mockSimulationData} />);
    
    const qualitySelect = screen.getByLabelText('Render Quality');
    fireEvent.change(qualitySelect, { target: { value: 'high' } });
    
    expect(qualitySelect).toHaveValue('high');
  });

  it('toggles play/pause state', () => {
    render(<GravitationalWaveEchoVisualizer simulationData={mockSimulationData} />);
    
    const playPauseButton = screen.getByRole('button', { name: /pause|play/i });
    const initialText = playPauseButton.textContent;
    
    fireEvent.click(playPauseButton);
    
    expect(playPauseButton.textContent).not.toBe(initialText);
  });

  it('adjusts time scale', () => {
    render(<GravitationalWaveEchoVisualizer simulationData={mockSimulationData} />);
    
    const timeScaleSlider = screen.getByLabelText('Time Scale');
    fireEvent.change(timeScaleSlider, { target: { value: '2' } });
    
    expect(timeScaleSlider).toHaveValue('2');
  });

  it('displays educational tooltip', async () => {
    render(<GravitationalWaveEchoVisualizer simulationData={mockSimulationData} />);
    
    const helpButton = screen.getByText('?');
    fireEvent.mouseEnter(helpButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Gravitational Wave Echoes/)).toBeInTheDocument();
    });
  });

  it('renders interference patterns when enabled', () => {
    render(<GravitationalWaveEchoVisualizer simulationData={mockSimulationData} />);
    
    expect(screen.getAllByTestId('sphere').length).toBeGreaterThan(0);
  });

  it('handles real-time updates', async () => {
    const { rerender } = render(<GravitationalWaveEchoVisualizer simulationData={mockSimulationData} />);
    
    const updatedData = {
      ...mockSimulationData,
      echoes: [
        ...mockSimulationData.echoes,
        {
          id: 'echo-3',
          timestamp: Date.now(),
          magnitude: 0.9,
          frequency: 200,
          position: new THREE.Vector3(2, 0, -1),
          waveform: 'burst' as const,
          informationContent: 0.8,
          memoryFieldCoupling: 0.7,
          coherence: 0.9,
          entropy: 0.2
        }
      ]
    };
    
    rerender(<GravitationalWaveEchoVisualizer simulationData={updatedData} />);
    
    await waitFor(() => {
      expect(screen.getByText(/Echo #3/)).toBeInTheDocument();
    });
  });
});