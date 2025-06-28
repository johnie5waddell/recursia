import { MemoryFieldEngine } from '../engines/MemoryFieldEngine';
import { EntropyCoherenceSolver } from '../engines/EntropyCoherenceSolver';
import { RSPEngine } from '../engines/RSPEngine';
import { ObserverEngine } from '../engines/ObserverEngine';
import { WavefunctionSimulator } from '../engines/WavefunctionSimulator';
import { SimulationHarness } from '../engines/SimulationHarness';
import { Complex } from '../utils/complex';

describe('OSH Engine Tests', () => {
  describe('MemoryFieldEngine', () => {
    let engine: MemoryFieldEngine;
    
    beforeEach(() => {
      engine = new MemoryFieldEngine();
    });
    
    it('should initialize with empty field', () => {
      const field = engine.getField();
      expect(field.fragments.length).toBe(0);
      expect(field.totalCoherence).toBe(0);
    });
    
    it('should add memory fragments', () => {
      const id = engine.addFragment([new Complex(1, 0)], [0, 0, 0]);
      expect(id).toBeTruthy();
      
      const field = engine.getField();
      expect(field.fragments.length).toBe(1);
      expect(field.fragments[0].position).toEqual([0, 0, 0]);
    });
    
    it('should update field with time evolution', () => {
      engine.addFragment([new Complex(1, 0)], [0, 0, 0]);
      const initialField = engine.getField();
      
      engine.updateField(0.1);
      const updatedField = engine.getField();
      
      expect(updatedField.fragments[0].coherence).toBeLessThan(initialField.fragments[0].coherence);
    });
  });
  
  describe('EntropyCoherenceSolver', () => {
    let solver: EntropyCoherenceSolver;
    
    beforeEach(() => {
      solver = new EntropyCoherenceSolver();
    });
    
    it('should calculate Shannon entropy', () => {
      const states = [
        [new Complex(1, 0)],
        [new Complex(0, 1)],
        [new Complex(0.707, 0.707)]
      ];
      
      const entropy = solver.calculateEntropy(states);
      expect(entropy.shannon).toBeGreaterThan(0);
      expect(entropy.total).toBeGreaterThan(0);
    });
    
    it('should calculate coherence metrics', () => {
      const memoryField = {
        fragments: [
          {
            id: '1',
            position: [0, 0, 0] as [number, number, number],
            state: [new Complex(1, 0)],
            coherence: 0.8,
            timestamp: Date.now()
          }
        ],
        totalCoherence: 0.8,
        averageCoherence: 0.8,
        lastDefragmentation: Date.now()
      };
      
      const coherence = solver.calculateCoherence(memoryField);
      expect(coherence.average).toBe(0.8);
      expect(coherence.gradients.length).toBeGreaterThan(0);
    });
  });
  
  describe('RSPEngine', () => {
    let engine: RSPEngine;
    
    beforeEach(() => {
      engine = new RSPEngine();
    });
    
    it('should calculate RSP correctly', () => {
      engine.update(0.5, 0.8, 0.1);
      const state = engine.getState();
      
      expect(state.rsp).toBe((state.information * state.coherence) / state.entropy);
    });
    
    it('should detect divergence', () => {
      // Force divergence by setting high coherence and low entropy
      for (let i = 0; i < 10; i++) {
        engine.update(0.1, 0.95, 0.1);
      }
      
      const state = engine.getState();
      expect(state.isDiverging).toBe(true);
    });
  });
  
  describe('ObserverEngine', () => {
    let engine: ObserverEngine;
    
    beforeEach(() => {
      engine = new ObserverEngine();
    });
    
    it('should add and retrieve observers', () => {
      const id = engine.addObserver({
        position: [0, 0, 0],
        focus: 0.8,
        phase: 0,
        threshold: 0.5
      });
      
      expect(id).toBeTruthy();
      expect(engine.getObservers().length).toBe(1);
    });
    
    it('should calculate collapse probability', () => {
      const observer = {
        id: '1',
        position: [0, 0, 0] as [number, number, number],
        focus: 0.8,
        phase: 0,
        threshold: 0.5,
        coherence: 0.9
      };
      
      const wavefunction = [
        new Complex(0.707, 0),
        new Complex(0.707, 0)
      ];
      
      const result = engine.processObservation(observer, wavefunction, 0.1);
      expect(result).toBeTruthy();
      expect(result!.probability).toBeGreaterThan(0);
    });
  });
  
  describe('WavefunctionSimulator', () => {
    let simulator: WavefunctionSimulator;
    
    beforeEach(() => {
      simulator = new WavefunctionSimulator({
        sizeX: 8,
        sizeY: 8,
        sizeZ: 8,
        spacing: 0.1,
        boundaryCondition: 'periodic'
      });
    });
    
    it('should initialize with Gaussian wavepacket', () => {
      simulator.setGaussianWavepacket([4, 4, 4], [0, 0, 0], 2);
      const state = simulator.getState();
      
      expect(state.amplitude.length).toBe(512); // 8^3
      
      // Check normalization
      let norm = 0;
      for (const amp of state.amplitude) {
        norm += amp.real * amp.real + amp.imag * amp.imag;
      }
      expect(Math.abs(norm - 1)).toBeLessThan(0.01);
    });
    
    it('should propagate wavefunction', () => {
      simulator.setGaussianWavepacket([4, 4, 4], [1, 0, 0], 2);
      const initialState = simulator.getState();
      
      const potential = new Array(512).fill(new Complex(0, 0));
      simulator.propagate(0.1, potential);
      
      const finalState = simulator.getState();
      expect(finalState).not.toEqual(initialState);
    });
  });
  
  describe('SimulationHarness', () => {
    let harness: SimulationHarness;
    
    beforeEach(() => {
      harness = new SimulationHarness({
        gridSize: 8,
        timeStep: 0.1,
        memoryDecayRate: 0.05,
        coherenceDiffusion: 0.1,
        observerThreshold: 0.7,
        quantumCoupling: 0.5,
        entropyWeight: 1.0,
        informationFlow: 0.8
      });
    });
    
    it('should initialize simulation', async () => {
      await harness.initialize();
      const state = harness.getState();
      
      expect(state.time).toBe(0);
      expect(state.step).toBe(0);
      expect(state.memoryField.fragments.length).toBeGreaterThan(0);
    });
    
    it('should step simulation forward', async () => {
      await harness.initialize();
      const initialState = harness.getState();
      
      await harness.step();
      const newState = harness.getState();
      
      expect(newState.time).toBeGreaterThan(initialState.time);
      expect(newState.step).toBe(initialState.step + 1);
    });
    
    it('should generate predictions', async () => {
      await harness.initialize();
      
      // Run several steps to generate data
      for (let i = 0; i < 10; i++) {
        await harness.step();
      }
      
      const predictions = harness.getPredictions();
      expect(Array.isArray(predictions)).toBe(true);
    });
  });
});