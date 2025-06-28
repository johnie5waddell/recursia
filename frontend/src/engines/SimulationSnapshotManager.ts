/**
 * Simulation Snapshot Manager
 * Captures and manages simulation state snapshots for timeline playback
 */

import { MemoryField } from './MemoryFieldEngine';
import { RSPState } from './RSPEngine';
import { WavefunctionState } from './WavefunctionSimulator';

export interface TimelineEvent {
  timestamp: number;
  type: 'collapse' | 'entanglement' | 'divergence' | 'attractor' | 'measurement';
  description: string;
  data: any;
}

export interface SimulationSnapshot {
  timestamp: number;
  memoryField: MemoryField;
  rspState: RSPState;
  wavefunction: WavefunctionState;
  events: TimelineEvent[];
}

export class SimulationSnapshotManager {
  private snapshots: SimulationSnapshot[] = [];
  private maxSnapshots: number = 1000;
  private snapshotInterval: number = 100; // ms
  private lastSnapshotTime: number = 0;
  private events: TimelineEvent[] = [];
  private recordingEnabled: boolean = true;
  
  /**
   * Record a new snapshot if interval has passed
   */
  recordSnapshot(
    memoryField: MemoryField,
    rspState: RSPState,
    wavefunction: WavefunctionState
  ): void {
    if (!this.recordingEnabled) return;
    
    const now = Date.now();
    if (now - this.lastSnapshotTime < this.snapshotInterval) return;
    
    // Deep clone the states to prevent reference issues
    const snapshot: SimulationSnapshot = {
      timestamp: now,
      memoryField: this.cloneMemoryField(memoryField),
      rspState: this.cloneRSPState(rspState),
      wavefunction: this.cloneWavefunction(wavefunction),
      events: [...this.events] // Copy current events
    };
    
    this.snapshots.push(snapshot);
    this.lastSnapshotTime = now;
    
    // Clear events for next snapshot
    this.events = [];
    
    // Maintain max snapshots
    if (this.snapshots.length > this.maxSnapshots) {
      this.snapshots.shift();
    }
  }
  
  /**
   * Record an event
   */
  recordEvent(event: TimelineEvent): void {
    this.events.push(event);
    
    // Also check if this is a significant event that warrants immediate snapshot
    if (event.type === 'collapse' || event.type === 'divergence') {
      this.lastSnapshotTime = 0; // Force next snapshot
    }
  }
  
  /**
   * Get snapshots for timeline player
   */
  getSnapshots(): SimulationSnapshot[] {
    return [...this.snapshots];
  }
  
  /**
   * Get snapshot at specific time
   */
  getSnapshotAt(timestamp: number): SimulationSnapshot | null {
    // Find closest snapshot
    let closest: SimulationSnapshot | null = null;
    let minDiff = Infinity;
    
    this.snapshots.forEach(snapshot => {
      const diff = Math.abs(snapshot.timestamp - timestamp);
      if (diff < minDiff) {
        minDiff = diff;
        closest = snapshot;
      }
    });
    
    return closest;
  }

  /**
   * Get events in time range
   */
  getEventsInRange(startTime: number, endTime: number): TimelineEvent[] {
    const events: TimelineEvent[] = [];
    
    this.snapshots.forEach(snapshot => {
      if (snapshot.timestamp >= startTime && snapshot.timestamp <= endTime) {
        events.push(...snapshot.events);
      }
    });
    
    return events.sort((a, b) => a.timestamp - b.timestamp);
  }
  
  /**
   * Clear all snapshots
   */
  clear(): void {
    this.snapshots = [];
    this.events = [];
    this.lastSnapshotTime = 0;
  }
  
  /**
   * Enable/disable recording
   */
  setRecording(enabled: boolean): void {
    this.recordingEnabled = enabled;
  }
  
  /**
   * Set snapshot interval
   */
  setSnapshotInterval(interval: number): void {
    this.snapshotInterval = Math.max(10, interval); // Minimum 10ms
  }
  
  /**
   * Get memory usage estimate
   */
  getMemoryUsage(): number {
    // Rough estimate: each snapshot ~1KB base + fragment data
    const baseSize = 1024;
    const fragmentSize = 256; // bytes per fragment
    
    let totalSize = 0;
    this.snapshots.forEach(snapshot => {
      totalSize += baseSize;
      totalSize += snapshot.memoryField.fragments.length * fragmentSize;
      totalSize += (snapshot.wavefunction.amplitude?.length || 0) * 16; // Complex = 16 bytes
    });
    
    return totalSize;
  }
  
  /**
   * Clone memory field (deep copy)
   */
  private cloneMemoryField(field: MemoryField): MemoryField {
    return {
      fragments: field.fragments.map(f => ({
        id: f.id,
        timestamp: f.timestamp,
        position: [...f.position] as [number, number, number],
        coherence: f.coherence,
        state: f.state.map(c => new (c.constructor as any)(c.real, c.imag)),
        strain: f.strain,
        entropy: f.entropy,
        parentFragments: f.parentFragments,
        childFragments: f.childFragments
      })),
      totalCoherence: field.totalCoherence,
      averageCoherence: field.averageCoherence,
      totalEntropy: field.totalEntropy,
      lastDefragmentation: field.lastDefragmentation
    };
  }
  
  /**
   * Clone RSP state (deep copy)
   */
  private cloneRSPState(state: RSPState): RSPState {
    return {
      value: state.value,
      rsp: state.rsp,
      information: state.information,
      coherence: state.coherence,
      entropy: state.entropy,
      timestamp: state.timestamp,
      isDiverging: state.isDiverging,
      attractors: state.attractors.map(a => ({
        id: a.id,
        position: [...a.position] as [number, number, number],
        strength: a.strength,
        radius: a.radius,
        capturedFragments: [...a.capturedFragments],
        rspDensity: a.rspDensity
      })),
      derivatives: { ...state.derivatives }
    };
  }
  
  /**
   * Clone wavefunction (deep copy)
   */
  private cloneWavefunction(wavefunction: WavefunctionState): WavefunctionState {
    return {
      amplitude: wavefunction.amplitude ? wavefunction.amplitude.map(c => 
        new (c.constructor as any)(c.real, c.imag)
      ) : [],
      grid: wavefunction.grid ? wavefunction.grid.map(plane =>
        plane.map(row => row.map(c => new (c.constructor as any)(c.real, c.imag)))
      ) : [],
      gridSize: wavefunction.gridSize || 0,
      time: wavefunction.time || 0,
      totalProbability: wavefunction.totalProbability || 0,
      coherenceField: wavefunction.coherenceField ? wavefunction.coherenceField.map(plane =>
        plane.map(row => [...row])
      ) : [],
      phaseField: wavefunction.phaseField ? wavefunction.phaseField.map(plane =>
        plane.map(row => [...row])
      ) : []
    };
  }
  
  /**
   * Export snapshots for analysis
   */
  exportSnapshots(): any {
    return {
      metadata: {
        count: this.snapshots.length,
        interval: this.snapshotInterval,
        memoryUsage: this.getMemoryUsage(),
        timeRange: {
          start: this.snapshots[0]?.timestamp || 0,
          end: this.snapshots[this.snapshots.length - 1]?.timestamp || 0
        }
      },
      snapshots: this.snapshots.map(s => ({
        timestamp: s.timestamp,
        memoryFragmentCount: s.memoryField.fragments.length,
        rsp: s.rspState.rsp,
        coherence: s.memoryField.totalCoherence,
        eventCount: s.events.length,
        events: s.events
      }))
    };
  }
  
  /**
   * Clear old snapshots to free memory
   */
  clearOldSnapshots(keepLast: number = 100): void {
    if (this.snapshots.length > keepLast) {
      const removed = this.snapshots.splice(0, this.snapshots.length - keepLast);
      console.log(`Cleared ${removed.length} old snapshots, kept last ${keepLast}`);
    }
    
    // Also clear old events
    const cutoffTime = this.snapshots[0]?.timestamp || 0;
    this.events = this.events.filter(e => e.timestamp >= cutoffTime);
  }
}