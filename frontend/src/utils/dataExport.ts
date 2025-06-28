import type { MemoryField } from '../engines/MemoryFieldEngine';
import type { RSPState } from '../engines/RSPEngine';
import type { WavefunctionState } from '../engines/WavefunctionSimulator';
import type { ObservationEvent } from '../engines/ObserverEngine';
import { Complex } from './complex';

export interface ExportData {
  timestamp: string;
  version: string;
  simulation: {
    memoryField: MemoryField;
    rspState: RSPState;
    wavefunction: WavefunctionState;
    events: ObservationEvent[];
  };
  metadata: {
    duration: number;
    steps: number;
    parameters: Record<string, any>;
  };
}

export interface DashboardExportData {
  timestamp: string;
  version: string;
  simulation: {
    type: string;
    engine: string;
  };
  rspState: {
    current: RSPState;
    metrics: any;
    attractors: any[];
  };
  configuration: {
    timeRange: number;
    chartMode: string;
  };
  metadata: {
    exported_by: string;
    format: string;
  };
}

export class DataExporter {
  static exportToJSON(data: ExportData | DashboardExportData | any): string {
    return JSON.stringify(data, (key, value) => {
      // Handle Complex numbers
      if (value instanceof Complex) {
        return { real: value.real, imag: value.imag, _type: 'Complex' };
      }
      // Handle typed arrays
      if (value instanceof Float32Array || value instanceof Float64Array) {
        return { data: Array.from(value), _type: value.constructor.name };
      }
      return value;
    }, 2);
  }

  static exportToCSV(rspStates: RSPState[]): string {
    const headers = ['timestamp', 'rsp', 'information', 'coherence', 'entropy', 'isDiverging', 'attractorCount'];
    const rows = rspStates.map(state => [
      state.timestamp,
      state.rsp,
      state.information,
      state.coherence,
      state.entropy,
      state.isDiverging ? 1 : 0,
      state.attractors.length
    ]);

    return [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');
  }

  static async exportToHDF5(data: ExportData): Promise<ArrayBuffer> {
    // Note: Full HDF5 implementation would require a library like h5wasm
    // This is a simplified binary format that captures the structure
    
    const encoder = new TextEncoder();
    const chunks: ArrayBuffer[] = [];
    
    // Header
    const header = encoder.encode('OSH-SIM-v1.0');
    chunks.push(header.buffer);
    
    // Metadata
    const metadataJson = JSON.stringify(data.metadata);
    const metadataBytes = encoder.encode(metadataJson);
    const metadataLength = new Uint32Array([metadataBytes.length]);
    chunks.push(metadataLength.buffer);
    chunks.push(metadataBytes.buffer);
    
    // Memory field data
    const memoryData = this.serializeMemoryField(data.simulation.memoryField);
    chunks.push(memoryData);
    
    // Wavefunction data
    const wavefunctionData = this.serializeWavefunction(data.simulation.wavefunction);
    chunks.push(wavefunctionData);
    
    // Combine all chunks
    const totalLength = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
    const result = new ArrayBuffer(totalLength);
    const view = new Uint8Array(result);
    
    let offset = 0;
    for (const chunk of chunks) {
      view.set(new Uint8Array(chunk), offset);
      offset += chunk.byteLength;
    }
    
    return result;
  }

  private static serializeMemoryField(field: MemoryField): ArrayBuffer {
    // Simplified serialization - in practice would be more complex
    const fragmentCount = field.fragments.length;
    const buffer = new ArrayBuffer(4 + fragmentCount * 32); // Simplified size
    const view = new DataView(buffer);
    
    view.setUint32(0, fragmentCount, true);
    
    field.fragments.forEach((fragment, i) => {
      const offset = 4 + i * 32;
      view.setFloat32(offset, fragment.position[0], true);
      view.setFloat32(offset + 4, fragment.position[1], true);
      view.setFloat32(offset + 8, fragment.position[2], true);
      view.setFloat32(offset + 12, fragment.coherence, true);
      view.setFloat32(offset + 16, fragment.timestamp, true);
    });
    
    return buffer;
  }

  private static serializeWavefunction(wavefunction: WavefunctionState): ArrayBuffer {
    const size = wavefunction.amplitude.length;
    const buffer = new ArrayBuffer(4 + size * 8); // 4 bytes for size + 8 bytes per complex number
    const view = new DataView(buffer);
    
    view.setUint32(0, size, true);
    
    wavefunction.amplitude.forEach((complex, i) => {
      const offset = 4 + i * 8;
      view.setFloat32(offset, complex.real, true);
      view.setFloat32(offset + 4, complex.imag, true);
    });
    
    return buffer;
  }

  static async captureVisualizationFrame(canvas: HTMLCanvasElement): Promise<Blob> {
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        resolve(blob!);
      }, 'image/png');
    });
  }

  static async exportToVideo(frames: Blob[], fps: number = 30): Promise<Blob> {
    // Note: Full video encoding would require a library like ffmpeg.wasm
    // This creates a simple animated WebP as a placeholder
    
    // For now, return the first frame as a static image
    return frames[0] || new Blob();
  }

  static downloadFile(content: string | ArrayBuffer | Blob, filename: string, mimeType: string) {
    let blob: Blob;
    
    if (content instanceof Blob) {
      blob = content;
    } else if (content instanceof ArrayBuffer) {
      blob = new Blob([content], { type: mimeType });
    } else {
      blob = new Blob([content], { type: mimeType });
    }
    
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  static generateExportData(
    memoryField: MemoryField,
    rspState: RSPState,
    wavefunction: WavefunctionState,
    events: ObservationEvent[],
    duration: number,
    steps: number,
    parameters: Record<string, any>
  ): ExportData {
    return {
      timestamp: new Date().toISOString(),
      version: '1.0.0',
      simulation: {
        memoryField,
        rspState,
        wavefunction,
        events
      },
      metadata: {
        duration,
        steps,
        parameters
      }
    };
  }
}