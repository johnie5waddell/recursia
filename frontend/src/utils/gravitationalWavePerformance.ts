import * as THREE from 'three';

interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  drawCalls: number;
  triangles: number;
  points: number;
  lines: number;
  memoryUsage: {
    geometries: number;
    textures: number;
    programs: number;
    materials: number;
  };
  shaderCompilations: number;
  renderTime: number;
}

interface PerformanceThresholds {
  minFPS: number;
  maxFrameTime: number;
  maxDrawCalls: number;
  maxTriangles: number;
  maxMemoryMB: number;
}

export class GravitationalWavePerformanceMonitor {
  private metrics: PerformanceMetrics;
  private thresholds: PerformanceThresholds;
  private frameCount: number = 0;
  private lastTime: number = performance.now();
  private frameTimes: number[] = [];
  private maxFrameSamples: number = 60;
  private renderer: THREE.WebGLRenderer | null = null;
  private performanceCallbacks: Set<(metrics: PerformanceMetrics) => void> = new Set();

  constructor(thresholds?: Partial<PerformanceThresholds>) {
    this.thresholds = {
      minFPS: 30,
      maxFrameTime: 33.33, // 30 FPS
      maxDrawCalls: 100,
      maxTriangles: 100000,
      maxMemoryMB: 256,
      ...thresholds
    };

    this.metrics = this.createEmptyMetrics();
  }

  private createEmptyMetrics(): PerformanceMetrics {
    return {
      fps: 0,
      frameTime: 0,
      drawCalls: 0,
      triangles: 0,
      points: 0,
      lines: 0,
      memoryUsage: {
        geometries: 0,
        textures: 0,
        programs: 0,
        materials: 0
      },
      shaderCompilations: 0,
      renderTime: 0
    };
  }

  public setRenderer(renderer: THREE.WebGLRenderer): void {
    this.renderer = renderer;
  }

  public startFrame(): void {
    this.frameCount++;
  }

  public endFrame(): void {
    const currentTime = performance.now();
    const deltaTime = currentTime - this.lastTime;
    
    // Update frame times
    this.frameTimes.push(deltaTime);
    if (this.frameTimes.length > this.maxFrameSamples) {
      this.frameTimes.shift();
    }

    // Calculate metrics
    this.updateMetrics();
    
    // Check performance thresholds
    this.checkThresholds();
    
    // Notify callbacks
    this.notifyCallbacks();
    
    this.lastTime = currentTime;
  }

  private updateMetrics(): void {
    // Calculate FPS and frame time
    if (this.frameTimes.length > 0) {
      const avgFrameTime = this.frameTimes.reduce((a, b) => a + b) / this.frameTimes.length;
      this.metrics.frameTime = avgFrameTime;
      this.metrics.fps = 1000 / avgFrameTime;
    }

    // Update renderer stats if available
    if (this.renderer) {
      const info = this.renderer.info;
      
      this.metrics.drawCalls = info.render.calls;
      this.metrics.triangles = info.render.triangles;
      this.metrics.points = info.render.points;
      this.metrics.lines = info.render.lines;
      
      this.metrics.memoryUsage = {
        geometries: info.memory.geometries,
        textures: info.memory.textures,
        programs: info.programs?.length || 0,
        materials: 0 // THREE.js doesn't provide material count directly
      };
    }
  }

  private checkThresholds(): void {
    const warnings: string[] = [];

    if (this.metrics.fps < this.thresholds.minFPS && this.metrics.fps > 0) {
      warnings.push(`Low FPS: ${this.metrics.fps.toFixed(1)} (min: ${this.thresholds.minFPS})`);
    }

    if (this.metrics.frameTime > this.thresholds.maxFrameTime) {
      warnings.push(`High frame time: ${this.metrics.frameTime.toFixed(1)}ms`);
    }

    if (this.metrics.drawCalls > this.thresholds.maxDrawCalls) {
      warnings.push(`Too many draw calls: ${this.metrics.drawCalls} (max: ${this.thresholds.maxDrawCalls})`);
    }

    if (this.metrics.triangles > this.thresholds.maxTriangles) {
      warnings.push(`Too many triangles: ${this.metrics.triangles} (max: ${this.thresholds.maxTriangles})`);
    }

    if (warnings.length > 0) {
      console.warn('[GW Performance]', warnings.join(', '));
    }
  }

  private notifyCallbacks(): void {
    this.performanceCallbacks.forEach(callback => callback(this.metrics));
  }

  public onMetricsUpdate(callback: (metrics: PerformanceMetrics) => void): () => void {
    this.performanceCallbacks.add(callback);
    return () => this.performanceCallbacks.delete(callback);
  }

  public getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  public reset(): void {
    this.frameCount = 0;
    this.frameTimes = [];
    this.metrics = this.createEmptyMetrics();
    this.lastTime = performance.now();
  }

  public suggestOptimizations(): string[] {
    const suggestions: string[] = [];

    if (this.metrics.fps < this.thresholds.minFPS) {
      suggestions.push('Reduce render quality or echo count');
      suggestions.push('Disable interference patterns or memory field visualization');
      suggestions.push('Use lower resolution for wave simulations');
    }

    if (this.metrics.drawCalls > this.thresholds.maxDrawCalls) {
      suggestions.push('Use instanced rendering for repeated geometries');
      suggestions.push('Merge similar meshes');
      suggestions.push('Use LOD (Level of Detail) for distant objects');
    }

    if (this.metrics.triangles > this.thresholds.maxTriangles) {
      suggestions.push('Reduce subdivision levels for spheres and waves');
      suggestions.push('Use simpler geometries for distant echoes');
      suggestions.push('Implement frustum culling');
    }

    const totalMemory = Object.values(this.metrics.memoryUsage).reduce((a, b) => a + b, 0);
    if (totalMemory > this.thresholds.maxMemoryMB * 1024 * 1024) {
      suggestions.push('Dispose unused geometries and materials');
      suggestions.push('Share geometries between similar objects');
      suggestions.push('Reduce texture resolutions');
    }

    return suggestions;
  }

  public getPerformanceReport(): string {
    const report = [
      '=== Gravitational Wave Visualizer Performance Report ===',
      `FPS: ${this.metrics.fps.toFixed(1)} (Target: ${this.thresholds.minFPS}+)`,
      `Frame Time: ${this.metrics.frameTime.toFixed(2)}ms`,
      `Draw Calls: ${this.metrics.drawCalls}`,
      `Triangles: ${this.metrics.triangles.toLocaleString()}`,
      `Points: ${this.metrics.points.toLocaleString()}`,
      `Lines: ${this.metrics.lines.toLocaleString()}`,
      '',
      'Memory Usage:',
      `  Geometries: ${this.metrics.memoryUsage.geometries}`,
      `  Textures: ${this.metrics.memoryUsage.textures}`,
      `  Programs: ${this.metrics.memoryUsage.programs}`,
      '',
      'Optimization Suggestions:',
      ...this.suggestOptimizations().map(s => `  - ${s}`)
    ];

    return report.join('\n');
  }
}

// React hook for performance monitoring
import { useEffect, useState, useRef } from 'react';

export function useGravitationalWavePerformance(
  renderer: THREE.WebGLRenderer | null,
  thresholds?: Partial<PerformanceThresholds>
) {
  const monitorRef = useRef<GravitationalWavePerformanceMonitor>();
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 0,
    frameTime: 0,
    drawCalls: 0,
    triangles: 0,
    points: 0,
    lines: 0,
    memoryUsage: {
      geometries: 0,
      textures: 0,
      programs: 0,
      materials: 0
    },
    shaderCompilations: 0,
    renderTime: 0
  });

  useEffect(() => {
    if (!monitorRef.current) {
      monitorRef.current = new GravitationalWavePerformanceMonitor(thresholds);
    }

    if (renderer) {
      monitorRef.current.setRenderer(renderer);
    }

    const unsubscribe = monitorRef.current.onMetricsUpdate(setMetrics);

    return () => {
      unsubscribe();
    };
  }, [renderer, thresholds]);

  return {
    monitor: monitorRef.current,
    metrics,
    startFrame: () => monitorRef.current?.startFrame(),
    endFrame: () => monitorRef.current?.endFrame(),
    getReport: () => monitorRef.current?.getPerformanceReport() || '',
    getSuggestions: () => monitorRef.current?.suggestOptimizations() || []
  };
}

// Automatic quality adjustment based on performance
export class QualityAutoAdjuster {
  private performanceHistory: number[] = [];
  private qualityLevel: 'low' | 'medium' | 'high' = 'high';
  private adjustmentCallbacks: Set<(quality: 'low' | 'medium' | 'high') => void> = new Set();

  public update(fps: number): void {
    this.performanceHistory.push(fps);
    if (this.performanceHistory.length > 30) { // 30 samples = ~0.5 seconds
      this.performanceHistory.shift();
    }

    // Auto-adjust quality based on average FPS
    if (this.performanceHistory.length >= 10) {
      const avgFPS = this.performanceHistory.reduce((a, b) => a + b) / this.performanceHistory.length;
      
      let newQuality = this.qualityLevel;
      
      if (avgFPS < 25 && this.qualityLevel !== 'low') {
        newQuality = this.qualityLevel === 'high' ? 'medium' : 'low';
      } else if (avgFPS > 55 && this.qualityLevel !== 'high') {
        newQuality = this.qualityLevel === 'low' ? 'medium' : 'high';
      }

      if (newQuality !== this.qualityLevel) {
        this.qualityLevel = newQuality;
        this.notifyCallbacks();
      }
    }
  }

  public onQualityChange(callback: (quality: 'low' | 'medium' | 'high') => void): () => void {
    this.adjustmentCallbacks.add(callback);
    return () => this.adjustmentCallbacks.delete(callback);
  }

  private notifyCallbacks(): void {
    this.adjustmentCallbacks.forEach(callback => callback(this.qualityLevel));
  }

  public getQuality(): 'low' | 'medium' | 'high' {
    return this.qualityLevel;
  }

  public reset(): void {
    this.performanceHistory = [];
    this.qualityLevel = 'high';
  }
}