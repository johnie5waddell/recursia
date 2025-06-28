# OSH Quantum Engine System Requirements

## Overview
The OSH Quantum Engine is a computationally intensive quantum simulation platform that performs real-time 3D wavefunction evolution, memory field dynamics, and complex mathematical operations. The memory requirements scale exponentially with grid size and simulation complexity.

## Minimum System Requirements

### Hardware
- **CPU**: Quad-core processor (4+ cores) @ 2.5 GHz or higher
  - Intel Core i5 (8th gen) or AMD Ryzen 5 2600 equivalent
  - The engine uses multi-threading for quantum calculations
  
- **RAM**: 8 GB minimum (16 GB strongly recommended)
  - JavaScript heap limit is typically 2-4 GB in browsers
  - Chrome/Edge provide better memory management than Firefox
  
- **GPU**: Dedicated graphics with WebGL 2.0 support
  - NVIDIA GTX 1050 / AMD RX 560 or better
  - Integrated graphics may struggle with 3D visualizations
  
- **Storage**: 500 MB free space for browser cache

### Software
- **Browser**: 
  - Chrome 90+ or Edge 90+ (recommended for performance.memory API)
  - Firefox 88+ (limited memory monitoring)
  - Safari 14+ (may have memory constraints)
  
- **Operating System**:
  - Windows 10/11 (best performance)
  - macOS 10.15+ 
  - Linux with modern desktop environment

## Recommended System Requirements

### Hardware
- **CPU**: 6+ core processor @ 3.0 GHz
  - Intel Core i7 (10th gen) or AMD Ryzen 7 3700X
  - More cores allow better parallel processing
  
- **RAM**: 16-32 GB
  - Allows larger grid sizes and longer simulations
  - Reduces garbage collection pauses
  
- **GPU**: Mid-range dedicated graphics
  - NVIDIA RTX 3060 / AMD RX 6600 or better
  - Hardware acceleration for Three.js rendering

## Memory Usage Breakdown

### Grid Size Impact
The wavefunction simulator uses a 3D complex number grid:
- **4×4×4 grid**: ~64 cells × 16 bytes = 1 KB base + overhead ≈ 100 KB total
- **8×8×8 grid**: ~512 cells × 16 bytes = 8 KB base + overhead ≈ 1 MB total
- **16×16×16 grid**: ~4,096 cells × 16 bytes = 64 KB base + overhead ≈ 10 MB total
- **30×30×30 grid**: ~27,000 cells × 16 bytes = 432 KB base + overhead ≈ 50 MB total

### Component Memory Usage
1. **WavefunctionSimulator**: 10-50 MB (grid size dependent)
2. **MemoryFieldEngine**: 5-20 MB (fragment count dependent)
3. **Three.js 3D Visualization**: 50-200 MB
4. **Worker Threads**: 10-30 MB each (2-4 workers)
5. **UI Components**: 20-50 MB
6. **Engine Overhead**: 30-50 MB

**Total Active Memory**: 200-500 MB typical, up to 1 GB for complex simulations

## Performance Optimization Settings

### Based on Available Memory

#### Low Memory Mode (<8 GB RAM or <1 GB available)
```javascript
// Automatically selected by getMemorySafeGridSize()
- Grid Size: 3×3×3 to 4×4×4
- Workers: 1
- Update Interval: 50ms (20 FPS)
- Visualization: Simplified
- Features: Core only
```

#### Standard Mode (8-16 GB RAM)
```javascript
- Grid Size: 6×6×6 to 8×8×8
- Workers: 2
- Update Interval: 33ms (30 FPS)
- Visualization: Full
- Features: All enabled
```

#### Performance Mode (16+ GB RAM)
```javascript
- Grid Size: 12×12×12 to 16×16×16
- Workers: 4
- Update Interval: 16ms (60 FPS)
- Visualization: Enhanced
- Features: All + experimental
```

## Browser Memory Limits

### Chrome/Edge
- Default heap limit: 2-4 GB (varies by system RAM)
- Can be increased with flags: `--max-old-space-size=4096`
- Best memory profiling tools

### Firefox
- Generally higher memory ceiling
- Less detailed memory API
- May handle large arrays better

### Safari
- More restrictive memory limits
- Limited performance monitoring APIs
- May require lower grid sizes

## Diagnosing Memory Issues

### Check Current Usage
1. Open browser DevTools (F12)
2. Go to Memory/Performance tab
3. Look for:
   - JS Heap Size > 1.5 GB (warning)
   - JS Heap Size > 2 GB (critical)

### Common Memory Leak Sources
1. **Uncleared animation frames** - Fixed with proper cleanup
2. **Event listener accumulation** - Fixed with removeEventListener
3. **Growing arrays** - Fixed with circular buffers
4. **Three.js geometries** - Fixed with dispose() calls
5. **Worker thread leaks** - Fixed with proper termination

### Performance Monitoring
The app includes built-in monitoring:
- Memory usage gauge in UI
- FPS counter
- Resource manager throttling
- Automatic quality reduction under pressure

## Recommended Actions

### If You Experience Memory Issues:

1. **Reduce Grid Size**
   - The app auto-detects available memory
   - You can manually set smaller grids in code

2. **Close Other Tabs/Apps**
   - Each Chrome tab uses separate memory
   - Other apps compete for RAM

3. **Use Performance Mode**
   - Chrome: Settings → System → Use hardware acceleration
   - Enable GPU acceleration

4. **Increase Browser Memory**
   - Windows: Create shortcut with `--max-old-space-size=4096`
   - Mac/Linux: Launch from terminal with flag

5. **Monitor Performance**
   - Use built-in diagnostics (`/test-simulation-debug.html`)
   - Watch for memory warnings in console

## Benchmark Results

### Test System: Intel i7-10700K, 32GB RAM, RTX 3070
- 4×4×4 grid: 60 FPS, 150 MB memory
- 8×8×8 grid: 60 FPS, 250 MB memory
- 16×16×16 grid: 45 FPS, 500 MB memory
- 30×30×30 grid: 15 FPS, 1.2 GB memory (not recommended)

### Test System: Intel i5-8250U, 8GB RAM, Integrated Graphics
- 4×4×4 grid: 30 FPS, 200 MB memory
- 8×8×8 grid: 20 FPS, 350 MB memory
- 16×16×16 grid: Out of memory errors

## Conclusion

For optimal experience:
- **Minimum**: 8 GB RAM, use 4×4×4 to 6×6×6 grids
- **Recommended**: 16 GB RAM, use 8×8×8 grids
- **Ideal**: 32 GB RAM, can handle 16×16×16 grids

The application includes automatic memory management and will adapt to your system's capabilities. If you're experiencing crashes, the built-in safety mechanisms should prevent them by reducing quality before hitting limits.