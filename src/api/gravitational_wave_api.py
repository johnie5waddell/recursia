"""
Gravitational Wave Echo API

Provides REST API endpoints for gravitational wave echo simulation and analysis.
Connects the frontend visualization to the physics engine implementation.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback for numpy operations if needed
    class _NumpyFallback:
        # Add ndarray type for compatibility
        ndarray = list
        def array(self, x): return x
        def zeros(self, shape): return [0] * (shape if isinstance(shape, int) else shape[0])
        def ones(self, shape): return [1] * (shape if isinstance(shape, int) else shape[0])
        @property
        def pi(self): return 3.14159265359
    np = _NumpyFallback()
from datetime import datetime
import json
import asyncio
import logging
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

# Import physics modules
from ..physics.gravitational_wave_echoes import (
    GravitationalWaveEchoSimulator,
    BinaryParameters,
    OSHEchoParameters,
    MergerType,
    EchoMechanism
)
# OSH calculations now done in VM only
# from ..physics.memory_field_proper import MemoryFieldProper  # Not used

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/gravitational-waves", tags=["gravitational-waves"])

# Global simulator instance (initialized on first use)
_simulator: Optional[GravitationalWaveEchoSimulator] = None


def get_simulator() -> GravitationalWaveEchoSimulator:
    """Get or create the global simulator instance."""
    global _simulator
    if _simulator is None:
        _simulator = GravitationalWaveEchoSimulator()
        logger.info("Initialized gravitational wave echo simulator")
    return _simulator


# Pydantic models for API

class BinaryParametersRequest(BaseModel):
    """Binary system parameters for API requests."""
    mass1: float = Field(gt=0, le=1000, description="First object mass in solar masses")
    mass2: float = Field(gt=0, le=1000, description="Second object mass in solar masses")
    distance: float = Field(gt=0, le=10000, description="Luminosity distance in Mpc")
    spin1: float = Field(ge=-1, le=1, description="Dimensionless spin of first object")
    spin2: float = Field(ge=-1, le=1, description="Dimensionless spin of second object")
    inclination: float = Field(ge=0, le=np.pi, description="Orbital inclination in radians")
    eccentricity: float = Field(ge=0, lt=1, description="Orbital eccentricity")
    
    def to_physics_params(self) -> BinaryParameters:
        """Convert to physics module parameters."""
        return BinaryParameters(
            mass1=self.mass1,
            mass2=self.mass2,
            spin1=np.array([0, 0, self.spin1]),  # Aligned spins
            spin2=np.array([0, 0, self.spin2]),
            eccentricity=self.eccentricity,
            inclination=self.inclination,
            distance=self.distance
        )


class OSHParametersRequest(BaseModel):
    """OSH echo parameters for API requests."""
    memoryStrainThreshold: float = Field(ge=0, le=1, default=0.85)
    informationCurvatureCoupling: float = Field(ge=0, le=1, default=0.3)
    rspAmplification: float = Field(ge=0, le=10, default=2.5)
    coherenceDecayTime: float = Field(gt=0, le=1, default=0.1)
    entropyBarrierHeight: float = Field(ge=0, le=1, default=0.95)
    echoDelayFactor: float = Field(ge=1, le=2, default=1.1)
    maxEchoOrders: int = Field(ge=1, le=20, default=5)
    
    def to_physics_params(self) -> OSHEchoParameters:
        """Convert to physics module parameters."""
        return OSHEchoParameters(
            memory_strain_threshold=self.memoryStrainThreshold,
            information_curvature_coupling=self.informationCurvatureCoupling,
            rsp_amplification=self.rspAmplification,
            coherence_decay_time=self.coherenceDecayTime,
            entropy_barrier_height=self.entropyBarrierHeight,
            echo_delay_factor=self.echoDelayFactor,
            max_echo_orders=self.maxEchoOrders
        )


class SimulationRequest(BaseModel):
    """Complete simulation request."""
    binaryParams: BinaryParametersRequest
    oshParams: OSHParametersRequest = OSHParametersRequest()
    includeNoise: bool = True
    samplingRate: Optional[float] = None
    duration: Optional[float] = None


class WaveformResponse(BaseModel):
    """Gravitational waveform response."""
    times: List[float]
    strainPlus: List[float]
    strainCross: List[float]
    frequency: List[float]
    amplitude: List[float]
    phase: List[float]
    echoTimes: Optional[List[float]] = None
    evidenceScore: Optional[float] = None
    rsp: Optional[float] = None
    infoCurvature: Optional[float] = None
    diagnostics: Optional[Dict[str, Any]] = None


class AnalysisRequest(BaseModel):
    """Echo analysis request."""
    times: List[float]
    strainPlus: List[float]
    strainCross: List[float]
    frequency: Optional[List[float]] = None
    amplitude: Optional[List[float]] = None
    phase: Optional[List[float]] = None


class AnalysisResponse(BaseModel):
    """Echo analysis response."""
    evidenceScore: float
    echoTimes: List[float]
    nEchoesDetected: int
    maxCorrelation: float
    patternScore: float
    spacingRegularity: float
    statisticalSignificance: float
    spectralFeatures: Optional[Dict[str, Any]] = None


class CatalogEntry(BaseModel):
    """Pre-computed waveform catalog entry."""
    id: str
    name: str
    description: str
    binaryParams: BinaryParametersRequest
    oshParams: OSHParametersRequest
    evidenceScore: float
    nEchoes: int
    category: str  # "strong", "moderate", "weak", "none"


# API Endpoints

@router.post("/simulate", response_model=WaveformResponse)
async def simulate_waveform(request: SimulationRequest) -> WaveformResponse:
    """
    Simulate gravitational wave signal with OSH echoes.
    
    This endpoint generates a complete gravitational wave signal including
    inspiral, merger, ringdown, and OSH-predicted echo patterns.
    """
    try:
        simulator = get_simulator()
        
        # Convert request parameters
        binary_params = request.binaryParams.to_physics_params()
        osh_params = request.oshParams.to_physics_params()
        
        # Override sampling rate and duration if specified
        if request.samplingRate:
            simulator.sampling_rate = request.samplingRate
        if request.duration:
            simulator.segment_duration = request.duration
        
        # Generate waveform
        logger.info(f"Simulating waveform for M1={binary_params.mass1}, M2={binary_params.mass2}")
        signal = simulator.simulate_merger_with_echoes(
            binary_params=binary_params,
            osh_params=osh_params,
            include_noise=request.includeNoise
        )
        
        # Downsample for response (keep every Nth point)
        downsample_factor = max(1, len(signal.times) // 4096)
        indices = slice(0, len(signal.times), downsample_factor)
        
        # Extract echo analysis results
        echo_analysis = None
        if hasattr(signal, 'diagnostics') and signal.diagnostics:
            echo_analysis = signal.diagnostics
        
        response = WaveformResponse(
            times=signal.times[indices].tolist(),
            strainPlus=signal.strain_plus[indices].tolist(),
            strainCross=signal.strain_cross[indices].tolist(),
            frequency=signal.frequency[indices].tolist(),
            amplitude=signal.amplitude[indices].tolist(),
            phase=signal.phase[indices].tolist(),
            echoTimes=echo_analysis.get('echo_times', []) if echo_analysis else None,
            evidenceScore=echo_analysis.get('echo_snr', 0) / 10 if echo_analysis else None,  # Normalize
            rsp=echo_analysis['osh_metrics'].get('rsp') if echo_analysis else None,
            infoCurvature=echo_analysis['osh_metrics'].get('info_curvature') if echo_analysis else None,
            diagnostics=echo_analysis
        )
        
        logger.info(f"Simulation complete: {len(response.echoTimes or [])} echoes detected")
        return response
        
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_waveform(request: AnalysisRequest) -> AnalysisResponse:
    """
    Analyze existing waveform for OSH echo signatures.
    
    This endpoint performs echo detection and analysis on provided
    gravitational wave data.
    """
    try:
        simulator = get_simulator()
        
        # Convert lists to numpy arrays
        signal_data = {
            'times': np.array(request.times),
            'strain_plus': np.array(request.strainPlus),
            'strain_cross': np.array(request.strainCross)
        }
        
        # Add optional fields if provided
        if request.frequency:
            signal_data['frequency'] = np.array(request.frequency)
        else:
            # Estimate frequency from phase derivative
            if request.phase:
                phase = np.array(request.phase)
                dt = signal_data['times'][1] - signal_data['times'][0]
                signal_data['frequency'] = np.gradient(phase) / (2 * np.pi * dt)
            else:
                signal_data['frequency'] = np.zeros_like(signal_data['times'])
        
        if request.amplitude:
            signal_data['amplitude'] = np.array(request.amplitude)
        else:
            # Calculate from strain
            signal_data['amplitude'] = np.sqrt(
                signal_data['strain_plus']**2 + signal_data['strain_cross']**2
            )
        
        if request.phase:
            signal_data['phase'] = np.array(request.phase)
        else:
            # Calculate from strain
            signal_data['phase'] = np.angle(
                signal_data['strain_plus'] + 1j * signal_data['strain_cross']
            )
        
        # Create signal object
        from ..physics.gravitational_wave_echoes import GravitationalWaveSignal
        signal = GravitationalWaveSignal(**signal_data)
        
        # Perform analysis
        analysis_results = simulator.analyze_echo_evidence(signal)
        
        response = AnalysisResponse(
            evidenceScore=analysis_results['evidence_score'],
            echoTimes=analysis_results['echo_times'],
            nEchoesDetected=analysis_results['n_echoes_detected'],
            maxCorrelation=analysis_results['max_correlation'],
            patternScore=analysis_results['pattern_score'],
            spacingRegularity=analysis_results['spacing_regularity'],
            statisticalSignificance=analysis_results['statistical_significance'],
            spectralFeatures=analysis_results.get('spectral_features')
        )
        
        logger.info(f"Analysis complete: evidence_score={response.evidenceScore:.3f}")
        return response
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/catalog", response_model=List[CatalogEntry])
async def get_waveform_catalog(
    category: Optional[str] = None,
    min_evidence: Optional[float] = None,
    max_evidence: Optional[float] = None
) -> List[CatalogEntry]:
    """
    Get catalog of pre-computed interesting waveforms.
    
    Returns a curated list of waveforms with various echo signatures
    for demonstration and testing purposes.
    """
    catalog = []
    
    # Example catalog entries
    examples = [
        {
            "id": "gw150914-like",
            "name": "GW150914-like with Strong Echoes",
            "description": "Binary black hole similar to GW150914 with strong OSH echoes",
            "binary": {"mass1": 36, "mass2": 29, "distance": 410, "spin1": 0.3, "spin2": -0.1},
            "osh": {"rspAmplification": 3.5, "maxEchoOrders": 7},
            "evidenceScore": 0.85,
            "nEchoes": 5,
            "category": "strong"
        },
        {
            "id": "intermediate-mass",
            "name": "Intermediate Mass BBH",
            "description": "Intermediate mass binary with moderate echo signature",
            "binary": {"mass1": 85, "mass2": 66, "distance": 1000, "spin1": 0.7, "spin2": 0.5},
            "osh": {"memoryStrainThreshold": 0.9, "coherenceDecayTime": 0.15},
            "evidenceScore": 0.65,
            "nEchoes": 3,
            "category": "moderate"
        },
        {
            "id": "neutron-star",
            "name": "Binary Neutron Star",
            "description": "BNS merger with weak echo signature",
            "binary": {"mass1": 1.4, "mass2": 1.35, "distance": 40, "spin1": 0.05, "spin2": 0.02},
            "osh": {"informationCurvatureCoupling": 0.5},
            "evidenceScore": 0.25,
            "nEchoes": 1,
            "category": "weak"
        },
        {
            "id": "high-spin",
            "name": "High Spin BBH",
            "description": "Rapidly spinning black holes with complex echo pattern",
            "binary": {"mass1": 25, "mass2": 20, "distance": 800, "spin1": 0.95, "spin2": 0.85},
            "osh": {"echoDelayFactor": 1.3, "entropyBarrierHeight": 0.98},
            "evidenceScore": 0.75,
            "nEchoes": 4,
            "category": "strong"
        }
    ]
    
    for example in examples:
        # Apply filters
        if category and example["category"] != category:
            continue
        if min_evidence and example["evidenceScore"] < min_evidence:
            continue
        if max_evidence and example["evidenceScore"] > max_evidence:
            continue
        
        # Create catalog entry
        entry = CatalogEntry(
            id=example["id"],
            name=example["name"],
            description=example["description"],
            binaryParams=BinaryParametersRequest(
                **example["binary"],
                inclination=0.0,
                eccentricity=0.0
            ),
            oshParams=OSHParametersRequest(**example.get("osh", {})),
            evidenceScore=example["evidenceScore"],
            nEchoes=example["nEchoes"],
            category=example["category"]
        )
        catalog.append(entry)
    
    return catalog


@router.post("/visualize/spectrogram")
async def generate_spectrogram(request: AnalysisRequest) -> StreamingResponse:
    """
    Generate spectrogram visualization of gravitational wave signal.
    
    Returns a PNG image showing time-frequency representation with
    echo signatures highlighted.
    """
    try:
        # Convert to numpy arrays
        times = np.array(request.times)
        strain = np.array(request.strainPlus)
        
        # Create figure
        fig = Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # Compute spectrogram
        from scipy import signal
        f, t, Sxx = signal.spectrogram(
            strain,
            fs=1.0 / (times[1] - times[0]),
            nperseg=256,
            noverlap=240
        )
        
        # Plot with log scale
        im = ax.pcolormesh(
            t + times[0], f, 10 * np.log10(Sxx + 1e-10),
            shading='gouraud',
            cmap='viridis',
            vmin=-200,
            vmax=-140
        )
        
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_ylim(10, 512)
        ax.set_yscale('log')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, label='Power [dB]')
        
        # Find and mark merger time (max amplitude)
        merger_idx = np.argmax(np.abs(strain))
        merger_time = times[merger_idx]
        ax.axvline(merger_time, color='red', linestyle='--', alpha=0.7, label='Merger')
        
        # Mark echo times if we can detect them
        try:
            simulator = get_simulator()
            from ..physics.gravitational_wave_echoes import GravitationalWaveSignal
            signal_obj = GravitationalWaveSignal(
                times=times,
                strain_plus=strain,
                strain_cross=np.array(request.strainCross) if request.strainCross else strain,
                frequency=np.zeros_like(times),
                amplitude=np.abs(strain),
                phase=np.angle(strain + 0j)
            )
            echo_times = simulator._detect_echo_times(signal_obj, merger_idx)
            
            for i, echo_time in enumerate(echo_times[:5]):  # Show up to 5 echoes
                ax.axvline(echo_time, color='yellow', linestyle=':', alpha=0.5)
                ax.text(echo_time, 400, f'E{i+1}', color='yellow', ha='center', va='bottom')
        except:
            pass  # Skip echo marking if detection fails
        
        ax.legend()
        ax.set_title('Gravitational Wave Spectrogram with OSH Echo Analysis')
        
        # Save to bytes buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/parameters/limits")
async def get_parameter_limits() -> Dict[str, Any]:
    """
    Get valid ranges for all simulation parameters.
    
    Returns the physical and computational limits for all parameters
    to help with UI validation.
    """
    return {
        "binary": {
            "mass1": {"min": 0.1, "max": 1000, "unit": "M_sun", "description": "Primary mass"},
            "mass2": {"min": 0.1, "max": 1000, "unit": "M_sun", "description": "Secondary mass"},
            "distance": {"min": 1, "max": 10000, "unit": "Mpc", "description": "Luminosity distance"},
            "spin1": {"min": -1, "max": 1, "unit": "dimensionless", "description": "Primary spin"},
            "spin2": {"min": -1, "max": 1, "unit": "dimensionless", "description": "Secondary spin"},
            "inclination": {"min": 0, "max": np.pi, "unit": "radians", "description": "Orbital inclination"},
            "eccentricity": {"min": 0, "max": 0.9, "unit": "dimensionless", "description": "Orbital eccentricity"}
        },
        "osh": {
            "memoryStrainThreshold": {"min": 0, "max": 1, "default": 0.85, "description": "Memory field activation threshold"},
            "informationCurvatureCoupling": {"min": 0, "max": 1, "default": 0.3, "description": "Coupling to information geometry"},
            "rspAmplification": {"min": 0, "max": 10, "default": 2.5, "description": "RSP effect strength"},
            "coherenceDecayTime": {"min": 0.001, "max": 1, "default": 0.1, "unit": "seconds", "description": "Coherence decay timescale"},
            "entropyBarrierHeight": {"min": 0, "max": 1, "default": 0.95, "description": "Entropy threshold for echoes"},
            "echoDelayFactor": {"min": 1, "max": 2, "default": 1.1, "description": "Echo delay scaling factor"},
            "maxEchoOrders": {"min": 1, "max": 20, "default": 5, "description": "Maximum number of echo reflections"}
        },
        "simulation": {
            "samplingRate": {"min": 512, "max": 16384, "default": 4096, "unit": "Hz", "description": "Time series sampling rate"},
            "duration": {"min": 0.1, "max": 128, "default": 32, "unit": "seconds", "description": "Total simulation duration"}
        }
    }


@router.post("/batch/simulate")
async def batch_simulate(
    requests: List[SimulationRequest],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Queue multiple simulations for batch processing.
    
    Returns a job ID that can be used to check status and retrieve results.
    """
    job_id = f"batch_{datetime.now().timestamp()}"
    
    # In a real implementation, this would use a task queue like Celery
    # For now, we'll use FastAPI's background tasks
    async def run_batch():
        results = []
        for i, request in enumerate(requests):
            try:
                result = await simulate_waveform(request)
                results.append({"index": i, "status": "success", "data": result})
            except Exception as e:
                results.append({"index": i, "status": "error", "error": str(e)})
        
        # Store results (in production, use a database or cache)
        # For now, just log completion
        logger.info(f"Batch job {job_id} completed: {len(results)} simulations")
    
    background_tasks.add_task(run_batch)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "total_simulations": len(requests),
        "message": "Batch simulation job queued successfully"
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Check health status of the gravitational wave API.
    
    Verifies that the simulator is initialized and functional.
    """
    try:
        simulator = get_simulator()
        
        # Perform a quick test simulation
        test_params = BinaryParameters(mass1=30, mass2=30, distance=100)
        test_signal = simulator._generate_base_waveform(test_params)
        
        return {
            "status": "healthy",
            "simulator_initialized": True,
            "test_signal_length": len(test_signal.times),
            "sampling_rate": simulator.sampling_rate,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }