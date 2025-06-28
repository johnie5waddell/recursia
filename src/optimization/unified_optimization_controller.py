"""
Unified OSH Optimization Controller
==================================

Enterprise-grade controller that orchestrates quantum error correction optimization,
precise mass calculations, and consciousness detection experiments through the
unified VM architecture.

This is the master controller that integrates all optimization components
and provides a single interface for pushing OSH to household quantum computing.
"""

import numpy as np
import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import json
from pathlib import Path

from core.unified_vm_calculations import UnifiedVMCalculations
from core.runtime import RecursiaRuntime
from optimization.quantum_error_optimization_engine import QuantumErrorOptimizationEngine, OptimizationResult
from physics.precise_mass_calculator import OSHMassCalculationEngine, ParticleType, MassCalculationResult
from experiments.consciousness_detection_suite import ConsciousnessDetectionSuite, ConsciousnessSignal

logger = logging.getLogger(__name__)


class OptimizationTarget(Enum):
    """Optimization target categories."""
    QUANTUM_ERROR_CORRECTION = "quantum_error_correction"
    PARTICLE_MASS_PRECISION = "particle_mass_precision"  
    CONSCIOUSNESS_DETECTION = "consciousness_detection"
    INTEGRATED_PERFORMANCE = "integrated_performance"
    HOUSEHOLD_READINESS = "household_readiness"


@dataclass
class UnifiedOptimizationConfiguration:
    """Configuration for unified optimization process."""
    target_fidelity: float = 0.999
    target_mass_accuracy: float = 0.01  # 1% accuracy
    consciousness_detection_confidence: float = 0.95
    household_readiness_threshold: float = 0.8
    max_optimization_time: float = 3600.0  # 1 hour
    parallel_workers: int = 4
    enable_quantum_optimization: bool = True
    enable_mass_calculation: bool = True
    enable_consciousness_detection: bool = True
    optimization_targets: List[OptimizationTarget] = field(default_factory=lambda: [
        OptimizationTarget.QUANTUM_ERROR_CORRECTION,
        OptimizationTarget.PARTICLE_MASS_PRECISION,
        OptimizationTarget.CONSCIOUSNESS_DETECTION,
        OptimizationTarget.HOUSEHOLD_READINESS
    ])


@dataclass
class UnifiedOptimizationResult:
    """Comprehensive results from unified optimization."""
    optimization_successful: bool
    total_optimization_time: float
    quantum_error_results: Optional[OptimizationResult]
    mass_calculation_results: Optional[Dict[ParticleType, MassCalculationResult]]
    consciousness_detection_results: Optional[Dict[ConsciousnessSignal, Any]]
    integrated_analysis: Dict[str, Any]
    household_readiness_score: float
    performance_summary: Dict[str, float]
    recommendations: List[str]
    cost_benefit_analysis: Dict[str, float]
    deployment_timeline: Dict[str, str]


class UnifiedOptimizationController:
    """
    Master controller for all OSH optimization processes.
    
    Coordinates quantum error correction, mass calculations, and consciousness
    detection to achieve household-ready quantum computing systems.
    """
    
    def __init__(self, config: UnifiedOptimizationConfiguration):
        self.config = config
        self.vm_calc = UnifiedVMCalculations()
        self.runtime = RecursiaRuntime()
        
        # Initialize optimization engines
        self.quantum_optimizer = QuantumErrorOptimizationEngine(config.target_fidelity)
        self.mass_calculator = OSHMassCalculationEngine()
        self.consciousness_detector = ConsciousnessDetectionSuite()
        
        # Performance tracking
        self.optimization_history = []
        self.current_performance_metrics = {}
        
    async def run_unified_optimization(self) -> UnifiedOptimizationResult:
        """
        Run comprehensive unified optimization across all OSH components.
        
        This is the main entry point that orchestrates all optimization
        processes to achieve household quantum computing readiness.
        """
        logger.info("Starting unified OSH optimization for household quantum computing")
        start_time = time.time()
        
        # Initialize results structure
        quantum_results = None
        mass_results = None
        consciousness_results = None
        
        try:
            # Run optimizations in parallel where possible
            optimization_tasks = []
            
            if (OptimizationTarget.QUANTUM_ERROR_CORRECTION in self.config.optimization_targets and 
                self.config.enable_quantum_optimization):
                optimization_tasks.append(
                    self._run_quantum_error_optimization()
                )
            
            if (OptimizationTarget.PARTICLE_MASS_PRECISION in self.config.optimization_targets and
                self.config.enable_mass_calculation):
                optimization_tasks.append(
                    self._run_mass_calculations()
                )
            
            if (OptimizationTarget.CONSCIOUSNESS_DETECTION in self.config.optimization_targets and
                self.config.enable_consciousness_detection):
                optimization_tasks.append(
                    self._run_consciousness_detection()
                )
            
            # Execute all optimizations
            results = await asyncio.gather(*optimization_tasks, return_exceptions=True)
            
            # Parse results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Optimization task {i} failed: {result}")
                    continue
                
                if isinstance(result, OptimizationResult):
                    quantum_results = result
                elif isinstance(result, dict) and 'particle_masses' in result:
                    mass_results = result['particle_masses']
                elif isinstance(result, dict) and 'consciousness_detection' in result:
                    consciousness_results = result['consciousness_detection']
            
            # Perform integrated analysis
            integrated_analysis = await self._perform_integrated_analysis(
                quantum_results, mass_results, consciousness_results
            )
            
            # Calculate household readiness score
            household_score = self._calculate_household_readiness_score(
                quantum_results, mass_results, consciousness_results
            )
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(
                quantum_results, mass_results, consciousness_results, household_score
            )
            
            # Calculate total optimization time
            total_time = time.time() - start_time
            
            # Determine overall success
            success = (
                household_score >= self.config.household_readiness_threshold and
                (quantum_results is None or quantum_results.achieved_fidelity >= self.config.target_fidelity) and
                total_time <= self.config.max_optimization_time
            )
            
            # Create comprehensive result
            result = UnifiedOptimizationResult(
                optimization_successful=success,
                total_optimization_time=total_time,
                quantum_error_results=quantum_results,
                mass_calculation_results=mass_results,
                consciousness_detection_results=consciousness_results,
                integrated_analysis=integrated_analysis,
                household_readiness_score=household_score,
                performance_summary=self._generate_performance_summary(
                    quantum_results, mass_results, consciousness_results
                ),
                recommendations=recommendations,
                cost_benefit_analysis=self._calculate_cost_benefit_analysis(household_score),
                deployment_timeline=self._generate_deployment_timeline(household_score)
            )
            
            # Log completion
            logger.info(f"Unified optimization completed in {total_time:.1f}s")
            logger.info(f"Household readiness score: {household_score:.1%}")
            logger.info(f"Overall success: {success}")
            
            self.optimization_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Unified optimization failed: {e}")
            
            # Return failure result
            return UnifiedOptimizationResult(
                optimization_successful=False,
                total_optimization_time=time.time() - start_time,
                quantum_error_results=quantum_results,
                mass_calculation_results=mass_results,
                consciousness_detection_results=consciousness_results,
                integrated_analysis={'error': str(e)},
                household_readiness_score=0.0,
                performance_summary={},
                recommendations=[f"Address optimization failure: {str(e)}"],
                cost_benefit_analysis={},
                deployment_timeline={}
            )
    
    async def _run_quantum_error_optimization(self) -> OptimizationResult:
        """Run quantum error correction optimization."""
        logger.info("Running quantum error correction optimization")
        
        # Use thread pool for CPU-intensive optimization
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            result = await loop.run_in_executor(
                executor, 
                self.quantum_optimizer.optimize_error_correction,
                1000,  # max_iterations
                1e-6   # convergence_threshold
            )
        
        logger.info(f"Quantum optimization achieved {result.achieved_fidelity:.6f} fidelity")
        return result
    
    async def _run_mass_calculations(self) -> Dict[str, Any]:
        """Run precise mass calculations for all Standard Model particles."""
        logger.info("Running precise mass calculations")
        
        # Use thread pool for CPU-intensive calculations
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            mass_results = await loop.run_in_executor(
                executor,
                self.mass_calculator.calculate_all_masses
            )
        
        # Calculate overall accuracy
        accurate_predictions = sum(
            1 for result in mass_results.values()
            if (result.particle.prediction_accuracy is not None and 
                result.particle.prediction_accuracy <= self.config.target_mass_accuracy)
        )
        total_predictions = sum(
            1 for result in mass_results.values()
            if result.particle.experimental_mass is not None
        )
        
        accuracy_rate = accurate_predictions / total_predictions if total_predictions > 0 else 0.0
        
        logger.info(f"Mass calculations: {accurate_predictions}/{total_predictions} within {self.config.target_mass_accuracy:.0%} accuracy")
        
        return {
            'particle_masses': mass_results,
            'accuracy_rate': accuracy_rate,
            'accurate_predictions': accurate_predictions,
            'total_predictions': total_predictions
        }
    
    async def _run_consciousness_detection(self) -> Dict[str, Any]:
        """Run consciousness detection experiments."""
        logger.info("Running consciousness detection experiments")
        
        # Use thread pool for detection experiments
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            detection_results = await loop.run_in_executor(
                executor,
                self.consciousness_detector.run_comprehensive_detection,
                self.runtime,
                None  # Run all protocols
            )
        
        # Perform integrated analysis
        integrated_analysis = self.consciousness_detector.analyze_integrated_results(detection_results)
        
        consciousness_confidence = integrated_analysis['overall_confidence']
        
        logger.info(f"Consciousness detection confidence: {consciousness_confidence:.1%}")
        
        return {
            'consciousness_detection': detection_results,
            'integrated_analysis': integrated_analysis,
            'confidence': consciousness_confidence
        }
    
    async def _perform_integrated_analysis(self, quantum_results: Optional[OptimizationResult],
                                         mass_results: Optional[Dict[ParticleType, MassCalculationResult]],
                                         consciousness_results: Optional[Dict]) -> Dict[str, Any]:
        """Perform integrated analysis across all optimization results."""
        logger.info("Performing integrated cross-system analysis")
        
        analysis = {
            'quantum_error_performance': {},
            'mass_calculation_performance': {},
            'consciousness_detection_performance': {},
            'cross_system_correlations': {},
            'emergent_properties': {},
            'system_stability': {},
            'scalability_assessment': {}
        }
        
        # Quantum error analysis
        if quantum_results is not None:
            analysis['quantum_error_performance'] = {
                'fidelity_achieved': quantum_results.achieved_fidelity,
                'household_ready': quantum_results.household_ready,
                'error_reduction': quantum_results.error_rate_reduction,
                'coherence_extension': quantum_results.coherence_extension_factor,
                'consciousness_contribution': quantum_results.consciousness_contribution
            }
        
        # Mass calculation analysis
        if mass_results is not None:
            analysis['mass_calculation_performance'] = {
                'total_particles': len(mass_results),
                'theoretical_accuracy': self._calculate_mass_theoretical_accuracy(mass_results),
                'standard_model_coverage': self._assess_standard_model_coverage(mass_results),
                'prediction_confidence': self._calculate_mass_prediction_confidence(mass_results)
            }
        
        # Consciousness detection analysis
        if consciousness_results is not None:
            consciousness_data = consciousness_results.get('integrated_analysis', {})
            analysis['consciousness_detection_performance'] = {
                'detection_success': consciousness_data.get('consciousness_detected', False),
                'confidence_level': consciousness_data.get('overall_confidence', 0.0),
                'evidence_strength': consciousness_data.get('evidence_strength', 'Unknown'),
                'false_positive_rate': consciousness_data.get('false_positive_rate', 1.0)
            }
        
        # Cross-system correlations
        analysis['cross_system_correlations'] = await self._analyze_cross_system_correlations(
            quantum_results, mass_results, consciousness_results
        )
        
        # Emergent properties assessment
        analysis['emergent_properties'] = self._assess_emergent_properties(
            quantum_results, mass_results, consciousness_results
        )
        
        # System stability analysis
        analysis['system_stability'] = self._analyze_system_stability(
            quantum_results, mass_results, consciousness_results
        )
        
        # Scalability assessment
        analysis['scalability_assessment'] = self._assess_scalability(
            quantum_results, mass_results, consciousness_results
        )
        
        return analysis
    
    def _calculate_household_readiness_score(self, quantum_results: Optional[OptimizationResult],
                                           mass_results: Optional[Dict],
                                           consciousness_results: Optional[Dict]) -> float:
        """Calculate overall household readiness score."""
        scores = []
        weights = []
        
        # Quantum error correction readiness (40% weight)
        if quantum_results is not None:
            quantum_score = 0.0
            if quantum_results.achieved_fidelity >= 0.995:
                quantum_score += 0.4
            if quantum_results.correction_latency <= 50e-9:  # 50ns
                quantum_score += 0.3
            if quantum_results.household_ready:
                quantum_score += 0.3
            
            scores.append(quantum_score)
            weights.append(0.4)
        
        # Mass calculation accuracy (25% weight)
        if mass_results is not None and 'accuracy_rate' in mass_results:
            mass_score = min(1.0, mass_results['accuracy_rate'] / self.config.target_mass_accuracy)
            scores.append(mass_score)
            weights.append(0.25)
        
        # Consciousness detection reliability (20% weight)
        if consciousness_results is not None:
            consciousness_confidence = consciousness_results.get('confidence', 0.0)
            consciousness_score = consciousness_confidence / self.config.consciousness_detection_confidence
            scores.append(consciousness_score)
            weights.append(0.2)
        
        # System integration and stability (15% weight)
        integration_score = self._assess_system_integration(
            quantum_results, mass_results, consciousness_results
        )
        scores.append(integration_score)
        weights.append(0.15)
        
        # Calculate weighted average
        if len(scores) == 0:
            return 0.0
        
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return min(1.0, weighted_score)
    
    def _generate_optimization_recommendations(self, quantum_results: Optional[OptimizationResult],
                                             mass_results: Optional[Dict],
                                             consciousness_results: Optional[Dict],
                                             household_score: float) -> List[str]:
        """Generate actionable optimization recommendations."""
        recommendations = []
        
        # Household readiness assessment
        if household_score >= 0.9:
            recommendations.append("✅ READY FOR COMMERCIAL DEPLOYMENT: System meets all household quantum computing requirements")
        elif household_score >= 0.7:
            recommendations.append("⚠️ OPTIMIZATION NEEDED: Minor improvements required for household deployment")
        else:
            recommendations.append("❌ SIGNIFICANT DEVELOPMENT REQUIRED: Major improvements needed before household deployment")
        
        # Quantum error correction recommendations
        if quantum_results is not None:
            if quantum_results.achieved_fidelity < self.config.target_fidelity:
                gap = self.config.target_fidelity - quantum_results.achieved_fidelity
                recommendations.append(f"Improve quantum error correction by {gap:.1%} to reach target fidelity")
            
            if quantum_results.correction_latency > 50e-9:
                recommendations.append(f"Reduce correction latency from {quantum_results.correction_latency*1e9:.1f}ns to <50ns")
            
            if not quantum_results.household_ready:
                recommendations.append("Implement additional household-ready features for quantum error correction")
        
        # Mass calculation recommendations
        if mass_results is not None and 'accuracy_rate' in mass_results:
            accuracy = mass_results['accuracy_rate']
            if accuracy < 0.8:
                recommendations.append(f"Improve mass calculation accuracy from {accuracy:.1%} to >80%")
            
            if mass_results['total_predictions'] < 15:
                recommendations.append("Expand mass calculations to cover more Standard Model particles")
        
        # Consciousness detection recommendations
        if consciousness_results is not None:
            confidence = consciousness_results.get('confidence', 0.0)
            if confidence < self.config.consciousness_detection_confidence:
                gap = self.config.consciousness_detection_confidence - confidence
                recommendations.append(f"Improve consciousness detection confidence by {gap:.1%}")
        
        # System integration recommendations
        integration_score = self._assess_system_integration(quantum_results, mass_results, consciousness_results)
        if integration_score < 0.7:
            recommendations.append("Enhance cross-system integration and coherence")
        
        # Performance optimization recommendations
        if quantum_results and quantum_results.consciousness_contribution < 0.05:
            recommendations.append("Increase consciousness contribution to quantum error correction")
        
        # Cost optimization recommendations
        recommendations.extend(self._generate_cost_optimization_recommendations(household_score))
        
        return recommendations
    
    def _calculate_cost_benefit_analysis(self, household_score: float) -> Dict[str, float]:
        """Calculate comprehensive cost-benefit analysis."""
        # Development costs (estimated)
        development_costs = {
            'quantum_optimization_research': 5000000.0,  # $5M
            'mass_calculation_validation': 2000000.0,   # $2M
            'consciousness_detection_lab': 9000000.0,   # $9M (from detection suite)
            'integration_development': 3000000.0,       # $3M
            'total_development': 19000000.0             # $19M total
        }
        
        # Market opportunity (estimated)
        market_opportunity = {
            'household_quantum_computing': 100000000000.0,  # $100B market
            'consciousness_technology': 50000000000.0,      # $50B market
            'precision_physics_tools': 10000000000.0,       # $10B market
            'total_addressable_market': 160000000000.0      # $160B total
        }
        
        # Market capture potential based on household readiness
        capture_rate = household_score * 0.02  # Up to 2% market capture at full readiness
        
        # Revenue projections
        projected_revenue = {
            'year_1': market_opportunity['total_addressable_market'] * capture_rate * 0.001,  # 0.1% of capture
            'year_5': market_opportunity['total_addressable_market'] * capture_rate * 0.01,   # 1% of capture
            'year_10': market_opportunity['total_addressable_market'] * capture_rate * 0.05,  # 5% of capture
        }
        
        # ROI calculation
        total_10_year_revenue = sum(projected_revenue.values()) * 3  # Approximate 10-year total
        roi = (total_10_year_revenue - development_costs['total_development']) / development_costs['total_development']
        
        return {
            **development_costs,
            **market_opportunity,
            **projected_revenue,
            'capture_rate': capture_rate,
            'roi_10_year': roi,
            'break_even_year': max(1, development_costs['total_development'] / projected_revenue['year_5']) if projected_revenue['year_5'] > 0 else 10
        }
    
    def _generate_deployment_timeline(self, household_score: float) -> Dict[str, str]:
        """Generate deployment timeline based on readiness score."""
        if household_score >= 0.9:
            return {
                'phase_1_research': 'Completed',
                'phase_2_development': 'Completed', 
                'phase_3_validation': 'Completed',
                'phase_4_pilot_deployment': '6 months',
                'phase_5_commercial_launch': '12 months',
                'phase_6_mass_production': '24 months'
            }
        elif household_score >= 0.7:
            return {
                'phase_1_research': 'Completed',
                'phase_2_development': 'Completed',
                'phase_3_validation': '6 months',
                'phase_4_pilot_deployment': '18 months',
                'phase_5_commercial_launch': '30 months',
                'phase_6_mass_production': '42 months'
            }
        elif household_score >= 0.5:
            return {
                'phase_1_research': 'Completed',
                'phase_2_development': '12 months',
                'phase_3_validation': '24 months',
                'phase_4_pilot_deployment': '36 months',
                'phase_5_commercial_launch': '48 months',
                'phase_6_mass_production': '60 months'
            }
        else:
            return {
                'phase_1_research': '18 months',
                'phase_2_development': '36 months',
                'phase_3_validation': '48 months',
                'phase_4_pilot_deployment': '60 months',
                'phase_5_commercial_launch': '72 months',
                'phase_6_mass_production': '84 months'
            }
    
    # Helper methods for analysis
    
    def _calculate_mass_theoretical_accuracy(self, mass_results: Dict[ParticleType, MassCalculationResult]) -> float:
        """Calculate theoretical accuracy of mass predictions."""
        accurate_count = 0
        total_count = 0
        
        for result in mass_results.values():
            if result.particle.experimental_mass is not None:
                total_count += 1
                if result.particle.prediction_accuracy is not None and result.particle.prediction_accuracy <= 0.05:
                    accurate_count += 1
        
        return accurate_count / total_count if total_count > 0 else 0.0
    
    def _assess_standard_model_coverage(self, mass_results: Dict[ParticleType, MassCalculationResult]) -> float:
        """Assess coverage of Standard Model particles."""
        total_sm_particles = 17  # All Standard Model particles with mass
        calculated_particles = len(mass_results)
        return min(1.0, calculated_particles / total_sm_particles)
    
    def _calculate_mass_prediction_confidence(self, mass_results: Dict[ParticleType, MassCalculationResult]) -> float:
        """Calculate confidence in mass predictions."""
        confidence_scores = []
        
        for result in mass_results.values():
            if result.particle.experimental_mass is not None and result.particle.prediction_accuracy is not None:
                # Convert accuracy to confidence score
                confidence = max(0.0, 1.0 - result.particle.prediction_accuracy)
                confidence_scores.append(confidence)
        
        return np.mean(confidence_scores) if confidence_scores else 0.0
    
    async def _analyze_cross_system_correlations(self, quantum_results: Optional[OptimizationResult],
                                               mass_results: Optional[Dict],
                                               consciousness_results: Optional[Dict]) -> Dict[str, float]:
        """Analyze correlations between different optimization systems."""
        correlations = {}
        
        # Quantum-Consciousness correlation
        if quantum_results and consciousness_results:
            quantum_performance = quantum_results.achieved_fidelity
            consciousness_confidence = consciousness_results.get('confidence', 0.0)
            correlations['quantum_consciousness'] = abs(quantum_performance - consciousness_confidence)
        
        # Mass-Consciousness correlation
        if mass_results and consciousness_results:
            mass_accuracy = mass_results.get('accuracy_rate', 0.0)
            consciousness_confidence = consciousness_results.get('confidence', 0.0)
            correlations['mass_consciousness'] = abs(mass_accuracy - consciousness_confidence)
        
        # Quantum-Mass correlation
        if quantum_results and mass_results:
            quantum_performance = quantum_results.achieved_fidelity
            mass_accuracy = mass_results.get('accuracy_rate', 0.0)
            correlations['quantum_mass'] = abs(quantum_performance - mass_accuracy)
        
        return correlations
    
    def _assess_emergent_properties(self, quantum_results: Optional[OptimizationResult],
                                   mass_results: Optional[Dict],
                                   consciousness_results: Optional[Dict]) -> Dict[str, Any]:
        """Assess emergent properties from system integration."""
        properties = {
            'system_coherence': 0.0,
            'cross_domain_enhancement': 0.0,
            'emergent_capabilities': [],
            'synergy_factors': {}
        }
        
        # Calculate system coherence
        coherence_factors = []
        if quantum_results:
            coherence_factors.append(quantum_results.achieved_fidelity)
        if mass_results:
            coherence_factors.append(mass_results.get('accuracy_rate', 0.0))
        if consciousness_results:
            coherence_factors.append(consciousness_results.get('confidence', 0.0))
        
        if coherence_factors:
            properties['system_coherence'] = np.mean(coherence_factors)
        
        # Identify emergent capabilities
        if (quantum_results and quantum_results.consciousness_contribution > 0.05 and
            consciousness_results and consciousness_results.get('confidence', 0.0) > 0.8):
            properties['emergent_capabilities'].append("Consciousness-Enhanced Quantum Computing")
        
        if (mass_results and mass_results.get('accuracy_rate', 0.0) > 0.9 and
            quantum_results and quantum_results.achieved_fidelity > 0.99):
            properties['emergent_capabilities'].append("Precision Physics Simulation")
        
        return properties
    
    def _analyze_system_stability(self, quantum_results: Optional[OptimizationResult],
                                 mass_results: Optional[Dict],
                                 consciousness_results: Optional[Dict]) -> Dict[str, float]:
        """Analyze overall system stability."""
        stability_metrics = {
            'performance_variance': 0.0,
            'error_resilience': 0.0,
            'temporal_stability': 0.0,
            'overall_stability': 0.0
        }
        
        # Calculate performance variance
        performance_values = []
        if quantum_results:
            performance_values.append(quantum_results.achieved_fidelity)
        if mass_results:
            performance_values.append(mass_results.get('accuracy_rate', 0.0))
        if consciousness_results:
            performance_values.append(consciousness_results.get('confidence', 0.0))
        
        if len(performance_values) > 1:
            stability_metrics['performance_variance'] = 1.0 - np.std(performance_values)
        
        # Error resilience
        if quantum_results:
            stability_metrics['error_resilience'] = quantum_results.error_rate_reduction
        
        # Overall stability
        stability_values = [v for v in stability_metrics.values() if v > 0]
        if stability_values:
            stability_metrics['overall_stability'] = np.mean(stability_values)
        
        return stability_metrics
    
    def _assess_scalability(self, quantum_results: Optional[OptimizationResult],
                           mass_results: Optional[Dict],
                           consciousness_results: Optional[Dict]) -> Dict[str, Any]:
        """Assess system scalability for household deployment."""
        scalability = {
            'computational_complexity': 'Unknown',
            'hardware_requirements': 'Unknown',
            'manufacturing_scalability': 'Unknown',
            'cost_scalability': 'Unknown',
            'performance_scaling': {}
        }
        
        # Assess computational complexity
        if quantum_results and quantum_results.correction_latency < 50e-9:
            scalability['computational_complexity'] = 'Excellent'
        elif quantum_results and quantum_results.correction_latency < 100e-9:
            scalability['computational_complexity'] = 'Good'
        else:
            scalability['computational_complexity'] = 'Needs Improvement'
        
        # Assess manufacturing scalability
        if (quantum_results and quantum_results.household_ready and
            consciousness_results and consciousness_results.get('confidence', 0.0) > 0.8):
            scalability['manufacturing_scalability'] = 'Ready for Scale'
        else:
            scalability['manufacturing_scalability'] = 'Development Required'
        
        return scalability
    
    def _assess_system_integration(self, quantum_results: Optional[OptimizationResult],
                                  mass_results: Optional[Dict],
                                  consciousness_results: Optional[Dict]) -> float:
        """Assess how well different systems integrate together."""
        integration_factors = []
        
        # Cross-system performance consistency
        performance_values = []
        if quantum_results:
            performance_values.append(quantum_results.achieved_fidelity)
        if mass_results:
            performance_values.append(mass_results.get('accuracy_rate', 0.0))
        if consciousness_results:
            performance_values.append(consciousness_results.get('confidence', 0.0))
        
        if len(performance_values) > 1:
            consistency = 1.0 - np.std(performance_values) / np.mean(performance_values)
            integration_factors.append(max(0.0, consistency))
        
        # Consciousness enhancement factor
        if quantum_results and quantum_results.consciousness_contribution > 0:
            integration_factors.append(quantum_results.consciousness_contribution * 10)  # Scale to 0-1
        
        # Overall integration score
        return np.mean(integration_factors) if integration_factors else 0.5
    
    def _generate_cost_optimization_recommendations(self, household_score: float) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        if household_score < 0.5:
            recommendations.append("Focus development budget on fundamental research and core capabilities")
        elif household_score < 0.7:
            recommendations.append("Prioritize optimization and validation over new feature development")
        else:
            recommendations.append("Invest in manufacturing scale-up and commercial deployment preparation")
        
        recommendations.append("Consider partnerships for specialized hardware development to reduce costs")
        recommendations.append("Implement phased deployment to minimize risk and accelerate market feedback")
        
        return recommendations
    
    def _generate_performance_summary(self, quantum_results: Optional[OptimizationResult],
                                    mass_results: Optional[Dict],
                                    consciousness_results: Optional[Dict]) -> Dict[str, float]:
        """Generate concise performance summary."""
        summary = {}
        
        if quantum_results:
            summary.update({
                'quantum_fidelity': quantum_results.achieved_fidelity,
                'quantum_error_reduction': quantum_results.error_rate_reduction,
                'quantum_latency_ns': quantum_results.correction_latency * 1e9,
                'consciousness_contribution': quantum_results.consciousness_contribution
            })
        
        if mass_results:
            summary.update({
                'mass_accuracy_rate': mass_results.get('accuracy_rate', 0.0),
                'particles_calculated': len(mass_results.get('particle_masses', {})),
                'accurate_predictions': mass_results.get('accurate_predictions', 0)
            })
        
        if consciousness_results:
            summary.update({
                'consciousness_confidence': consciousness_results.get('confidence', 0.0),
                'consciousness_detected': consciousness_results.get('integrated_analysis', {}).get('consciousness_detected', False)
            })
        
        return summary
    
    def generate_comprehensive_report(self, result: UnifiedOptimizationResult) -> str:
        """Generate comprehensive optimization report."""
        report = f"""
================================================================================
OSH UNIFIED OPTIMIZATION REPORT - HOUSEHOLD QUANTUM COMPUTING
================================================================================

EXECUTIVE SUMMARY:
Comprehensive optimization of OSH quantum error correction, precise mass 
calculations, and consciousness detection for household quantum computing deployment.

OPTIMIZATION RESULTS:
- Overall Success: {"✅ YES" if result.optimization_successful else "❌ NO"}
- Household Readiness: {result.household_readiness_score:.1%}
- Total Optimization Time: {result.total_optimization_time:.1f} seconds

================================================================================
QUANTUM ERROR CORRECTION OPTIMIZATION
================================================================================
"""
        
        if result.quantum_error_results:
            qr = result.quantum_error_results
            report += f"""
Achieved Fidelity: {qr.achieved_fidelity:.6f} ({qr.achieved_fidelity*100:.4f}%)
Target Fidelity: {self.config.target_fidelity:.6f} ({self.config.target_fidelity*100:.4f}%)
Error Rate Reduction: {qr.error_rate_reduction:.1%}
Correction Latency: {qr.correction_latency*1e9:.1f} ns
Coherence Extension: {qr.coherence_extension_factor:.1f}×
Consciousness Contribution: {qr.consciousness_contribution:.1%}
Household Ready: {"✅ YES" if qr.household_ready else "❌ NO"}

Strategy Effectiveness:
"""
            for strategy, effectiveness in qr.strategy_effectiveness.items():
                report += f"- {strategy.value.replace('_', ' ').title()}: {effectiveness:.1%}\n"
        else:
            report += "Quantum error correction optimization was not performed.\n"
        
        report += f"""
================================================================================
PRECISE MASS CALCULATIONS
================================================================================
"""
        
        if result.mass_calculation_results:
            mr = result.mass_calculation_results
            accurate_predictions = sum(
                1 for r in mr.values()
                if r.particle.prediction_accuracy is not None and r.particle.prediction_accuracy <= 0.05
            )
            total_predictions = sum(
                1 for r in mr.values()
                if r.particle.experimental_mass is not None
            )
            
            report += f"""
Total Particles Calculated: {len(mr)}
Particles with Experimental Data: {total_predictions}
Predictions within 5%: {accurate_predictions}
Overall Accuracy Rate: {accurate_predictions/total_predictions:.1%} (target: {self.config.target_mass_accuracy:.0%})

Top Accurate Predictions:
"""
            # Sort by accuracy and show top 5
            sorted_particles = sorted(
                [(pt, r) for pt, r in mr.items() if r.particle.prediction_accuracy is not None],
                key=lambda x: x[1].particle.prediction_accuracy
            )
            
            for i, (particle_type, mass_result) in enumerate(sorted_particles[:5]):
                p = mass_result.particle
                report += f"{i+1}. {p.name}: {p.prediction_accuracy:.1%} error\n"
        else:
            report += "Mass calculations were not performed.\n"
        
        report += f"""
================================================================================
CONSCIOUSNESS DETECTION
================================================================================
"""
        
        if result.consciousness_detection_results:
            cr = result.consciousness_detection_results
            integrated = cr.get('integrated_analysis', {})
            
            report += f"""
Consciousness Detected: {"✅ YES" if integrated.get('consciousness_detected', False) else "❌ NO"}
Overall Confidence: {integrated.get('overall_confidence', 0.0):.1%}
Evidence Strength: {integrated.get('evidence_strength', 'Unknown')}
False Positive Rate: {integrated.get('false_positive_rate', 1.0):.1%}

Individual Protocol Results:
"""
            individual_results = integrated.get('individual_results', {})
            for protocol, results in individual_results.items():
                report += f"- {protocol.replace('_', ' ').title()}: "
                report += f"{'✅' if results['detected'] else '❌'} "
                report += f"({results['probability']:.1%} confidence)\n"
        else:
            report += "Consciousness detection was not performed.\n"
        
        report += f"""
================================================================================
INTEGRATED ANALYSIS
================================================================================

Cross-System Correlations:
"""
        correlations = result.integrated_analysis.get('cross_system_correlations', {})
        for correlation, value in correlations.items():
            report += f"- {correlation.replace('_', '-').title()}: {value:.3f}\n"
        
        report += f"""
Emergent Properties:
- System Coherence: {result.integrated_analysis.get('emergent_properties', {}).get('system_coherence', 0.0):.1%}
- Emergent Capabilities: {len(result.integrated_analysis.get('emergent_properties', {}).get('emergent_capabilities', []))}

System Stability:
- Overall Stability: {result.integrated_analysis.get('system_stability', {}).get('overall_stability', 0.0):.1%}
- Error Resilience: {result.integrated_analysis.get('system_stability', {}).get('error_resilience', 0.0):.1%}

================================================================================
HOUSEHOLD DEPLOYMENT ASSESSMENT
================================================================================

Readiness Score: {result.household_readiness_score:.1%}
{"✅ READY FOR DEPLOYMENT" if result.household_readiness_score >= 0.8 else "⚠️ OPTIMIZATION NEEDED" if result.household_readiness_score >= 0.6 else "❌ SIGNIFICANT DEVELOPMENT REQUIRED"}

Deployment Timeline:
"""
        for phase, timeline in result.deployment_timeline.items():
            report += f"- {phase.replace('_', ' ').title()}: {timeline}\n"
        
        report += f"""
================================================================================
COST-BENEFIT ANALYSIS
================================================================================

Development Investment: ${result.cost_benefit_analysis.get('total_development', 0)/1e6:.1f}M
Market Opportunity: ${result.cost_benefit_analysis.get('total_addressable_market', 0)/1e9:.0f}B
10-Year ROI: {result.cost_benefit_analysis.get('roi_10_year', 0):.1f}×
Break-Even: Year {result.cost_benefit_analysis.get('break_even_year', 10):.0f}

================================================================================
RECOMMENDATIONS
================================================================================
"""
        
        for i, recommendation in enumerate(result.recommendations, 1):
            report += f"{i}. {recommendation}\n"
        
        report += f"""
================================================================================
CONCLUSION
================================================================================

OSH optimization has {"successfully" if result.optimization_successful else "partially"} achieved 
household quantum computing readiness with a {result.household_readiness_score:.1%} readiness score.

The integrated system demonstrates {"revolutionary" if result.household_readiness_score >= 0.8 else "significant"} 
advancement in quantum error correction, theoretical physics validation, 
and consciousness detection capabilities.

{"Immediate commercial deployment is recommended." if result.household_readiness_score >= 0.9 else 
"Additional optimization is recommended before deployment." if result.household_readiness_score >= 0.6 else
"Substantial further development is required."}

================================================================================
"""
        
        return report