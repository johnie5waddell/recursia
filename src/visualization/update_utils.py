"""
update_utils.py - Real-Time Phenomena Detection and Report Synchronization Engine

This module provides critical update hooks for the Recursia runtime system, enabling:
- Real-time emergent phenomena detection and classification
- Structured scientific report data collection and validation
- Advanced OSH metric analysis and correlation tracking
- Performance-optimized state synchronization
- Comprehensive diagnostic and validation systems

Integrates with EmergentPhenomenaDetector, ReportBuilder, and global metrics tracking.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
from datetime import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

# Performance monitoring
from src.core.utils import performance_profiler, global_error_manager

logger = logging.getLogger(__name__)


@dataclass
class UpdateMetrics:
    """Comprehensive metrics tracking for update operations"""
    total_updates: int = 0
    phenomena_updates: int = 0
    report_updates: int = 0
    failed_updates: int = 0
    average_update_time: float = 0.0
    peak_update_time: float = 0.0
    last_update_time: float = 0.0
    update_frequency: float = 0.0
    error_rate: float = 0.0
    data_validation_errors: int = 0
    correlation_analysis_count: int = 0
    anomaly_detections: int = 0


@dataclass
class PhenomenaValidationResult:
    """Results from phenomena detection validation"""
    is_valid: bool
    detected_count: int
    strength_distribution: Dict[str, float] = field(default_factory=dict)
    temporal_consistency: float = 0.0
    correlation_score: float = 0.0
    anomaly_flags: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class ReportDataValidation:
    """Validation result for report data processing."""
    is_valid: bool = True
    data_completeness: float = 1.0
    temporal_continuity: float = 1.0
    metric_consistency: float = 1.0
    data_quality_score: float = 1.0
    validation_errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
class AdvancedPhenomenaAnalyzer:
    """Advanced analyzer for emergent phenomena detection and classification"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.detection_history = deque(maxlen=self.config.get('history_length', 1000))
        self.correlation_tracker = defaultdict(list)
        self.anomaly_baseline = {}
        self.pattern_cache = {}
        self.lock = threading.RLock()
        
        # Statistical thresholds
        self.coherence_anomaly_threshold = self.config.get('coherence_anomaly_threshold', 0.3)
        self.entropy_spike_threshold = self.config.get('entropy_spike_threshold', 0.5)
        self.strain_critical_threshold = self.config.get('strain_critical_threshold', 0.8)
        self.consensus_stability_threshold = self.config.get('consensus_stability_threshold', 0.7)
        
        # Performance optimization
        self.batch_size = self.config.get('batch_size', 50)
        self.enable_async_analysis = self.config.get('enable_async_analysis', True)
        
        logger.info("Advanced Phenomena Analyzer initialized with enhanced detection capabilities")
    
    def analyze_phenomena_patterns(self, phenomena_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced pattern analysis on detected phenomena"""
        try:
            with self.lock:
                analysis_results = {
                    'pattern_classification': self._classify_phenomena_patterns(phenomena_data),
                    'temporal_correlation': self._analyze_temporal_correlations(phenomena_data),
                    'anomaly_detection': self._detect_statistical_anomalies(phenomena_data),
                    'emergence_indicators': self._calculate_emergence_indicators(phenomena_data),
                    'stability_metrics': self._assess_system_stability(phenomena_data),
                    'prediction_confidence': self._calculate_prediction_confidence(phenomena_data)
                }
                
                # Update baseline for future comparisons
                self._update_anomaly_baseline(phenomena_data)
                
                return analysis_results
                
        except Exception as e:
            logger.error(f"Error in phenomena pattern analysis: {e}")
            return {'error': str(e), 'analysis_timestamp': time.time()}
    
    def _classify_phenomena_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify phenomena into OSH-aligned categories"""
        classifications = {
            'coherence_wave_patterns': [],
            'entropy_cascade_events': [],
            'recursive_boundary_phenomena': [],
            'observer_consensus_formations': [],
            'memory_strain_propagations': [],
            'emergence_indicators': []
        }
        
        for phenomenon_name, phenomenon_data in data.items():
            if isinstance(phenomenon_data, dict):
                strength = phenomenon_data.get('strength', 0.0)
                frequency = phenomenon_data.get('frequency', 0.0)
                
                # Classify based on OSH theoretical framework
                if 'coherence' in phenomenon_name and strength > 0.4:
                    classifications['coherence_wave_patterns'].append({
                        'name': phenomenon_name,
                        'strength': strength,
                        'frequency': frequency,
                        'classification': 'high_coherence_emergence'
                    })
                
                elif 'entropy' in phenomenon_name and strength > 0.6:
                    classifications['entropy_cascade_events'].append({
                        'name': phenomenon_name,
                        'strength': strength,
                        'rate': phenomenon_data.get('rate', 0.0),
                        'classification': 'entropy_destabilization'
                    })
                
                elif 'boundary' in phenomenon_name or 'recursive' in phenomenon_name:
                    classifications['recursive_boundary_phenomena'].append({
                        'name': phenomenon_name,
                        'depth': phenomenon_data.get('depth', 0),
                        'crossings': phenomenon_data.get('crossings', 0),
                        'classification': 'recursive_instability'
                    })
                
                elif 'consensus' in phenomenon_name or 'observer' in phenomenon_name:
                    classifications['observer_consensus_formations'].append({
                        'name': phenomenon_name,
                        'participants': phenomenon_data.get('observer_count', 0),
                        'strength': strength,
                        'classification': 'consciousness_emergence'
                    })
        
        return classifications
    
    def _analyze_temporal_correlations(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze temporal correlations between phenomena"""
        correlations = {}
        
        try:
            # Extract time-series data for correlation analysis
            time_series_data = {}
            for phenomenon_name, phenomenon_data in data.items():
                if isinstance(phenomenon_data, dict) and 'time' in phenomenon_data:
                    time_series_data[phenomenon_name] = phenomenon_data.get('strength', 0.0)
            
            # Calculate cross-correlations
            phenomena_names = list(time_series_data.keys())
            for i, name1 in enumerate(phenomena_names):
                for name2 in phenomena_names[i+1:]:
                    correlation_key = f"{name1}_vs_{name2}"
                    
                    # Simple correlation calculation
                    if len(self.detection_history) > 10:
                        correlation = self._calculate_correlation(name1, name2)
                        correlations[correlation_key] = correlation
            
        except Exception as e:
            logger.error(f"Error in temporal correlation analysis: {e}")
        
        return correlations
    
    def _calculate_correlation(self, name1: str, name2: str) -> float:
        """Calculate correlation between two phenomena over time"""
        try:
            values1 = []
            values2 = []
            
            for historical_data in list(self.detection_history)[-50:]:  # Last 50 entries
                if name1 in historical_data and name2 in historical_data:
                    val1 = historical_data[name1].get('strength', 0.0) if isinstance(historical_data[name1], dict) else 0.0
                    val2 = historical_data[name2].get('strength', 0.0) if isinstance(historical_data[name2], dict) else 0.0
                    values1.append(val1)
                    values2.append(val2)
            
            if len(values1) > 3:
                correlation = np.corrcoef(values1, values2)[0, 1]
                return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating correlation between {name1} and {name2}: {e}")
        
        return 0.0
    
    def _detect_statistical_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect statistical anomalies in phenomena data"""
        anomalies = {
            'detected_anomalies': [],
            'anomaly_score': 0.0,
            'baseline_deviations': {},
            'critical_thresholds_exceeded': []
        }
        
        try:
            for phenomenon_name, phenomenon_data in data.items():
                if isinstance(phenomenon_data, dict):
                    strength = phenomenon_data.get('strength', 0.0)
                    
                    # Check against baseline
                    if phenomenon_name in self.anomaly_baseline:
                        baseline = self.anomaly_baseline[phenomenon_name]
                        deviation = abs(strength - baseline.get('mean', 0.0))
                        threshold = baseline.get('std', 0.1) * 2.5  # 2.5 sigma threshold
                        
                        if deviation > threshold:
                            anomalies['detected_anomalies'].append({
                                'phenomenon': phenomenon_name,
                                'current_strength': strength,
                                'baseline_mean': baseline.get('mean', 0.0),
                                'deviation': deviation,
                                'severity': 'high' if deviation > threshold * 1.5 else 'moderate'
                            })
                    
                    # Check critical thresholds
                    if 'coherence' in phenomenon_name and strength < 0.3:
                        anomalies['critical_thresholds_exceeded'].append({
                            'type': 'coherence_collapse',
                            'phenomenon': phenomenon_name,
                            'value': strength,
                            'threshold': 0.3
                        })
                    elif 'entropy' in phenomenon_name and strength > 0.8:
                        anomalies['critical_thresholds_exceeded'].append({
                            'type': 'entropy_explosion',
                            'phenomenon': phenomenon_name,
                            'value': strength,
                            'threshold': 0.8
                        })
            
            # Calculate overall anomaly score
            anomaly_count = len(anomalies['detected_anomalies'])
            critical_count = len(anomalies['critical_thresholds_exceeded'])
            anomalies['anomaly_score'] = min(1.0, (anomaly_count * 0.1) + (critical_count * 0.2))
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
        
        return anomalies
    
    def _calculate_emergence_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate indicators of emergent behavior"""
        indicators = {
            'complexity_increase': 0.0,
            'information_integration': 0.0,
            'recursive_depth_expansion': 0.0,
            'consciousness_emergence_potential': 0.0,
            'system_criticality': 0.0
        }
        
        try:
            total_phenomena = len(data)
            if total_phenomena == 0:
                return indicators
            
            # Complexity increase indicator
            high_strength_phenomena = sum(1 for d in data.values() 
                                        if isinstance(d, dict) and d.get('strength', 0) > 0.6)
            indicators['complexity_increase'] = high_strength_phenomena / max(1, total_phenomena)
            
            # Information integration (based on correlations)
            correlation_strength = 0.0
            correlation_count = 0
            for phenomenon_name in data.keys():
                for other_name in data.keys():
                    if phenomenon_name != other_name:
                        corr = self._calculate_correlation(phenomenon_name, other_name)
                        if abs(corr) > 0.3:  # Significant correlation
                            correlation_strength += abs(corr)
                            correlation_count += 1
            
            if correlation_count > 0:
                indicators['information_integration'] = correlation_strength / correlation_count
            
            # Recursive depth expansion
            recursive_phenomena = [d for d in data.values() 
                                 if isinstance(d, dict) and 'depth' in d]
            if recursive_phenomena:
                max_depth = max(d.get('depth', 0) for d in recursive_phenomena)
                indicators['recursive_depth_expansion'] = min(1.0, max_depth / 10.0)
            
            # Consciousness emergence potential
            observer_phenomena = [d for name, d in data.items() 
                                if isinstance(d, dict) and 'observer' in name.lower()]
            if observer_phenomena:
                avg_strength = np.mean([d.get('strength', 0) for d in observer_phenomena])
                indicators['consciousness_emergence_potential'] = avg_strength
            
            # System criticality
            critical_phenomena = sum(1 for d in data.values() 
                                   if isinstance(d, dict) and d.get('strength', 0) > 0.8)
            indicators['system_criticality'] = critical_phenomena / max(1, total_phenomena)
            
        except Exception as e:
            logger.error(f"Error calculating emergence indicators: {e}")
        
        return indicators
    
    def _assess_system_stability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system stability based on phenomena"""
        stability_assessment = {
            'overall_stability': 1.0,
            'stability_factors': {},
            'risk_indicators': [],
            'stability_trend': 'stable'
        }
        
        try:
            risk_score = 0.0
            factor_count = 0
            
            for phenomenon_name, phenomenon_data in data.items():
                if isinstance(phenomenon_data, dict):
                    strength = phenomenon_data.get('strength', 0.0)
                    
                    # Assess stability impact
                    if 'collapse' in phenomenon_name or 'critical' in phenomenon_name:
                        risk_score += strength * 0.3
                        stability_assessment['risk_indicators'].append({
                            'type': 'collapse_risk',
                            'phenomenon': phenomenon_name,
                            'risk_level': strength
                        })
                    
                    elif 'instability' in phenomenon_name or 'boundary' in phenomenon_name:
                        risk_score += strength * 0.2
                        stability_assessment['risk_indicators'].append({
                            'type': 'boundary_instability',
                            'phenomenon': phenomenon_name,
                            'risk_level': strength
                        })
                    
                    # Track stability factors
                    if 'coherence' in phenomenon_name:
                        stability_assessment['stability_factors']['coherence'] = 1.0 - strength
                    elif 'consensus' in phenomenon_name:
                        stability_assessment['stability_factors']['consensus'] = strength
                    
                    factor_count += 1
            
            # Calculate overall stability
            stability_assessment['overall_stability'] = max(0.0, 1.0 - risk_score)
            
            # Determine trend
            if len(self.detection_history) > 5:
                recent_stability = [self._calculate_historical_stability(h) 
                                  for h in list(self.detection_history)[-5:]]
                if len(recent_stability) > 2:
                    trend_slope = np.polyfit(range(len(recent_stability)), recent_stability, 1)[0]
                    if trend_slope > 0.1:
                        stability_assessment['stability_trend'] = 'improving'
                    elif trend_slope < -0.1:
                        stability_assessment['stability_trend'] = 'degrading'
            
        except Exception as e:
            logger.error(f"Error in stability assessment: {e}")
        
        return stability_assessment
    
    def _calculate_historical_stability(self, historical_data: Dict[str, Any]) -> float:
        """Calculate stability score for historical data point"""
        try:
            risk_factors = 0.0
            total_phenomena = len(historical_data)
            
            if total_phenomena == 0:
                return 1.0
            
            for phenomenon_data in historical_data.values():
                if isinstance(phenomenon_data, dict):
                    strength = phenomenon_data.get('strength', 0.0)
                    if strength > 0.7:
                        risk_factors += 0.1
            
            return max(0.0, 1.0 - risk_factors)
            
        except Exception:
            return 1.0
    
    def _calculate_prediction_confidence(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence levels for predictions based on phenomena"""
        confidence_metrics = {
            'short_term_prediction': 0.5,
            'medium_term_prediction': 0.3,
            'long_term_prediction': 0.1,
            'pattern_recognition_confidence': 0.5,
            'anomaly_prediction_confidence': 0.4
        }
        
        try:
            # Base confidence on data consistency and historical patterns
            if len(self.detection_history) > 20:
                consistency_score = self._calculate_data_consistency()
                pattern_strength = self._calculate_pattern_strength(data)
                
                confidence_metrics['short_term_prediction'] = min(0.9, consistency_score * 0.8)
                confidence_metrics['medium_term_prediction'] = min(0.7, consistency_score * 0.6)
                confidence_metrics['long_term_prediction'] = min(0.5, consistency_score * 0.4)
                confidence_metrics['pattern_recognition_confidence'] = pattern_strength
                confidence_metrics['anomaly_prediction_confidence'] = min(0.8, consistency_score * pattern_strength)
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {e}")
        
        return confidence_metrics
    
    def _calculate_data_consistency(self) -> float:
        """Calculate consistency score across historical data"""
        try:
            if len(self.detection_history) < 5:
                return 0.5
            
            # Analyze variance in phenomena detection
            phenomenon_counts = defaultdict(int)
            for historical_data in self.detection_history:
                for phenomenon_name in historical_data.keys():
                    phenomenon_counts[phenomenon_name] += 1
            
            # Calculate consistency based on regular detection patterns
            total_history_points = len(self.detection_history)
            consistency_scores = []
            
            for phenomenon_name, count in phenomenon_counts.items():
                consistency = count / total_history_points
                consistency_scores.append(consistency)
            
            if consistency_scores:
                return np.mean(consistency_scores)
            
        except Exception as e:
            logger.debug(f"Error calculating data consistency: {e}")
        
        return 0.5
    
    def _calculate_pattern_strength(self, data: Dict[str, Any]) -> float:
        """Calculate strength of detected patterns"""
        try:
            if not data:
                return 0.0
            
            strengths = []
            for phenomenon_data in data.values():
                if isinstance(phenomenon_data, dict):
                    strength = phenomenon_data.get('strength', 0.0)
                    strengths.append(strength)
            
            if strengths:
                return np.mean(strengths)
            
        except Exception:
            pass
        
        return 0.0
    
    def _update_anomaly_baseline(self, data: Dict[str, Any]):
        """Update baseline statistics for anomaly detection"""
        try:
            for phenomenon_name, phenomenon_data in data.items():
                if isinstance(phenomenon_data, dict):
                    strength = phenomenon_data.get('strength', 0.0)
                    
                    if phenomenon_name not in self.anomaly_baseline:
                        self.anomaly_baseline[phenomenon_name] = {
                            'values': deque(maxlen=100),
                            'mean': 0.0,
                            'std': 0.1
                        }
                    
                    baseline = self.anomaly_baseline[phenomenon_name]
                    baseline['values'].append(strength)
                    
                    # Update statistics
                    if len(baseline['values']) > 5:
                        values_array = np.array(list(baseline['values']))
                        baseline['mean'] = np.mean(values_array)
                        baseline['std'] = max(0.01, np.std(values_array))  # Minimum std to avoid division by zero
        
        except Exception as e:
            logger.debug(f"Error updating anomaly baseline: {e}")
    
    def record_detection(self, phenomena_data: Dict[str, Any]):
        """Record phenomena detection for historical analysis"""
        try:
            with self.lock:
                timestamped_data = {
                    'timestamp': time.time(),
                    'phenomena': phenomena_data.copy()
                }
                self.detection_history.append(timestamped_data)
                
        except Exception as e:
            logger.error(f"Error recording phenomena detection: {e}")


class AdvancedReportDataProcessor:
    """Advanced processor for report data validation and enhancement"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data_history = deque(maxlen=self.config.get('history_length', 2000))
        self.validation_cache = {}
        self.data_quality_metrics = defaultdict(list)
        self.lock = threading.RLock()
        
        # Validation thresholds
        self.completeness_threshold = self.config.get('completeness_threshold', 0.8)
        self.consistency_threshold = self.config.get('consistency_threshold', 0.7)
        self.temporal_gap_threshold = self.config.get('temporal_gap_threshold', 5.0)
        
        logger.info("Advanced Report Data Processor initialized")
    
    def validate_report_data(self, step_results: Dict[str, Any]) -> ReportDataValidation:
        """Comprehensive validation of report data"""
        try:
            validation = ReportDataValidation(is_valid=True, data_completeness=1.0, temporal_continuity=True)
            
            # Check data completeness
            required_fields = ['changes']
            missing_fields = []
            
            for field in required_fields:
                if field not in step_results:
                    missing_fields.append(field)
            
            validation.missing_fields = missing_fields
            validation.data_completeness = 1.0 - (len(missing_fields) / len(required_fields))
            
            # Validate changes structure
            if 'changes' in step_results:
                changes = step_results['changes']
                validation.metric_consistency = self._validate_metric_consistency(changes)
            
            # Check temporal continuity
            validation.temporal_continuity = self._check_temporal_continuity(step_results)
            
            # Calculate overall data quality score
            validation.data_quality_score = self._calculate_data_quality_score(validation)
            
            # Determine overall validity
            validation.is_valid = (
                validation.data_completeness >= self.completeness_threshold and
                validation.temporal_continuity and
                validation.data_quality_score >= self.consistency_threshold
            )
            
            # Record validation metrics
            self._record_validation_metrics(validation)
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating report data: {e}")
            return ReportDataValidation(
                is_valid=False,
                data_completeness=0.0,
                temporal_continuity=False,
                validation_errors=[str(e)]
            )
    
    def _validate_metric_consistency(self, changes: Dict[str, Any]) -> Dict[str, float]:
        """Validate consistency of metrics within the changes data"""
        consistency_scores = {}
        
        try:
            # Check coherence metrics
            if 'coherence' in changes:
                coherence_value = changes['coherence']
                if isinstance(coherence_value, (int, float)):
                    consistency_scores['coherence'] = 1.0 if 0.0 <= coherence_value <= 1.0 else 0.5
                else:
                    consistency_scores['coherence'] = 0.0
            
            # Check memory metrics
            if 'memory' in changes and isinstance(changes['memory'], dict):
                memory_data = changes['memory']
                strain_valid = True
                
                if 'strain' in memory_data:
                    strain = memory_data['strain']
                    strain_valid = isinstance(strain, (int, float)) and 0.0 <= strain <= 1.0
                
                consistency_scores['memory'] = 1.0 if strain_valid else 0.5
            
            # Check observer metrics
            if 'observers' in changes and isinstance(changes['observers'], dict):
                observer_data = changes['observers']
                observer_valid = True
                
                if 'count' in observer_data:
                    count = observer_data['count']
                    observer_valid = isinstance(count, int) and count >= 0
                
                if 'consensus' in observer_data and observer_valid:
                    consensus = observer_data['consensus']
                    observer_valid = isinstance(consensus, (int, float)) and 0.0 <= consensus <= 1.0
                
                consistency_scores['observers'] = 1.0 if observer_valid else 0.5
            
            # Check recursive metrics
            if 'recursive' in changes and isinstance(changes['recursive'], dict):
                recursive_data = changes['recursive']
                recursive_valid = True
                
                if 'depth' in recursive_data:
                    depth = recursive_data['depth']
                    recursive_valid = isinstance(depth, int) and depth >= 0
                
                consistency_scores['recursive'] = 1.0 if recursive_valid else 0.5
            
            # Check field metrics
            if 'field' in changes and isinstance(changes['field'], dict):
                field_data = changes['field']
                field_valid = True
                
                if 'energy' in field_data:
                    energy = field_data['energy']
                    field_valid = isinstance(energy, (int, float)) and energy >= 0.0
                
                consistency_scores['field'] = 1.0 if field_valid else 0.5
            
        except Exception as e:
            logger.error(f"Error validating metric consistency: {e}")
        
        return consistency_scores
    
    def _check_temporal_continuity(self, step_results: Dict[str, Any]) -> bool:
        """Check temporal continuity of the data"""
        try:
            current_timestamp = time.time()
            
            # Check if we have recent data to compare against
            if len(self.data_history) > 0:
                last_entry = self.data_history[-1]
                last_timestamp = last_entry.get('timestamp', current_timestamp)
                
                time_gap = current_timestamp - last_timestamp
                
                # Check if the time gap is reasonable
                if time_gap > self.temporal_gap_threshold:
                    logger.warning(f"Large temporal gap detected: {time_gap:.2f} seconds")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking temporal continuity: {e}")
            return False
    
    def _calculate_data_quality_score(self, validation: ReportDataValidation) -> float:
        """Calculate overall data quality score"""
        try:
            quality_factors = []
            
            # Data completeness factor
            quality_factors.append(validation.data_completeness * 0.3)
            
            # Temporal continuity factor
            quality_factors.append(1.0 if validation.temporal_continuity else 0.0) * 0.2
            
            # Metric consistency factor
            if validation.metric_consistency:
                consistency_avg = np.mean(list(validation.metric_consistency.values()))
                quality_factors.append(consistency_avg * 0.3)
            
            # Historical consistency factor (if we have history)
            if len(self.data_history) > 5:
                historical_consistency = self._calculate_historical_consistency()
                quality_factors.append(historical_consistency * 0.2)
            else:
                quality_factors.append(0.5 * 0.2)  # Neutral score for insufficient history
            
            return sum(quality_factors)
            
        except Exception as e:
            logger.error(f"Error calculating data quality score: {e}")
            return 0.5
    
    def _calculate_historical_consistency(self) -> float:
        """Calculate consistency with historical data patterns"""
        try:
            if len(self.data_history) < 5:
                return 0.5
            
            # Analyze variance in key metrics over time
            coherence_values = []
            strain_values = []
            
            for entry in list(self.data_history)[-20:]:  # Last 20 entries
                step_results = entry.get('step_results', {})
                changes = step_results.get('changes', {})
                
                if 'coherence' in changes and isinstance(changes['coherence'], (int, float)):
                    coherence_values.append(changes['coherence'])
                
                if 'memory' in changes and isinstance(changes['memory'], dict):
                    strain = changes['memory'].get('strain')
                    if isinstance(strain, (int, float)):
                        strain_values.append(strain)
            
            consistency_scores = []
            
            # Check coherence consistency
            if len(coherence_values) > 3:
                coherence_variance = np.var(coherence_values)
                consistency_scores.append(max(0.0, 1.0 - coherence_variance))  # Lower variance = higher consistency
            
            # Check strain consistency
            if len(strain_values) > 3:
                strain_variance = np.var(strain_values)
                consistency_scores.append(max(0.0, 1.0 - strain_variance))
            
            return np.mean(consistency_scores) if consistency_scores else 0.5
            
        except Exception as e:
            logger.debug(f"Error calculating historical consistency: {e}")
            return 0.5
    
    def _record_validation_metrics(self, validation: ReportDataValidation):
        """Record validation metrics for analysis"""
        try:
            with self.lock:
                self.data_quality_metrics['completeness'].append(validation.data_completeness)
                self.data_quality_metrics['quality_score'].append(validation.data_quality_score)
                self.data_quality_metrics['temporal_continuity'].append(1.0 if validation.temporal_continuity else 0.0)
                
                # Limit history size
                for metric_list in self.data_quality_metrics.values():
                    if len(metric_list) > 1000:
                        metric_list.pop(0)
        
        except Exception as e:
            logger.debug(f"Error recording validation metrics: {e}")
    
    def enhance_report_data(self, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance report data with additional analytical information"""
        try:
            enhanced_data = step_results.copy()
            
            # Add timestamp if not present
            if 'timestamp' not in enhanced_data:
                enhanced_data['timestamp'] = time.time()
            
            # Add data quality metrics
            validation = self.validate_report_data(step_results)
            enhanced_data['data_quality'] = {
                'completeness': validation.data_completeness,
                'quality_score': validation.data_quality_score,
                'temporal_continuity': validation.temporal_continuity,
                'validation_errors': validation.validation_errors
            }
            
            # Add trend analysis if we have sufficient history
            if len(self.data_history) > 10:
                enhanced_data['trend_analysis'] = self._calculate_trend_analysis()
            
            # Add correlation analysis
            if len(self.data_history) > 5:
                enhanced_data['correlation_analysis'] = self._calculate_correlation_analysis(step_results)
            
            # Record the data
            self.record_data_point(enhanced_data)
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error enhancing report data: {e}")
            return step_results
    
    def _calculate_trend_analysis(self) -> Dict[str, Any]:
        """Calculate trend analysis from historical data"""
        trends = {
            'coherence_trend': 'stable',
            'entropy_trend': 'stable',
            'strain_trend': 'stable',
            'trend_confidence': 0.5
        }
        
        try:
            # Extract time series for key metrics
            coherence_series = []
            strain_series = []
            timestamps = []
            
            for entry in list(self.data_history)[-50:]:  # Last 50 entries
                timestamp = entry.get('timestamp', 0)
                step_results = entry.get('step_results', {})
                changes = step_results.get('changes', {})
                
                timestamps.append(timestamp)
                
                # Extract coherence
                coherence = changes.get('coherence', 0.5)
                if isinstance(coherence, (int, float)):
                    coherence_series.append(coherence)
                else:
                    coherence_series.append(0.5)
                
                # Extract strain
                memory_data = changes.get('memory', {})
                strain = memory_data.get('strain', 0.5) if isinstance(memory_data, dict) else 0.5
                if isinstance(strain, (int, float)):
                    strain_series.append(strain)
                else:
                    strain_series.append(0.5)
            
            # Calculate trends using simple linear regression
            if len(coherence_series) > 5:
                coherence_slope = np.polyfit(range(len(coherence_series)), coherence_series, 1)[0]
                if coherence_slope > 0.01:
                    trends['coherence_trend'] = 'increasing'
                elif coherence_slope < -0.01:
                    trends['coherence_trend'] = 'decreasing'
            
            if len(strain_series) > 5:
                strain_slope = np.polyfit(range(len(strain_series)), strain_series, 1)[0]
                if strain_slope > 0.01:
                    trends['strain_trend'] = 'increasing'
                elif strain_slope < -0.01:
                    trends['strain_trend'] = 'decreasing'
            
            # Calculate trend confidence based on data consistency
            trends['trend_confidence'] = min(1.0, len(coherence_series) / 20.0)
            
        except Exception as e:
            logger.debug(f"Error calculating trend analysis: {e}")
        
        return trends
    
    def _calculate_correlation_analysis(self, current_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate correlations between different metrics"""
        correlations = {}
        
        try:
            # Extract current metrics
            changes = current_data.get('changes', {})
            current_coherence = changes.get('coherence', 0.5)
            current_strain = changes.get('memory', {}).get('strain', 0.5) if isinstance(changes.get('memory'), dict) else 0.5
            
            # Extract historical data for correlation
            coherence_values = []
            strain_values = []
            observer_counts = []
            
            for entry in list(self.data_history)[-30:]:
                step_results = entry.get('step_results', {})
                changes = step_results.get('changes', {})
                
                coherence = changes.get('coherence', 0.5)
                if isinstance(coherence, (int, float)):
                    coherence_values.append(coherence)
                
                memory_data = changes.get('memory', {})
                strain = memory_data.get('strain', 0.5) if isinstance(memory_data, dict) else 0.5
                if isinstance(strain, (int, float)):
                    strain_values.append(strain)
                
                observer_data = changes.get('observers', {})
                count = observer_data.get('count', 0) if isinstance(observer_data, dict) else 0
                if isinstance(count, int):
                    observer_counts.append(count)
            
            # Calculate correlations
            if len(coherence_values) > 5 and len(strain_values) > 5:
                min_length = min(len(coherence_values), len(strain_values))
                coherence_array = np.array(coherence_values[:min_length])
                strain_array = np.array(strain_values[:min_length])
                
                correlation = np.corrcoef(coherence_array, strain_array)[0, 1]
                if not np.isnan(correlation):
                    correlations['coherence_strain_correlation'] = float(correlation)
            
            if len(coherence_values) > 5 and len(observer_counts) > 5:
                min_length = min(len(coherence_values), len(observer_counts))
                coherence_array = np.array(coherence_values[:min_length])
                observer_array = np.array(observer_counts[:min_length])
                
                correlation = np.corrcoef(coherence_array, observer_array)[0, 1]
                if not np.isnan(correlation):
                    correlations['coherence_observer_correlation'] = float(correlation)
            
        except Exception as e:
            logger.debug(f"Error calculating correlation analysis: {e}")
        
        return correlations
    
    def record_data_point(self, enhanced_data: Dict[str, Any]):
        """Record enhanced data point for historical analysis"""
        try:
            with self.lock:
                data_point = {
                    'timestamp': enhanced_data.get('timestamp', time.time()),
                    'step_results': enhanced_data
                }
                self.data_history.append(data_point)
                
        except Exception as e:
            logger.error(f"Error recording data point: {e}")
    
    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get summary of data quality metrics"""
        try:
            with self.lock:
                summary = {
                    'average_completeness': 0.0,
                    'average_quality_score': 0.0,
                    'temporal_continuity_rate': 0.0,
                    'total_data_points': len(self.data_history),
                    'validation_error_rate': 0.0
                }
                
                if self.data_quality_metrics:
                    if 'completeness' in self.data_quality_metrics and self.data_quality_metrics['completeness']:
                        summary['average_completeness'] = np.mean(self.data_quality_metrics['completeness'])
                    
                    if 'quality_score' in self.data_quality_metrics and self.data_quality_metrics['quality_score']:
                        summary['average_quality_score'] = np.mean(self.data_quality_metrics['quality_score'])
                    
                    if 'temporal_continuity' in self.data_quality_metrics and self.data_quality_metrics['temporal_continuity']:
                        summary['temporal_continuity_rate'] = np.mean(self.data_quality_metrics['temporal_continuity'])
                
                return summary
                
        except Exception as e:
            logger.error(f"Error getting data quality summary: {e}")
            return {'error': str(e)}


class UpdateSystemManager:
    """Central manager for all update operations with comprehensive monitoring"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = UpdateMetrics()
        self.phenomena_analyzer = AdvancedPhenomenaAnalyzer(self.config.get('phenomena_config', {}))
        self.report_processor = AdvancedReportDataProcessor(self.config.get('report_config', {}))
        
        # Performance monitoring
        self.update_times = deque(maxlen=100)
        self.error_history = deque(maxlen=50)
        self.last_update_timestamp = 0.0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Async processing
        self.enable_async = self.config.get('enable_async_processing', False)
        self.thread_pool = ThreadPoolExecutor(max_workers=2) if self.enable_async else None
        
        logger.info("Update System Manager initialized with advanced monitoring capabilities")
    
    def update_phenomena_detector_advanced(self, current_metrics, phenomena_detector) -> PhenomenaValidationResult:
        """Enhanced phenomena detector update with comprehensive validation"""
        start_time = time.time()
        result = PhenomenaValidationResult(is_valid=False, detected_count=0)
        
        try:
            with self.lock:
                self.metrics.total_updates += 1
                self.metrics.phenomena_updates += 1
                
                # Validate input parameters
                if not self._validate_metrics_object(current_metrics):
                    result.validation_errors.append("Invalid current_metrics object")
                    return result
                
                if not self._validate_phenomena_detector(phenomena_detector):
                    result.validation_errors.append("Invalid phenomena_detector object")
                    return result
                
                # Extract comprehensive metrics data
                metrics_data = self._extract_comprehensive_metrics(current_metrics)
                
                # Record state in phenomena detector
                phenomena_detector.record_state(
                    time=metrics_data['timestamp'],
                    coherence_values=metrics_data['coherence_values'],
                    entropy_values=metrics_data['entropy_values'],
                    strain_values=metrics_data['strain_values'],
                    observer_data=metrics_data['observer_data'],
                    field_data=metrics_data['field_data']
                )
                
                # Detect phenomena
                detected_phenomena = phenomena_detector.detect_phenomena()
                result.detected_count = len(detected_phenomena)
                
                # Advanced analysis
                if detected_phenomena:
                    analysis_results = self.phenomena_analyzer.analyze_phenomena_patterns(detected_phenomena)
                    
                    # Update current metrics with enhanced data
                    self._update_metrics_with_phenomena(current_metrics, detected_phenomena, analysis_results)
                    
                    # Validate results
                    result = self._validate_phenomena_results(detected_phenomena, analysis_results)
                    
                    # Record for historical analysis
                    self.phenomena_analyzer.record_detection(detected_phenomena)
                
                # Update performance metrics
                update_time = time.time() - start_time
                self._update_performance_metrics(update_time, True, 'phenomena')
                
                result.is_valid = True
                
        except Exception as e:
            error_msg = f"Error in advanced phenomena detector update: {e}"
            logger.error(error_msg)
            global_error_manager.error("update_utils", 0, 0, error_msg)
            
            result.validation_errors.append(error_msg)
            self.metrics.failed_updates += 1
            self._record_error(e, 'phenomena_update')
            
            update_time = time.time() - start_time
            self._update_performance_metrics(update_time, False, 'phenomena')
        
        return result
    
    def update_report_builder_advanced(self, current_metrics, report_builder, event) -> ReportDataValidation:
        """Enhanced report builder update with comprehensive data validation"""
        start_time = time.time()
        validation = ReportDataValidation(is_valid=False, data_completeness=0.0, temporal_continuity=False)
        
        try:
            with self.lock:
                self.metrics.total_updates += 1
                self.metrics.report_updates += 1
                
                # Validate inputs
                if not self._validate_metrics_object(current_metrics):
                    validation.validation_errors.append("Invalid current_metrics object")
                    return validation
                
                if not self._validate_report_builder(report_builder):
                    validation.validation_errors.append("Invalid report_builder object")
                    return validation
                
                # Create comprehensive step results
                step_results = self._create_comprehensive_step_results(current_metrics, event)
                
                # Validate and enhance data
                validation = self.report_processor.validate_report_data(step_results)
                
                if validation.is_valid:
                    # Enhance data with additional analytics
                    enhanced_step_results = self.report_processor.enhance_report_data(step_results)
                    
                    # Collect data in report builder
                    timestamp = enhanced_step_results.get('timestamp', time.time())
                    report_builder.collect_step_data(timestamp, enhanced_step_results)
                    
                    # Update correlation tracking
                    self.metrics.correlation_analysis_count += 1
                else:
                    logger.warning(f"Report data validation failed: {validation.validation_errors}")
                    self.metrics.data_validation_errors += 1
                
                # Update performance metrics
                update_time = time.time() - start_time
                self._update_performance_metrics(update_time, validation.is_valid, 'report')
                
        except Exception as e:
            error_msg = f"Error in advanced report builder update: {e}"
            logger.error(error_msg)
            global_error_manager.error("update_utils", 0, 0, error_msg)
            
            validation.validation_errors.append(error_msg)
            self.metrics.failed_updates += 1
            self._record_error(e, 'report_update')
            
            update_time = time.time() - start_time
            self._update_performance_metrics(update_time, False, 'report')
        
        return validation
    
    def _validate_metrics_object(self, metrics) -> bool:
        """Validate that the metrics object has required attributes"""
        try:
            required_attrs = ['timestamp', 'coherence', 'entropy', 'strain']
            for attr in required_attrs:
                if not hasattr(metrics, attr):
                    logger.debug(f"Metrics object missing required attribute: {attr}")
                    return False
            return True
        except Exception:
            return False
    
    def _validate_phenomena_detector(self, detector) -> bool:
        """Validate that the phenomena detector has required methods"""
        try:
            required_methods = ['record_state', 'detect_phenomena']
            for method in required_methods:
                if not hasattr(detector, method) or not callable(getattr(detector, method)):
                    logger.debug(f"Phenomena detector missing required method: {method}")
                    return False
            return True
        except Exception:
            return False
    
    def _validate_report_builder(self, builder) -> bool:
        """Validate that the report builder has required methods"""
        try:
            required_methods = ['collect_step_data']
            for method in required_methods:
                if not hasattr(builder, method) or not callable(getattr(builder, method)):
                    logger.debug(f"Report builder missing required method: {method}")
                    return False
            return True
        except Exception:
            return False
    
    def _extract_comprehensive_metrics(self, current_metrics) -> Dict[str, Any]:
        """Extract comprehensive metrics data with safe attribute access"""
        try:
            metrics_data = {
                'timestamp': getattr(current_metrics, 'timestamp', time.time()),
                'coherence_values': {},
                'entropy_values': {},
                'strain_values': {},
                'observer_data': {},
                'field_data': {}
            }
            
            # Extract core metrics
            coherence = getattr(current_metrics, 'coherence', 0.5)
            entropy = getattr(current_metrics, 'entropy', 0.5)
            strain = getattr(current_metrics, 'strain', 0.5)
            
            metrics_data['coherence_values']['global'] = float(coherence) if isinstance(coherence, (int, float)) else 0.5
            metrics_data['entropy_values']['global'] = float(entropy) if isinstance(entropy, (int, float)) else 0.5
            metrics_data['strain_values']['global'] = float(strain) if isinstance(strain, (int, float)) else 0.5
            
            # Extract observer data
            observer_count = getattr(current_metrics, 'observer_count', 0)
            observer_consensus = getattr(current_metrics, 'observer_consensus', 0.5)
            active_observers = getattr(current_metrics, 'active_observers', 0)
            
            metrics_data['observer_data'] = {
                'count': int(observer_count) if isinstance(observer_count, (int, float)) else 0,
                'consensus': float(observer_consensus) if isinstance(observer_consensus, (int, float)) else 0.5,
                'active_count': int(active_observers) if isinstance(active_observers, (int, float)) else 0
            }
            
            # Extract field data
            field_count = getattr(current_metrics, 'field_count', 0)
            total_field_energy = getattr(current_metrics, 'total_field_energy', 0.0)
            
            metrics_data['field_data'] = {
                'count': int(field_count) if isinstance(field_count, (int, float)) else 0,
                'total_energy': float(total_field_energy) if isinstance(total_field_energy, (int, float)) else 0.0
            }
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive metrics: {e}")
            return {
                'timestamp': time.time(),
                'coherence_values': {'global': 0.5},
                'entropy_values': {'global': 0.5},
                'strain_values': {'global': 0.5},
                'observer_data': {'count': 0, 'consensus': 0.5, 'active_count': 0},
                'field_data': {'count': 0, 'total_energy': 0.0}
            }
    
    def _update_metrics_with_phenomena(self, current_metrics, phenomena_data: Dict[str, Any], analysis_results: Dict[str, Any]):
        """Update current metrics with detected phenomena and analysis results"""
        try:
            # Update emergent phenomena list
            phenomena_names = list(phenomena_data.keys())
            if hasattr(current_metrics, 'emergent_phenomena'):
                current_metrics.emergent_phenomena = phenomena_names
            
            # Calculate average phenomena strength
            strengths = []
            for phenomenon_data in phenomena_data.values():
                if isinstance(phenomenon_data, dict) and 'strength' in phenomenon_data:
                    strength = phenomenon_data['strength']
                    if isinstance(strength, (int, float)):
                        strengths.append(strength)
            
            if strengths and hasattr(current_metrics, 'phenomena_strength'):
                current_metrics.phenomena_strength = np.mean(strengths)
            
            # Update with analysis results
            if 'emergence_indicators' in analysis_results:
                emergence = analysis_results['emergence_indicators']
                if hasattr(current_metrics, 'emergence_index'):
                    current_metrics.emergence_index = emergence.get('complexity_increase', 0.0)
                if hasattr(current_metrics, 'consciousness_quotient'):
                    current_metrics.consciousness_quotient = emergence.get('consciousness_emergence_potential', 0.0)
            
            # Update stability metrics
            if 'stability_metrics' in analysis_results:
                stability = analysis_results['stability_metrics']
                if hasattr(current_metrics, 'temporal_stability'):
                    current_metrics.temporal_stability = stability.get('overall_stability', 1.0)
            
        except Exception as e:
            logger.error(f"Error updating metrics with phenomena: {e}")
    
    def _validate_phenomena_results(self, phenomena_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> PhenomenaValidationResult:
        """Validate phenomena detection results"""
        result = PhenomenaValidationResult(is_valid=True, detected_count=len(phenomena_data))
        
        try:
            # Validate strength distribution
            strengths = {}
            for name, data in phenomena_data.items():
                if isinstance(data, dict) and 'strength' in data:
                    strength = data['strength']
                    if isinstance(strength, (int, float)) and 0.0 <= strength <= 1.0:
                        strengths[name] = strength
                    else:
                        result.anomaly_flags.append(f"Invalid strength for {name}: {strength}")
            
            result.strength_distribution = strengths
            
            # Validate temporal consistency
            if 'temporal_correlation' in analysis_results:
                correlations = analysis_results['temporal_correlation']
                if isinstance(correlations, dict):
                    correlation_values = [v for v in correlations.values() if isinstance(v, (int, float))]
                    if correlation_values:
                        result.temporal_consistency = np.mean([abs(v) for v in correlation_values])
            
            # Validate correlation score
            if 'pattern_classification' in analysis_results:
                patterns = analysis_results['pattern_classification']
                if isinstance(patterns, dict):
                    pattern_count = sum(len(v) for v in patterns.values() if isinstance(v, list))
                    result.correlation_score = min(1.0, pattern_count / 10.0)  # Normalize to [0,1]
            
            # Check for anomalies
            if 'anomaly_detection' in analysis_results:
                anomalies = analysis_results['anomaly_detection']
                if isinstance(anomalies, dict):
                    detected_anomalies = anomalies.get('detected_anomalies', [])
                    if detected_anomalies:
                        result.anomaly_flags.extend([a.get('phenomenon', 'unknown') for a in detected_anomalies])
            
            # Overall validation
            result.is_valid = (
                len(result.validation_errors) == 0 and
                result.detected_count >= 0 and
                result.temporal_consistency >= 0.0
            )
            
        except Exception as e:
            logger.error(f"Error validating phenomena results: {e}")
            result.validation_errors.append(str(e))
            result.is_valid = False
        
        return result
    
    def _create_comprehensive_step_results(self, current_metrics, event) -> Dict[str, Any]:
        """Create comprehensive step results for report builder"""
        try:
            step_results = {
                'timestamp': getattr(current_metrics, 'timestamp', time.time()),
                'event': event,
                'changes': {}
            }
            
            # Core metrics
            changes = step_results['changes']
            changes['coherence'] = getattr(current_metrics, 'coherence', 0.5)
            
            # Memory data
            memory_data = {}
            memory_data['strain'] = getattr(current_metrics, 'strain', 0.5)
            memory_data['regions'] = getattr(current_metrics, 'memory_regions', 0)
            changes['memory'] = memory_data
            
            # Observer data
            observer_data = {}
            observer_data['count'] = getattr(current_metrics, 'observer_count', 0)
            observer_data['consensus'] = getattr(current_metrics, 'observer_consensus', 0.5)
            changes['observers'] = observer_data
            
            # Recursive data
            recursive_data = {}
            recursive_data['depth'] = getattr(current_metrics, 'recursion_depth', 0)
            recursive_data['boundary_crossings'] = getattr(current_metrics, 'boundary_crossings', 0)
            changes['recursive'] = recursive_data
            
            # Field data
            field_data = {}
            field_data['energy'] = getattr(current_metrics, 'total_field_energy', 0.0)
            field_data['evolution_steps'] = getattr(current_metrics, 'field_evolution_steps', 0)
            changes['field'] = field_data
            
            return step_results
            
        except Exception as e:
            logger.error(f"Error creating comprehensive step results: {e}")
            return {
                'timestamp': time.time(),
                'event': event,
                'changes': {
                    'coherence': 0.5,
                    'memory': {'strain': 0.5, 'regions': 0},
                    'observers': {'count': 0, 'consensus': 0.5},
                    'recursive': {'depth': 0, 'boundary_crossings': 0},
                    'field': {'energy': 0.0, 'evolution_steps': 0}
                }
            }
    
    def _update_performance_metrics(self, update_time: float, success: bool, update_type: str):
        """Update performance metrics"""
        try:
            with self.lock:
                self.update_times.append(update_time)
                
                # Update averages
                if len(self.update_times) > 0:
                    self.metrics.average_update_time = np.mean(list(self.update_times))
                    self.metrics.peak_update_time = max(self.metrics.peak_update_time, update_time)
                
                # Update frequency
                current_time = time.time()
                if self.last_update_timestamp > 0:
                    time_delta = current_time - self.last_update_timestamp
                    if time_delta > 0:
                        self.metrics.update_frequency = 1.0 / time_delta
                
                self.last_update_timestamp = current_time
                self.metrics.last_update_time = update_time
                
                # Update error rate
                if not success:
                    self.metrics.failed_updates += 1
                
                total_updates = max(1, self.metrics.total_updates)
                self.metrics.error_rate = self.metrics.failed_updates / total_updates
                
        except Exception as e:
            logger.debug(f"Error updating performance metrics: {e}")
    
    def _record_error(self, error: Exception, context: str):
        """Record error for analysis"""
        try:
            error_record = {
                'timestamp': time.time(),
                'error': str(error),
                'context': context,
                'traceback': traceback.format_exc()
            }
            self.error_history.append(error_record)
        except Exception:
            pass  # Don't let error recording cause additional errors
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            with self.lock:
                summary = {
                    'total_updates': self.metrics.total_updates,
                    'phenomena_updates': self.metrics.phenomena_updates,
                    'report_updates': self.metrics.report_updates,
                    'failed_updates': self.metrics.failed_updates,
                    'success_rate': 1.0 - self.metrics.error_rate,
                    'average_update_time': self.metrics.average_update_time,
                    'peak_update_time': self.metrics.peak_update_time,
                    'update_frequency': self.metrics.update_frequency,
                    'data_validation_errors': self.metrics.data_validation_errors,
                    'correlation_analysis_count': self.metrics.correlation_analysis_count,
                    'anomaly_detections': self.metrics.anomaly_detections
                }
                
                # Add data quality summary
                if self.report_processor:
                    summary['data_quality'] = self.report_processor.get_data_quality_summary()
                
                return summary
                
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            logger.info("Update System Manager cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global update system manager instance
_update_manager = None
_manager_lock = threading.Lock()


def get_update_manager(config: Optional[Dict[str, Any]] = None) -> UpdateSystemManager:
    """Get or create the global update system manager"""
    global _update_manager
    
    with _manager_lock:
        if _update_manager is None:
            _update_manager = UpdateSystemManager(config)
    
    return _update_manager


# Main update functions (backward compatibility)
def update_phenomena_detector(current_metrics, phenomena_detector) -> None:
    """
    Enhanced phenomena detector update with comprehensive validation and analysis.
    
    Args:
        current_metrics: Object containing live simulation metrics
        phenomena_detector: Phenomena detector with record_state and detect_phenomena methods
    """
    try:
        manager = get_update_manager()
        result = manager.update_phenomena_detector_advanced(current_metrics, phenomena_detector)
        
        if not result.is_valid:
            logger.warning(f"Phenomena detector update validation failed: {result.validation_errors}")
            
    except Exception as e:
        error_msg = f"Error in phenomena detector update: {e}"
        logger.error(error_msg)
        global_error_manager.error("update_utils", 0, 0, error_msg)


def update_report_builder(current_metrics, report_builder, event) -> None:
    """
    Enhanced report builder update with comprehensive data validation and enhancement.
    
    Args:
        current_metrics: Object containing live simulation metrics
        report_builder: Report builder with collect_step_data method
        event: The triggering simulation event
    """
    try:
        manager = get_update_manager()
        validation = manager.update_report_builder_advanced(current_metrics, report_builder, event)
        
        if not validation.is_valid:
            logger.warning(f"Report builder update validation failed: {validation.validation_errors}")
            
    except Exception as e:
        error_msg = f"Error in report builder update: {e}"
        logger.error(error_msg)
        global_error_manager.error("update_utils", 0, 0, error_msg)


def get_update_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary for update operations"""
    try:
        manager = get_update_manager()
        return manager.get_performance_summary()
    except Exception as e:
        logger.error(f"Error getting update performance summary: {e}")
        return {'error': str(e)}


def cleanup_update_system():
    """Cleanup the update system resources"""
    global _update_manager
    
    try:
        with _manager_lock:
            if _update_manager:
                _update_manager.cleanup()
                _update_manager = None
        logger.info("Update system cleanup completed")
    except Exception as e:
        logger.error(f"Error during update system cleanup: {e}")


# Performance monitoring decorator
def monitor_update_performance(func):
    """Decorator to monitor update function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            with performance_profiler.timed_step(f"update_{func.__name__}"):
                result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in monitored update function {func.__name__}: {e}")
            raise
        finally:
            execution_time = time.time() - start_time
            logger.debug(f"Update function {func.__name__} executed in {execution_time:.4f} seconds")
    
    return wrapper


# Initialize logging
logger.info("Update utilities module loaded with enterprise-grade capabilities")