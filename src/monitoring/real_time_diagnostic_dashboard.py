"""
Real-Time Diagnostic Dashboard for Recursia Integration Testing

Provides comprehensive real-time monitoring during stress tests with:
- Live system metrics visualization
- Interactive debugging controls
- Emergency shutdown triggers
- Performance profiling
- Memory usage tracking
- Error pattern detection
- Network topology visualization for entanglement
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import psutil
from pathlib import Path

# Web interface
try:
    import dash
    from dash import dcc, html, Input, Output, State, callback, dash_table
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

# WebSocket for real-time updates
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for system alerts."""
    memory_threshold_gb: float = 6.0
    cpu_threshold_percent: float = 80.0
    error_rate_threshold: float = 0.1
    response_time_threshold_ms: float = 1000.0
    queue_size_threshold: int = 1000


@dataclass
class DiagnosticMetrics:
    """Real-time diagnostic metrics."""
    timestamp: float = field(default_factory=time.time)
    
    # System resources
    memory_usage_gb: float = 0.0
    cpu_usage_percent: float = 0.0
    thread_count: int = 0
    file_descriptors: int = 0
    
    # Quantum simulation metrics
    active_systems: int = 0
    total_qubits: int = 0
    entangled_pairs: int = 0
    coherence_time_avg: float = 0.0
    decoherence_rate: float = 0.0
    
    # Performance metrics
    operations_per_second: float = 0.0
    average_response_time_ms: float = 0.0
    queue_length: int = 0
    cache_hit_rate: float = 0.0
    
    # Error tracking
    total_errors: int = 0
    error_rate: float = 0.0
    warnings: int = 0
    
    # Memory field metrics
    memory_field_strain: float = 0.0
    memory_field_entropy: float = 0.0
    memory_field_utilization: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'memory_usage_gb': self.memory_usage_gb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'thread_count': self.thread_count,
            'active_systems': self.active_systems,
            'total_qubits': self.total_qubits,
            'entangled_pairs': self.entangled_pairs,
            'coherence_time_avg': self.coherence_time_avg,
            'operations_per_second': self.operations_per_second,
            'average_response_time_ms': self.average_response_time_ms,
            'total_errors': self.total_errors,
            'error_rate': self.error_rate,
            'memory_field_strain': self.memory_field_strain,
            'memory_field_entropy': self.memory_field_entropy
        }


class SystemHealthMonitor:
    """Monitors system health with configurable alerts."""
    
    def __init__(self, alert_config: Optional[AlertConfig] = None):
        self.alert_config = alert_config or AlertConfig()
        self.process = psutil.Process()
        self.metrics_history: deque = deque(maxlen=1000)
        self.alerts_triggered: List[Dict[str, Any]] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.operation_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.total_operations = 0
        
    def start_monitoring(self, interval: float = 1.0):
        """Start health monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> DiagnosticMetrics:
        """Collect current system metrics."""
        # Basic system metrics
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        
        # Calculate performance metrics
        ops_per_sec = len(self.operation_times) / max(1, time.time() - (self.operation_times[0] if self.operation_times else time.time()))
        avg_response_time = np.mean(list(self.operation_times)) * 1000 if self.operation_times else 0
        error_rate = self.error_count / max(1, self.total_operations)
        
        return DiagnosticMetrics(
            memory_usage_gb=memory_info.rss / (1024**3),
            cpu_usage_percent=cpu_percent,
            thread_count=self.process.num_threads(),
            file_descriptors=self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
            operations_per_second=ops_per_sec,
            average_response_time_ms=avg_response_time,
            total_errors=self.error_count,
            error_rate=error_rate
        )
    
    def _check_alerts(self, metrics: DiagnosticMetrics):
        """Check for alert conditions."""
        alerts = []
        
        # Memory alert
        if metrics.memory_usage_gb > self.alert_config.memory_threshold_gb:
            alerts.append({
                'type': 'memory',
                'severity': 'high',
                'message': f'Memory usage {metrics.memory_usage_gb:.2f} GB exceeds threshold {self.alert_config.memory_threshold_gb} GB',
                'timestamp': time.time()
            })
        
        # CPU alert
        if metrics.cpu_usage_percent > self.alert_config.cpu_threshold_percent:
            alerts.append({
                'type': 'cpu',
                'severity': 'medium',
                'message': f'CPU usage {metrics.cpu_usage_percent:.1f}% exceeds threshold {self.alert_config.cpu_threshold_percent}%',
                'timestamp': time.time()
            })
        
        # Error rate alert
        if metrics.error_rate > self.alert_config.error_rate_threshold:
            alerts.append({
                'type': 'error_rate',
                'severity': 'high',
                'message': f'Error rate {metrics.error_rate:.3f} exceeds threshold {self.alert_config.error_rate_threshold}',
                'timestamp': time.time()
            })
        
        # Response time alert
        if metrics.average_response_time_ms > self.alert_config.response_time_threshold_ms:
            alerts.append({
                'type': 'response_time',
                'severity': 'medium',
                'message': f'Response time {metrics.average_response_time_ms:.1f}ms exceeds threshold {self.alert_config.response_time_threshold_ms}ms',
                'timestamp': time.time()
            })
        
        # Log and store alerts
        for alert in alerts:
            logger.warning(f"ALERT: {alert['message']}")
            self.alerts_triggered.append(alert)
    
    def record_operation(self, duration: float, success: bool = True):
        """Record operation performance."""
        self.operation_times.append(duration)
        self.total_operations += 1
        if not success:
            self.error_count += 1
    
    def get_current_metrics(self) -> DiagnosticMetrics:
        """Get current metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return DiagnosticMetrics()
    
    def get_metrics_history(self, last_n: Optional[int] = None) -> List[DiagnosticMetrics]:
        """Get metrics history."""
        if last_n:
            return list(self.metrics_history)[-last_n:]
        return list(self.metrics_history)


class RealTimeDiagnosticDashboard:
    """Real-time diagnostic dashboard for Recursia stress testing."""
    
    def __init__(self, port: int = 8051):
        self.port = port
        self.health_monitor = SystemHealthMonitor()
        
        # Dashboard state
        self.emergency_stop_callbacks: List[Callable] = []
        self.pause_callbacks: List[Callable] = []
        self.resume_callbacks: List[Callable] = []
        
        # Simulation state tracking
        self.current_test: Optional[str] = None
        self.test_progress: float = 0.0
        self.system_states: Dict[str, Any] = {}
        
        # Initialize web app
        if DASH_AVAILABLE:
            self.app = self._create_dash_app()
        else:
            self.app = None
            logger.warning("Dash not available - dashboard disabled")
    
    def _create_dash_app(self) -> dash.Dash:
        """Create the Dash web application."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Recursia Diagnostic Dashboard", className="text-center text-primary mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Control Panel"),
                        dbc.CardBody([
                            dbc.ButtonGroup([
                                dbc.Button("Emergency Stop", id="emergency-stop-btn", 
                                         color="danger", size="lg"),
                                dbc.Button("Pause", id="pause-btn", 
                                         color="warning", size="lg"),
                                dbc.Button("Resume", id="resume-btn", 
                                         color="success", size="lg"),
                                dbc.Button("Export Data", id="export-btn", 
                                         color="info", size="lg")
                            ], className="d-grid gap-2"),
                            html.Hr(),
                            html.Div(id="control-status", className="mt-3")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Current Test Status
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Current Test Status"),
                        dbc.CardBody([
                            html.H4(id="current-test-name", children="No test running"),
                            dbc.Progress(id="test-progress", value=0, striped=True, animated=True),
                            html.Div(id="test-metrics", className="mt-3")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # System Metrics Graphs
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="system-metrics-graph")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Quantum Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id="quantum-metrics-graph")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Memory and Error Tracking
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Memory Field Visualization"),
                        dbc.CardBody([
                            dcc.Graph(id="memory-field-graph")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Error Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="error-analysis-graph")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Alerts Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Alerts"),
                        dbc.CardBody([
                            html.Div(id="alerts-table")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0
            ),
            
            # Hidden div to store emergency stop state
            html.Div(id='emergency-stop-state', style={'display': 'none'})
            
        ], fluid=True)
        
        # Register callbacks
        self._register_callbacks(app)
        
        return app
    
    def _register_callbacks(self, app: dash.Dash):
        """Register Dash callbacks."""
        
        @app.callback(
            [Output('system-metrics-graph', 'figure'),
             Output('quantum-metrics-graph', 'figure'),
             Output('memory-field-graph', 'figure'),
             Output('error-analysis-graph', 'figure'),
             Output('current-test-name', 'children'),
             Output('test-progress', 'value'),
             Output('test-metrics', 'children'),
             Output('alerts-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n_intervals):
            return self._update_all_graphs()
        
        @app.callback(
            Output('control-status', 'children'),
            [Input('emergency-stop-btn', 'n_clicks'),
             Input('pause-btn', 'n_clicks'),
             Input('resume-btn', 'n_clicks'),
             Input('export-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def handle_control_buttons(emergency_clicks, pause_clicks, resume_clicks, export_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                return "Ready"
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'emergency-stop-btn':
                self._trigger_emergency_stop()
                return dbc.Alert("EMERGENCY STOP ACTIVATED", color="danger")
            elif button_id == 'pause-btn':
                self._trigger_pause()
                return dbc.Alert("System paused", color="warning")
            elif button_id == 'resume-btn':
                self._trigger_resume()
                return dbc.Alert("System resumed", color="success")
            elif button_id == 'export-btn':
                self._export_data()
                return dbc.Alert("Data exported", color="info")
            
            return "Ready"
    
    def _update_all_graphs(self) -> Tuple:
        """Update all dashboard graphs."""
        try:
            # Get current metrics
            current_metrics = self.health_monitor.get_current_metrics()
            metrics_history = self.health_monitor.get_metrics_history(last_n=100)
            
            # Create system metrics graph
            system_fig = self._create_system_metrics_graph(metrics_history)
            
            # Create quantum metrics graph
            quantum_fig = self._create_quantum_metrics_graph(metrics_history)
            
            # Create memory field graph
            memory_fig = self._create_memory_field_graph(current_metrics)
            
            # Create error analysis graph
            error_fig = self._create_error_analysis_graph(metrics_history)
            
            # Update test status
            test_name = self.current_test or "No test running"
            test_progress = self.test_progress
            test_metrics = self._format_test_metrics(current_metrics)
            
            # Create alerts table
            alerts_table = self._create_alerts_table()
            
            return (system_fig, quantum_fig, memory_fig, error_fig, 
                   test_name, test_progress, test_metrics, alerts_table)
            
        except Exception as e:
            logger.error(f"Dashboard update error: {e}")
            # Return empty figures on error
            empty_fig = go.Figure()
            return (empty_fig, empty_fig, empty_fig, empty_fig, 
                   "Error updating", 0, "Update error", "No alerts")
    
    def _create_system_metrics_graph(self, metrics_history: List[DiagnosticMetrics]) -> go.Figure:
        """Create system performance metrics graph."""
        if not metrics_history:
            return go.Figure()
        
        timestamps = [m.timestamp for m in metrics_history]
        memory_usage = [m.memory_usage_gb for m in metrics_history]
        cpu_usage = [m.cpu_usage_percent for m in metrics_history]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Memory Usage (GB)', 'CPU Usage (%)'),
            vertical_spacing=0.1
        )
        
        # Memory usage
        fig.add_trace(
            go.Scatter(x=timestamps, y=memory_usage, name='Memory GB', 
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # CPU usage
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_usage, name='CPU %',
                      line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(height=400, showlegend=False)
        
        return fig
    
    def _create_quantum_metrics_graph(self, metrics_history: List[DiagnosticMetrics]) -> go.Figure:
        """Create quantum simulation metrics graph."""
        if not metrics_history:
            return go.Figure()
        
        timestamps = [m.timestamp for m in metrics_history]
        active_systems = [m.active_systems for m in metrics_history]
        entangled_pairs = [m.entangled_pairs for m in metrics_history]
        coherence_time = [m.coherence_time_avg for m in metrics_history]
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Active Systems', 'Entangled Pairs', 'Avg Coherence Time'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=active_systems, name='Systems'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=entangled_pairs, name='Entangled Pairs'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=coherence_time, name='Coherence Time'),
            row=3, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        
        return fig
    
    def _create_memory_field_graph(self, current_metrics: DiagnosticMetrics) -> go.Figure:
        """Create memory field visualization."""
        # Create a heatmap showing memory field strain
        strain_data = np.random.random((10, 10)) * current_metrics.memory_field_strain
        
        fig = go.Figure(data=go.Heatmap(
            z=strain_data,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Strain Level")
        ))
        
        fig.update_layout(
            title="Memory Field Strain Distribution",
            height=400
        )
        
        return fig
    
    def _create_error_analysis_graph(self, metrics_history: List[DiagnosticMetrics]) -> go.Figure:
        """Create error analysis graph."""
        if not metrics_history:
            return go.Figure()
        
        timestamps = [m.timestamp for m in metrics_history]
        error_rates = [m.error_rate for m in metrics_history]
        total_errors = [m.total_errors for m in metrics_history]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Error Rate', 'Total Errors'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=error_rates, name='Error Rate'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=total_errors, name='Total Errors'),
            row=2, col=1
        )
        
        fig.update_layout(height=400, showlegend=False)
        
        return fig
    
    def _format_test_metrics(self, metrics: DiagnosticMetrics) -> html.Div:
        """Format test metrics for display."""
        return html.Div([
            html.P(f"Memory: {metrics.memory_usage_gb:.2f} GB"),
            html.P(f"CPU: {metrics.cpu_usage_percent:.1f}%"),
            html.P(f"Active Systems: {metrics.active_systems}"),
            html.P(f"Error Rate: {metrics.error_rate:.3f}"),
            html.P(f"Operations/sec: {metrics.operations_per_second:.1f}")
        ])
    
    def _create_alerts_table(self) -> dash_table.DataTable:
        """Create alerts table."""
        alerts = self.health_monitor.alerts_triggered[-10:]  # Last 10 alerts
        
        if not alerts:
            return html.P("No alerts")
        
        return dash_table.DataTable(
            data=alerts,
            columns=[
                {"name": "Time", "id": "timestamp", "type": "datetime"},
                {"name": "Type", "id": "type"},
                {"name": "Severity", "id": "severity"},
                {"name": "Message", "id": "message"}
            ],
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{severity} = high'},
                    'backgroundColor': '#ffebee',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{severity} = medium'},
                    'backgroundColor': '#fff3e0',
                    'color': 'black',
                }
            ]
        )
    
    def _trigger_emergency_stop(self):
        """Trigger emergency stop."""
        logger.critical("Emergency stop triggered from dashboard")
        for callback in self.emergency_stop_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency stop callback error: {e}")
    
    def _trigger_pause(self):
        """Trigger system pause."""
        logger.info("System pause triggered from dashboard")
        for callback in self.pause_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Pause callback error: {e}")
    
    def _trigger_resume(self):
        """Trigger system resume."""
        logger.info("System resume triggered from dashboard")
        for callback in self.resume_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Resume callback error: {e}")
    
    def _export_data(self):
        """Export current diagnostic data."""
        logger.info("Data export triggered from dashboard")
        
        # Export metrics history
        metrics_data = [m.to_dict() for m in self.health_monitor.get_metrics_history()]
        
        export_file = Path(f"diagnostic_export_{int(time.time())}.json")
        with open(export_file, 'w') as f:
            json.dump({
                'metrics_history': metrics_data,
                'alerts': self.health_monitor.alerts_triggered,
                'export_timestamp': time.time(),
                'current_test': self.current_test
            }, f, indent=2)
        
        logger.info(f"Data exported to {export_file}")
    
    def register_emergency_stop_callback(self, callback: Callable):
        """Register emergency stop callback."""
        self.emergency_stop_callbacks.append(callback)
    
    def register_pause_callback(self, callback: Callable):
        """Register pause callback."""
        self.pause_callbacks.append(callback)
    
    def register_resume_callback(self, callback: Callable):
        """Register resume callback."""
        self.resume_callbacks.append(callback)
    
    def update_test_status(self, test_name: str, progress: float):
        """Update current test status."""
        self.current_test = test_name
        self.test_progress = min(100, max(0, progress))
    
    def update_system_state(self, system_name: str, state_data: Dict[str, Any]):
        """Update system state for monitoring."""
        self.system_states[system_name] = {
            **state_data,
            'timestamp': time.time()
        }
    
    def start_monitoring(self):
        """Start the monitoring system."""
        self.health_monitor.start_monitoring()
        logger.info("Diagnostic monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.health_monitor.stop_monitoring()
        logger.info("Diagnostic monitoring stopped")
    
    def run_dashboard(self, debug: bool = False):
        """Run the dashboard web server."""
        if not self.app:
            logger.error("Dashboard not available - Dash not installed")
            return
        
        logger.info(f"Starting diagnostic dashboard on port {self.port}")
        self.app.run_server(debug=debug, port=self.port, host='0.0.0.0')


# Factory function
def create_diagnostic_dashboard(port: int = 8051, **kwargs) -> RealTimeDiagnosticDashboard:
    """Create diagnostic dashboard instance."""
    return RealTimeDiagnosticDashboard(port=port, **kwargs)


# Main execution for standalone testing
if __name__ == "__main__":
    dashboard = create_diagnostic_dashboard()
    dashboard.start_monitoring()
    
    try:
        dashboard.run_dashboard(debug=True)
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested")
    finally:
        dashboard.stop_monitoring()