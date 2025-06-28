import datetime
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# import numpy as np  # REMOVED - import where needed

# Configure logging with cross-platform path handling
project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / "recursia.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=str(log_file)
)
logger = logging.getLogger('src.interpreter')

# Add console handler for interactive use
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# ANSI color codes for syntax highlighting
COLORS = {
    'RESET': '\033[0m',
    'KEYWORD': '\033[1;36m',    # Bright Cyan
    'IDENTIFIER': '\033[0;37m',  # White
    'STRING': '\033[0;32m',     # Green
    'NUMBER': '\033[0;33m',     # Yellow
    'COMMENT': '\033[0;90m',    # Dark Gray
    'OPERATOR': '\033[0;35m',   # Purple
    'ERROR': '\033[1;31m',      # Bright Red
    'GATE': '\033[1;33m',       # Bright Yellow
    'QUANTUM': '\033[1;35m',    # Bright Purple
    'SUCCESS': '\033[0;32m',    # Green
    'WARNING': '\033[0;33m',    # Yellow
    'INFO': '\033[0;34m',       # Blue
    'DEBUG': '\033[0;36m',      # Cyan
    'CRITICAL': '\033[0;41m',   # Red background
    'HEADER': '\033[1;37;44m',  # White on blue background
    'HIGHLIGHT': '\033[1;30;47m'  # Black on white background
}

class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, profiler: 'PerformanceProfiler', name: str):
        self.profiler = profiler
        self.name = name
        self.start_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        """Start the timer when entering the context."""
        self.start_time = time.time()
        if self.name not in self.profiler.timers:
            self.profiler.timers[self.name] = 0.0
        
        # Set as total timer if it's the first timer started
        if self.profiler.total_timer is None:
            self.profiler.total_timer = self.name
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer when exiting the context."""
        if self.start_time is not None:
            self.elapsed_time = time.time() - self.start_time
            elapsed_ms = self.elapsed_time * 1000.0  # Convert to milliseconds
            self.profiler.timers[self.name] += elapsed_ms

class ErrorManager:
    """
    Manager for compilation errors and warnings.
    
    Provides methods for recording, categorizing, and retrieving errors and warnings
    that occur during compilation, interpretation, and execution of Recursia programs.
    """
    
    def __init__(self):
        """Initialize the error manager"""
        self.errors = []
        self.warnings = []
        self.error_counts = {
            'general': 0,
            'syntax': 0,
            'semantic': 0,
            'type': 0,
            'runtime': 0,
            'quantum': 0
        }
        self.warning_counts = {
            'general': 0,
            'syntax': 0,
            'semantic': 0,
            'type': 0,
            'runtime': 0,
            'quantum': 0
        }
        self.error_locations = set()  # Track unique error locations
    
    def error(self, message, error_type='general'):
        """
        Add a general error
        
        Args:
            message: Error message
            error_type: Error category type
        """
        self.errors.append({
            'type': error_type,
            'message': message,
            'timestamp': datetime.datetime.now(),
        })
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log the error
        logger.error(f"{error_type.capitalize()} error: {message}")
    
    def warning(self, message, warning_type='general'):
        """
        Add a general warning
        
        Args:
            message: Warning message
            warning_type: Warning category type
        """
        self.warnings.append({
            'type': warning_type,
            'message': message,
            'timestamp': datetime.datetime.now(),
        })
        self.warning_counts[warning_type] = self.warning_counts.get(warning_type, 0) + 1
        
        # Log the warning
        logger.warning(f"{warning_type.capitalize()} warning: {message}")
    
    def syntax_error(self, filename, line, column, message):
        """
        Add a syntax error
        
        Args:
            filename: Source filename
            line: Line number
            column: Column number
            message: Error message
        """
        error = {
            'type': 'syntax',
            'filename': filename,
            'line': line,
            'column': column,
            'message': message,
            'timestamp': datetime.datetime.now(),
        }
        self.errors.append(error)
        self.error_counts['syntax'] = self.error_counts.get('syntax', 0) + 1
        
        # Track location to avoid duplicate errors
        location = (filename, line, column)
        self.error_locations.add(location)
        
        # Log the error
        logger.error(f"Syntax error at {filename}:{line}:{column}: {message}")
    
    def semantic_error(self, filename, line, column, message):
        """
        Add a semantic error
        
        Args:
            filename: Source filename
            line: Line number
            column: Column number
            message: Error message
        """
        error = {
            'type': 'semantic',
            'filename': filename,
            'line': line,
            'column': column,
            'message': message,
            'timestamp': datetime.datetime.now(),
        }
        self.errors.append(error)
        self.error_counts['semantic'] = self.error_counts.get('semantic', 0) + 1
        
        # Track location to avoid duplicate errors
        location = (filename, line, column)
        self.error_locations.add(location)
        
        # Log the error
        logger.error(f"Semantic error at {filename}:{line}:{column}: {message}")
    
    def type_error(self, filename, line, column, message):
        """
        Add a type error
        
        Args:
            filename: Source filename
            line: Line number
            column: Column number
            message: Error message
        """
        error = {
            'type': 'type',
            'filename': filename,
            'line': line,
            'column': column,
            'message': message,
            'timestamp': datetime.datetime.now(),
        }
        self.errors.append(error)
        self.error_counts['type'] = self.error_counts.get('type', 0) + 1
        
        # Track location to avoid duplicate errors
        location = (filename, line, column)
        self.error_locations.add(location)
        
        # Log the error
        logger.error(f"Type error at {filename}:{line}:{column}: {message}")
    
    def quantum_error(self, filename, line, column, message):
        """
        Add a quantum operation error
        
        Args:
            filename: Source filename
            line: Line number
            column: Column number
            message: Error message
        """
        error = {
            'type': 'quantum',
            'filename': filename,
            'line': line,
            'column': column,
            'message': message,
            'timestamp': datetime.datetime.now(),
        }
        self.errors.append(error)
        self.error_counts['quantum'] = self.error_counts.get('quantum', 0) + 1
        
        # Track location to avoid duplicate errors
        location = (filename, line, column)
        self.error_locations.add(location)
        
        # Log the error
        logger.error(f"Quantum error at {filename}:{line}:{column}: {message}")
    
    def runtime_error(self, message, traceback_info=None):
        """
        Add a runtime error
        
        Args:
            message: Error message
            traceback_info: Optional traceback information
        """
        error = {
            'type': 'runtime',
            'message': message,
            'traceback': traceback_info or traceback.format_exc(),
            'timestamp': datetime.datetime.now(),
        }
        self.errors.append(error)
        self.error_counts['runtime'] = self.error_counts.get('runtime', 0) + 1
        
        # Log the error
        logger.error(f"Runtime error: {message}")
        if traceback_info:
            logger.debug(traceback_info)
    
    def has_errors(self):
        """
        Check if there are any errors
        
        Returns:
            bool: True if there are errors
        """
        return len(self.errors) > 0
    
    def has_errors_of_type(self, error_type):
        """
        Check if there are errors of a specific type
        
        Args:
            error_type: Error type to check
            
        Returns:
            bool: True if there are errors of this type
        """
        return self.error_counts.get(error_type, 0) > 0
    
    def get_errors(self, error_type=None):
        """
        Get all errors or errors of a specific type
        
        Args:
            error_type: Optional error type filter
            
        Returns:
            list: List of errors
        """
        if error_type:
            return [e for e in self.errors if e['type'] == error_type]
        return self.errors.copy()
    
    def get_warnings(self, warning_type=None):
        """
        Get all warnings or warnings of a specific type
        
        Args:
            warning_type: Optional warning type filter
            
        Returns:
            list: List of warnings
        """
        if warning_type:
            return [w for w in self.warnings if w['type'] == warning_type]
        return self.warnings.copy()
    
    def get_error_summary(self):
        """
        Get a summary of errors
        
        Returns:
            dict: Error summary
        """
        return {
            'total': len(self.errors),
            'by_type': self.error_counts,
            'unique_locations': len(self.error_locations)
        }
    
    def get_warning_summary(self):
        """
        Get a summary of warnings
        
        Returns:
            dict: Warning summary
        """
        return {
            'total': len(self.warnings),
            'by_type': self.warning_counts
        }
    
    def clear(self):
        """Clear all errors and warnings"""
        self.errors = []
        self.warnings = []
        self.error_counts = {k: 0 for k in self.error_counts}
        self.warning_counts = {k: 0 for k in self.warning_counts}
        self.error_locations = set()
    
    def clear_errors(self, error_type=None):
        """
        Clear errors of a specific type or all errors
        
        Args:
            error_type: Optional error type to clear
        """
        if error_type:
            count = self.error_counts.get(error_type, 0)
            self.errors = [e for e in self.errors if e['type'] != error_type]
            self.error_counts[error_type] = 0
            return count
        else:
            count = len(self.errors)
            self.errors = []
            self.error_counts = {k: 0 for k in self.error_counts}
            self.error_locations = set()
            return count
    
    def get_most_recent_error(self):
        """
        Get the most recent error
        
        Returns:
            dict: Most recent error or None
        """
        if not self.errors:
            return None
        return self.errors[-1]
    
    def format_error_message(self, error, show_location=True, include_timestamp=False,
                            colorize=False):
        """
        Format an error message
        
        Args:
            error: Error dictionary
            show_location: Whether to include location info
            include_timestamp: Whether to include timestamp
            colorize: Whether to use ANSI colors
            
        Returns:
            str: Formatted error message
        """
        if colorize:
            type_color = COLORS.get('ERROR', '')
            reset = COLORS.get('RESET', '')
        else:
            type_color = ''
            reset = ''
        
        # Build message components
        message_parts = []
        
        if include_timestamp and 'timestamp' in error:
            message_parts.append(f"[{error['timestamp']}]")
        
        message_parts.append(f"{type_color}{error['type'].capitalize()} Error{reset}:")
        
        if show_location and all(k in error for k in ('filename', 'line', 'column')):
            message_parts.append(f"at {error['filename']}:{error['line']}:{error['column']}")
        
        message_parts.append(error['message'])
        
        return " ".join(message_parts)
    
    def print_errors(self, error_type=None, colorize=True):
        """
        Print errors to console
        
        Args:
            error_type: Optional error type filter
            colorize: Whether to use ANSI colors
        """
        errors = self.get_errors(error_type)
        
        if not errors:
            if colorize:
                print(f"{COLORS['SUCCESS']}No errors{COLORS['RESET']}")
            else:
                print("No errors")
            return
        
        for error in errors:
            print(self.format_error_message(error, colorize=colorize))


class ConfigManager:
    """
    Manages configuration settings for the Recursia system.
    
    Provides methods for loading, saving, and accessing configuration settings,
    with support for different profiles and environment-specific overrides.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to configuration file or directory
        """
        self.config = {
            'system': {
                'log_level': 'info',
                'max_history': 1000,
                'max_recursion_depth': 20,
                'debug_mode': False
            },
            'quantum': {
                'default_backend': 'simulator',
                'hardware_provider': 'auto',
                'simulator_precision': 'double',
                'max_qubits': 24
            },
            'visualization': {
                'enabled': True,
                'use_color': True,
                'output_format': 'text',
                'auto_visualize': False
            },
            'compiler': {
                'optimization_level': 2,
                'emit_warnings': True,
                'enable_extensions': True,
                'debug_symbols': False
            },
            'runtime': {
                'memory_limit': 1024,  # in MB
                'execution_timeout': 30,  # in seconds
                'error_handling': 'strict',
                'event_logging': True
            }
        }
        
        self.config_path = config_path or os.path.join(os.path.expanduser('~'), '.recursia', 'config.json')
        self.config_dir = os.path.dirname(self.config_path)
        self.loaded_from = None
        self.save_on_exit = False
        
        # Try to load config from file
        self._ensure_config_dir()
        self._load_config()
    
    def _ensure_config_dir(self):
        """Ensure the configuration directory exists"""
        if not os.path.exists(self.config_dir):
            try:
                os.makedirs(self.config_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create config directory {self.config_dir}: {e}")
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                # Update config with loaded values
                self._recursive_update(self.config, loaded_config)
                self.loaded_from = self.config_path
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.info(f"No configuration file found at {self.config_path}, using defaults")
        except Exception as e:
            logger.warning(f"Error loading configuration: {e}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration with all required sections.
        
        Returns:
            Dictionary containing default configuration
        """
        return {
            'system': {
                'max_qubits': 25,
                'precision': 'double',
                'use_gpu': False,
                'log_level': 'INFO',
                'log_to_console': True,
                'log_to_file': True,
                'log_file_path': 'recursia.log',
                'memory_limit_mb': 2048,
                'max_threads': 4
            },
            'quantum': {
                'backend_type': 'simulator',
                'default_shots': 1024,
                'measurement_basis': 'Z_basis',
                'enable_noise': False,
                'noise_model': 'default',
                'coherence_time': 1000.0,
                'gate_fidelity': 0.99,
                'measurement_fidelity': 0.95
            },
            'runtime': {
                'simulation_time_step': 0.01,
                'max_simulation_time': 1000.0,
                'auto_coherence_tracking': True,
                'observer_dynamics_enabled': True,
                'recursive_mechanics_enabled': True,
                'memory_field_enabled': True,
                'event_system_enabled': True,
                'performance_profiling': False
            },
            'compiler': {
                'optimization_level': 1,
                'target_backend': 'quantum_simulator',
                'enable_caching': True,
                'validate_semantics': True,
                'syntax_highlighting': True,
                'debug_mode': False,
                'verbose_compilation': False
            },
            'visualization': {
                'enable_dashboard': True,
                'dashboard_port': 8080,
                'dashboard_host': 'localhost',
                'real_time_updates': True,
                'update_interval': 0.1,
                'enable_3d_visualization': True,
                'color_scheme': 'viridis',
                'export_format': 'png',
                'export_dpi': 300,
                'theme': 'dark'
            },
            'hardware': {
                'providers': {
                    'ibm': {
                        'enabled': False,
                        'token': '',
                        'hub': 'ibm-q',
                        'group': 'open',
                        'project': 'main'
                    },
                    'rigetti': {
                        'enabled': False,
                        'api_key': '',
                        'user_id': '',
                        'endpoint': 'https://api.rigetti.com'
                    },
                    'google': {
                        'enabled': False,
                        'project_id': '',
                        'processor_id': ''
                    },
                    'ionq': {
                        'enabled': False,
                        'api_key': '',
                        'endpoint': 'https://api.ionq.co'
                    }
                },
                'default_provider': 'ibm',
                'connection_timeout': 30.0,
                'retry_attempts': 3
            },
            'osh': {
                'validation_enabled': True,
                'coherence_threshold': 0.7,
                'entropy_threshold': 0.3,
                'strain_threshold': 0.8,
                'rsp_threshold': 0.5,
                'emergence_detection': True,
                'consciousness_analysis': True,
                'recursive_depth_limit': 10,
                'memory_strain_tracking': True,
                'observer_consensus_tracking': True
            },
            'export': {
                'default_format': 'json',
                'include_metadata': True,
                'compression': 'gzip',
                'scientific_notation': True,
                'precision_digits': 6,
                'export_directory': 'exports',
                'auto_timestamp': True
            }
        }

    def _recursive_update(self, target_dict, update_dict):
        """
        Recursively update a dictionary
        
        Args:
            target_dict: Dictionary to update
            update_dict: Dictionary with updates
        """
        for k, v in update_dict.items():
            if k in target_dict and isinstance(target_dict[k], dict) and isinstance(v, dict):
                self._recursive_update(target_dict[k], v)
            else:
                target_dict[k] = v
    
    def save_config(self, config_path=None):
        """
        Save configuration to file
        
        Args:
            config_path: Optional path to save configuration
            
        Returns:
            bool: True if saved successfully
        """
        save_path = config_path or self.config_path
        
        try:
            # Ensure directory exists
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # Write config file
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, section, key=None, default=None):
        """
        Get a configuration value or entire section
        
        Args:
            section: Configuration section
            key: Configuration key (optional - if None, returns entire section)
            default: Default value if not found
            
        Returns:
            Configuration value, entire section, or default
        """
        # Handle the common pattern get(section, default_value) 
        # If only two args provided and second arg is not a string, treat it as default
        if default is None and key is not None and not isinstance(key, str):
            return self.config.get(section, key)
        
        if key is None:
            # If no key provided, return the entire section
            return self.config.get(section, default if default is not None else {})
        else:
            # If key provided, get specific value within section
            section_config = self.config.get(section, {})
            if isinstance(section_config, dict):
                return section_config.get(key, default)
            else:
                return default
            
    def set(self, section, key, value):
        """
        Set a configuration value
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
            
        Returns:
            bool: True if value was set
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
        return True
    
    def enable_auto_save(self):
        """Enable automatic saving of configuration on exit"""
        self.save_on_exit = True
    
    def reset_to_defaults(self, section=None):
        """
        Reset configuration to defaults
        
        Args:
            section: Optional section to reset
        """
        default_config = {
            'system': {
                'log_level': 'info',
                'max_history': 1000,
                'max_recursion_depth': 20,
                'debug_mode': False
            },
            'quantum': {
                'default_backend': 'simulator',
                'hardware_provider': 'auto',
                'simulator_precision': 'double',
                'max_qubits': 24
            },
            'visualization': {
                'enabled': True,
                'use_color': True,
                'output_format': 'text',
                'auto_visualize': False
            },
            'compiler': {
                'optimization_level': 2,
                'emit_warnings': True,
                'enable_extensions': True,
                'debug_symbols': False
            },
            'runtime': {
                'memory_limit': 1024,  # in MB
                'execution_timeout': 30,  # in seconds
                'error_handling': 'strict',
                'event_logging': True
            }
        }
        
        if section:
            if section in default_config:
                self.config[section] = default_config[section].copy()
        else:
            self.config = default_config.copy()
    
    def __enter__(self):
        """Context manager enter method"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit method
        
        Saves configuration if auto-save is enabled
        """
        if self.save_on_exit:
            self.save_config()


class PathManager:
    """
    Manages file paths for the Recursia system.
    
    Provides methods for resolving, validating, and manipulating file paths,
    including support for standard library paths and project-specific paths.
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize the path manager
        
        Args:
            base_dir: Base directory for relative paths
        """
        self.base_dir = base_dir or os.getcwd()
        self.std_lib_paths = []
        
        # Add standard search paths
        recursia_home = os.environ.get('RECURSIA_HOME')
        if recursia_home:
            std_lib_path = os.path.join(recursia_home, 'lib')
            if os.path.exists(std_lib_path):
                self.std_lib_paths.append(std_lib_path)
        
        # Add user library path
        user_lib_path = os.path.join(os.path.expanduser('~'), '.recursia', 'lib')
        if os.path.exists(user_lib_path):
            self.std_lib_paths.append(user_lib_path)
        
        # Add current directory as a fallback
        self.std_lib_paths.append(os.path.join(self.base_dir, 'lib'))
    
    def resolve_path(self, path, relative_to=None):
        """
        Resolve a path to an absolute path
        
        Args:
            path: Path to resolve
            relative_to: Path to resolve relative to
            
        Returns:
            str: Absolute path
        """
        if os.path.isabs(path):
            return path
        
        if relative_to:
            if not os.path.isabs(relative_to):
                relative_to = os.path.join(self.base_dir, relative_to)
            if os.path.isfile(relative_to):
                relative_to = os.path.dirname(relative_to)
            return os.path.normpath(os.path.join(relative_to, path))
        
        return os.path.normpath(os.path.join(self.base_dir, path))
    
    def find_library_file(self, filename):
        """
        Find a library file in standard library paths
        
        Args:
            filename: Library filename to find
            
        Returns:
            str: Path to library file or None if not found
        """
        # Check if it's a direct path
        if os.path.exists(filename):
            return os.path.abspath(filename)
        
        # Check in standard library paths
        for path in self.std_lib_paths:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                return full_path
        
        return None
    
    def ensure_directory(self, directory):
        """
        Ensure a directory exists
        
        Args:
            directory: Directory to ensure
            
        Returns:
            bool: True if directory exists or was created
        """
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                return True
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {e}")
                return False
        return True
    
    def file_exists(self, path):
        """
        Check if a file exists
        
        Args:
            path: Path to check
            
        Returns:
            bool: True if file exists
        """
        resolved_path = self.resolve_path(path)
        return os.path.isfile(resolved_path)
    
    def get_modification_time(self, path):
        """
        Get the modification time of a file
        
        Args:
            path: Path to check
            
        Returns:
            float: Modification time or None if file doesn't exist
        """
        resolved_path = self.resolve_path(path)
        if os.path.exists(resolved_path):
            return os.path.getmtime(resolved_path)
        return None
    
    def is_newer_than(self, path1, path2):
        """
        Check if one file is newer than another
        
        Args:
            path1: First path
            path2: Second path
            
        Returns:
            bool: True if path1 is newer than path2
        """
        mtime1 = self.get_modification_time(path1)
        mtime2 = self.get_modification_time(path2)
        
        if mtime1 is None or mtime2 is None:
            return False
        
        return mtime1 > mtime2
    
    def get_file_hash(self, path):
        """
        Get the SHA-256 hash of a file
        
        Args:
            path: Path to file
            
        Returns:
            str: File hash or None if file doesn't exist
        """
        resolved_path = self.resolve_path(path)
        if not os.path.exists(resolved_path):
            return None
        
        try:
            hash_obj = hashlib.sha256()
            with open(resolved_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash for {resolved_path}: {e}")
            return None


class PerformanceProfiler:
    """
    Performance profiling utility for tracking execution times across
    different components of the Recursia framework.
    
    This class provides methods to start and stop timers, collect
    elapsed times, and generate performance reports for analysis.
    """
    
    def __init__(self):
        """Initialize the performance profiler"""
        self.timers = {}
        self.start_times = {}
        self.active_timers = set()
        self.total_timer = None
        
    def start_timer(self, name: str) -> 'TimerContext':
        """
        Start a named timer and return a context manager
        
        Args:
            name: Name of the timer to start
            
        Returns:
            TimerContext: Context manager for this timer
        """
        return TimerContext(self, name)

    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and return elapsed time in milliseconds
        
        Args:
            name: Name of the timer to stop
            
        Returns:
            Elapsed time in milliseconds
        """
        import time
        
        if name not in self.active_timers:
            return 0.0  # Timer not running
            
        elapsed = time.time() - self.start_times[name]
        elapsed_ms = elapsed * 1000.0  # Convert to milliseconds
        
        self.timers[name] += elapsed_ms
        self.active_timers.remove(name)
        
        return elapsed_ms
        
    def get_timer_data(self) -> dict:
        """
        Get all timer data
        
        Returns:
            Dictionary mapping timer names to elapsed times in milliseconds
        """
        return self.timers.copy()
        
    def reset(self) -> None:
        """Reset all timers"""
        self.timers = {}
        self.start_times = {}
        self.active_timers = set()
        self.total_timer = None
        
    def get_timer(self, name: str) -> float:
        """
        Get the elapsed time for a specific timer
        
        Args:
            name: Name of the timer
            
        Returns:
            Elapsed time in milliseconds
        """
        return self.timers.get(name, 0.0)
        
    def get_total_time(self) -> float:
        """
        Get the total elapsed time
        
        Returns:
            Total elapsed time in milliseconds
        """
        if self.total_timer:
            return self.timers.get(self.total_timer, 0.0)
        return 0.0
        
    def stop_all_timers(self) -> None:
        """Stop all active timers"""
        active_timers = list(self.active_timers)
        for timer in active_timers:
            self.stop_timer(timer)
            
    def get_timer_summary(self) -> str:
        """
        Get a formatted summary of all timers
        
        Returns:
            Formatted string with timer information
        """
        if not self.timers:
            return "No timing data available"
            
        total = self.get_total_time()
        if total <= 0:
            total = sum(self.timers.values())
            
        lines = [f"Performance Summary:"]
        
        # Sort timers by elapsed time (descending)
        sorted_timers = sorted(self.timers.items(), key=lambda x: x[1], reverse=True)
        
        for name, elapsed in sorted_timers:
            percentage = (elapsed / total * 100) if total > 0 else 0
            lines.append(f"  {name}: {elapsed:.2f} ms ({percentage:.1f}%)")
            
        return "\n".join(lines)
        
    def __del__(self):
        """Ensure all timers are stopped when the profiler is destroyed"""
        self.stop_all_timers()
        
class VisualizationHelper:
    """
    Helper for visualizing quantum states and other Recursia objects.
    
    Provides methods for creating text-based and other visualizations of quantum
    states, observers, entanglement, and other aspects of Recursia programs.
    """
    
    def __init__(self, use_color=True):
        """
        Initialize the visualization helper
        
        Args:
            use_color: Whether to use ANSI colors
        """
        self.use_color = use_color
        self.visualization_data = {}
        self.last_update_time = None
        self.strain = {}
        self.coherence = {}
        self.entropy = {}
        self.memory_field = self

    def update(self, simulation_data):
        """
        Update the visualization with new simulation data
        
        Args:
            simulation_data: Dictionary containing current simulation state
            
        Returns:
            bool: True if update was successful
        """
        import time
        
        # Store the current time
        self.last_update_time = time.time()
        
        # Store the simulation data
        self.visualization_data = simulation_data.copy() if simulation_data else {}
        
        # Process any visualization-specific updates
        if self.visualization_data:
            # Extract states and observers if present
            states = self.visualization_data.get('states', {})
            observers = self.visualization_data.get('observers', {})
            
            # Store processed visualization data for each state
            for state_name, state_data in states.items():
                if 'state_vector' in state_data:
                    # Generate visualization string for state vector
                    state_data['visualization'] = {
                        'state_vector_text': self.state_vector_to_string(state_data['state_vector']),
                        'generated_at': self.last_update_time
                    }
            
            # Store processed visualization data for observer network
            if observers:
                observed_states = {}
                for obs_name, obs_data in observers.items():
                    if 'observed_states' in obs_data:
                        observed_states[obs_name] = obs_data['observed_states']
                        
                self.visualization_data['observer_network'] = {
                    'text': self.observer_network(observers, observed_states),
                    'generated_at': self.last_update_time
                }
        
        return True
    
    def get_visualization_data(self):
        """
        Get the current visualization data
        
        Returns:
            dict: Current visualization data
        """
        return self.visualization_data
    
    def state_vector_to_string(self, state_vector, num_qubits=None, precision=4,
                              show_phases=True, threshold=1e-10):
        """
        Convert a state vector to a string representation
        
        Args:
            state_vector: Quantum state vector (numpy array)
            num_qubits: Number of qubits (inferred if None)
            precision: Decimal precision
            show_phases: Whether to show phases
            threshold: Amplitude threshold for showing basis states
            
        Returns:
            str: String representation of state vector
        """
        if state_vector is None:
            return "None"
        
        # Ensure state_vector is a numpy array
        import numpy as np
        if not isinstance(state_vector, np.ndarray):
            try:
                state_vector = np.array(state_vector, dtype=complex)
            except:
                return str(state_vector)
        
        # Determine number of qubits if not provided
        if num_qubits is None:
            dim = len(state_vector)
            num_qubits = int(np.log2(dim)) if dim > 0 else 0
            if 2**num_qubits != dim:
                # Not a power of 2, can't determine number of qubits
                return str(state_vector)
        
        # Format each basis state
        result_parts = []
        for i, amplitude in enumerate(state_vector):
            if abs(amplitude) > threshold:
                # Format the amplitude
                if show_phases:
                    # Show in a+bi form
                    amplitude_str = self._format_complex(amplitude, precision)
                else:
                    # Show as magnitude
                    amplitude_str = f"{abs(amplitude):.{precision}f}"
                
                # Add color if enabled
                if self.use_color:
                    if abs(amplitude) > 0.5:
                        amplitude_str = f"{COLORS['QUANTUM']}{amplitude_str}{COLORS['RESET']}"
                
                # Get binary representation of basis state
                basis_str = format(i, f'0{num_qubits}b')
                
                # Format the ket
                ket_str = f"|{basis_str}⟩"
                
                result_parts.append(f"{amplitude_str}{ket_str}")
        
        if not result_parts:
            return "0"
        
        return " + ".join(result_parts)
    
    def _format_complex(self, value, precision=4):
        """
        Format a complex number
        
        Args:
            value: Complex number
            precision: Decimal precision
            
        Returns:
            str: Formatted complex number
        """
        real = value.real
        imag = value.imag
        
        if abs(real) < 1e-10 and abs(imag) < 1e-10:
            return "0"
        elif abs(real) < 1e-10:
            return f"{imag:.{precision}f}i"
        elif abs(imag) < 1e-10:
            return f"{real:.{precision}f}"
        elif imag < 0:
            return f"{real:.{precision}f}{imag:.{precision}f}i"
        else:
            return f"{real:.{precision}f}+{imag:.{precision}f}i"
    
    def density_matrix_to_string(self, density_matrix, num_qubits=None, precision=4,
                                threshold=1e-10):
        """
        Convert a density matrix to a string representation
        
        Args:
            density_matrix: Density matrix (numpy array)
            num_qubits: Number of qubits (inferred if None)
            precision: Decimal precision
            threshold: Value threshold for showing matrix elements
            
        Returns:
            str: String representation of density matrix
        """
        if density_matrix is None:
            return "None"
        
        # Ensure density_matrix is a numpy array
        import numpy as np
        if not isinstance(density_matrix, np.ndarray):
            try:
                density_matrix = np.array(density_matrix, dtype=complex)
            except:
                return str(density_matrix)
        
        # Check if the matrix is square
        if density_matrix.shape[0] != density_matrix.shape[1]:
            return str(density_matrix)
        
        # Determine number of qubits if not provided
        dim = density_matrix.shape[0]
        if num_qubits is None:
            num_qubits = int(np.log2(dim)) if dim > 0 else 0
            if 2**num_qubits != dim:
                # Not a power of 2, can't determine number of qubits
                return str(density_matrix)
        
        # Format as a matrix
        result_lines = []
        
        for i in range(dim):
            row_parts = []
            for j in range(dim):
                value = density_matrix[i, j]
                if abs(value) > threshold:
                    value_str = self._format_complex(value, precision)
                    
                    # Add color if enabled
                    if self.use_color:
                        if i == j:  # Diagonal elements
                            value_str = f"{COLORS['QUANTUM']}{value_str}{COLORS['RESET']}"
                        elif abs(value) > 0.1:  # Significant off-diagonal elements
                            value_str = f"{COLORS['NUMBER']}{value_str}{COLORS['RESET']}"
                else:
                    value_str = "0"
                
                row_parts.append(value_str.rjust(precision + 5))
            
            result_lines.append("[ " + " ".join(row_parts) + " ]")
        
        return "\n".join(result_lines)
    
    def bloch_sphere_coordinates(self, qubit_state):
        """
        Calculate Bloch sphere coordinates for a qubit state
        
        Args:
            qubit_state: Single-qubit state vector [alpha, beta]
            
        Returns:
            tuple: (x, y, z) Bloch sphere coordinates
        """
        # Ensure qubit_state is a numpy array
        import numpy as np
        if not isinstance(qubit_state, np.ndarray):
            try:
                qubit_state = np.array(qubit_state, dtype=complex)
            except:
                return (0, 0, 1)  # Default to |0⟩ state
        
        # Normalize if needed
        norm = np.linalg.norm(qubit_state)
        if norm > 0:
            qubit_state = qubit_state / norm
        
        # Extract alpha and beta
        if len(qubit_state) >= 2:
            alpha, beta = qubit_state[0], qubit_state[1]
        else:
            return (0, 0, 1)  # Default to |0⟩ state
        
        # Calculate Bloch sphere coordinates
        # x = 2 * Re(alpha* * beta)
        # y = 2 * Im(alpha* * beta)
        # z = |alpha|^2 - |beta|^2
        x = 2 * (alpha.conjugate() * beta).real
        y = 2 * (alpha.conjugate() * beta).imag
        z = (abs(alpha) ** 2) - (abs(beta) ** 2)
        
        return (x, y, z)
    
    def bloch_sphere_text(self, qubit_state, resolution=10):
        """
        Create a text-based Bloch sphere visualization
        
        Args:
            qubit_state: Single-qubit state vector
            resolution: Sphere resolution
            
        Returns:
            str: Text Bloch sphere
        """
        x, y, z = self.bloch_sphere_coordinates(qubit_state)
        
        # Create a text-based sphere
        sphere = []
        
        # Calculate the position of the state vector on the sphere
        import numpy as np
        phi = np.arctan2(y, x)  # Azimuthal angle
        theta = np.arccos(z)    # Polar angle
        
        for i in range(resolution * 2 + 1):
            row = []
            for j in range(resolution * 2 + 1):
                # Map (i,j) to spherical coordinates
                i_norm = (i - resolution) / resolution  # -1 to 1
                j_norm = (j - resolution) / resolution  # -1 to 1
                
                # Skip points outside the circle
                if i_norm**2 + j_norm**2 > 1:
                    row.append(' ')
                    continue
                
                # Determine the point's position on the sphere
                point_phi = np.arctan2(j_norm, i_norm) if (i_norm != 0 or j_norm != 0) else 0
                point_theta = np.pi * np.sqrt(i_norm**2 + j_norm**2)
                
                # Check if this point is close to the state vector
                angle_diff = np.sqrt(
                    min((point_phi - phi) % (2*np.pi), (phi - point_phi) % (2*np.pi))**2 + 
                    (point_theta - theta)**2
                )
                
                if angle_diff < 0.5:
                    if self.use_color:
                        row.append(f"{COLORS['QUANTUM']}●{COLORS['RESET']}")
                    else:
                        row.append('●')
                else:
                    row.append('·')
            
            sphere.append(''.join(row))
        
        # Add labels
        sphere_with_labels = []
        sphere_with_labels.append("z↑")
        sphere_with_labels.extend(sphere)
        sphere_with_labels.append("z↓")
        
        middle_row = resolution + 1
        sphere_with_labels[middle_row] = "x← " + sphere_with_labels[middle_row] + " →x"
        
        # Add state information
        bloch_info = [
            "Bloch Sphere Coordinates:",
            f"x: {x:.4f}",
            f"y: {y:.4f}",
            f"z: {z:.4f}"
        ]
        
        # Add some padding between the sphere and info
        return "\n".join(sphere_with_labels) + "\n\n" + "\n".join(bloch_info)
    
    def entanglement_graph(self, quantum_states_or_registry, state_names=None):
        """
        Create a text-based graph of entanglement relationships
        
        Args:
            quantum_states_or_registry: Either an entanglement registry (dict mapping (state1, state2) to strength)
                                    or a quantum_states dictionary
            state_names: Optional list of state names to include
            
        Returns:
            str: Text entanglement graph
        """
        # Handle both quantum_states dict and entanglement registry
        entanglement_registry = {}
        
        if isinstance(quantum_states_or_registry, dict):
            # Check if this looks like quantum states or entanglement registry
            if quantum_states_or_registry and next(iter(quantum_states_or_registry.values())) is not None:
                first_key = next(iter(quantum_states_or_registry.keys()))
                first_value = quantum_states_or_registry[first_key]
                
                # If the key is a tuple, assume it's an entanglement registry
                if isinstance(first_key, tuple) and len(first_key) == 2:
                    entanglement_registry = quantum_states_or_registry
                else:
                    # It's quantum states - extract entanglement relationships
                    for state_name, state_obj in quantum_states_or_registry.items():
                        if hasattr(state_obj, 'entangled_with'):
                            entangled_states = getattr(state_obj, 'entangled_with', [])
                            if isinstance(entangled_states, (list, set)):
                                for entangled_state in entangled_states:
                                    # Create a consistent key (sorted tuple)
                                    key = tuple(sorted([state_name, entangled_state]))
                                    if key not in entanglement_registry:
                                        entanglement_registry[key] = 1.0  # Default strength
                            elif isinstance(entangled_states, dict):
                                for entangled_state, strength in entangled_states.items():
                                    key = tuple(sorted([state_name, entangled_state]))
                                    entanglement_registry[key] = strength
        
        # Extract unique state names from the registry if not provided
        if state_names is None:
            state_names = set()
            for key in entanglement_registry:
                if isinstance(key, tuple) and len(key) == 2:
                    state_names.add(key[0])
                    state_names.add(key[1])
            state_names = sorted(state_names)
        
        if not state_names:
            return "No entangled states"
        
        # Create an adjacency matrix
        n = len(state_names)
        adjacency = [[0 for _ in range(n)] for _ in range(n)]
        
        # Map state names to indices
        name_to_index = {name: i for i, name in enumerate(state_names)}
        
        # Fill the adjacency matrix
        for key, strength in entanglement_registry.items():
            if isinstance(key, tuple) and len(key) == 2:
                state1, state2 = key
                if state1 in name_to_index and state2 in name_to_index:
                    i, j = name_to_index[state1], name_to_index[state2]
                    adjacency[i][j] = adjacency[j][i] = strength
        
        # Create a text-based graph
        lines = ["Entanglement Graph:"]
        lines.append("  " + " ".join(name[:3].ljust(3) for name in state_names))
        
        for i, name in enumerate(state_names):
            row = [name[:3].ljust(3)]
            for j in range(n):
                strength = adjacency[i][j]
                if i == j:
                    cell = "   "
                elif strength == 0:
                    cell = "   "
                else:
                    # Represent strength with different characters
                    if strength > 0.8:
                        cell = "███"
                    elif strength > 0.5:
                        cell = "▓▓▓"
                    elif strength > 0.2:
                        cell = "▒▒▒"
                    else:
                        cell = "░░░"
                    
                    # Add color if enabled
                    if self.use_color:
                        if strength > 0.8:
                            cell = f"{COLORS['QUANTUM']}{cell}{COLORS['RESET']}"
                        elif strength > 0.5:
                            cell = f"{COLORS['STRING']}{cell}{COLORS['RESET']}"
                        elif strength > 0.2:
                            cell = f"{COLORS['NUMBER']}{cell}{COLORS['RESET']}"
                
                row.append(cell)
            
            lines.append(" ".join(row))
        
        return "\n".join(lines)

    def observer_network(self, observers, observed_states):
        """
        Create a text-based visualization of observers and observed states
        
        Args:
            observers: Dictionary mapping observer names to observer properties
            observed_states: Dictionary mapping observer names to lists of observed state names
            
        Returns:
            str: Text observer network
        """
        if not observers:
            return "No observers"
        
        # Create a text-based network
        lines = ["Observer Network:"]
        
        for observer_name, properties in sorted(observers.items()):
            # Get observer properties
            observer_type = properties.get('observer_type', 'standard_observer')
            focus = properties.get('observer_focus', 'none')
            
            # Format observer line
            observer_line = observer_name
            
            # Add color if enabled
            if self.use_color:
                observer_line = f"{COLORS['KEYWORD']}{observer_line}{COLORS['RESET']}"
            
            observer_line += f" ({observer_type})"
            
            # Add observed states
            states = observed_states.get(observer_name, [])
            if states:
                if self.use_color:
                    states_str = ", ".join(f"{COLORS['IDENTIFIER']}{s}{COLORS['RESET']}" for s in states)
                else:
                    states_str = ", ".join(states)
                
                observer_line += f" → {states_str}"
            
            # Add focus if different from observed states
            if focus and focus not in states:
                if self.use_color:
                    focus = f"{COLORS['QUANTUM']}{focus}{COLORS['RESET']}"
                observer_line += f" (focus: {focus})"
            
            lines.append(observer_line)
        
        return "\n".join(lines)
    
    def format_code(self, code, highlight=True):
        """
        Format Recursia code with syntax highlighting
        
        Args:
            code: Recursia code
            highlight: Whether to apply syntax highlighting
            
        Returns:
            str: Formatted code
        """
        try:
            from src.core.lexer import RecursiaLexer
        except ImportError:
            RecursiaLexer = None
            
        if not highlight or not self.use_color:
            return code
        
        # Simple syntax highlighting for keywords, strings, etc.
        lines = code.split('\n')
        highlighted_lines = []
        
        for line in lines:
            # Check if the line is a comment
            if line.lstrip().startswith('//'):
                highlighted_lines.append(f"{COLORS['COMMENT']}{line}{COLORS['RESET']}")
                continue
            
            # Process tokens in the line
            remaining = line
            tokens = []
            
            while remaining:
                # Check for strings
                if remaining[0] in ('"', "'", '`'):
                    quote = remaining[0]
                    end = 1
                    while end < len(remaining) and remaining[end] != quote:
                        if remaining[end] == '\\' and end + 1 < len(remaining):
                            end += 2  # Skip the escaped character
                        else:
                            end += 1
                    
                    if end < len(remaining):
                        tokens.append((f"{COLORS['STRING']}{remaining[:end+1]}{COLORS['RESET']}", end + 1))
                    else:
                        tokens.append((f"{COLORS['STRING']}{remaining}{COLORS['RESET']}", len(remaining)))
                
                # Check for keywords
                elif remaining[0].isalpha() or remaining[0] == '_':
                    end = 1
                    while end < len(remaining) and (remaining[end].isalnum() or remaining[end] == '_'):
                        end += 1
                    
                    word = remaining[:end]
                    keywords = (RecursiaLexer.KEYWORDS if RecursiaLexer is not None else [
                        "state", "observer", "pattern", "apply", "render", "cohere", 
                        "if", "when", "while", "for", "function", "return", "import", 
                        "export", "let", "const", "measure", "entangle", "teleport",
                        "hook", "visualize", "simulate", "align", "defragment", "print",
                        "log", "reset", "qubit", "qubits", "control", "params", "basis"
                    ])
                    gate_types = (RecursiaLexer.GATE_TYPES if RecursiaLexer is not None else [
                        "H_gate", "X_gate", "Y_gate", "Z_gate", "CNOT_gate"
                    ])
                    
                    if word in keywords:
                        tokens.append((f"{COLORS['KEYWORD']}{word}{COLORS['RESET']}", end))
                    elif word in gate_types:
                        tokens.append((f"{COLORS['GATE']}{word}{COLORS['RESET']}", end))
                    else:
                        tokens.append((f"{COLORS['IDENTIFIER']}{word}{COLORS['RESET']}", end))
                
                # Check for numbers
                elif remaining[0].isdigit() or (remaining[0] == '.' and len(remaining) > 1 and remaining[1].isdigit()):
                    end = 1
                    while end < len(remaining) and (remaining[end].isdigit() or remaining[end] == '.'):
                        end += 1
                    
                    tokens.append((f"{COLORS['NUMBER']}{remaining[:end]}{COLORS['RESET']}", end))
                
                # Check for operators
                elif remaining[0] in "+-*/%=<>!&|^~?:;,.(){}[]":
                    tokens.append((f"{COLORS['OPERATOR']}{remaining[0]}{COLORS['RESET']}", 1))
                
                # Skip whitespace and other characters
                else:
                    tokens.append((remaining[0], 1))
                
                # Update remaining text
                offset = tokens[-1][1]
                remaining = remaining[offset:]
            
            # Join tokens for this line
            highlighted_lines.append(''.join(token[0] for token in tokens))
        
        return '\n'.join(highlighted_lines)

    def progress_bar(self, value, maximum, width=40, label=None):
        """
        Create a text-based progress bar
        
        Args:
            value: Current value
            maximum: Maximum value
            width: Bar width
            label: Optional label
            
        Returns:
            str: Text progress bar
        """
        # Calculate percentage
        if maximum <= 0:
            percentage = 0
        else:
            percentage = min(100, int(100 * value / maximum))
        
        # Calculate the number of filled units
        filled_width = int(width * percentage / 100)
        
        # Create the bar
        bar = "█" * filled_width + "░" * (width - filled_width)
        
        # Add percentage
        output = f"[{bar}] {percentage}%"
        
        # Add label
        if label:
            output = f"{label}: {output}"
        
        # Add color if enabled
        if self.use_color:
            if percentage < 30:
                color = COLORS['WARNING']
            elif percentage < 70:
                color = COLORS['INFO']
            else:
                color = COLORS['SUCCESS']
            
            output = f"{color}{output}{COLORS['RESET']}"
        
        return output

def time_function(func):
    """
    Decorator to time a function's execution
    
    Args:
        func: Function to time
        
    Returns:
        callable: Timed function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(f"Function {func.__name__} took {end_time - start_time:.6f} seconds")
        return result
    return wrapper


def catch_exceptions(func):
    """
    Decorator to catch and log exceptions
    
    Args:
        func: Function to wrap
        
    Returns:
        callable: Exception-catching function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}")
            logger.debug(traceback.format_exc())
            raise
    return wrapper


def format_binary(value, bits=None):
    """
    Format a value as a binary string
    
    Args:
        value: Integer value
        bits: Number of bits (autodetected if None)
        
    Returns:
        str: Formatted binary string
    """
    if bits is None:
        # Calculate required bits
        bits = max(1, value.bit_length())
    
    return format(value, f'0{bits}b')


def format_matrix(matrix, precision=4, threshold=1e-10):
    """
    Format a matrix as a string
    
    Args:
        matrix: 2D numpy array
        precision: Decimal precision
        threshold: Value threshold for showing matrix elements
        
    Returns:
        str: Formatted matrix
    """
    if matrix is None:
        return "None"
    
    # Ensure matrix is a numpy array
    import numpy as np
    if not isinstance(matrix, np.ndarray):
        try:
            matrix = np.array(matrix)
        except:
            return str(matrix)
    
    # Check dimensions
    if matrix.ndim != 2:
        return str(matrix)
    
    rows, cols = matrix.shape
    
    # Format each element
    formatted_rows = []
    for i in range(rows):
        formatted_row = []
        for j in range(cols):
            value = matrix[i, j]
            if abs(value) < threshold:
                formatted_row.append("0".rjust(precision + 2))
            elif np.iscomplex(value):
                real = value.real
                imag = value.imag
                if abs(real) < threshold:
                    formatted_row.append(f"{imag:.{precision}f}j".rjust(precision + 5))
                elif abs(imag) < threshold:
                    formatted_row.append(f"{real:.{precision}f}".rjust(precision + 2))
                elif imag < 0:
                    formatted_row.append(f"{real:.{precision}f}{imag:.{precision}f}j".rjust(precision + 6))
                else:
                    formatted_row.append(f"{real:.{precision}f}+{imag:.{precision}f}j".rjust(precision + 6))
            else:
                formatted_row.append(f"{value:.{precision}f}".rjust(precision + 2))
        
        formatted_rows.append("[" + ", ".join(formatted_row) + "]")
    
    return "[" + ",\n ".join(formatted_rows) + "]"


def create_global_error_manager():
    """
    Create a global error manager
    
    Returns:
        ErrorManager: Global error manager
    """
    global_error_manager = ErrorManager()
    return global_error_manager


def create_global_config_manager():
    """
    Create a global configuration manager
    
    Returns:
        ConfigManager: Global configuration manager
    """
    global_config_manager = ConfigManager()
    return global_config_manager


def setup_logging(log_file=None, log_level=None, enable_console=True):
    """
    Set up logging configuration
    
    Args:
        log_file: Optional log file path
        log_level: Optional log level
        enable_console: Whether to enable console logging
        
    Returns:
        logging.Logger: Configured logger
    """
    # Get the log level
    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    level = level_map.get(log_level.lower(), logging.INFO) if log_level else logging.INFO
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add file handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Add console handler if enabled
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        root_logger.addHandler(console_handler)
    
    # Create and return our specific logger
    logger = logging.getLogger('src.interpreter')
    logger.debug("Logging configured")
    
    return logger


def colorize_text(text, color_name):
    """
    Apply ANSI color to text
    
    Args:
        text: Text to colorize
        color_name: Color name from COLORS dictionary
        
    Returns:
        str: Colorized text
    """
    if color_name in COLORS:
        return f"{COLORS[color_name]}{text}{COLORS['RESET']}"
    return text


def disable_colors():
    """Disable all ANSI colors by setting them to empty strings"""
    for key in COLORS:
        COLORS[key] = ''


def enable_colors():
    """Re-enable ANSI colors with their default values"""
    global COLORS
    COLORS = {
        'RESET': '\033[0m',
        'KEYWORD': '\033[1;36m',    # Bright Cyan
        'IDENTIFIER': '\033[0;37m',  # White
        'STRING': '\033[0;32m',     # Green
        'NUMBER': '\033[0;33m',     # Yellow
        'COMMENT': '\033[0;90m',    # Dark Gray
        'OPERATOR': '\033[0;35m',   # Purple
        'ERROR': '\033[1;31m',      # Bright Red
        'GATE': '\033[1;33m',       # Bright Yellow
        'QUANTUM': '\033[1;35m',    # Bright Purple
        'SUCCESS': '\033[0;32m',    # Green
        'WARNING': '\033[0;33m',    # Yellow
        'INFO': '\033[0;34m',       # Blue
        'DEBUG': '\033[0;36m',      # Cyan
        'CRITICAL': '\033[0;41m',   # Red background
        'HEADER': '\033[1;37;44m',  # White on blue background
        'HIGHLIGHT': '\033[1;30;47m'  # Black on white background
    }


def supports_ansi_colors():
    """
    Check if the current terminal supports ANSI colors
    
    Returns:
        bool: True if ANSI colors are supported
    """
    # Check if we're running in a terminal
    if not sys.stdout.isatty():
        return False
    
    # Check platform-specific support
    platform = sys.platform
    if platform == 'win32':
        # On Windows, check if we're using a modern terminal that supports ANSI
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            return kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7) != 0
        except:
            # If we can't check, assume no support
            return False
    
    # Most Unix-like platforms support ANSI colors
    return platform != 'Pocket PC'


# Initialize global utility instances
global_error_manager = create_global_error_manager()
global_config_manager = create_global_config_manager()
performance_profiler = PerformanceProfiler()
path_manager = PathManager()
visualization_helper = VisualizationHelper(use_color=supports_ansi_colors())

# Disable colors if not supported
if not supports_ansi_colors():
    disable_colors()