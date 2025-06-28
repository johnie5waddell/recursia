"""
Error Throttler Utility
Prevents memory exhaustion from repeated error logging
"""

import time
import logging
from typing import Dict, Optional, Callable
from collections import defaultdict


class ErrorThrottler:
    """
    Enterprise-grade error throttling to prevent log spam and memory issues.
    
    Tracks error frequencies and suppresses repeated errors after thresholds.
    Provides detailed reporting on error patterns.
    """
    
    def __init__(
        self,
        max_errors_per_type: int = 100,
        time_window_seconds: float = 60.0,
        log_first_n: int = 5,
        log_every_n: int = 100
    ):
        """
        Initialize error throttler.
        
        Args:
            max_errors_per_type: Maximum errors before complete suppression
            time_window_seconds: Time window for rate limiting
            log_first_n: Log first N occurrences of each error
            log_every_n: After first_n, log every Nth occurrence
        """
        self.max_errors_per_type = max_errors_per_type
        self.time_window_seconds = time_window_seconds
        self.log_first_n = log_first_n
        self.log_every_n = log_every_n
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_timestamps: Dict[str, list] = defaultdict(list)
        self.suppressed_counts: Dict[str, int] = defaultdict(int)
        self.last_logged: Dict[str, float] = {}
        
        # Global stats
        self.total_errors = 0
        self.total_suppressed = 0
        
    def should_log(self, error_key: str, error_message: str = "") -> bool:
        """
        Determine if an error should be logged.
        
        Args:
            error_key: Unique identifier for error type
            error_message: Optional error message for context
            
        Returns:
            True if error should be logged, False if it should be suppressed
        """
        current_time = time.time()
        self.total_errors += 1
        
        # Clean old timestamps
        self._clean_old_timestamps(error_key, current_time)
        
        # Update counts
        self.error_counts[error_key] += 1
        self.error_timestamps[error_key].append(current_time)
        
        count = self.error_counts[error_key]
        
        # Check if exceeded maximum
        if count > self.max_errors_per_type:
            self.suppressed_counts[error_key] += 1
            self.total_suppressed += 1
            return False
            
        # Log first N
        if count <= self.log_first_n:
            self.last_logged[error_key] = current_time
            return True
            
        # Log every Nth after that
        if count % self.log_every_n == 0:
            self.last_logged[error_key] = current_time
            return True
            
        # Otherwise suppress
        self.suppressed_counts[error_key] += 1
        self.total_suppressed += 1
        return False
        
    def log_error(
        self,
        logger: logging.Logger,
        error_key: str,
        error_message: str,
        level: int = logging.ERROR,
        exc_info: Optional[Exception] = None
    ):
        """
        Log an error with throttling.
        
        Args:
            logger: Logger instance to use
            error_key: Unique identifier for error type
            error_message: Error message to log
            level: Logging level
            exc_info: Optional exception info
        """
        if self.should_log(error_key, error_message):
            count = self.error_counts[error_key]
            suppressed = self.suppressed_counts[error_key]
            
            # Add context about throttling
            if count == self.log_first_n:
                error_message += f" (Will suppress similar errors, logging every {self.log_every_n})"
            elif count > self.log_first_n:
                error_message += f" (Error #{count}, suppressed {suppressed} similar)"
                
            if count == self.max_errors_per_type:
                error_message += f" (FINAL: Suppressing all future occurrences)"
                
            logger.log(level, error_message, exc_info=exc_info)
            
    def _clean_old_timestamps(self, error_key: str, current_time: float):
        """Remove timestamps outside the time window."""
        cutoff_time = current_time - self.time_window_seconds
        self.error_timestamps[error_key] = [
            ts for ts in self.error_timestamps[error_key]
            if ts > cutoff_time
        ]
        
    def get_error_rate(self, error_key: str) -> float:
        """Get current error rate for a specific error type."""
        current_time = time.time()
        self._clean_old_timestamps(error_key, current_time)
        
        timestamps = self.error_timestamps[error_key]
        if not timestamps:
            return 0.0
            
        time_span = current_time - timestamps[0]
        if time_span <= 0:
            return 0.0
            
        return len(timestamps) / time_span
        
    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive error statistics."""
        stats = {
            'total_errors': self.total_errors,
            'total_suppressed': self.total_suppressed,
            'suppression_rate': self.total_suppressed / max(1, self.total_errors),
            'error_types': len(self.error_counts),
            'active_suppressions': sum(
                1 for count in self.error_counts.values()
                if count > self.max_errors_per_type
            ),
            'top_errors': sorted(
                [
                    {
                        'key': key,
                        'count': count,
                        'suppressed': self.suppressed_counts[key],
                        'rate': self.get_error_rate(key)
                    }
                    for key, count in self.error_counts.items()
                ],
                key=lambda x: x['count'],
                reverse=True
            )[:10]
        }
        return stats
        
    def reset(self, error_key: Optional[str] = None):
        """Reset error tracking for specific error or all errors."""
        if error_key:
            self.error_counts[error_key] = 0
            self.error_timestamps[error_key] = []
            self.suppressed_counts[error_key] = 0
            if error_key in self.last_logged:
                del self.last_logged[error_key]
        else:
            self.error_counts.clear()
            self.error_timestamps.clear()
            self.suppressed_counts.clear()
            self.last_logged.clear()
            self.total_errors = 0
            self.total_suppressed = 0
            
    def create_throttled_logger(
        self,
        logger: logging.Logger,
        error_key_fn: Optional[Callable[[str], str]] = None
    ) -> 'ThrottledLogger':
        """
        Create a logger wrapper that automatically throttles.
        
        Args:
            logger: Base logger to wrap
            error_key_fn: Optional function to extract error key from message
            
        Returns:
            ThrottledLogger instance
        """
        return ThrottledLogger(self, logger, error_key_fn)


class ThrottledLogger:
    """Logger wrapper that applies error throttling."""
    
    def __init__(
        self,
        throttler: ErrorThrottler,
        logger: logging.Logger,
        error_key_fn: Optional[Callable[[str], str]] = None
    ):
        self.throttler = throttler
        self.logger = logger
        self.error_key_fn = error_key_fn or self._default_key_fn
        
    def _default_key_fn(self, message: str) -> str:
        """Default error key extraction - first 50 chars."""
        return message[:50] if message else "unknown"
        
    def error(self, message: str, exc_info: Optional[Exception] = None):
        """Log error with throttling."""
        error_key = self.error_key_fn(message)
        self.throttler.log_error(
            self.logger,
            error_key,
            message,
            logging.ERROR,
            exc_info
        )
        
    def warning(self, message: str, exc_info: Optional[Exception] = None):
        """Log warning with throttling."""
        error_key = self.error_key_fn(message)
        self.throttler.log_error(
            self.logger,
            error_key,
            message,
            logging.WARNING,
            exc_info
        )
        
    def info(self, message: str):
        """Pass through info messages without throttling."""
        self.logger.info(message)
        
    def debug(self, message: str):
        """Pass through debug messages without throttling."""
        self.logger.debug(message)


# Global throttler instance
_global_throttler = ErrorThrottler()


def get_throttler() -> ErrorThrottler:
    """Get the global error throttler instance."""
    return _global_throttler


def throttled_error(
    logger: logging.Logger,
    error_key: str,
    message: str,
    exc_info: Optional[Exception] = None
):
    """Convenience function for throttled error logging."""
    _global_throttler.log_error(logger, error_key, message, logging.ERROR, exc_info)


def throttled_warning(
    logger: logging.Logger,
    error_key: str,
    message: str,
    exc_info: Optional[Exception] = None
):
    """Convenience function for throttled warning logging."""
    _global_throttler.log_error(logger, error_key, message, logging.WARNING, exc_info)


def reset_throttling(error_key: Optional[str] = None):
    """Reset throttling for specific error or all errors."""
    _global_throttler.reset(error_key)


def get_throttling_stats() -> Dict[str, any]:
    """Get current throttling statistics."""
    return _global_throttler.get_stats()