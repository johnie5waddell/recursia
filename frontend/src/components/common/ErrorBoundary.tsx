/**
 * Enterprise Error Boundary Component
 * Provides graceful error handling and recovery for all components
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home, Mail } from 'lucide-react';

interface Props {
  children: ReactNode;
  componentName?: string;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  fallbackUI?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorCount: number;
}

/**
 * Production-ready error boundary with comprehensive error handling
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorCount: 0
    };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error,
      errorInfo: null,
      errorCount: 0
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('Error caught by boundary:', error, errorInfo);
    }

    // Update state with error details
    this.setState(prevState => ({
      error,
      errorInfo,
      errorCount: prevState.errorCount + 1
    }));

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Send error to monitoring service in production
    if (process.env.NODE_ENV === 'production') {
      this.logErrorToService(error, errorInfo);
    }
  }

  /**
   * Log error to monitoring service (e.g., Sentry, LogRocket)
   */
  private logErrorToService(error: Error, errorInfo: ErrorInfo) {
    // Implementation would go here
    console.error('[ErrorBoundary] Production error:', {
      component: this.props.componentName,
      error: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack
    });
  }

  /**
   * Reset error state and try rendering again
   */
  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorCount: 0
    });
  };

  /**
   * Reload the entire application
   */
  handleReload = () => {
    window.location.reload();
  };

  /**
   * Navigate to home page
   */
  handleHome = () => {
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError) {
      // Use custom fallback UI if provided
      if (this.props.fallbackUI) {
        return this.props.fallbackUI;
      }

      // Default error UI
      return (
        <div className="error-boundary-container">
          <div className="error-content">
            <div className="error-icon">
              <AlertTriangle size={64} />
            </div>
            
            <h1 className="error-title">Something went wrong</h1>
            
            <p className="error-message">
              {this.props.componentName 
                ? `An error occurred in the ${this.props.componentName} component.`
                : 'An unexpected error has occurred.'}
            </p>

            {/* Show error details in development */}
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="error-details">
                <summary>Error Details</summary>
                <div className="error-stack">
                  <p className="error-name">{this.state.error.name}: {this.state.error.message}</p>
                  <pre>{this.state.error.stack}</pre>
                  {this.state.errorInfo && (
                    <pre>{this.state.errorInfo.componentStack}</pre>
                  )}
                </div>
              </details>
            )}

            {/* Error count warning */}
            {this.state.errorCount > 2 && (
              <div className="error-warning">
                <p>This component has crashed {this.state.errorCount} times. 
                   You may need to reload the page.</p>
              </div>
            )}

            {/* Action buttons */}
            <div className="error-actions">
              <button 
                onClick={this.handleReset}
                className="error-button primary"
              >
                <RefreshCw size={16} />
                Try Again
              </button>
              
              <button 
                onClick={this.handleReload}
                className="error-button secondary"
              >
                <RefreshCw size={16} />
                Reload Page
              </button>
              
              <button 
                onClick={this.handleHome}
                className="error-button secondary"
              >
                <Home size={16} />
                Go Home
              </button>
            </div>

            {/* Support contact */}
            <div className="error-support">
              <p>If this problem persists, please contact support:</p>
              <a href="mailto:johnie5waddell@outlook.com" className="error-email">
                <Mail size={16} />
                johnie5waddell@outlook.com
              </a>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Hook to wrap functional components with error boundary
 */
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  componentName?: string,
  fallbackUI?: ReactNode
) {
  return (props: P) => (
    <ErrorBoundary componentName={componentName} fallbackUI={fallbackUI}>
      <Component {...props} />
    </ErrorBoundary>
  );
}