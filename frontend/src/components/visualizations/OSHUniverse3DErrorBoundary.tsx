/**
 * Error Boundary for OSH Universe 3D Visualization
 * Provides graceful error handling and recovery for 3D rendering issues
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertCircle, RefreshCw, Activity } from 'lucide-react';

interface Props {
  children: ReactNode;
  primaryColor: string;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  retryCount: number;
}

/**
 * Error boundary specifically designed for 3D visualization components
 * Handles WebGL context loss, shader compilation errors, and resource loading failures
 */
export class OSHUniverse3DErrorBoundary extends Component<Props, State> {
  private retryTimeoutId: number | null = null;

  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0
    };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error: error,
      errorInfo: null,
      retryCount: 0
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error details
    console.error('OSH Universe 3D Error:', error, errorInfo);
    
    // Update state with error info
    this.setState({
      errorInfo: errorInfo
    });
    
    // Call error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
    
    // Check if it's a WebGL context loss error
    if (this.isWebGLContextLossError(error)) {
      this.scheduleRetry(2000); // Retry after 2 seconds
    }
  }

  componentWillUnmount() {
    if (this.retryTimeoutId) {
      clearTimeout(this.retryTimeoutId);
    }
  }

  /**
   * Check if the error is related to WebGL context loss
   */
  private isWebGLContextLossError(error: Error): boolean {
    const errorMessage = error.message.toLowerCase();
    return errorMessage.includes('webgl') || 
           errorMessage.includes('context') ||
           errorMessage.includes('shader') ||
           errorMessage.includes('three');
  }

  /**
   * Schedule an automatic retry
   */
  private scheduleRetry(delay: number) {
    if (this.state.retryCount < 3) {
      this.retryTimeoutId = window.setTimeout(() => {
        this.handleRetry();
      }, delay);
    }
  }

  /**
   * Handle retry attempt
   */
  private handleRetry = () => {
    this.setState(prevState => ({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: prevState.retryCount + 1
    }));
  };

  /**
   * Render error details in development mode
   */
  private renderErrorDetails() {
    if (import.meta.env.MODE !== 'development' || !this.state.error || !this.state.errorInfo) {
      return null;
    }

    return (
      <details style={{
        marginTop: '16px',
        padding: '12px',
        background: 'rgba(255, 255, 255, 0.05)',
        borderRadius: '8px',
        fontSize: '12px',
        color: '#aaa',
        maxHeight: '200px',
        overflowY: 'auto'
      }}>
        <summary style={{ cursor: 'pointer', marginBottom: '8px', color: this.props.primaryColor }}>
          Error Details
        </summary>
        <pre style={{ 
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          margin: 0,
          fontFamily: '"JetBrains Mono", monospace'
        }}>
          {this.state.error.toString()}
          {'\n\n'}
          {this.state.errorInfo.componentStack}
        </pre>
      </details>
    );
  }

  render() {
    if (this.state.hasError) {
      const { primaryColor } = this.props;
      const isWebGLError = this.state.error && this.isWebGLContextLossError(this.state.error);
      
      return (
        <div style={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#0a0a0a',
          color: '#fff',
          padding: '24px',
          textAlign: 'center'
        }}>
          <div style={{
            maxWidth: '500px',
            width: '100%'
          }}>
            {/* Error Icon */}
            <div style={{
              marginBottom: '24px',
              display: 'flex',
              justifyContent: 'center'
            }}>
              <div style={{
                width: '80px',
                height: '80px',
                borderRadius: '50%',
                background: `${primaryColor}20`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative'
              }}>
                <AlertCircle size={40} color={primaryColor} />
                {isWebGLError && (
                  <div style={{
                    position: 'absolute',
                    bottom: '-4px',
                    right: '-4px',
                    background: '#ff4444',
                    borderRadius: '50%',
                    width: '24px',
                    height: '24px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <span style={{ fontSize: '12px', fontWeight: 'bold' }}>!</span>
                  </div>
                )}
              </div>
            </div>
            
            {/* Error Message */}
            <h3 style={{
              fontSize: '20px',
              marginBottom: '12px',
              color: primaryColor
            }}>
              {isWebGLError ? 'WebGL Context Lost' : '3D Visualization Error'}
            </h3>
            
            <p style={{
              fontSize: '14px',
              color: '#aaa',
              marginBottom: '24px',
              lineHeight: 1.6
            }}>
              {isWebGLError 
                ? 'The 3D rendering context was lost. This can happen due to GPU driver issues or resource constraints.'
                : 'An error occurred while rendering the OSH Universe visualization.'}
            </p>
            
            {/* Action Buttons */}
            <div style={{
              display: 'flex',
              gap: '12px',
              justifyContent: 'center',
              flexWrap: 'wrap'
            }}>
              <button
                onClick={this.handleRetry}
                style={{
                  padding: '10px 20px',
                  background: primaryColor,
                  color: '#000',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '14px',
                  fontWeight: 600,
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = `0 4px 12px ${primaryColor}40`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <RefreshCw size={16} />
                Retry Visualization
              </button>
              
              <button
                onClick={() => window.location.reload()}
                style={{
                  padding: '10px 20px',
                  background: 'transparent',
                  color: primaryColor,
                  border: `1px solid ${primaryColor}`,
                  borderRadius: '6px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '14px',
                  fontWeight: 600,
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = `${primaryColor}20`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                }}
              >
                Reload Page
              </button>
            </div>
            
            {/* Retry Status */}
            {this.state.retryCount > 0 && (
              <p style={{
                marginTop: '16px',
                fontSize: '12px',
                color: '#888'
              }}>
                Retry attempt: {this.state.retryCount}/3
              </p>
            )}
            
            {/* Alternative View */}
            <div style={{
              marginTop: '32px',
              padding: '16px',
              background: 'rgba(255, 255, 255, 0.05)',
              borderRadius: '8px',
              border: `1px solid ${primaryColor}30`
            }}>
              <p style={{
                fontSize: '12px',
                color: '#aaa',
                marginBottom: '8px'
              }}>
                Alternative: View simplified metrics
              </p>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                color: primaryColor
              }}>
                <Activity size={16} />
                <span style={{ fontSize: '14px' }}>
                  OSH Universe Status: Active
                </span>
              </div>
            </div>
            
            {/* Error Details (Dev Only) */}
            {this.renderErrorDetails()}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default OSHUniverse3DErrorBoundary;