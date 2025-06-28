import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class GravitationalWaveEchoVisualizerErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('GravitationalWaveEchoVisualizer error:', error);
    console.error('Error info:', errorInfo);
    
    // Log specific chart.js or settings errors
    if (error.message.includes('settings is not defined')) {
      console.error('Settings prop was not properly passed to a component');
    }
    if (error.message.includes('logarithmic')) {
      console.error('Chart.js scale registration issue detected');
    }
  }

  public render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div style={{
          padding: '20px',
          background: '#1a1a1a',
          border: '1px solid #ff0000',
          borderRadius: '8px',
          color: '#ffffff',
          fontFamily: 'monospace'
        }}>
          <h3 style={{ color: '#ff6b6b', marginBottom: '10px' }}>
            Gravitational Wave Echo Visualizer Error
          </h3>
          <p style={{ marginBottom: '10px' }}>
            {this.state.error?.message || 'An unexpected error occurred'}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: undefined })}
            style={{
              padding: '8px 16px',
              background: '#333',
              border: '1px solid #666',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer'
            }}
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default GravitationalWaveEchoVisualizerErrorBoundary;