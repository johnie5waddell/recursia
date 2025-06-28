import React, { Component, ErrorInfo, ReactNode } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

// Ensure all scales are registered
ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class TheoryOfEverythingErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('TheoryOfEverythingPanel error:', error);
    console.error('Error info:', errorInfo);
    
    // Handle specific chart.js errors
    if (error.message.includes('logarithmic') && error.message.includes('not a registered scale')) {
      console.error('Logarithmic scale not registered. Attempting to register...');
      try {
        // Try to register the scale again
        ChartJS.register(LogarithmicScale);
      } catch (e) {
        console.error('Failed to register logarithmic scale:', e);
      }
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
            Theory of Everything Panel Error
          </h3>
          <p style={{ marginBottom: '10px' }}>
            {this.state.error?.message || 'An unexpected error occurred'}
          </p>
          {this.state.error?.message.includes('logarithmic') && (
            <p style={{ fontSize: '12px', color: '#888', marginBottom: '10px' }}>
              This may be a Chart.js scale registration issue. The component will attempt to recover.
            </p>
          )}
          <button
            onClick={() => {
              // Re-register scales before retry
              try {
                ChartJS.register(
                  CategoryScale,
                  LinearScale,
                  LogarithmicScale,
                  RadialLinearScale,
                  PointElement,
                  LineElement,
                  ArcElement,
                  Title,
                  Tooltip,
                  Legend,
                  Filler
                );
              } catch (e) {
                console.error('Failed to re-register Chart.js components:', e);
              }
              this.setState({ hasError: false, error: undefined });
            }}
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

export default TheoryOfEverythingErrorBoundary;