/**
 * Visualization Wrapper Component
 * Provides error boundaries, loading states, and data validation for all visualizations
 */

import React, { Suspense } from 'react';
import { ErrorBoundary } from './ErrorBoundary';
import { ComponentLoading, EmptyState } from './LoadingStates';
import { useEngineAPIContext } from '../../contexts/EngineAPIContext';
import { Activity } from 'lucide-react';

interface VisualizationWrapperProps {
  children: React.ReactNode;
  componentName: string;
  requiresData?: boolean;
  minHeight?: string | number;
  className?: string;
  primaryColor?: string;
}

/**
 * HOC that wraps visualizations with error handling and loading states
 */
export const VisualizationWrapper: React.FC<VisualizationWrapperProps> = ({
  children,
  componentName,
  requiresData = false,
  minHeight = 400,
  className = '',
  primaryColor = '#4fc3f7'
}) => {
  const { metrics, isConnected } = useEngineAPIContext();

  // Check if we have required data
  const hasData = !requiresData || (metrics && Object.keys(metrics).length > 0);

  return (
    <ErrorBoundary 
      componentName={componentName}
      onError={(error, errorInfo) => {
        console.error(`[${componentName}] Error:`, error, errorInfo);
      }}
    >
      <Suspense 
        fallback={
          <ComponentLoading 
            componentName={componentName} 
            primaryColor={primaryColor}
          />
        }
      >
        <div 
          className={`visualization-wrapper ${className}`}
          style={{ minHeight }}
        >
          {!isConnected && requiresData ? (
            <EmptyState
              icon={Activity}
              title="No Connection"
              message="Unable to connect to the backend. Please check your connection and try again."
              primaryColor={primaryColor}
              action={{
                label: "Retry Connection",
                onClick: () => window.location.reload()
              }}
            />
          ) : !hasData && requiresData ? (
            <EmptyState
              icon={Activity}
              title="No Data Available"
              message="Waiting for data from the quantum engine. Start a simulation to see visualizations."
              primaryColor={primaryColor}
            />
          ) : (
            children
          )}
        </div>
      </Suspense>
    </ErrorBoundary>
  );
};

interface WithVisualizationWrapperOptions {
  requiresData?: boolean;
  minHeight?: string | number;
  className?: string;
}

/**
 * HOC to wrap any component with visualization wrapper
 */
export function withVisualizationWrapper<P extends { primaryColor?: string }>(
  Component: React.ComponentType<P>,
  componentName: string,
  options: WithVisualizationWrapperOptions = {}
) {
  return React.forwardRef<any, P>((props, ref) => (
    <VisualizationWrapper
      componentName={componentName}
      primaryColor={props.primaryColor}
      {...options}
    >
      <Component {...(props as P)} />
    </VisualizationWrapper>
  ));
}