/**
 * Component Health Check Utility
 * Validates that all frontend components are properly integrated
 * with the unified API and displaying data correctly
 */

import { BackendMetrics } from './backendMetricsSync';

export interface ComponentHealth {
  name: string;
  status: 'healthy' | 'warning' | 'error';
  message: string;
  checks: HealthCheck[];
}

export interface HealthCheck {
  name: string;
  passed: boolean;
  message: string;
}

/**
 * Check if a component is using the unified API properly
 */
export function checkComponentIntegration(
  componentName: string,
  metrics: BackendMetrics | null,
  isConnected: boolean,
  additionalChecks?: HealthCheck[]
): ComponentHealth {
  const checks: HealthCheck[] = [];
  
  // Check API connection
  checks.push({
    name: 'API Connection',
    passed: isConnected,
    message: isConnected ? 'Connected to backend' : 'Not connected to backend API'
  });
  
  // Check metrics availability
  checks.push({
    name: 'Metrics Available',
    passed: metrics !== null && Object.keys(metrics).length > 0,
    message: metrics ? 'Receiving metrics from backend' : 'No metrics available'
  });
  
  // Check metrics validity
  if (metrics) {
    const coreMetrics = ['rsp', 'coherence', 'entropy', 'information'];
    const missingMetrics = coreMetrics.filter(key => 
      !(key in metrics) || !isFinite(metrics[key as keyof BackendMetrics] as number)
    );
    
    checks.push({
      name: 'Core Metrics Valid',
      passed: missingMetrics.length === 0,
      message: missingMetrics.length === 0 
        ? 'All core metrics valid' 
        : `Missing/invalid metrics: ${missingMetrics.join(', ')}`
    });
  }
  
  // Add custom checks
  if (additionalChecks) {
    checks.push(...additionalChecks);
  }
  
  // Determine overall status
  const failedChecks = checks.filter(c => !c.passed);
  const status: ComponentHealth['status'] = 
    failedChecks.length === 0 ? 'healthy' :
    failedChecks.some(c => c.name.includes('Connection')) ? 'error' : 'warning';
  
  const message = 
    status === 'healthy' ? 'All checks passed' :
    status === 'error' ? 'Critical issues detected' :
    `${failedChecks.length} warning(s)`;
  
  return {
    name: componentName,
    status,
    message,
    checks
  };
}

/**
 * Check if a component has proper error handling
 */
export function checkErrorHandling(
  componentName: string,
  hasErrorBoundary: boolean,
  hasLoadingState: boolean,
  hasEmptyState: boolean
): HealthCheck[] {
  return [
    {
      name: 'Error Boundary',
      passed: hasErrorBoundary,
      message: hasErrorBoundary 
        ? 'Component wrapped in error boundary' 
        : 'No error boundary protection'
    },
    {
      name: 'Loading State',
      passed: hasLoadingState,
      message: hasLoadingState 
        ? 'Loading state implemented' 
        : 'Missing loading state'
    },
    {
      name: 'Empty State',
      passed: hasEmptyState,
      message: hasEmptyState 
        ? 'Empty state implemented' 
        : 'Missing empty state handling'
    }
  ];
}

/**
 * Check if a component is displaying all available data
 */
export function checkDataDisplay(
  displayedFields: string[],
  availableFields: string[]
): HealthCheck {
  const missingFields = availableFields.filter(field => !displayedFields.includes(field));
  
  return {
    name: 'Data Display Completeness',
    passed: missingFields.length === 0,
    message: missingFields.length === 0
      ? 'Displaying all available fields'
      : `Missing fields: ${missingFields.join(', ')}`
  };
}

/**
 * Check component performance
 */
export function checkPerformance(
  renderTime: number,
  updateFrequency: number,
  memoryUsage?: number
): HealthCheck[] {
  const checks: HealthCheck[] = [];
  
  checks.push({
    name: 'Render Performance',
    passed: renderTime < 16, // 60fps threshold
    message: renderTime < 16 
      ? `Rendering at ${Math.round(1000/renderTime)}fps`
      : `Slow rendering: ${renderTime.toFixed(1)}ms`
  });
  
  checks.push({
    name: 'Update Frequency',
    passed: updateFrequency < 100, // Should update at least every 100ms
    message: updateFrequency < 100
      ? `Updates every ${updateFrequency}ms`
      : `Infrequent updates: ${updateFrequency}ms`
  });
  
  if (memoryUsage !== undefined) {
    checks.push({
      name: 'Memory Usage',
      passed: memoryUsage < 100, // 100MB threshold
      message: memoryUsage < 100
        ? `Using ${memoryUsage.toFixed(1)}MB`
        : `High memory usage: ${memoryUsage.toFixed(1)}MB`
    });
  }
  
  return checks;
}

/**
 * Run comprehensive health check on all components
 */
export function runFullSystemHealthCheck(
  components: Array<{
    name: string;
    metrics: BackendMetrics | null;
    isConnected: boolean;
    hasErrorBoundary?: boolean;
    hasLoadingState?: boolean;
    hasEmptyState?: boolean;
    displayedFields?: string[];
    availableFields?: string[];
    renderTime?: number;
    updateFrequency?: number;
  }>
): ComponentHealth[] {
  return components.map(component => {
    const checks: HealthCheck[] = [];
    
    // Basic integration checks
    const integrationHealth = checkComponentIntegration(
      component.name,
      component.metrics,
      component.isConnected
    );
    checks.push(...integrationHealth.checks);
    
    // Error handling checks
    if (component.hasErrorBoundary !== undefined) {
      checks.push(...checkErrorHandling(
        component.name,
        component.hasErrorBoundary,
        component.hasLoadingState || false,
        component.hasEmptyState || false
      ));
    }
    
    // Data display checks
    if (component.displayedFields && component.availableFields) {
      checks.push(checkDataDisplay(
        component.displayedFields,
        component.availableFields
      ));
    }
    
    // Performance checks
    if (component.renderTime !== undefined) {
      checks.push(...checkPerformance(
        component.renderTime,
        component.updateFrequency || 50
      ));
    }
    
    // Determine overall status
    const failedChecks = checks.filter(c => !c.passed);
    const status: ComponentHealth['status'] = 
      failedChecks.length === 0 ? 'healthy' :
      failedChecks.some(c => c.name.includes('Connection') || c.name.includes('Error')) ? 'error' : 'warning';
    
    return {
      name: component.name,
      status,
      message: `${failedChecks.length} issue(s) found`,
      checks
    };
  });
}