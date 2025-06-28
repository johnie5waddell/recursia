/**
 * Chart utility functions for type-safe Chart.js configuration
 * Provides common chart options and type helpers
 */

import { ChartOptions } from 'chart.js';

/**
 * Base chart options that work across all chart types
 */
export const createBaseChartOptions = (primaryColor: string) => ({
  responsive: true,
  maintainAspectRatio: false,
  interaction: {
    mode: 'index' as const,
    intersect: false
  },
  plugins: {
    legend: {
      display: true,
      position: 'top' as const,
      labels: {
        color: '#ffffff',
        font: {
          size: 11
        },
        usePointStyle: true,
        padding: 15
      }
    },
    tooltip: {
      mode: 'index' as const,
      intersect: false,
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      titleColor: '#ffffff',
      bodyColor: '#ffffff',
      borderColor: primaryColor,
      borderWidth: 1,
      padding: 10,
      displayColors: true,
      callbacks: {
        label: function(context: any) {
          let label = context.dataset.label || '';
          if (label) {
            label += ': ';
          }
          if (context.parsed.y !== null) {
            label += context.parsed.y.toFixed(3);
          }
          return label;
        }
      }
    }
  },
  scales: {
    x: {
      grid: {
        display: true,
        color: 'rgba(255, 255, 255, 0.05)'
      },
      ticks: {
        color: '#666666',
        font: {
          size: 10
        }
      }
    },
    y: {
      grid: {
        display: true,
        color: 'rgba(255, 255, 255, 0.05)'
      },
      ticks: {
        color: '#666666',
        font: {
          size: 10
        }
      }
    }
  }
});

/**
 * Create line chart options
 */
export const createLineChartOptions = (primaryColor: string): ChartOptions<'line'> => {
  return createBaseChartOptions(primaryColor) as ChartOptions<'line'>;
};

/**
 * Create scatter chart options
 */
export const createScatterChartOptions = (primaryColor: string): ChartOptions<'scatter'> => {
  const baseOptions = createBaseChartOptions(primaryColor);
  return {
    ...baseOptions,
    scales: {
      x: {
        ...baseOptions.scales.x,
        type: 'linear' as const,
        position: 'bottom' as const
      },
      y: {
        ...baseOptions.scales.y,
        type: 'linear' as const,
        position: 'left' as const
      }
    }
  } as ChartOptions<'scatter'>;
};

/**
 * Create bar chart options
 */
export const createBarChartOptions = (primaryColor: string): ChartOptions<'bar'> => {
  const baseOptions = createBaseChartOptions(primaryColor);
  return {
    ...baseOptions,
    indexAxis: 'x' as const,
    plugins: {
      ...baseOptions.plugins,
      legend: {
        display: false
      }
    }
  } as ChartOptions<'bar'>;
};