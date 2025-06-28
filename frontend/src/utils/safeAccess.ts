/**
 * Safe access utilities for preventing runtime errors
 * Provides type-safe methods to access nested properties without crashes
 */

/**
 * Safely access a nested property with a default value
 */
export function safeGet<T>(
  obj: any,
  path: string,
  defaultValue: T
): T {
  try {
    const keys = path.split('.');
    let result = obj;
    
    for (const key of keys) {
      if (result === null || result === undefined) {
        return defaultValue;
      }
      result = result[key];
    }
    
    return result !== undefined && result !== null ? result : defaultValue;
  } catch {
    return defaultValue;
  }
}

/**
 * Safely call a method with error handling
 */
export function safeCall<T>(
  fn: () => T,
  defaultValue: T,
  errorHandler?: (error: any) => void
): T {
  try {
    return fn();
  } catch (error) {
    if (errorHandler) {
      errorHandler(error);
    }
    return defaultValue;
  }
}

/**
 * Ensure a number is finite and valid
 */
export function ensureFinite(value: number, defaultValue: number = 0): number {
  return isFinite(value) && !isNaN(value) ? value : defaultValue;
}

/**
 * Ensure an array is valid and not empty
 */
export function ensureArray<T>(value: T[] | undefined | null, defaultValue: T[] = []): T[] {
  return Array.isArray(value) && value.length > 0 ? value : defaultValue;
}

/**
 * Format a number safely for display
 */
export function safeFormat(
  value: number | undefined | null,
  decimals: number = 2,
  defaultValue: string = '0'
): string {
  if (value === null || value === undefined || !isFinite(value) || isNaN(value)) {
    return defaultValue;
  }
  
  // Handle very large numbers with scientific notation
  if (value >= 1e9) {
    // For values >= 1 billion, use scientific notation
    if (value >= 1e21) {
      // For universe-scale values, use condensed notation
      const exponent = Math.floor(Math.log10(value));
      const mantissa = value / Math.pow(10, exponent);
      return `${mantissa.toFixed(1)}e${exponent}`;
    }
    return value.toExponential(decimals);
  }
  
  // Handle moderately large numbers with K/M/B suffixes
  if (value >= 1e6) {
    return (value / 1e6).toFixed(1) + 'M';
  }
  if (value >= 1e3) {
    return (value / 1e3).toFixed(1) + 'K';
  }
  
  return value.toFixed(decimals);
}