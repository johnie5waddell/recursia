/**
 * Array Limiter Utility - Prevents unbounded array growth
 * 
 * Provides utilities to limit array sizes and prevent memory leaks
 * in quantum simulation engines.
 */

/**
 * Limit array size by removing oldest elements
 */
export function limitArraySize<T>(
  array: T[], 
  maxSize: number, 
  keepStrategy: 'newest' | 'oldest' | 'random' = 'newest'
): T[] {
  if (array.length <= maxSize) {
    return array;
  }
  
  switch (keepStrategy) {
    case 'newest':
      // Keep the newest elements (end of array)
      return array.slice(-maxSize);
      
    case 'oldest':
      // Keep the oldest elements (start of array)
      return array.slice(0, maxSize);
      
    case 'random':
      // Keep random elements
      const shuffled = [...array].sort(() => Math.random() - 0.5);
      return shuffled.slice(0, maxSize);
      
    default:
      return array.slice(-maxSize);
  }
}

/**
 * Remove elements based on a condition
 */
export function filterByCondition<T>(
  array: T[],
  keepCondition: (item: T) => boolean,
  maxSize?: number
): T[] {
  let filtered = array.filter(keepCondition);
  
  if (maxSize && filtered.length > maxSize) {
    filtered = limitArraySize(filtered, maxSize);
  }
  
  return filtered;
}

/**
 * Remove old elements based on timestamp
 */
export function removeOldElements<T extends { timestamp: number }>(
  array: T[],
  maxAgeMs: number,
  maxSize?: number
): T[] {
  const now = Date.now();
  const cutoffTime = now - maxAgeMs;
  
  return filterByCondition(
    array,
    item => item.timestamp > cutoffTime,
    maxSize
  );
}

/**
 * Limit 2D array size
 */
export function limit2DArray<T>(
  array: T[][],
  maxRows: number,
  maxCols: number
): T[][] {
  // Limit rows
  const limitedRows = limitArraySize(array, maxRows);
  
  // Limit columns in each row
  return limitedRows.map(row => 
    limitArraySize(row, maxCols)
  );
}

/**
 * Limit 3D array size
 */
export function limit3DArray<T>(
  array: T[][][],
  maxX: number,
  maxY: number,
  maxZ: number
): T[][][] {
  // Limit X dimension
  const limitedX = limitArraySize(array, maxX);
  
  // Limit Y and Z dimensions
  return limitedX.map(plane =>
    limit2DArray(plane, maxY, maxZ)
  );
}

/**
 * Calculate memory usage estimate for an array
 */
export function estimateArrayMemory(
  elementCount: number,
  bytesPerElement: number = 8 // Default for number/Complex
): number {
  // Add overhead for array structure
  const overhead = 32 + elementCount * 4; // Rough estimate
  return overhead + (elementCount * bytesPerElement);
}

/**
 * Check if array size is safe for available memory
 */
export function isArraySizeMemorySafe(
  elementCount: number,
  bytesPerElement: number = 8,
  maxMemoryMB: number = 100
): boolean {
  const estimatedBytes = estimateArrayMemory(elementCount, bytesPerElement);
  const estimatedMB = estimatedBytes / (1024 * 1024);
  return estimatedMB <= maxMemoryMB;
}