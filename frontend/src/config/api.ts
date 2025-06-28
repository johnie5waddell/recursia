/**
 * API configuration for Recursia frontend
 * Handles all backend communication endpoints and utilities
 */

// Use Vite's import.meta.env for environment variables
// In production, set VITE_API_URL in your environment
// In development, use relative URLs to leverage Vite's proxy
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

/**
 * API endpoint configuration
 * Centralized location for all backend endpoints
 */
export const API_ENDPOINTS = {
  // Program loader endpoints
  loadProgram: `${API_BASE_URL}/api/load-program`,
  listPrograms: `${API_BASE_URL}/api/list-programs`,
  programInfo: `${API_BASE_URL}/api/program-info`,
  
  // Execution endpoints
  execute: `${API_BASE_URL}/api/execute`,
  compile: `${API_BASE_URL}/api/compile`,
  analyze: `${API_BASE_URL}/api/analyze`,
  
  // Metric endpoints
  metrics: `${API_BASE_URL}/api/metrics`,
  states: `${API_BASE_URL}/api/states`,
  observers: `${API_BASE_URL}/api/observers`,
  
  // Health check
  health: `${API_BASE_URL}/api/health`,
  
  // OSH calculation endpoints
  osh: {
    rsp: `${API_BASE_URL}/api/osh/rsp`,
    rspBound: `${API_BASE_URL}/api/osh/rsp-bound`,
    informationAction: `${API_BASE_URL}/api/osh/information-action`,
    cmbComplexity: `${API_BASE_URL}/api/osh/cmb-complexity`,
    gwEchoes: `${API_BASE_URL}/api/osh/gw-echoes`,
    constantDrift: `${API_BASE_URL}/api/osh/constant-drift`,
    eegCosmic: `${API_BASE_URL}/api/osh/eeg-cosmic`,
    voidEntropy: `${API_BASE_URL}/api/osh/void-entropy`,
    consciousnessMap: `${API_BASE_URL}/api/osh/consciousness-map`
  },
  
  // WebSocket endpoint
  websocket: API_BASE_URL ? API_BASE_URL.replace('http', 'ws').replace('https', 'wss') + '/ws' : '/ws'
};

/**
 * Response types for API calls
 */
export interface ProgramContent {
  content: string;
  path: string;
}

export interface ProgramInfo {
  path: string;
  name: string;
  size: number;
  modified: number;
  lines: number;
  description: string;
  tags: string[];
}

export interface ApiError {
  detail: string;
  status?: number;
}

/**
 * Custom error class for API errors
 */
export class APIError extends Error {
  status?: number;
  detail?: string;

  constructor(message: string, status?: number, detail?: string) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Fetch with timeout and error handling
 */
async function fetchWithTimeout(
  url: string, 
  options: RequestInit = {}, 
  timeout: number = 30000
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new APIError('Request timed out', 408);
    }
    throw error;
  }
}

/**
 * Fetch a Recursia program's content
 * @param path - Relative path to the program file
 * @returns The program content
 */
export const fetchProgram = async (path: string): Promise<string> => {
  try {
    const url = `${API_ENDPOINTS.loadProgram}?path=${encodeURIComponent(path)}`;
    // Load program from API endpoint
    
    const response = await fetchWithTimeout(url);
    
    if (!response.ok) {
      let errorDetail = response.statusText;
      let errorData: any = null;
      
      try {
        errorData = await response.json();
        errorDetail = errorData.detail || errorData.error || response.statusText;
      } catch (e) {
        // If JSON parsing fails, use status text
        // Failed to parse error response - use status text
      }
      
      // Log error details for debugging
      
      throw new APIError(
        `Failed to load program: ${errorDetail}`,
        response.status,
        errorDetail
      );
    }
    
    const data: ProgramContent = await response.json();
    if (!data.content) {
      throw new APIError('Invalid response: missing content field');
    }
    
    // Successfully loaded program content
    return data.content;
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    
    // Handle network errors
    
    // Provide more specific error messages
    if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
      throw new APIError(
        'Cannot connect to API server. Please ensure the backend is running on port 8080.',
        0,
        'Network connection failed'
      );
    }
    
    throw new APIError(`Network error loading program: ${error instanceof Error ? error.message : String(error)}`);
  }
};

/**
 * Fetch information about a Recursia program
 * @param path - Relative path to the program file
 * @returns Program metadata
 */
export const fetchProgramInfo = async (path: string): Promise<ProgramInfo> => {
  try {
    const response = await fetchWithTimeout(
      `${API_ENDPOINTS.programInfo}?path=${encodeURIComponent(path)}`
    );
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new APIError(
        `Failed to get program info: ${error.detail || response.statusText}`,
        response.status,
        error.detail
      );
    }
    
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(`Network error getting program info: ${error}`);
  }
};

/**
 * Execute Recursia code
 * @param code - The Recursia code to execute
 * @returns Execution result
 */
export const executeCode = async (code: string): Promise<any> => {
  try {
    const response = await fetchWithTimeout(
      API_ENDPOINTS.execute,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code })
      },
      120000 // 120 second timeout for execution
    );
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new APIError(
        `Execution failed: ${error.detail || response.statusText}`,
        response.status,
        error.detail
      );
    }
    
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(`Network error during execution: ${error}`);
  }
};

/**
 * Check API health status
 * @returns Health check result
 */
export const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await fetchWithTimeout(API_ENDPOINTS.health, {}, 5000);
    return response.ok;
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};

/**
 * Create WebSocket connection for real-time updates
 * @param onMessage - Message handler
 * @param onError - Error handler
 * @param onClose - Close handler
 * @returns WebSocket instance
 */
export const createWebSocket = (
  onMessage: (data: any) => void,
  onError?: (error: Event) => void,
  onClose?: (event: CloseEvent) => void
): WebSocket => {
  // For WebSocket, we need to construct the full URL
  const wsUrl = API_ENDPOINTS.websocket.startsWith('/')
    ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}${API_ENDPOINTS.websocket}`
    : API_ENDPOINTS.websocket;
  
  const ws = new WebSocket(wsUrl);
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (error) {
      console.error('WebSocket message parse error:', error);
    }
  };
  
  ws.onerror = onError || ((error) => console.error('WebSocket error:', error));
  ws.onclose = onClose || ((event) => console.log('WebSocket closed:', event));
  
  return ws;
};

// Export base URL for components that need it
export { API_BASE_URL };