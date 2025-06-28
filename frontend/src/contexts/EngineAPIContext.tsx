/**
 * Engine API Context Provider
 * 
 * Provides centralized access to the Engine API throughout the application.
 * This context wraps the useEngineAPI hook to provide a single source of truth
 * for all components that need access to engine metrics, states, and controls.
 */

import React, { createContext, useContext, ReactNode } from 'react';
import { useEngineAPI } from '../hooks/useEngineAPI';

// Re-export the hook's return type for convenience
type EngineAPIContextValue = ReturnType<typeof useEngineAPI>;

// Create the context with undefined default
const EngineAPIContext = createContext<EngineAPIContextValue | undefined>(undefined);

/**
 * Provider component that wraps the app with Engine API access
 */
export const EngineAPIProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const engineAPI = useEngineAPI();
  
  
  return (
    <EngineAPIContext.Provider value={engineAPI}>
      {children}
    </EngineAPIContext.Provider>
  );
};

/**
 * Hook to access the Engine API context
 * Throws if used outside of EngineAPIProvider
 */
export const useEngineAPIContext = (): EngineAPIContextValue => {
  const context = useContext(EngineAPIContext);
  
  if (!context) {
    throw new Error('useEngineAPIContext must be used within an EngineAPIProvider');
  }
  
  return context;
};

// Export the context itself for advanced use cases
export { EngineAPIContext };