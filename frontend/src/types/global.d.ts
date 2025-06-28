/**
 * Global type declarations for the frontend
 */

declare global {
  interface Window {
    DiagnosticsSystem: any;
  }
  
  declare var toast: {
    success: (message: string) => void;
    error: (message: string) => void;
    info: (message: string) => void;
    warning: (message: string) => void;
  };
}

export {};