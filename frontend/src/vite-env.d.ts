/// <reference types="vite/client" />

/**
 * Environment variable types for Vite
 */
interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  readonly MODE: 'development' | 'production' | 'test';
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly SSR: boolean;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/**
 * Worker module declarations
 */
declare module '*?worker' {
  const workerConstructor: {
    new (): Worker;
  };
  export default workerConstructor;
}

declare module '*.worker?worker' {
  const workerConstructor: {
    new (): Worker;
  };
  export default workerConstructor;
}