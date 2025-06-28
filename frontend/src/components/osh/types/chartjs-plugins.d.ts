// Type declarations for Chart.js plugins

declare module 'chartjs-plugin-zoom' {
  const plugin: any;
  export default plugin;
}

declare module 'chartjs-plugin-annotation' {
  const plugin: any;
  export default plugin;
}

declare module 'papaparse' {
  export interface ParseConfig {
    delimiter?: string;
    newline?: string;
    quoteChar?: string;
    escapeChar?: string;
    header?: boolean;
    transformHeader?: (header: string, index: number) => string;
    dynamicTyping?: boolean;
    preview?: number;
    encoding?: string;
    worker?: boolean;
    comments?: boolean | string;
    step?: (results: ParseResult<any>, parser: Parser) => void;
    complete?: (results: ParseResult<any>) => void;
    error?: (error: ParseError, file?: any) => void;
    download?: boolean;
    downloadRequestHeaders?: { [key: string]: string };
    skipEmptyLines?: boolean | 'greedy';
    chunk?: (results: ParseResult<any>, parser: Parser) => void;
    fastMode?: boolean;
    beforeFirstChunk?: (chunk: string) => string | void;
    withCredentials?: boolean;
    transform?: (value: string, field: string | number) => any;
    delimitersToGuess?: string[];
  }

  export interface ParseError {
    type: string;
    code: string;
    message: string;
    row?: number;
  }

  export interface ParseMeta {
    delimiter: string;
    linebreak: string;
    aborted: boolean;
    fields?: string[];
    truncated: boolean;
  }

  export interface ParseResult<T> {
    data: T[];
    errors: ParseError[];
    meta: ParseMeta;
  }

  export interface Parser {
    abort(): void;
    pause(): void;
    resume(): void;
  }

  export function parse<T>(
    input: string | File | NodeJS.ReadableStream,
    config?: ParseConfig
  ): ParseResult<T>;

  export function unparse(
    data: any[] | { fields: string[]; data: any[] },
    config?: UnparseConfig
  ): string;

  export interface UnparseConfig {
    quotes?: boolean | boolean[];
    quoteChar?: string;
    escapeChar?: string;
    delimiter?: string;
    header?: boolean;
    newline?: string;
    skipEmptyLines?: boolean | 'greedy';
    columns?: string[];
  }

  const Papa: {
    parse: typeof parse;
    unparse: typeof unparse;
    BAD_DELIMITERS: string[];
    RECORD_SEP: string;
    UNIT_SEP: string;
    WORKERS_SUPPORTED: boolean;
    LocalChunkSize: number;
    RemoteChunkSize: number;
    DefaultDelimiter: string;
  };

  export default Papa;
}