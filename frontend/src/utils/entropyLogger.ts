export interface LogEntry {
  timestamp: number;
  level: 'info' | 'warning' | 'error' | 'critical';
  category: 'quantum' | 'memory' | 'rsp' | 'observer' | 'simulation';
  message: string;
  data?: any;
  entropy?: number;
}

export class EntropyLogger {
  private logs: LogEntry[] = [];
  private entropyThreshold: number = 0.7;
  private maxLogs: number = 10000;
  private listeners: ((entry: LogEntry) => void)[] = [];
  
  constructor(entropyThreshold: number = 0.7) {
    this.entropyThreshold = entropyThreshold;
  }
  
  log(
    level: LogEntry['level'],
    category: LogEntry['category'],
    message: string,
    data?: any,
    entropy?: number
  ): void {
    const entry: LogEntry = {
      timestamp: Date.now(),
      level,
      category,
      message,
      data,
      entropy
    };
    
    // Only log if entropy is above threshold (if provided)
    if (entropy !== undefined && entropy < this.entropyThreshold) {
      return;
    }
    
    this.logs.push(entry);
    
    // Maintain max log size
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }
    
    // Notify listeners
    this.listeners.forEach(listener => listener(entry));
  }
  
  info(category: LogEntry['category'], message: string, data?: any, entropy?: number): void {
    this.log('info', category, message, data, entropy);
  }
  
  warning(category: LogEntry['category'], message: string, data?: any, entropy?: number): void {
    this.log('warning', category, message, data, entropy);
  }
  
  error(category: LogEntry['category'], message: string, data?: any, entropy?: number): void {
    this.log('error', category, message, data, entropy);
  }
  
  critical(category: LogEntry['category'], message: string, data?: any, entropy?: number): void {
    this.log('critical', category, message, data, entropy);
  }
  
  getLogs(filter?: {
    level?: LogEntry['level'];
    category?: LogEntry['category'];
    startTime?: number;
    endTime?: number;
    minEntropy?: number;
  }): LogEntry[] {
    let filtered = [...this.logs];
    
    if (filter) {
      if (filter.level) {
        filtered = filtered.filter(log => log.level === filter.level);
      }
      if (filter.category) {
        filtered = filtered.filter(log => log.category === filter.category);
      }
      if (filter.startTime !== undefined) {
        filtered = filtered.filter(log => log.timestamp >= filter.startTime!);
      }
      if (filter.endTime !== undefined) {
        filtered = filtered.filter(log => log.timestamp <= filter.endTime!);
      }
      if (filter.minEntropy !== undefined) {
        filtered = filtered.filter(log => log.entropy !== undefined && log.entropy >= filter.minEntropy);
      }
    }
    
    return filtered;
  }
  
  subscribe(listener: (entry: LogEntry) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }
  
  exportLogs(): string {
    return JSON.stringify(this.logs, null, 2);
  }
  
  clear(): void {
    this.logs = [];
  }
  
  setEntropyThreshold(threshold: number): void {
    this.entropyThreshold = threshold;
  }
  
  // Analyze log patterns
  analyzePatterns(): {
    totalLogs: number;
    logsByLevel: Record<LogEntry['level'], number>;
    logsByCategory: Record<LogEntry['category'], number>;
    averageEntropy: number;
    criticalEvents: LogEntry[];
  } {
    const logsByLevel: Record<LogEntry['level'], number> = {
      info: 0,
      warning: 0,
      error: 0,
      critical: 0
    };
    
    const logsByCategory: Record<LogEntry['category'], number> = {
      quantum: 0,
      memory: 0,
      rsp: 0,
      observer: 0,
      simulation: 0
    };
    
    let totalEntropy = 0;
    let entropyCount = 0;
    const criticalEvents: LogEntry[] = [];
    
    this.logs.forEach(log => {
      logsByLevel[log.level]++;
      logsByCategory[log.category]++;
      
      if (log.entropy !== undefined) {
        totalEntropy += log.entropy;
        entropyCount++;
      }
      
      if (log.level === 'critical') {
        criticalEvents.push(log);
      }
    });
    
    return {
      totalLogs: this.logs.length,
      logsByLevel,
      logsByCategory,
      averageEntropy: entropyCount > 0 ? totalEntropy / entropyCount : 0,
      criticalEvents
    };
  }
}