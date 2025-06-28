/**
 * Enhanced Execution Log Component
 * Provides comprehensive execution monitoring, analysis, and export capabilities
 * Features real-time filtering, search, statistics, and detailed execution traces
 */

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Tooltip } from './ui/Tooltip';
import { 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Info, 
  Download, 
  Search,
  Filter,
  BarChart,
  Code,
  Activity,
  Zap,
  Clock,
  Copy,
  FileDown,
  ChevronDown,
  ChevronUp,
  Trash2,
  Play,
  Square,
  TrendingUp,
  TrendingDown,
  GitBranch,
  Cpu,
  Brain,
  Atom
} from 'lucide-react';
import { saveAs } from 'file-saver';
import type { OSHEvidence } from '../utils/oshEvidenceEvaluator';
import { generateColorPalette, adjustLightness, mixColors } from '../utils/colorUtils';

/**
 * Extended log entry interface with comprehensive execution data
 */
export interface ExecutionLogEntry {
  id: string;
  timestamp: number;
  level: 'info' | 'success' | 'warning' | 'error' | 'debug';
  category: 'execution' | 'measurement' | 'analysis' | 'osh_evidence' | 'quantum_state' | 'optimization' | 'compilation' | 'editor' | 'performance';
  message: string;
  data?: any;
  oshEvidence?: OSHEvidence;
  details?: string;
  
  // Enhanced execution metadata
  executionContext?: {
    programName?: string;
    lineNumber?: number;
    columnNumber?: number;
    statementType?: string;
    executionTime?: number; // milliseconds
    iterations?: number;
  };
  
  // Quantum state information
  quantumState?: {
    stateName: string;
    qubits?: number;
    coherence: number;
    entropy?: number;
    fidelity?: number;
    measurement?: {
      state: string;
      qubit: number;
      outcome: number;
      probability: number;
    };
    amplitudes?: Complex[];
  };
  
  // Measurement results
  measurementResult?: {
    qubit: number;
    outcome: number;
    probability: number;
    collapseType?: 'projective' | 'weak' | 'continuous';
  };
  
  // Performance metrics
  performance?: {
    cpuUsage?: number;
    memoryUsage?: number;
    executionDuration?: number;
  };
  
  // OSH metrics from analysis
  oshMetrics?: {
    depth: number;
    strain: number;
    focus: number;
    information: number;
  };
}

interface Complex {
  real: number;
  imag: number;
}

interface EnhancedExecutionLogProps {
  entries: ExecutionLogEntry[];
  primaryColor: string;
  maxEntries?: number;
  onClearLog?: () => void;
  isExecuting?: boolean;
}

/**
 * Statistics panel showing execution summary
 */
const ExecutionStatistics: React.FC<{
  entries: ExecutionLogEntry[];
  primaryColor: string;
}> = ({ entries, primaryColor }) => {
  const palette = useMemo(() => generateColorPalette(primaryColor), [primaryColor]);
  const stats = useMemo(() => {
    const total = entries.length;
    const byLevel = entries.reduce((acc, entry) => {
      acc[entry.level] = (acc[entry.level] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const byCategory = entries.reduce((acc, entry) => {
      acc[entry.category] = (acc[entry.category] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const executionTimes = entries
      .filter(e => e.executionContext?.executionTime)
      .map(e => e.executionContext!.executionTime!);
    
    const avgExecutionTime = executionTimes.length > 0
      ? executionTimes.reduce((a, b) => a + b, 0) / executionTimes.length
      : 0;
    
    const quantumStates = entries.filter(e => e.quantumState).length;
    const measurements = entries.filter(e => e.measurementResult).length;
    const oshEvidence = entries.filter(e => e.oshEvidence).length;
    
    return {
      total,
      byLevel,
      byCategory,
      avgExecutionTime,
      quantumStates,
      measurements,
      oshEvidence
    };
  }, [entries]);
  
  return (
    <div style={{
      padding: '12px',
      background: palette.bgSecondary,
      borderRadius: '8px',
      marginBottom: '12px'
    }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
        gap: '12px'
      }}>
        <div>
          <div style={{ fontSize: '12px', color: palette.textSecondary, marginBottom: '4px' }}>Total Entries</div>
          <div style={{ fontSize: '20px', fontWeight: 'bold', color: primaryColor }}>{stats.total}</div>
        </div>
        
        <div>
          <div style={{ fontSize: '12px', color: palette.textSecondary, marginBottom: '4px' }}>Errors</div>
          <div style={{ fontSize: '20px', fontWeight: 'bold', color: palette.error }}>
            {stats.byLevel.error || 0}
          </div>
        </div>
        
        <div>
          <div style={{ fontSize: '12px', color: palette.textSecondary, marginBottom: '4px' }}>Quantum States</div>
          <div style={{ fontSize: '20px', fontWeight: 'bold', color: palette.success }}>
            {stats.quantumStates}
          </div>
        </div>
        
        <div>
          <div style={{ fontSize: '12px', color: palette.textSecondary, marginBottom: '4px' }}>Measurements</div>
          <div style={{ fontSize: '20px', fontWeight: 'bold', color: palette.info }}>
            {stats.measurements}
          </div>
        </div>
        
        <div>
          <div style={{ fontSize: '12px', color: palette.textSecondary, marginBottom: '4px' }}>Avg Exec Time</div>
          <div style={{ fontSize: '20px', fontWeight: 'bold', color: palette.warning }}>
            {stats.avgExecutionTime.toFixed(1)}ms
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Enhanced log entry component with expandable details
 * Memoized to prevent unnecessary re-renders
 */
const LogEntryComponent = React.memo<{
  entry: ExecutionLogEntry;
  primaryColor: string;
  onCopy: (text: string, entryId?: string) => void;
  copiedId: string | null;
}>(({ entry, primaryColor, onCopy, copiedId }) => {
  const [expanded, setExpanded] = useState(
    (entry.category === 'analysis' && entry.oshMetrics !== undefined) ||
    (entry.data?.rsp !== undefined) ||
    (entry.data?.measurement_count !== undefined)
  );
  
  const palette = useMemo(() => generateColorPalette(primaryColor), [primaryColor]);
  
  const getLevelIcon = () => {
    switch (entry.level) {
      case 'success': return <CheckCircle size={16} color={palette.success} />;
      case 'error': return <XCircle size={16} color={palette.error} />;
      case 'warning': return <AlertCircle size={16} color={palette.warning} />;
      case 'debug': return <Code size={16} color={palette.textTertiary} />;
      default: return <Info size={16} color={palette.info} />;
    }
  };
  
  const getCategoryIcon = () => {
    switch (entry.category) {
      case 'execution': return <Play size={14} />;
      case 'measurement': return <Activity size={14} />;
      case 'analysis': return <BarChart size={14} />;
      case 'quantum_state': return <Atom size={14} />;
      case 'optimization': return <Zap size={14} />;
      case 'compilation': return <Cpu size={14} />;
      case 'osh_evidence': return <Brain size={14} />;
      case 'editor': return <Code size={14} />;
      default: return <Info size={14} />;
    }
  };
  
  const getCategoryColor = () => {
    // For analysis category, use fixed colors based on success/error level
    if (entry.category === 'analysis') {
      if (entry.level === 'success') {
        return '#10b981'; // Fixed green for success
      } else if (entry.level === 'error') {
        return '#ef4444'; // Fixed red for error
      }
      return palette.warning; // Default yellow for other levels
    }
    
    // For other categories, use the existing color scheme
    switch (entry.category) {
      case 'execution': return palette.info;
      case 'measurement': return palette.success;
      case 'quantum_state': return adjustLightness(primaryColor, 20);
      case 'optimization': return mixColors(primaryColor, palette.error, 0.5);
      case 'compilation': return adjustLightness(palette.info, 10);
      case 'editor': return palette.textTertiary;
      case 'osh_evidence': return primaryColor;
      default: return palette.textTertiary;
    }
  };
  
  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3
    });
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      style={{
        marginBottom: '8px',
        border: `1px solid ${getCategoryColor()}30`,
        borderRadius: '6px',
        background: `linear-gradient(135deg, ${getCategoryColor()}10, ${palette.bgSecondary})`,
        boxShadow: `0 2px 8px ${getCategoryColor()}10`,
        overflow: 'hidden'
      }}
    >
      {/* Main entry header */}
      <div
        style={{
          padding: '8px 12px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          cursor: 'pointer',
          userSelect: 'none'
        }}
        onClick={() => setExpanded(!expanded)}
      >
        {getLevelIcon()}
        
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '4px',
          color: getCategoryColor(),
          fontSize: '11px',
          fontWeight: '600',
          textTransform: 'uppercase'
        }}>
          {getCategoryIcon()}
          {entry.category.replace('_', ' ')}
        </div>
        
        {entry.executionContext?.programName && (
          <div style={{ 
            fontSize: '11px', 
            color: palette.textSecondary,
            padding: '2px 6px',
            background: palette.bgHover,
            borderRadius: '4px'
          }}>
            {entry.executionContext.programName}
          </div>
        )}
        
        {entry.executionContext?.lineNumber && (
          <div style={{ 
            fontSize: '11px', 
            color: palette.textSecondary 
          }}>
            L{entry.executionContext.lineNumber}:{entry.executionContext.columnNumber || 0}
          </div>
        )}
        
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
          {/* Indicator for entries with metrics */}
          {entry.oshMetrics && (
            <div style={{
              fontSize: '10px',
              color: primaryColor,
              background: `${primaryColor}20`,
              padding: '2px 6px',
              borderRadius: '4px',
              fontWeight: '500',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}>
              <BarChart size={10} />
              {entry.data?.rsp !== undefined ? (
                <>RSP: {typeof entry.data.rsp === 'number' ? entry.data.rsp.toFixed(2) : entry.data.rsp}</>
              ) : (
                'METRICS'
              )}
            </div>
          )}
          
          {/* Measurements indicator */}
          {entry.data?.measurement_count !== undefined && entry.data.measurement_count > 0 && (
            <div style={{
              fontSize: '10px',
              color: palette.success,
              background: palette.successLight + '20',
              padding: '2px 6px',
              borderRadius: '4px',
              fontWeight: '500',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}>
              <Activity size={10} />
              {entry.data.measurement_count} MEAS
            </div>
          )}
          
          {entry.executionContext?.executionTime && (
            <div style={{ 
              fontSize: '11px', 
              color: palette.textTertiary,
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}>
              <Clock size={12} />
              {entry.executionContext.executionTime}ms
            </div>
          )}
          
          <div style={{ fontSize: '11px', color: palette.textTertiary }}>
            {formatTimestamp(entry.timestamp)}
          </div>
          
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </div>
      </div>
      
      {/* Message */}
      <Tooltip content="Click to copy message">
        <div 
          onClick={(e) => {
            e.stopPropagation();
            onCopy(entry.message, `msg-${entry.id}`);
          }}
          style={{
            padding: '0 12px 8px 12px',
            fontSize: '13px',
            color: '#fff',
            lineHeight: '1.5',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            cursor: 'pointer',
            position: 'relative',
            transition: 'background 0.2s ease',
            borderRadius: '4px'
          }}
        >
        {entry.message}
        {copiedId === `msg-${entry.id}` && (
          <span style={{
            position: 'absolute',
            top: '0',
            right: '12px',
            fontSize: '10px',
            color: '#10b981',
            background: 'rgba(0, 0, 0, 0.8)',
            padding: '2px 6px',
            borderRadius: '4px'
          }}>
            Copied!
          </span>
        )}
        </div>
      </Tooltip>
      
      {/* Expanded details */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            style={{
              borderTop: `1px solid ${getCategoryColor()}20`,
              background: 'rgba(0, 0, 0, 0.3)'
            }}
          >
            <div style={{ padding: '12px' }}>
              {/* Quantum State Details */}
              {entry.quantumState && (
                <div style={{ marginBottom: '12px' }}>
                  <h4 style={{ 
                    fontSize: '12px', 
                    fontWeight: '600', 
                    marginBottom: '8px',
                    color: getCategoryColor()
                  }}>
                    Quantum State: {entry.quantumState.stateName}
                  </h4>
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                    gap: '8px',
                    fontSize: '11px'
                  }}>
                    <div>
                      <span style={{ color: '#999' }}>Qubits:</span> {entry.quantumState.qubits}
                    </div>
                    <div>
                      <span style={{ color: '#999' }}>Coherence:</span> {entry.quantumState.coherence.toFixed(4)}
                    </div>
                    <div>
                      <span style={{ color: '#999' }}>Entropy:</span> {entry.quantumState.entropy.toFixed(4)}
                    </div>
                    {entry.quantumState.fidelity !== undefined && (
                      <div>
                        <span style={{ color: '#999' }}>Fidelity:</span> {entry.quantumState.fidelity.toFixed(4)}
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* Measurement Result */}
              {entry.measurementResult && (
                <div style={{ marginBottom: '12px' }}>
                  <h4 style={{ 
                    fontSize: '12px', 
                    fontWeight: '600', 
                    marginBottom: '8px',
                    color: getCategoryColor()
                  }}>
                    Measurement Result
                  </h4>
                  <div style={{ fontSize: '11px' }}>
                    <div>
                      <span style={{ color: '#999' }}>Qubit:</span> {entry.measurementResult.qubit} → 
                      <span style={{ fontWeight: '600', marginLeft: '4px' }}>
                        |{entry.measurementResult.outcome}⟩
                      </span>
                    </div>
                    <div>
                      <span style={{ color: '#999' }}>Probability:</span> {
                        (entry.measurementResult.probability * 100).toFixed(2)
                      }%
                    </div>
                    {entry.measurementResult.collapseType && (
                      <div>
                        <span style={{ color: '#999' }}>Type:</span> {entry.measurementResult.collapseType}
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* OSH Metrics - Enhanced Display */}
              {entry.oshMetrics && (
                <div style={{ marginBottom: '12px' }}>
                  <h4 style={{ 
                    fontSize: '12px', 
                    fontWeight: '600', 
                    marginBottom: '8px',
                    color: getCategoryColor(),
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    <BarChart size={14} />
                    OSH Analysis Results
                  </h4>
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                    gap: '12px',
                    fontSize: '12px',
                    padding: '8px',
                    background: 'rgba(255, 255, 255, 0.03)',
                    borderRadius: '4px',
                    border: `1px solid ${getCategoryColor()}20`
                  }}>
                    {/* Primary Metrics Row */}
                    {entry.data?.rsp !== undefined && (
                      <div style={{ 
                        padding: '4px 8px',
                        background: `${getCategoryColor()}15`,
                        borderRadius: '4px',
                        border: `1px solid ${getCategoryColor()}30`
                      }}>
                        <span style={{ color: '#999', fontSize: '10px', display: 'block' }}>RSP (Recursive Sim)</span>
                        <span style={{ fontWeight: '600', fontSize: '16px', color: getCategoryColor() }}>
                          {typeof entry.data.rsp === 'number' ? entry.data.rsp.toFixed(2) : entry.data.rsp}
                        </span>
                      </div>
                    )}
                    
                    {entry.data?.measurement_count !== undefined && (
                      <div style={{ 
                        padding: '4px 8px',
                        background: `${getCategoryColor()}15`,
                        borderRadius: '4px',
                        border: `1px solid ${getCategoryColor()}30`
                      }}>
                        <span style={{ color: '#999', fontSize: '10px', display: 'block' }}>Measurements</span>
                        <span style={{ fontWeight: '600', fontSize: '16px', color: getCategoryColor() }}>
                          {entry.data.measurement_count}
                        </span>
                      </div>
                    )}
                    
                    {/* Core OSH Metrics */}
                    <div>
                      <span style={{ color: '#999', fontSize: '10px' }}>Recursion Depth</span>
                      <div style={{ fontWeight: '500' }}>{entry.oshMetrics.depth}</div>
                    </div>
                    
                    <div>
                      <span style={{ color: '#999', fontSize: '10px' }}>Memory Strain</span>
                      <div style={{ fontWeight: '500' }}>{entry.oshMetrics.strain.toFixed(4)}</div>
                    </div>
                    
                    <div>
                      <span style={{ color: '#999', fontSize: '10px' }}>Observer Focus</span>
                      <div style={{ fontWeight: '500' }}>{entry.oshMetrics.focus.toFixed(4)}</div>
                    </div>
                    
                    <div>
                      <span style={{ color: '#999', fontSize: '10px' }}>Information</span>
                      <div style={{ fontWeight: '500' }}>{entry.oshMetrics.information.toFixed(2)} bits</div>
                    </div>
                    
                    {/* Additional metrics from data if available */}
                    {entry.data?.coherence !== undefined && (
                      <div>
                        <span style={{ color: '#999', fontSize: '10px' }}>Coherence</span>
                        <div style={{ fontWeight: '500' }}>{entry.data.coherence.toFixed(4)}</div>
                      </div>
                    )}
                    
                    {entry.data?.entropy !== undefined && (
                      <div>
                        <span style={{ color: '#999', fontSize: '10px' }}>Entropy</span>
                        <div style={{ fontWeight: '500' }}>{entry.data.entropy.toFixed(4)}</div>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* OSH Evidence */}
              {entry.oshEvidence && (
                <div style={{ marginBottom: '12px' }}>
                  <h4 style={{ 
                    fontSize: '12px', 
                    fontWeight: '600', 
                    marginBottom: '8px',
                    color: getCategoryColor()
                  }}>
                    OSH Evidence Analysis
                  </h4>
                  <div style={{ 
                    padding: '8px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '4px',
                    fontSize: '11px'
                  }}>
                    <div style={{ marginBottom: '4px' }}>
                      <strong>{entry.oshEvidence.evidenceType.toUpperCase()}</strong> - 
                      Strength: {(entry.oshEvidence.strength * 100).toFixed(0)}%, 
                      Confidence: {(entry.oshEvidence.confidence * 100).toFixed(0)}%
                    </div>
                    <div style={{ color: '#ccc' }}>{entry.oshEvidence.verdict}</div>
                  </div>
                </div>
              )}
              
              {/* Raw Data */}
              {entry.data && (
                <div>
                  <h4 style={{ 
                    fontSize: '12px', 
                    fontWeight: '600', 
                    marginBottom: '8px',
                    color: getCategoryColor(),
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between'
                  }}>
                    Raw Data
                    <button
                      onClick={() => onCopy(JSON.stringify(entry.data, null, 2), entry.id)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: copiedId === entry.id ? '#10b981' : '#999',
                        cursor: 'pointer',
                        padding: '4px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        transition: 'color 0.2s ease'
                      }}
                      title={copiedId === entry.id ? 'Copied!' : 'Copy to clipboard'}
                    >
                      <Copy size={12} />
                      {copiedId === entry.id && <span style={{ fontSize: '10px' }}>Copied!</span>}
                    </button>
                  </h4>
                  <pre style={{
                    fontSize: '10px',
                    color: '#ccc',
                    background: 'rgba(0, 0, 0, 0.5)',
                    padding: '8px',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '200px',
                    margin: 0
                  }}>
                    {JSON.stringify(entry.data, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison for memo optimization
  return (
    prevProps.entry.id === nextProps.entry.id &&
    prevProps.primaryColor === nextProps.primaryColor &&
    prevProps.copiedId === nextProps.copiedId
  );
});

/**
 * Main Enhanced Execution Log Component
 */
export const EnhancedExecutionLog: React.FC<EnhancedExecutionLogProps> = ({
  entries = [],
  primaryColor = '#ffd700',
  maxEntries = 2000,
  onClearLog,
  isExecuting = false
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedLevels, setSelectedLevels] = useState<Set<string>>(
    new Set(['info', 'success', 'warning', 'error', 'debug'])
  );
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(
    new Set(['execution', 'measurement', 'analysis', 'osh_evidence', 'quantum_state', 'optimization', 'compilation'])
  );
  const [autoScroll, setAutoScroll] = useState(true);
  const [showStatistics, setShowStatistics] = useState(true);
  const [categoryDropdownOpen, setCategoryDropdownOpen] = useState(false);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  
  // Generate color palette from primary color
  const palette = useMemo(() => generateColorPalette(primaryColor), [primaryColor]);
  
  // Filter entries based on search and selections
  const filteredEntries = useMemo(() => {
    return entries
      .filter(entry => {
        // Level filter
        if (!selectedLevels.has(entry.level)) return false;
        
        // Category filter
        if (!selectedCategories.has(entry.category)) return false;
        
        // Search filter
        if (searchTerm) {
          const searchLower = searchTerm.toLowerCase();
          return (
            entry.message.toLowerCase().includes(searchLower) ||
            entry.executionContext?.programName?.toLowerCase().includes(searchLower) ||
            entry.quantumState?.stateName?.toLowerCase().includes(searchLower) ||
            JSON.stringify(entry.data).toLowerCase().includes(searchLower)
          );
        }
        
        return true;
      })
      .slice(-maxEntries);
  }, [entries, searchTerm, selectedLevels, selectedCategories, maxEntries]);
  
  // State for copy feedback
  const [copiedId, setCopiedId] = useState<string | null>(null);
  
  // Export functions - defined before useEffect that uses them
  const exportAsJSON = useCallback(() => {
    const data = {
      exportDate: new Date().toISOString(),
      totalEntries: filteredEntries.length,
      entries: filteredEntries
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    saveAs(blob, `execution_log_${Date.now()}.json`);
  }, [filteredEntries]);
  
  const exportAsCSV = useCallback(() => {
    const headers = ['Timestamp', 'Level', 'Category', 'Message', 'Program', 'Line', 'Execution Time'];
    const rows = filteredEntries.map(entry => [
      new Date(entry.timestamp).toISOString(),
      entry.level,
      entry.category,
      entry.message.replace(/"/g, '""'), // Escape quotes
      entry.executionContext?.programName || '',
      entry.executionContext?.lineNumber || '',
      entry.executionContext?.executionTime || ''
    ]);
    
    const csv = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    saveAs(blob, `execution_log_${Date.now()}.csv`);
  }, [filteredEntries]);
  
  const copyToClipboard = useCallback((text: string, entryId?: string) => {
    // Check if clipboard API is available
    if (!navigator.clipboard) {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.opacity = '0';
      document.body.appendChild(textArea);
      textArea.select();
      
      try {
        document.execCommand('copy');
        if (entryId) {
          setCopiedId(entryId);
          setTimeout(() => setCopiedId(null), 2000);
        }
      } catch (err) {
        console.error('Failed to copy to clipboard (fallback):', err);
      } finally {
        document.body.removeChild(textArea);
      }
      return;
    }
    
    // Modern clipboard API
    navigator.clipboard.writeText(text).then(() => {
      if (entryId) {
        setCopiedId(entryId);
        setTimeout(() => setCopiedId(null), 2000);
      }
    }).catch(err => {
      console.error('Failed to copy to clipboard:', err);
      // Try fallback method on error
      const textArea = document.createElement('textarea');
      textArea.value = text;
      textArea.style.position = 'fixed';
      textArea.style.opacity = '0';
      document.body.appendChild(textArea);
      textArea.select();
      
      try {
        document.execCommand('copy');
        if (entryId) {
          setCopiedId(entryId);
          setTimeout(() => setCopiedId(null), 2000);
        }
      } catch (fallbackErr) {
        console.error('Failed to copy to clipboard (fallback):', fallbackErr);
      } finally {
        document.body.removeChild(textArea);
      }
    });
  }, []);
  
  // Auto-scroll effect
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [filteredEntries, autoScroll]);
  
  // Click outside handler for category dropdown
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      // Check if click is outside dropdown
      if (categoryDropdownOpen && 
          !target.closest('[data-category-dropdown]') && 
          !target.closest('[data-category-dropdown-button]')) {
        setCategoryDropdownOpen(false);
      }
    };
    
    if (categoryDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [categoryDropdownOpen]);
  
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl/Cmd + F - Focus search
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
      // Ctrl/Cmd + L - Clear logs
      else if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        if (onClearLog) {
          onClearLog();
        }
      }
      // Ctrl/Cmd + E - Export JSON
      else if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault();
        exportAsJSON();
      }
      // Escape - Clear search or close dropdown
      else if (e.key === 'Escape') {
        if (categoryDropdownOpen) {
          setCategoryDropdownOpen(false);
        } else if (searchTerm) {
          setSearchTerm('');
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [searchTerm, onClearLog, exportAsJSON, categoryDropdownOpen]);
  
  return (
    <div style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      background: `linear-gradient(135deg, rgba(0, 0, 0, 0.9), ${adjustLightness(primaryColor, -80)}20)`,
      borderRadius: '8px',
      overflow: 'hidden',
      boxShadow: `inset 0 1px 0 ${primaryColor}10`
    }}>
      {/* Header */}
      <div style={{
        padding: '12px',
        borderBottom: `1px solid ${primaryColor}40`,
        background: 'rgba(0, 0, 0, 0.4)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
          marginBottom: '10px'
        }}>
          {/* Search bar */}
          <div style={{
            position: 'relative',
            flex: 1,
            maxWidth: '300px'
          }}>
            <Search size={14} style={{
              position: 'absolute',
              left: '10px',
              top: '50%',
              transform: 'translateY(-50%)',
              color: '#666'
            }} />
            <input
              ref={searchInputRef}
              type="text"
              placeholder="Search logs... (Ctrl+F)"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              aria-label="Search execution logs"
              style={{
                width: '100%',
                padding: '6px 10px 6px 32px',
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '4px',
                color: '#fff',
                fontSize: '12px',
                outline: 'none',
                transition: 'all 0.2s ease'
              }}
              onFocus={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.2)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
              }}
            />
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            {isExecuting && (
              <div style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: '#10b981',
                animation: 'pulse 2s infinite',
                marginRight: '4px'
              }} />
            )}
            
            <button
              onClick={() => setShowStatistics(!showStatistics)}
              style={{
                padding: '4px 10px',
                background: showStatistics ? `${primaryColor}20` : 'transparent',
                border: `1px solid ${showStatistics ? primaryColor : `${primaryColor}40`}`,
                borderRadius: '4px',
                color: primaryColor,
                cursor: 'pointer',
                fontSize: '11px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                transition: 'all 0.2s ease',
                fontWeight: showStatistics ? '500' : '400'
              }}
              title={showStatistics ? 'Hide Statistics' : 'Show Statistics'}
            >
              <BarChart size={12} />
              Stats
            </button>
            
            <Tooltip content="Export logs as JSON (Ctrl+E)">
              <button
                onClick={exportAsJSON}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = `${primaryColor}20`;
                  e.currentTarget.style.borderColor = primaryColor;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.borderColor = `${primaryColor}40`;
                }}
                style={{
                  padding: '4px 10px',
                  background: 'transparent',
                  border: `1px solid ${primaryColor}40`,
                  borderRadius: '4px',
                  color: primaryColor,
                  cursor: 'pointer',
                  fontSize: '11px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  transition: 'all 0.2s ease'
                }}
              >
                <FileDown size={12} />
                JSON
              </button>
            </Tooltip>
            
            <Tooltip content="Export logs as CSV">
              <button
                onClick={exportAsCSV}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = `${primaryColor}20`;
                  e.currentTarget.style.borderColor = primaryColor;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.borderColor = `${primaryColor}40`;
                }}
                style={{
                  padding: '4px 10px',
                  background: 'transparent',
                  border: `1px solid ${primaryColor}40`,
                  borderRadius: '4px',
                  color: primaryColor,
                  cursor: 'pointer',
                  fontSize: '11px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  transition: 'all 0.2s ease'
                }}
              >
                <Download size={12} />
                CSV
              </button>
            </Tooltip>
            
            <button
              onClick={() => {
                if (onClearLog) {
                  onClearLog();
                }
              }}
              disabled={!onClearLog}
              style={{
                padding: '4px 10px',
                background: 'transparent',
                border: `1px solid #ef444440`,
                borderRadius: '4px',
                color: '#ef4444',
                cursor: onClearLog ? 'pointer' : 'not-allowed',
                fontSize: '11px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                opacity: onClearLog ? 1 : 0.5,
                transition: 'all 0.2s ease'
              }}
              title={onClearLog ? 'Clear all logs (Ctrl+L)' : 'Clear function not provided'}
            >
              <Trash2 size={12} />
              Clear
            </button>
          </div>
        </div>
        
        {/* Filters */}
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          {/* Level filters */}
          <div>
            <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>Levels</div>
            <div style={{ display: 'flex', gap: '4px' }}>
              {['info', 'success', 'warning', 'error', 'debug'].map(level => (
                <label
                  key={level}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '2px',
                    fontSize: '11px',
                    color: selectedLevels.has(level) ? palette.textPrimary : palette.textTertiary,
                    cursor: 'pointer'
                  }}
                >
                  <input
                    type="checkbox"
                    checked={selectedLevels.has(level)}
                    onChange={(e) => {
                      const newLevels = new Set(selectedLevels);
                      if (e.target.checked) {
                        newLevels.add(level);
                      } else {
                        newLevels.delete(level);
                      }
                      setSelectedLevels(newLevels);
                    }}
                    style={{ 
                      width: '12px', 
                      height: '12px',
                      accentColor: primaryColor,
                      cursor: 'pointer'
                    }}
                  />
                  {level}
                </label>
              ))}
            </div>
          </div>
          
          {/* Category filters dropdown */}
          <div style={{ position: 'relative' }}>
            <div style={{ fontSize: '11px', color: palette.textTertiary, marginBottom: '4px' }}>Categories</div>
            <button
              data-category-dropdown-button
              onClick={() => setCategoryDropdownOpen(!categoryDropdownOpen)}
              style={{
                padding: '6px 12px',
                background: palette.bgHover,
                border: `1px solid ${palette.borderPrimary}`,
                borderRadius: '4px',
                color: palette.textPrimary,
                fontSize: '12px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                minWidth: '180px',
                justifyContent: 'space-between',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                e.currentTarget.style.borderColor = palette.borderSecondary;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = palette.bgHover;
                e.currentTarget.style.borderColor = palette.borderPrimary;
              }}
            >
              <span>
                {selectedCategories.size === 0 ? 'All Categories' : 
                 selectedCategories.size === 7 ? 'All Categories' :
                 `${selectedCategories.size} selected`}
              </span>
              <ChevronDown size={14} style={{ 
                transform: categoryDropdownOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s ease'
              }} />
            </button>
            
            {/* Dropdown menu */}
            <AnimatePresence>
              {categoryDropdownOpen && (
                <motion.div
                  data-category-dropdown
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  style={{
                    position: 'absolute',
                    top: '100%',
                    left: 0,
                    marginTop: '4px',
                    background: palette.bgPrimary + 'f2',
                    border: `1px solid ${primaryColor}40`,
                    borderRadius: '4px',
                    padding: '8px',
                    minWidth: '200px',
                    zIndex: 10,
                    boxShadow: `0 4px 12px ${palette.bgPrimary}66`
                  }}
                >
                  {/* All/None selector */}
                  <label
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      padding: '6px 8px',
                      fontSize: '12px',
                      color: palette.textPrimary,
                      cursor: 'pointer',
                      borderRadius: '4px',
                      transition: 'background 0.2s ease',
                      borderBottom: `1px solid ${palette.borderPrimary}`,
                      marginBottom: '4px'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                  >
                    <input
                      type="checkbox"
                      checked={selectedCategories.size === 7}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedCategories(new Set(['execution', 'measurement', 'analysis', 'osh_evidence', 'quantum_state', 'optimization', 'compilation']));
                        } else {
                          setSelectedCategories(new Set());
                        }
                      }}
                      style={{ width: '14px', height: '14px', cursor: 'pointer' }}
                    />
                    <span style={{ fontWeight: '600' }}>Select All</span>
                  </label>
                  
                  {/* Individual category options */}
                  {['execution', 'measurement', 'analysis', 'osh_evidence', 'quantum_state', 'optimization', 'compilation'].map(category => {
                    const count = entries.filter(e => e.category === category).length;
                    return (
                      <label
                        key={category}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          gap: '8px',
                          padding: '6px 8px',
                          fontSize: '12px',
                          color: selectedCategories.has(category) ? '#fff' : '#999',
                          cursor: 'pointer',
                          borderRadius: '4px',
                          transition: 'background 0.2s ease'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)'}
                        onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <input
                            type="checkbox"
                            checked={selectedCategories.has(category)}
                            onChange={(e) => {
                              const newCategories = new Set(selectedCategories);
                              if (e.target.checked) {
                                newCategories.add(category);
                              } else {
                                newCategories.delete(category);
                              }
                              setSelectedCategories(newCategories);
                            }}
                            style={{ width: '14px', height: '14px', cursor: 'pointer' }}
                          />
                          <span style={{ textTransform: 'capitalize' }}>
                            {category.replace('_', ' ')}
                          </span>
                        </div>
                        <span style={{ 
                          fontSize: '11px', 
                          color: '#666',
                          fontFamily: 'monospace'
                        }}>
                          {count}
                        </span>
                      </label>
                    );
                  })}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
      
      {/* Statistics Panel */}
      {showStatistics && (
        <div style={{ padding: '0 16px' }}>
          <ExecutionStatistics entries={filteredEntries} primaryColor={primaryColor} />
        </div>
      )}
      
      {/* Log entries */}
      <div
        ref={logContainerRef}
        style={{
          flex: 1,
          overflow: 'auto',
          padding: '16px'
        }}
        onScroll={(e) => {
          const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
          setAutoScroll(scrollTop + clientHeight >= scrollHeight - 10);
        }}
      >
        <AnimatePresence>
          {filteredEntries.map((entry) => (
            <LogEntryComponent
              key={entry.id}
              entry={entry}
              primaryColor={primaryColor}
              onCopy={copyToClipboard}
              copiedId={copiedId}
            />
          ))}
        </AnimatePresence>
        
        {filteredEntries.length === 0 && (
          <div style={{
            textAlign: 'center',
            padding: '48px',
            color: '#666'
          }}>
            <Activity size={48} style={{ marginBottom: '16px', opacity: 0.3 }} />
            <div>No log entries match your filters</div>
          </div>
        )}
      </div>
      
      {/* Footer */}
      <div style={{
        padding: '12px 16px',
        borderTop: `1px solid ${primaryColor}40`,
        background: 'rgba(0, 0, 0, 0.4)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ fontSize: '12px', color: '#666' }}>
          Showing {filteredEntries.length} of {entries.length} entries
        </div>
        
        <label style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '12px',
          color: primaryColor,
          cursor: 'pointer'
        }}>
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
            style={{ accentColor: primaryColor }}
          />
          Auto-scroll
        </label>
      </div>
      
      <style>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
      `}</style>
    </div>
  );
};

export default EnhancedExecutionLog;