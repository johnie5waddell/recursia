import React, { useState, useRef, useEffect, useCallback } from 'react';
import Editor from '@monaco-editor/react';
import type { Monaco as MonacoType } from '@monaco-editor/react';
import { 
  Code, Play, Save, FileText, Package, Zap, Cpu, 
  Brain, GitBranch, Copy, Download, Upload, Info,
  Database, Activity, FlaskConical, Layers, Settings2,
  FilePlus, FileX, AlertTriangle, Microscope
} from 'lucide-react';
import { Tooltip } from './ui/Tooltip';
import { SaveAsDialog } from './ui/SaveAsDialog';
import { advancedQuantumPrograms } from '../data/advancedQuantumPrograms';
import quantumSnippets from '../data/quantumCodeSnippets.json';
import { loadSavedCircuits, circuitToCodeBlock, parseRecursiaToCircuit } from '../utils/circuitIntegration';
import { 
  saveCustomProgram, 
  updateCustomProgram, 
  loadCustomPrograms,
  CustomProgram 
} from '../utils/programStorage';

interface QuantumCodeEditorProps {
  onRun?: (code: string) => void;
  onSave?: (code: string, programInfo?: { id: string; name: string; isCustom: boolean }) => void;
  onNewFile?: () => void;
  onProgramSelect?: (programName: string, programId?: string, isCustom?: boolean) => void;
  primaryColor: string;
  initialCode?: string;
  onCodeChange?: (code: string) => void;
  currentProgramName?: string;
  currentProgramId?: string;
  isCurrentProgramCustom?: boolean;
}

/**
 * Confirmation dialog component for unsaved changes
 */
const ConfirmationDialog: React.FC<{
  isOpen: boolean;
  title: string;
  message: string;
  onSave: () => void;
  onDontSave: () => void;
  onCancel: () => void;
  primaryColor: string;
}> = ({ isOpen, title, message, onSave, onDontSave, onCancel, primaryColor }) => {
  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="confirmation-backdrop"
        onClick={onCancel}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          backdropFilter: 'blur(4px)'
        }}
      />
      
      {/* Dialog */}
      <div 
        className="confirmation-dialog"
        style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: '#1a1a1a',
          border: `1px solid ${primaryColor}40`,
          borderRadius: '8px',
          padding: '2rem',
          minWidth: '400px',
          maxWidth: '500px',
          zIndex: 10000,
          boxShadow: `0 0 40px ${primaryColor}40`
        }}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
          <AlertTriangle size={24} color={primaryColor} style={{ marginRight: '1rem', flexShrink: 0 }} />
          <div>
            <h3 style={{ margin: 0, marginBottom: '0.5rem', color: primaryColor }}>{title}</h3>
            <p style={{ margin: 0, color: '#ccc', fontSize: '0.875rem', lineHeight: '1.5' }}>{message}</p>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
          <button
            onClick={onCancel}
            className="dialog-btn"
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#333',
              border: '1px solid #666',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.875rem',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#444';
              e.currentTarget.style.borderColor = '#888';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#333';
              e.currentTarget.style.borderColor = '#666';
            }}
          >
            Cancel
          </button>
          
          <button
            onClick={onDontSave}
            className="dialog-btn"
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#444',
              border: '1px solid #666',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.875rem',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#555';
              e.currentTarget.style.borderColor = '#888';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#444';
              e.currentTarget.style.borderColor = '#666';
            }}
          >
            Don't Save
          </button>
          
          <button
            onClick={onSave}
            className="dialog-btn primary"
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: primaryColor,
              border: `1px solid ${primaryColor}`,
              borderRadius: '4px',
              color: '#000',
              cursor: 'pointer',
              fontSize: '0.875rem',
              fontWeight: '500',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = primaryColor + 'cc';
              e.currentTarget.style.transform = 'translateY(-1px)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = primaryColor;
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            Save Changes
          </button>
        </div>
      </div>
    </>
  );
};

// Icon mapping for categories
const categoryIcons: Record<string, React.ReactNode> = {
  quantum_basics: <Cpu size={14} />,
  entanglement: <GitBranch size={14} />,
  observers: <Brain size={14} />,
  memory_fields: <Database size={14} />,
  osh_physics: <Zap size={14} />,
  quantum_algorithms: <Activity size={14} />,
  consciousness_experiments: <FlaskConical size={14} />,
  advanced_osh: <Layers size={14} />,
  control_flow: <GitBranch size={14} />,
  advanced_statements: <Layers size={14} />,
  utilities: <Settings2 size={14} />,
  experiments: <FlaskConical size={14} />
};

// Transform JSON data to component format
const codeSnippets = Object.entries(quantumSnippets.categories).reduce((acc, [key, category]) => {
  acc[key] = {
    title: category.title,
    icon: categoryIcons[key] || <Code size={14} />,
    description: category.description,
    snippets: category.snippets
  };
  return acc;
}, {} as Record<string, any>);

/**
 * Truncate text to a maximum number of characters
 */
const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
};

export const QuantumCodeEditor: React.FC<QuantumCodeEditorProps> = ({ 
  onRun, 
  onSave, 
  onNewFile,
  onProgramSelect, 
  primaryColor, 
  initialCode = '', 
  onCodeChange,
  currentProgramName,
  currentProgramId,
  isCurrentProgramCustom = false
}) => {
  const defaultCode = `// Quantum OSH Program
// Use the building blocks on the right to get started

// Define a quantum state
state QuantumSystem : quantum {
  state_qubits: 3
}

// Define an observer
observer ConsciousnessProbe {
  focus: 0.9,
  phase: 0.0,
  collapse_threshold: 0.5
}

// Your quantum algorithm here...
print "Hello, Quantum World!";

// Apply quantum gates
apply H to QuantumSystem qubit 0;
apply CNOT to QuantumSystem qubits 0, 1;

// Measure quantum state
measure QuantumSystem qubit 0 into result;
`;

  const [code, setCode] = useState(initialCode || defaultCode);
  const [selectedCategory, setSelectedCategory] = useState<keyof typeof codeSnippets>('quantum_basics');
  const [showInfo, setShowInfo] = useState(false);
  const [isModified, setIsModified] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [pendingAction, setPendingAction] = useState<{ type: 'newFile' | 'selectProgram' | 'custom'; payload?: any } | null>(null);
  const [lastSavedCode, setLastSavedCode] = useState(initialCode || defaultCode);
  const [showSaveAsDialog, setShowSaveAsDialog] = useState(false);
  const editorRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Update code when initialCode changes
  useEffect(() => {
    if (initialCode) {
      setCode(initialCode);
      setLastSavedCode(initialCode);
      setIsModified(false);
    }
  }, [initialCode]);

  /**
   * Check for unsaved changes and show confirmation if needed
   */
  const checkUnsavedChanges = useCallback((action: { type: 'newFile' | 'selectProgram'; payload?: any }) => {
    if (isModified) {
      setPendingAction(action);
      setShowConfirmation(true);
      return true;
    }
    return false;
  }, [isModified]);

  // Track modifications
  useEffect(() => {
    const modified = code !== lastSavedCode;
    setIsModified(modified);
  }, [code, lastSavedCode]);

  // Listen for program selection requests from the library
  useEffect(() => {
    const handleProgramSelectRequest = (event: CustomEvent) => {
      const { program } = event.detail;
      if (program && program.name) {
        // Check for unsaved changes
        if (checkUnsavedChanges({ type: 'selectProgram', payload: program.name })) {
          return;
        }
        // If no unsaved changes, proceed with selection
        onProgramSelect?.(program.name);
      }
    };

    const container = containerRef.current;
    if (container) {
      container.addEventListener('program-select-request', handleProgramSelectRequest as EventListener);
      return () => {
        container.removeEventListener('program-select-request', handleProgramSelectRequest as EventListener);
      };
    }
  }, [checkUnsavedChanges, onProgramSelect]);

  // Listen for circuit designer code updates
  useEffect(() => {
    const handleUpdateCodeEditor = (event: CustomEvent) => {
      const { code: newCode, filename } = event.detail;
      if (newCode) {
        // Store the new code in pending action for handling after save decision
        const payload = { name: filename || 'circuit.recursia', code: newCode };
        
        // Check for unsaved changes
        if (isModified) {
          setPendingAction({ type: 'selectProgram', payload });
          setShowConfirmation(true);
          return;
        }
        
        // If no unsaved changes, update the code immediately
        setCode(newCode);
        setLastSavedCode(newCode);
        setIsModified(false);
        if (filename && onProgramSelect) {
          onProgramSelect(filename);
        }
      }
    };

    window.addEventListener('updateCodeEditor', handleUpdateCodeEditor as EventListener);
    return () => {
      window.removeEventListener('updateCodeEditor', handleUpdateCodeEditor as EventListener);
    };
  }, [isModified, onProgramSelect]);

  // Notify parent of code changes
  useEffect(() => {
    if (onCodeChange) {
      onCodeChange(code);
    }
  }, [code, onCodeChange]);

  /**
   * Handle new file creation
   */
  const handleNewFile = useCallback((checkChanges = true) => {
    if (checkChanges && checkUnsavedChanges({ type: 'newFile' })) {
      return;
    }

    const newFileCode = `// New Quantum Program
// Created: ${new Date().toLocaleString()}

state MyQuantumState : quantum {
  state_qubits: 2
}

// Start coding here...
`;
    
    setCode(newFileCode);
    setLastSavedCode(newFileCode);
    setIsModified(false);
    onNewFile?.();
  }, [checkUnsavedChanges, onNewFile]);

  /**
   * Handle save action from confirmation dialog
   */
  const handleConfirmSave = useCallback(() => {
    if (onSave) {
      onSave(code);
      setLastSavedCode(code);
      setIsModified(false);
    }
    setShowConfirmation(false);
    
    // Execute pending action
    if (pendingAction) {
      if (pendingAction.type === 'newFile') {
        handleNewFile(false);
      } else if (pendingAction.type === 'selectProgram' && pendingAction.payload) {
        // Handle program selection after save
        const { name, code } = pendingAction.payload;
        if (name && code) {
          setCode(code);
          setLastSavedCode(code);
          setIsModified(false);
          onProgramSelect?.(name);
        }
      }
    }
    setPendingAction(null);
  }, [code, onSave, pendingAction, onProgramSelect, handleNewFile]);

  /**
   * Handle don't save from confirmation dialog
   */
  const handleConfirmDontSave = useCallback(() => {
    setShowConfirmation(false);
    setIsModified(false);
    
    // Execute pending action without saving
    if (pendingAction) {
      if (pendingAction.type === 'newFile') {
        handleNewFile(false);
      } else if (pendingAction.type === 'selectProgram' && pendingAction.payload) {
        // Handle program selection without save
        const { name, code } = pendingAction.payload;
        if (name && code) {
          setCode(code);
          setLastSavedCode(code);
          setIsModified(false);
          onProgramSelect?.(name);
        }
      }
    }
    setPendingAction(null);
  }, [pendingAction, onProgramSelect, handleNewFile]);

  /**
   * Handle cancel from confirmation dialog
   */
  const handleConfirmCancel = useCallback(() => {
    setShowConfirmation(false);
    setPendingAction(null);
  }, []);

  /**
   * Handle save button click
   */
  const handleSave = useCallback(() => {
    // If it's a custom program that already exists, update it
    if (isCurrentProgramCustom && currentProgramId) {
      const updated = updateCustomProgram(currentProgramId, { code });
      if (updated && onSave) {
        onSave(code, { 
          id: updated.id, 
          name: updated.name, 
          isCustom: true 
        });
        setLastSavedCode(code);
        setIsModified(false);
      }
    } else {
      // For built-in programs or new programs, show save as dialog
      setShowSaveAsDialog(true);
    }
  }, [code, onSave, isCurrentProgramCustom, currentProgramId]);

  /**
   * Handle save as
   */
  const handleSaveAs = useCallback((name: string, description: string) => {
    const savedProgram = saveCustomProgram({
      name,
      description,
      code
    });
    
    if (onSave) {
      onSave(code, { 
        id: savedProgram.id, 
        name: savedProgram.name, 
        isCustom: true 
      });
    }
    
    // Update current program info
    if (onProgramSelect) {
      onProgramSelect(savedProgram.name, savedProgram.id, true);
    }
    
    setLastSavedCode(code);
    setIsModified(false);
    setShowSaveAsDialog(false);
  }, [code, onSave, onProgramSelect]);

  /**
   * Insert code snippet at cursor position
   */
  const insertSnippet = (snippet: string) => {
    if (editorRef.current) {
      const editor = editorRef.current;
      const position = editor.getPosition();
      editor.executeEdits('', [{
        range: {
          startLineNumber: position.lineNumber,
          startColumn: position.column,
          endLineNumber: position.lineNumber,
          endColumn: position.column
        },
        text: snippet + '\n\n'
      }]);
      editor.focus();
    }
  };

  const handleEditorMount = (editor: any, monaco: MonacoType) => {
    editorRef.current = editor;
    
    // Add custom language definition for Recursia
    monaco.languages.register({ id: 'recursia' });
    
    // Set token provider for syntax highlighting
    monaco.languages.setMonarchTokensProvider('recursia', {
      keywords: [
        // Core declarations (from grammar)
        'state', 'observer', 'pattern', 'field', 'measurement',
        // Types (complete from grammar)
        'quantum_type', 'observer_type', 'field_type', 'qubit_type', 'entangled_type', 
        'superposition_type', 'state_vector_type', 'density_matrix_type',
        'standard_observer', 'quantum_observer', 'meta_observer', 'holographic_observer',
        'number_type', 'string_type', 'boolean_type', 'null_type',
        'complex_type', 'vector_type', 'matrix_type', 'tensor_type',
        // Control flow (complete from grammar)
        'if', 'else', 'elseif', 'when', 'while', 'for', 'function', 'return',
        'break', 'continue', 'recursive',
        // Quantum operations (complete from grammar)
        'measure', 'entangle', 'teleport', 'apply', 'evolve', 'observe',
        'cohere', 'render', 'visualize', 'simulate', 'align', 'defragment',
        'hook', 'reset', 'print', 'log',
        // Keywords (complete from grammar)
        'let', 'const', 'import', 'export', 'as', 'with', 'to', 'from', 
        'using', 'in', 'by', 'until', 'all', 'any', 'each', 'group', 
        'self', 'system', 'null', 'true', 'false', 'undefined',
        'Infinity', 'NaN', 'default', 'at', 'and', 'or', 'xor', 'implies',
        'iff', 'not', 'complex', 'vec', 'mat', 'tensor', 'statevec',
        'density', 'anticontrol', 'protocol', 'into', 'phase', 'steps',
        'ticks', 'cycles', 'basis', 'formatted', 'scope', 'focus', 'target',
        'epoch', 'level', 'remove', 'exists', 'of', 'entanglement', 'network',
        'evolution', 'probability', 'distribution', 'wavefunction', 'matrix',
        'bloch', 'sphere', 'quantum', 'trajectory', 'circuit', 'correlation',
        'between', 'mode', 'external_computation', 'script', 'arguments', 'timeout',
        'execute',
        // Quantum specific (extended from grammar)
        'qubit', 'qubits', 'control', 'controls', 'params', 'anticontrol',
        // Access modifiers
        'public', 'private', 'protected', 'internal'
      ],
      
      operators: [
        '=', '>', '<', '!', '~', '?', ':', '==', '<=', '>=', '!=',
        '&&', '||', '++', '--', '+', '-', '*', '/', '&', '|', '^', '%',
        '<<', '>>', '>>>', '+=', '-=', '*=', '/=', '&=', '|=', '^=',
        '%=', '<<=', '>>=', '>>>='
      ],
      
      symbols: /[=><!~?:&|+\-*\/\^%]+/,
      
      tokenizer: {
        root: [
          // Quantum notation and mathematical symbols
          [/\|[0-9]+⟩/, 'quantum-state'],
          [/\|[01]+⟩/, 'quantum-state'],
          [/⟨[0-9]+\|/, 'quantum-bra'],
          [/√[0-9]*/, 'quantum-sqrt'],
          [/[πΣ∞αβγδεζηθικλμνξοπρστυφχψω]/, 'quantum-symbol'],
          [/[ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]/, 'quantum-symbol'],
          [/\d*\.?\d*[eE][\-+]?\d*[ij]/, 'complex-number'],
          [/[0-9]*\.?[0-9]+\s*\+\s*[0-9]*\.?[0-9]+[ij]/, 'complex-number'],
          
          // Gate names (special highlighting)
          [/[A-Z]+_gate/, 'gate-name'],
          [/H|X|Y|Z|S|T|CNOT|SWAP|TOFFOLI|RX|RY|RZ|CZ|PHASE/, 'gate-name'],
          
          // Keywords
          [/[a-z_$][\w$]*/, {
            cases: {
              '@keywords': 'keyword',
              '@default': 'identifier'
            }
          }],
          
          // Numbers
          [/\d*\.\d+([eE][\-+]?\d+)?/, 'number.float'],
          [/0[xX][0-9a-fA-F]+/, 'number.hex'],
          [/\d+/, 'number'],
          
          // Strings
          [/"([^"\\]|\\.)*$/, 'string.invalid'],
          [/"/, { token: 'string.quote', bracket: '@open', next: '@string' }],
          
          // Comments
          [/\/\/.*$/, 'comment'],
          [/\/\*/, 'comment', '@comment'],
          
          // Delimiters
          [/[{}()\[\]]/, '@brackets'],
          [/[<>](?!@symbols)/, '@brackets'],
          [/@symbols/, {
            cases: {
              '@operators': 'operator',
              '@default': ''
            }
          }]
        ],
        
        string: [
          [/[^\\"]+/, 'string'],
          [/\\./, 'string.escape.invalid'],
          [/"/, { token: 'string.quote', bracket: '@close', next: '@pop' }]
        ],
        
        comment: [
          [/[^\/*]+/, 'comment'],
          [/\/\*/, 'comment', '@push'],
          [/\*\//, 'comment', '@pop'],
          [/[\/*]/, 'comment']
        ]
      }
    });
    
    // Set theme colors
    monaco.editor.defineTheme('quantum-dark', {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'quantum-state', foreground: '00d4ff', fontStyle: 'bold' },
        { token: 'quantum-bra', foreground: '00d4ff', fontStyle: 'bold' },
        { token: 'quantum-sqrt', foreground: 'ff00ff', fontStyle: 'bold' },
        { token: 'quantum-symbol', foreground: 'ffd700', fontStyle: 'bold' },
        { token: 'complex-number', foreground: 'ff6b9d', fontStyle: 'italic' },
        { token: 'gate-name', foreground: 'ff8c00', fontStyle: 'bold' },
        { token: 'keyword', foreground: '00ff88', fontStyle: 'bold' },
        { token: 'comment', foreground: '6a9955', fontStyle: 'italic' },
        { token: 'string', foreground: 'ce9178' },
        { token: 'number', foreground: 'b5cea8' }
      ],
      colors: {
        'editor.background': '#000000',
        'editor.foreground': '#ffffff',
        'editor.lineHighlightBackground': '#1a1a1a66',
        'editor.lineHighlightBorder': 'transparent',
        'editorCursor.foreground': primaryColor,
        'editor.selectionBackground': primaryColor + '40',
        'editor.selectionHighlightBackground': primaryColor + '20',
        'editorLineNumber.foreground': '#666666',
        'editorLineNumber.activeForeground': primaryColor,
        'editorIndentGuide.background': '#1a1a1a',
        'editorIndentGuide.activeBackground': '#333333',
        'editorGutter.background': '#000000',
        'editorOverviewRuler.border': 'transparent',
        'scrollbar.shadow': 'transparent',
        'editor.overviewRulerBorder': 'transparent'
      }
    });
  };

  return (
    <div className="quantum-code-editor" ref={containerRef}>
      <div className="editor-toolbar">
        <div className="toolbar-left">
          <Tooltip primaryColor={primaryColor} content="Create new file and clear editor">
            <button 
              className="editor-btn"
              onClick={() => handleNewFile()}
            >
              <FilePlus size={14} />
              New
            </button>
          </Tooltip>

          <Tooltip primaryColor={primaryColor} content={`Run quantum program${currentProgramName ? ` (${currentProgramName})` : ''}`}>
            <button 
              className="editor-btn primary"
              onClick={() => onRun?.(code)}
            >
              <Play size={14} />
              Run
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content={`Save current program${currentProgramName ? ` (${currentProgramName})` : ''}${isModified ? ' (unsaved changes)' : ''}`}>
            <button 
              className={`editor-btn ${isModified ? 'unsaved' : ''}`}
              onClick={handleSave}
              style={{
                position: 'relative',
                ...(isModified ? {
                  borderColor: primaryColor + '80',
                  animation: 'pulse 2s infinite'
                } : {})
              }}
            >
              <Save size={14} />
              {isCurrentProgramCustom ? 'Save' : 'Save As'}
              {isModified && (
                <span 
                  style={{
                    position: 'absolute',
                    top: '-2px',
                    right: '-2px',
                    width: '6px',
                    height: '6px',
                    backgroundColor: primaryColor,
                    borderRadius: '50%',
                    boxShadow: `0 0 4px ${primaryColor}`
                  }}
                />
              )}
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Copy code to clipboard">
            <button 
              className="editor-btn"
              onClick={() => {
                navigator.clipboard.writeText(code);
                // Could add a toast notification here
              }}
            >
              <Copy size={14} />
              Copy
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Download program as .recursia file">
            <button 
              className="editor-btn"
              onClick={() => {
                const blob = new Blob([code], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = currentProgramName 
                  ? `${currentProgramName.replace(/\s+/g, '_').toLowerCase()}.recursia`
                  : 'quantum_program.recursia';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
              }}
            >
              <Download size={14} />
              Export
            </button>
          </Tooltip>

          <Tooltip primaryColor={primaryColor} content="Load program from file">
            <button 
              className="editor-btn"
              onClick={() => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.recursia,.txt';
                input.onchange = (e) => {
                  const file = (e.target as HTMLInputElement).files?.[0];
                  if (file) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                      const content = e.target?.result as string;
                      // Check for unsaved changes before loading new file
                      const loadFile = () => {
                        setCode(content);
                        setLastSavedCode(content);
                        setIsModified(false);
                      };
                      
                      if (isModified) {
                        // Show confirmation dialog
                        setPendingAction({ 
                          type: 'newFile', 
                          payload: loadFile 
                        });
                        setShowConfirmation(true);
                      } else {
                        // No unsaved changes, load directly
                        loadFile();
                      }
                    };
                    reader.readAsText(file);
                  }
                };
                input.click();
              }}
            >
              <Upload size={14} />
              Import
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Open in Circuit Designer">
            <button 
              className="editor-btn"
              onClick={() => {
                // Parse the current code to extract circuit information
                const circuit = parseRecursiaToCircuit(code);
                
                if (circuit) {
                  // Find the circuit designer window and update it
                  const circuitDesignerWindow = document.querySelector('[data-window-id="circuit-designer"]');
                  if (circuitDesignerWindow) {
                    // Dispatch custom event to update circuit designer
                    const event = new CustomEvent('updateCircuitDesigner', { 
                      detail: { circuit }
                    });
                    window.dispatchEvent(event);
                    // Show success notification
                    // Circuit sent to Circuit Designer
                  } else {
                    alert('Please open the Circuit Designer window first.');
                  }
                } else {
                  alert('Unable to parse the current code as a quantum circuit. Make sure your code follows the quantum circuit format.');
                }
              }}
            >
              <Cpu size={14} />
              To Circuit
            </button>
          </Tooltip>
          
          <Tooltip primaryColor={primaryColor} content="Import from Circuit Designer">
            <button 
              className="editor-btn"
              onClick={() => {
                const circuits = loadSavedCircuits();
                if (circuits.length === 0) {
                  alert('No saved circuits found. Create and save circuits in the Circuit Designer first.');
                  return;
                }
                
                // Create a simple selection dialog
                const circuitList = circuits.map((c, i) => `${i + 1}. ${c.name} (${c.wires.length} qubits, ${c.gates.length} gates)`).join('\n');
                const selection = prompt(`Select a circuit to import:\n\n${circuitList}\n\nEnter circuit number:`);
                
                if (selection) {
                  const index = parseInt(selection) - 1;
                  if (index >= 0 && index < circuits.length) {
                    const circuit = circuits[index];
                    const codeBlock = circuitToCodeBlock(circuit);
                    
                    // Insert circuit code at cursor position or append
                    const insertCode = () => {
                      if (editorRef.current) {
                        const editor = editorRef.current;
                        const position = editor.getPosition();
                        if (position) {
                          editor.executeEdits('', [{
                            range: {
                              startLineNumber: position.lineNumber,
                              startColumn: position.column,
                              endLineNumber: position.lineNumber,
                              endColumn: position.column
                            },
                            text: '\n' + codeBlock.code + '\n'
                          }]);
                        }
                      } else {
                        setCode(code + '\n' + codeBlock.code);
                      }
                    };
                    
                    if (isModified) {
                      setPendingAction({ 
                        type: 'custom', 
                        payload: insertCode 
                      });
                      setShowConfirmation(true);
                    } else {
                      insertCode();
                    }
                  }
                }
              }}
            >
              <Cpu size={14} />
              Circuit
            </button>
          </Tooltip>
        </div>
        
        <div className="toolbar-right">
          {currentProgramName && (
            <span style={{ 
              fontSize: '12px', 
              color: primaryColor, 
              marginRight: '1rem',
              opacity: 0.8 
            }}>
              {currentProgramName}
              {isModified && ' •'}
            </span>
          )}
          <Tooltip primaryColor={primaryColor} content="Show/hide language guide">
            <button 
              className={`editor-btn ${showInfo ? 'active' : ''}`}
              onClick={() => setShowInfo(!showInfo)}
            >
              <Info size={14} />
              Guide
            </button>
          </Tooltip>
        </div>
      </div>
      
      <div className="editor-content">
        <div className="editor-main">
          <div className="monaco-container">
            <div className="monaco-editor-wrapper">
              <Editor
                height="100%"
                width="100%"
                defaultLanguage="recursia"
                theme="quantum-dark"
                value={code}
                onChange={(value) => {
                  setCode(value || '');
                  onCodeChange?.(value || '');
                }}
                onMount={handleEditorMount}
                options={{
                minimap: { enabled: false }, // Removed minimap for cleaner look
                fontSize: 13,
                fontFamily: "'JetBrains Mono', 'Monaco', 'Menlo', monospace",
                wordWrap: 'on',
                automaticLayout: true,
                scrollBeyondLastLine: false,
                renderLineHighlight: 'all',
                renderLineHighlightOnlyWhenFocus: false,
                quickSuggestions: true,
                suggestOnTriggerCharacters: true,
                acceptSuggestionOnCommitCharacter: true,
                tabCompletion: 'on' as const,
                wordBasedSuggestions: 'currentDocument' as const,
                lineNumbers: 'on',
                lineNumbersMinChars: 3, // Consistent line number width
                glyphMargin: false, // Remove extra glyph margin
                folding: true,
                foldingStrategy: 'indentation' as const,
                renderWhitespace: 'none', // Cleaner without whitespace indicators
                bracketPairColorization: {
                  enabled: true
                },
                guides: {
                  indentation: true,
                  bracketPairs: true
                },
                padding: {
                  top: 8,
                  bottom: 8
                },
                lineHeight: 20,
                letterSpacing: 0.5,
                scrollbar: {
                  vertical: 'auto',
                  horizontal: 'auto',
                  useShadows: false,
                  verticalHasArrows: false,
                  horizontalHasArrows: false,
                  verticalScrollbarSize: 10,
                  horizontalScrollbarSize: 10,
                  arrowSize: 0
                },
                overviewRulerLanes: 0, // Hide overview ruler
                hideCursorInOverviewRuler: true,
                overviewRulerBorder: false
              }}
              />
            </div>
          </div>
          
          {showInfo && (
            <div className="language-guide" style={{ maxHeight: '800px', overflowY: 'auto' }}>
              <h3>Recursia Language Guide v3.2</h3>
              
              <div className="guide-section">
                <h4>🌌 Universe Declaration</h4>
                <p>Create quantum simulation environments</p>
                <code>universe MyQuantumLab {`{`}<br/>
                &nbsp;&nbsp;qubit q1 = |0&gt;;<br/>
                &nbsp;&nbsp;qubit q2 = |1&gt;;<br/>
                {`}`}</code>
              </div>

              <div className="guide-section">
                <h4>🔴 State Declaration</h4>
                <p>Define quantum states with comprehensive properties</p>
                <code>state QuantumSystem : quantum {`{`}<br/>
                &nbsp;&nbsp;state_qubits: 3,<br/>
                &nbsp;&nbsp;state_coherence: 0.95,<br/>
                &nbsp;&nbsp;state_entropy: 0.1,<br/>
                &nbsp;&nbsp;state_dimensions: 8<br/>
                {`}`};</code>
                <small>Required: state_qubits. Range: coherence/entropy [0.0-1.0]</small>
              </div>
              
              <div className="guide-section">
                <h4>👁️ Observer Declaration</h4>
                <p>Create consciousness observers for measurements</p>
                <code>observer ConsciousnessProbe {`{`}<br/>
                &nbsp;&nbsp;observer_type: "meta_observer",<br/>
                &nbsp;&nbsp;observer_focus: 0.9,<br/>
                &nbsp;&nbsp;observer_phase: 0.0,<br/>
                &nbsp;&nbsp;observer_collapse_threshold: 0.5,<br/>
                &nbsp;&nbsp;observer_self_awareness: 0.8<br/>
                {`}`};</code>
                <small>Required: observer_type, observer_focus</small>
              </div>

              <div className="guide-section">
                <h4>🔧 Pattern Declaration</h4>
                <p>Reusable quantum operation patterns</p>
                <code>pattern BellState : entangled_type {`{`}<br/>
                &nbsp;&nbsp;qubits: 2,<br/>
                &nbsp;&nbsp;entanglement_strength: 1.0<br/>
                {`}`}</code>
              </div>
              
              <div className="guide-section">
                <h4>⚛️ Quantum Operations</h4>
                <p>Apply gates and quantum transformations</p>
                <code>// Single qubit gates<br/>
                apply H_gate to QuantumSystem qubit 0;<br/>
                apply X_gate to QuantumSystem qubit 1;<br/><br/>
                // Multi-qubit gates<br/>
                apply CNOT_gate to QuantumSystem qubits [0, 1];<br/>
                apply CCNOT_gate to QuantumSystem qubits [0, 1, 2];<br/><br/>
                // Controlled operations<br/>
                apply H_gate to QuantumSystem qubit 2 control 0;<br/>
                apply X_gate to QuantumSystem qubit 1 anticontrol 0;<br/><br/>
                // Pattern application<br/>
                apply BellState to QuantumSystem with {`{`} strength: 0.9 {`}`};</code>
              </div>

              <div className="guide-section">
                <h4>📊 Measurement Operations</h4>
                <p>Comprehensive measurement types for OSH validation</p>
                <code>// Basic measurements<br/>
                measure QuantumSystem qubit 0 into result;<br/>
                measure QuantumSystem in computational basis;<br/><br/>
                // OSH-specific measurements<br/>
                measure QuantumSystem by phi;<br/>
                measure QuantumSystem by integrated_information;<br/>
                measure QuantumSystem by recursive_simulation_potential;<br/>
                measure QuantumSystem by gravitational_anomaly;<br/>
                measure QuantumSystem by consciousness_emergence;<br/>
                measure QuantumSystem by entropy;<br/>
                measure QuantumSystem by coherence;<br/><br/>
                // Advanced measurements<br/>
                measure QuantumSystem by wave_echo_amplitude;<br/>
                measure QuantumSystem by information_flow_tensor;<br/>
                measure QuantumSystem by observer_influence;<br/>
                measure QuantumSystem by temporal_stability;</code>
              </div>

              <div className="guide-section">
                <h4>🔗 Entanglement Operations</h4>
                <p>Create quantum entanglements between states</p>
                <code>// Basic entanglement<br/>
                entangle QuantumSystem with AnotherSystem;<br/>
                entangle QuantumSystem, AnotherSystem;<br/><br/>
                // Qubit-specific entanglement<br/>
                entangle QuantumSystem qubit 0 with AnotherSystem qubit 1;<br/><br/>
                // Protocol-based entanglement<br/>
                entangle QuantumSystem with AnotherSystem using BellProtocol protocol;</code>
              </div>

              <div className="guide-section">
                <h4>📡 Teleportation</h4>
                <p>Quantum state teleportation</p>
                <code>teleport QuantumSystem qubit 0 -&gt; TargetSystem qubit 0;<br/>
                teleport QuantumSystem qubit 1 -&gt; TargetSystem qubit 1 using StandardProtocol protocol;</code>
              </div>

              <div className="guide-section">
                <h4>🌊 Coherence & Evolution</h4>
                <p>Manage coherence and time evolution</p>
                <code>// Coherence control<br/>
                cohere QuantumSystem to level 0.95;<br/><br/>
                // Time evolution<br/>
                evolve for 100 steps;<br/>
                evolve QuantumSystem with timestep 0.01;<br/>
                evolve [QuantumSystem, Observer] with timestep 0.005 using Hamiltonian;</code>
              </div>

              <div className="guide-section">
                <h4>🔄 Simulation Operations</h4>
                <p>Recursive simulation and OSH operations</p>
                <code>// Simulation control<br/>
                simulate using QuantumEngine for 1000 steps;<br/>
                simulate for 500 steps;<br/><br/>
                // Recursive simulation<br/>
                recurse QuantumSystem depth 5;<br/>
                recurse QuantumSystem;</code>
              </div>

              <div className="guide-section">
                <h4>📈 Visualization</h4>
                <p>Comprehensive visualization modes</p>
                <code>// State visualization<br/>
                visualize QuantumSystem mode quantum_circuit;<br/>
                visualize QuantumSystem mode bloch_sphere;<br/>
                visualize QuantumSystem mode probability_distribution;<br/><br/>
                // Network visualization<br/>
                visualize entanglement_network;<br/>
                visualize memory_field;<br/><br/>
                // Evolution visualization<br/>
                visualize QuantumSystem evolution;<br/>
                visualize state evolution of QuantumSystem;<br/>
                visualize correlation between SystemA and SystemB;<br/><br/>
                // Render operations<br/>
                render QuantumSystem;</code>
              </div>

              <div className="guide-section">
                <h4>🔧 Control Flow</h4>
                <p>Programming constructs and flow control</p>
                <code>// Conditional statements<br/>
                if (coherence &gt; 0.8) {`{`}<br/>
                &nbsp;&nbsp;apply H_gate to QuantumSystem qubit 0;<br/>
                {`}`}<br/><br/>
                // Observer-based conditions<br/>
                when observer.phase == "active" {`{`}<br/>
                &nbsp;&nbsp;measure QuantumSystem by phi;<br/>
                {`}`}<br/><br/>
                // Loops with range syntax<br/>
                for i from 0 to 10 step 2 {`{`}<br/>
                &nbsp;&nbsp;apply X_gate to QuantumSystem qubit i;<br/>
                {`}`}<br/><br/>
                // Iterator loops<br/>
                for qubit in QuantumSystem {`{`}<br/>
                &nbsp;&nbsp;measure qubit by coherence;<br/>
                {`}`}<br/><br/>
                // While loops<br/>
                while (entropy &lt; 0.5) {`{`}<br/>
                &nbsp;&nbsp;evolve for 1 step;<br/>
                {`}`}</code>
              </div>

              <div className="guide-section">
                <h4>🔢 Variables & Types</h4>
                <p>Variable declarations and type system</p>
                <code>// Variable declarations<br/>
                let coherence_value = 0.95;<br/>
                const max_qubits = 10;<br/>
                let message = "Quantum state ready";<br/><br/>
                // Type annotations<br/>
                let state: quantum_type = QuantumSystem;<br/>
                let observer: meta_observer = ConsciousnessProbe;<br/><br/>
                // Complex numbers<br/>
                let amplitude = 0.707 + 0.707i;</code>
              </div>

              <div className="guide-section">
                <h4>🧮 Functions</h4>
                <p>Function definitions and calls</p>
                <code>// Function declaration<br/>
                function create_bell_pair(qubits: quantum_type): entangled_type {`{`}<br/>
                &nbsp;&nbsp;apply H_gate to qubits qubit 0;<br/>
                &nbsp;&nbsp;apply CNOT_gate to qubits qubits [0, 1];<br/>
                &nbsp;&nbsp;return qubits;<br/>
                {`}`}<br/><br/>
                // Function call<br/>
                let bell_state = create_bell_pair(QuantumSystem);</code>
              </div>

              <div className="guide-section">
                <h4>📝 Literals & Data Types</h4>
                <p>Supported data types and literals</p>
                <code>// Numbers<br/>
                let decimal = 42.5;<br/>
                let hex = 0xFF;<br/>
                let binary = 0b1010;<br/>
                let octal = 0o755;<br/><br/>
                // Strings<br/>
                let message = "Hello Quantum";<br/>
                let path = 'quantum/state';<br/><br/>
                // Booleans<br/>
                let is_entangled = true;<br/>
                let is_collapsed = false;<br/><br/>
                // Quantum states<br/>
                let state = |0&gt;;<br/>
                let superposition = |0&gt; + |1&gt;;</code>
              </div>

              <div className="guide-section">
                <h4>⚠️ Critical Syntax Notes</h4>
                <div style={{ backgroundColor: '#2d1b69', padding: '12px', borderRadius: '6px', border: '1px solid #6b46c1' }}>
                  <strong>Semicolons Required:</strong><br/>
                  • State declarations: <code>state Name {`{...}`};</code><br/>
                  • Observer declarations: <code>observer Name {`{...}`};</code><br/>
                  • Measure statements: <code>measure State;</code><br/>
                  • Entangle statements: <code>entangle A, B;</code><br/><br/>
                  
                  <strong>No Semicolons:</strong><br/>
                  • Apply statements: <code>apply H_gate to State qubit 0</code><br/><br/>
                  
                  <strong>Measurement Types:</strong><br/>
                  • Use keywords after "by": <code>measure State by phi;</code><br/>
                  • Not string literals: <code>measure State "phi";</code> ❌<br/><br/>
                  
                  <strong>Identifiers:</strong><br/>
                  • Support underscores: <code>quantum_state</code>, <code>my_observer</code><br/>
                  • Keywords can be used as identifiers in specific contexts
                </div>
              </div>
            </div>
          )}
        </div>
        
        <div className="code-snippets">
          <div className="snippets-header">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <h3>Building Blocks</h3>
              <span style={{ fontSize: '0.625rem', color: 'var(--text-tertiary)', opacity: 0.7 }}>
                {Object.values(codeSnippets).reduce((sum, cat) => sum + cat.snippets.length, 0)}
              </span>
            </div>
            <div className="category-tabs">
              {Object.entries(codeSnippets).map(([key, category]) => (
                <Tooltip key={key} content={`${category.title} (${category.snippets.length} snippets)`}>
                  <button
                    className={`category-tab ${selectedCategory === key ? 'active' : ''}`}
                    onClick={() => setSelectedCategory(key as keyof typeof codeSnippets)}
                  >
                    {category.icon}
                  </button>
                </Tooltip>
              ))}
            </div>
          </div>
          
          <div className="snippets-list">
            <h4>{codeSnippets[selectedCategory].title}</h4>
            <p style={{ fontSize: '0.625rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem', lineHeight: '1.4' }}>
              {codeSnippets[selectedCategory].description}
            </p>
            {codeSnippets[selectedCategory].snippets.map((snippet, idx) => (
              <Tooltip key={idx} content={`${snippet.name}: ${snippet.description}`}>
                <div 
                  className="snippet-item"
                  onClick={() => insertSnippet(snippet.code)}
                >
                  <div className="snippet-name">{truncateText(snippet.name, 20)}</div>
                  <div className="snippet-preview">{truncateText(snippet.code.split('\n')[0], 35)}</div>
                </div>
              </Tooltip>
            ))}
          </div>
        </div>
      </div>
      
      {/* Confirmation Dialog */}
      <ConfirmationDialog
        isOpen={showConfirmation}
        title="Unsaved Changes"
        message={`You have unsaved changes${currentProgramName ? ` in "${currentProgramName}"` : ''}. Would you like to save them before continuing?`}
        onSave={handleConfirmSave}
        onDontSave={() => {
          handleConfirmDontSave();
          // If there's a custom payload function (like for file import), execute it
          if (pendingAction?.payload && typeof pendingAction.payload === 'function') {
            pendingAction.payload();
          }
        }}
        onCancel={handleConfirmCancel}
        primaryColor={primaryColor}
      />
      
      {/* Add pulse animation for unsaved indicator */}
      <style>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.6; }
          100% { opacity: 1; }
        }
      `}</style>
      
      {/* Save As Dialog */}
      <SaveAsDialog
        isOpen={showSaveAsDialog}
        onClose={() => setShowSaveAsDialog(false)}
        onSave={handleSaveAs}
        currentName={currentProgramName || 'New Quantum Program'}
        currentDescription=""
        primaryColor={primaryColor}
        isUpdate={false}
      />
    </div>
  );
};