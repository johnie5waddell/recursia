/**
 * Quantum Programs Library Component
 * Provides browsing, filtering, and selection of quantum programs
 * Integrates with code editor and execution system
 */

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { 
  Search, 
  Filter, 
  Code, 
  Play, 
  Copy, 
  ChevronDown, 
  ChevronUp,
  Sparkles,
  Brain,
  Shield,
  Cpu,
  Activity,
  Eye,
  AlertCircle,
  CheckCircle,
  XCircle,
  Atom,
  Globe,
  TestTube,
  Beaker,
  Calculator,
  Database,
  Zap,
  FileCode,
  FlaskConical,
  Microscope,
  RefreshCw,
  Loader,
  Save,
  Trash2
} from 'lucide-react';
import { Tooltip } from './ui/Tooltip';
import { fetchProgram, API_ENDPOINTS } from '../config/api';
import { loadCustomPrograms, deleteCustomProgram, CustomProgram } from '../utils/programStorage';
import '../styles/quantum-programs-library.css';

// Re-export the RecursiaProgram interface
export interface RecursiaProgram {
  id: string;
  name: string;
  category: string;
  subcategory?: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert' | 'variable';
  description: string;
  path: string;
  tags: string[];
  status: 'working' | 'experimental' | 'fixed' | 'unknown';
  code?: string; // Optional code property for embedded programs
}

interface QuantumProgramsLibraryProps {
  primaryColor?: string;
  onProgramSelect?: (program: RecursiaProgram) => void;
  onProgramLoad?: (code: string, programName: string, programPath: string) => void;
  onProgramExecute?: (program: RecursiaProgram, iterations?: number) => void;
  currentProgramId?: string;
  disabled?: boolean;
}

// Category icons mapping
const categoryIcons: Record<string, React.ReactNode> = {
  // Main categories
  custom: <Save size={16} />,
  quantum_programs: <Atom size={16} />,
  experiments: <TestTube size={16} />,
  
  // Subcategories
  basic: <FileCode size={16} />,
  complete: <CheckCircle size={16} />,
  osh: <Globe size={16} />,
  intermediate: <Activity size={16} />,
  advanced: <Zap size={16} />,
  consciousness: <Brain size={16} />,
  experimental: <FlaskConical size={16} />,
  osh_calculations: <Calculator size={16} />,
  osh_predictions: <Eye size={16} />,
  osh_testing: <Microscope size={16} />,
  osh_demonstration: <Sparkles size={16} />,
  theoretical: <Beaker size={16} />,
  // Experimental subcategories
  validation: <CheckCircle size={16} />,
  empirical_tests: <Microscope size={16} />,
  optimization: <Zap size={16} />,
  'error-correction': <Shield size={16} />,
  empirical: <Database size={16} />,
  demonstration: <Eye size={16} />
};

// Difficulty colors
const difficultyColors: Record<string, string> = {
  beginner: '#4ade80',
  intermediate: '#60a5fa',
  advanced: '#f59e0b',
  expert: '#ef4444',
  variable: '#a78bfa'
};

// Status indicators
const statusIcons: Record<string, React.ReactNode> = {
  working: <CheckCircle size={14} color="#4ade80" />,
  fixed: <CheckCircle size={14} color="#60a5fa" />,
  experimental: <FlaskConical size={14} color="#f59e0b" />,
  unknown: <AlertCircle size={14} color="#6b7280" />
};

// Status colors
const statusColors: Record<string, string> = {
  working: '#4ade80',
  fixed: '#60a5fa',
  experimental: '#f59e0b',
  unknown: '#6b7280'
};

export const QuantumProgramsLibrary: React.FC<QuantumProgramsLibraryProps> = ({
  primaryColor = '#ffd700',
  onProgramSelect,
  onProgramLoad,
  onProgramExecute,
  currentProgramId,
  disabled = false
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedSubcategory, setSelectedSubcategory] = useState<string>('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['quantum_programs']));
  const [showFilters, setShowFilters] = useState(true);
  const [selectedProgram, setSelectedProgram] = useState<RecursiaProgram | null>(null);
  const [copiedProgramId, setCopiedProgramId] = useState<string | null>(null);
  const [loadingProgramId, setLoadingProgramId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // All program code generation removed - programs loaded from files only
  
  // Dynamic programs state
  const [allPrograms, setAllPrograms] = useState<RecursiaProgram[]>([]);
  const [customPrograms, setCustomPrograms] = useState<CustomProgram[]>([]);
  const [isLoadingPrograms, setIsLoadingPrograms] = useState(false);
  
  // No hardcoded programs - all loaded from API
  
  // Define program categories structure
  const programCategories = [
    {
      id: 'custom',
      name: 'My Programs',
      subcategories: []
    },
    {
      id: 'quantum_programs',
      name: 'Quantum Programs',
      subcategories: ['basic', 'intermediate', 'advanced', 'consciousness', 'experimental']
    },
    {
      id: 'experiments',
      name: 'Experiments',
      subcategories: []
    }
  ];
  
  // Calculate program statistics dynamically
  const programStats = useMemo(() => {
    const combinedPrograms = [
      ...allPrograms, 
      ...customPrograms.map(cp => ({
        id: cp.id,
        category: 'custom',
        status: 'working' as const,
        difficulty: 'variable' as const
      }))
    ];
    
    const stats = {
      total: combinedPrograms.length,
      byCategory: {} as Record<string, { category: string; count: number }>,
      byStatus: {
        working: 0,
        fixed: 0,
        experimental: 0,
        unknown: 0
      },
      byDifficulty: {
        beginner: 0,
        intermediate: 0,
        advanced: 0,
        expert: 0,
        variable: 0
      }
    };
    
    // Count by category
    programCategories.forEach(cat => {
      if (cat.id === 'custom') {
        stats.byCategory[cat.id] = { category: cat.id, count: customPrograms.length };
      } else {
        const apiCount = allPrograms.filter(p => p.category === cat.id).length;
        stats.byCategory[cat.id] = { category: cat.id, count: apiCount };
      }
    });
    
    // Count by status
    combinedPrograms.forEach(program => {
      if (program.status in stats.byStatus) {
        stats.byStatus[program.status as keyof typeof stats.byStatus]++;
      }
    });
    
    // Count by difficulty
    combinedPrograms.forEach(program => {
      if (program.difficulty && program.difficulty in stats.byDifficulty) {
        stats.byDifficulty[program.difficulty as keyof typeof stats.byDifficulty]++;
      }
    });
    
    return stats;
  }, [allPrograms, customPrograms]);
  
  // Load custom programs
  useEffect(() => {
    const loadCustom = () => {
      const custom = loadCustomPrograms();
      setCustomPrograms(custom);
    };
    
    loadCustom();
    
    // Listen for custom programs updates
    const handleUpdate = () => {
      loadCustom();
    };
    
    window.addEventListener('customProgramsUpdated', handleUpdate);
    return () => {
      window.removeEventListener('customProgramsUpdated', handleUpdate);
    };
  }, []);

  // Fetch programs from API
  useEffect(() => {
    const fetchPrograms = async () => {
      setIsLoadingPrograms(true);
      try {
        const response = await fetch(API_ENDPOINTS.listPrograms);
        if (response.ok) {
          const data = await response.json();
          setAllPrograms(data.programs || []);
          setError(null);
        } else {
          console.error('[QuantumProgramsLibrary] Failed to fetch programs:', response.status);
          setError('Failed to load programs from server');
          setAllPrograms([]);
        }
      } catch (err) {
        console.error('[QuantumProgramsLibrary] Error fetching programs:', err);
        setError('Unable to connect to server');
        setAllPrograms([]);
      } finally {
        setIsLoadingPrograms(false);
      }
    };
    
    fetchPrograms();
  }, []); // Only fetch once on mount

  // Filter programs based on search and filters
  const filteredPrograms = useMemo(() => {
    // Combine all programs with custom and experimental programs
    const customAsRecursia: RecursiaProgram[] = customPrograms.map(cp => ({
      id: cp.id,
      name: cp.name,
      category: 'custom',
      subcategory: undefined,
      difficulty: 'variable' as const,
      description: cp.description,
      path: `custom:${cp.id}`,
      tags: ['custom', 'user-created'],
      status: 'working' as const
    }));
    
    let programs = [...allPrograms, ...customAsRecursia];

    // Category filter
    if (selectedCategory !== 'all') {
      programs = programs.filter(p => p.category === selectedCategory);
    }

    // Subcategory filter
    if (selectedSubcategory !== 'all') {
      programs = programs.filter(p => p.subcategory === selectedSubcategory);
    }

    // Difficulty filter
    if (selectedDifficulty !== 'all') {
      programs = programs.filter(p => p.difficulty === selectedDifficulty);
    }

    // Status filter
    if (selectedStatus !== 'all') {
      programs = programs.filter(p => p.status === selectedStatus);
    }

    // Search filter
    if (searchTerm) {
      const search = searchTerm.toLowerCase();
      programs = programs.filter(p => 
        p.name.toLowerCase().includes(search) ||
        p.description.toLowerCase().includes(search) ||
        p.tags.some(tag => tag.toLowerCase().includes(search)) ||
        p.path.toLowerCase().includes(search)
      );
    }

    return programs;
  }, [allPrograms, customPrograms, searchTerm, selectedCategory, selectedSubcategory, selectedDifficulty, selectedStatus]);

  // Group programs by category and subcategory
  const groupedPrograms = useMemo(() => {
    const groups: Record<string, Record<string, RecursiaProgram[]>> = {};
    
    filteredPrograms.forEach(program => {
      if (!groups[program.category]) {
        groups[program.category] = {};
      }
      const subcategory = program.subcategory || 'general';
      if (!groups[program.category][subcategory]) {
        groups[program.category][subcategory] = [];
      }
      groups[program.category][subcategory].push(program);
    });

    return groups;
  }, [filteredPrograms]);

  // Handle program selection - automatically load when clicked
  const handleProgramSelect = useCallback((program: RecursiaProgram) => {
    setSelectedProgram(program);
    onProgramSelect?.(program);
    // Automatically load the program when selected
    handleProgramLoad(program);
  }, [onProgramSelect]);

  // Handle program load
  const handleProgramLoad = useCallback(async (program: RecursiaProgram) => {
    // Prevent loading if already loading or disabled
    if (loadingProgramId || disabled) return;
    
    setLoadingProgramId(program.id);
    setError(null);
    
    try {
      let code: string;
      
      // Check if it's a custom program
      if (program.path.startsWith('custom:')) {
        const customId = program.path.substring(7); // Remove "custom:" prefix
        const customProgram = customPrograms.find(cp => cp.id === customId);
        if (customProgram) {
          code = customProgram.code;
        } else {
          throw new Error('Custom program not found');
        }
      } else {
        // Load the actual program content from the API
        code = await fetchProgram(program.path);
      }
      
      onProgramLoad?.(code, program.name, program.path);
      setSelectedProgram(program);
    } catch (error) {
      console.error('Error loading program:', error);
      
      // Show user-friendly error message
      const errorMessage = error instanceof Error ? error.message : 'Failed to load program';
      setError(errorMessage);
      
    } finally {
      setLoadingProgramId(null);
    }
  }, [onProgramLoad, loadingProgramId, disabled, customPrograms]);

  // Handle program copy
  const handleCopyProgram = useCallback((program: RecursiaProgram, event: React.MouseEvent) => {
    event.stopPropagation();
    navigator.clipboard.writeText(program.path);
    setCopiedProgramId(program.id);
    setTimeout(() => setCopiedProgramId(null), 2000);
  }, []);

  // Handle category toggle
  const toggleCategory = useCallback((category: string) => {
    setExpandedCategories(prev => {
      const newSet = new Set(prev);
      if (newSet.has(category)) {
        newSet.delete(category);
      } else {
        newSet.add(category);
      }
      return newSet;
    });
  }, []);

  // Handle search with debounce
  const handleSearch = useCallback((value: string) => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    searchTimeoutRef.current = setTimeout(() => {
      setSearchTerm(value);
    }, 300);
  }, []);

  // Get subcategories for selected category
  const availableSubcategories = useMemo(() => {
    if (selectedCategory === 'all') {
      return [];
    }
    const category = programCategories.find(c => c.id === selectedCategory);
    return category?.subcategories || [];
  }, [selectedCategory]);

  // Render program item
  const renderProgramItem = (program: RecursiaProgram) => (
    <div
      key={program.id}
      className={`program-item ${selectedProgram?.id === program.id || currentProgramId === program.id ? 'selected' : ''} ${disabled ? 'disabled' : ''}`}
      onClick={() => !disabled && handleProgramSelect(program)}
      onKeyDown={(e) => {
        if (!disabled && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          handleProgramSelect(program);
        }
      }}
      tabIndex={disabled ? -1 : 0}
      role="button"
      aria-label={`Select ${program.name}`}
      style={{
        '--primary-color': primaryColor,
        '--hover-color': `${primaryColor}20`
      } as React.CSSProperties}
    >
      <div className="program-header">
        <div className="program-title">
          <span className="program-name">{program.name}</span>
          <div className="program-badges">
            <Tooltip content={`Status: ${program.status}`}>
              <span className="status-badge" style={{ color: statusColors[program.status] }}>
                {statusIcons[program.status]}
              </span>
            </Tooltip>
            <span 
              className="difficulty-badge" 
              style={{ backgroundColor: difficultyColors[program.difficulty] }}
            >
              {program.difficulty}
            </span>
          </div>
        </div>
        <div className="program-actions">
          <Tooltip content={loadingProgramId === program.id ? "Loading..." : "Load in editor"}>
            <button
              className={`icon-button ${loadingProgramId === program.id ? 'loading' : ''}`}
              onClick={(e) => {
                e.stopPropagation();
                handleProgramLoad(program);
              }}
              disabled={disabled || loadingProgramId !== null}
            >
              {loadingProgramId === program.id ? (
                <div className="spinner" />
              ) : (
                <Code size={16} />
              )}
            </button>
          </Tooltip>
          <Tooltip content={copiedProgramId === program.id ? "Copied!" : "Copy path"}>
            <button
              className="icon-button"
              onClick={(e) => handleCopyProgram(program, e)}
              disabled={disabled}
            >
              <Copy size={16} />
            </button>
          </Tooltip>
          {onProgramExecute && (
            <Tooltip content="Execute program">
              <button
                className="icon-button"
                onClick={(e) => {
                  e.stopPropagation();
                  onProgramExecute(program, 1);
                }}
                disabled={disabled}
              >
                <Play size={16} />
              </button>
            </Tooltip>
          )}
          {program.category === 'custom' && (
            <Tooltip content="Delete custom program">
              <button
                className="icon-button delete"
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm(`Are you sure you want to delete "${program.name}"?`)) {
                    const customId = program.path.substring(7); // Remove "custom:" prefix
                    if (deleteCustomProgram(customId)) {
                      // Program will be removed from list via event listener
                      if (selectedProgram?.id === program.id) {
                        setSelectedProgram(null);
                      }
                    }
                  }
                }}
                disabled={disabled}
                style={{ color: '#ef4444' }}
              >
                <Trash2 size={16} />
              </button>
            </Tooltip>
          )}
        </div>
      </div>
      <p className="program-description">{program.description}</p>
      <div className="program-metadata">
        <span className="program-path">{program.path}</span>
        <div className="program-tags">
          {program.tags.slice(0, 3).map((tag, index) => (
            <span key={index} className="tag">
              {tag}
            </span>
          ))}
          {program.tags.length > 3 && (
            <span className="tag more">+{program.tags.length - 3}</span>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className={`quantum-programs-library ${disabled ? 'library-disabled' : ''}`}>
      {/* Header - Simplified without title */}
      <div className="library-header">
        <div className="library-stats">
          {isLoadingPrograms ? (
            <span><Loader size={14} className="loading-spinner" /> Loading programs...</span>
          ) : (
            <span>{filteredPrograms.length} of {allPrograms.length + customPrograms.length} programs</span>
          )}
        </div>
        <div className="header-actions">
          <Tooltip content="Refresh programs list">
            <button
              className="icon-button"
              onClick={async () => {
                const fetchPrograms = async () => {
                  setIsLoadingPrograms(true);
                  try {
                    const response = await fetch(API_ENDPOINTS.listPrograms);
                    if (response.ok) {
                      const data = await response.json();
                      // Successfully refreshed programs list
                      setAllPrograms(data.programs || []);
                      setError(null);
                    } else {
                      setError('Failed to refresh programs');
                    }
                  } catch (err) {
                    setError('Unable to connect to server');
                  } finally {
                    setIsLoadingPrograms(false);
                  }
                };
                await fetchPrograms();
              }}
              disabled={isLoadingPrograms}
              aria-label="Refresh programs"
            >
              <RefreshCw size={16} className={isLoadingPrograms ? 'rotating' : ''} />
            </button>
          </Tooltip>
          <button
            className="icon-button"
            onClick={() => setShowFilters(!showFilters)}
            aria-label={showFilters ? "Hide filters" : "Show filters"}
          >
            <Filter size={16} />
          </button>
        </div>
      </div>

      {/* Error notification */}
      {error && (
        <div className="error-notification">
          <AlertCircle size={16} />
          <span>{error}</span>
          <button
            className="error-close"
            onClick={() => setError(null)}
            aria-label="Close error"
          >
            ×
          </button>
        </div>
      )}

      {/* Search and Filters */}
      {showFilters && (
        <div className="library-controls">
          <div className="search-container">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search programs by name, description, tags, or path..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                handleSearch(e.target.value);
              }}
              disabled={disabled}
            />
            {searchTerm && (
              <button
                className="search-clear"
                onClick={() => {
                  setSearchTerm('');
                  handleSearch('');
                }}
                aria-label="Clear search"
              >
                ×
              </button>
            )}
          </div>
          
          <div className="filter-controls">
            <select
              value={selectedCategory}
              onChange={(e) => {
                setSelectedCategory(e.target.value);
                setSelectedSubcategory('all');
              }}
              disabled={disabled}
            >
              <option value="all">All Categories</option>
              {programCategories.map(cat => (
                <option key={cat.id} value={cat.id}>
                  {cat.name} ({programStats.byCategory[cat.id]?.count || 0})
                </option>
              ))}
            </select>

            {availableSubcategories.length > 0 && (
              <select
                value={selectedSubcategory}
                onChange={(e) => setSelectedSubcategory(e.target.value)}
                disabled={disabled}
              >
                <option value="all">All Subcategories</option>
                {availableSubcategories.map(subcat => (
                  <option key={subcat} value={subcat}>
                    {subcat.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </option>
                ))}
              </select>
            )}

            <select
              value={selectedDifficulty}
              onChange={(e) => setSelectedDifficulty(e.target.value)}
              disabled={disabled}
            >
              <option value="all">All Difficulties</option>
              <option value="beginner">Beginner ({programStats.byDifficulty.beginner})</option>
              <option value="intermediate">Intermediate ({programStats.byDifficulty.intermediate})</option>
              <option value="advanced">Advanced ({programStats.byDifficulty.advanced})</option>
              <option value="expert">Expert ({programStats.byDifficulty.expert})</option>
            </select>

            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              disabled={disabled}
            >
              <option value="all">All Status</option>
              <option value="working">Working ({programStats.byStatus.working})</option>
              <option value="fixed">Fixed ({programStats.byStatus.fixed})</option>
              <option value="experimental">Experimental ({programStats.byStatus.experimental})</option>
              <option value="unknown">Unknown ({programStats.byStatus.unknown})</option>
            </select>
          </div>
        </div>
      )}

      {/* Programs List */}
      <div className="programs-container">
        {Object.entries(groupedPrograms).map(([category, subcategories]) => {
          const categoryInfo = programCategories.find(c => c.id === category);
          const isExpanded = expandedCategories.has(category);

          return (
            <div key={category} className="category-section">
              <div 
                className="category-header"
                onClick={() => toggleCategory(category)}
              >
                <div className="category-title">
                  {categoryIcons[category]}
                  <span>{categoryInfo?.name || category}</span>
                  <span className="category-count">
                    ({Object.values(subcategories).flat().length})
                  </span>
                </div>
                {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </div>

              {isExpanded && (
                <div className="category-content">
                  {Object.entries(subcategories).map(([subcategory, programs]) => (
                    <div key={subcategory} className="subcategory-section">
                      <div className="subcategory-header">
                        {categoryIcons[subcategory] || <FileCode size={14} />}
                        <span>
                          {subcategory.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </span>
                        <span className="subcategory-count">({programs.length})</span>
                      </div>
                      <div className="programs-list">
                        {programs.map(renderProgramItem)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}

        {filteredPrograms.length === 0 && (
          <div className="no-programs">
            <AlertCircle size={48} />
            <p>No programs found matching your criteria</p>
            <button 
              onClick={() => {
                // Clear any pending search timeout
                if (searchTimeoutRef.current) {
                  clearTimeout(searchTimeoutRef.current);
                  searchTimeoutRef.current = null;
                }
                setSearchTerm('');
                setSelectedCategory('all');
                setSelectedSubcategory('all');
                setSelectedDifficulty('all');
                setSelectedStatus('all');
              }}
            >
              Clear filters
            </button>
          </div>
        )}

      </div>
      
      {/* Disabled state overlay for universe mode */}
      {disabled && (
        <div className="library-disabled-overlay" style={{ pointerEvents: 'none' }}>
          <div className="disabled-content">
            <div className="disabled-icon">
              <Code size={48} style={{ opacity: 0.6 }} />
            </div>
            <h3 className="disabled-title">Program Mode Required</h3>
            <p className="disabled-message">
              Switch to Program Mode to access and execute Recursia quantum programs.
            </p>
            <div className="disabled-hint">
              <span className="hint-text">Toggle execution mode in the header controls</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};