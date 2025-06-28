/**
 * Quantum Program Loader
 * Handles loading quantum programs from individual files
 * 
 * @module quantumProgramLoader
 * @author Johnie Waddell
 * @version 1.0.0
 */

import type { QuantumProgram } from '../data/advancedQuantumPrograms';

export interface ProgramCatalogEntry {
  id: string;
  name: string;
  file: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  qubits_required: number;
  concepts: string[];
  osh_score?: number;
}

export interface ProgramCategory {
  name: string;
  description: string;
  programs: ProgramCatalogEntry[];
}

export interface ProgramCatalog {
  version: string;
  last_updated: string;
  categories: Record<string, ProgramCategory>;
  metadata: {
    total_programs: number;
    language_version: string;
    author: string;
    organization: string;
  };
}

/**
 * Base URL for quantum programs
 * In production, this would be served from the backend
 */
const PROGRAMS_BASE_URL = '/quantum_programs';
const CATALOG_URL = `${PROGRAMS_BASE_URL}/catalog.json`;

/**
 * Cache for loaded programs to avoid redundant fetches
 */
const programCache = new Map<string, string>();

/**
 * Load the program catalog
 */
export async function loadProgramCatalog(): Promise<ProgramCatalog> {
  try {
    const response = await fetch(CATALOG_URL);
    if (!response.ok) {
      throw new Error(`Failed to load catalog: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error loading program catalog:', error);
    throw error;
  }
}

/**
 * Load a specific program's code from file
 */
export async function loadProgramCode(category: string, filename: string): Promise<string> {
  const cacheKey = `${category}/${filename}`;
  
  // Check cache first
  if (programCache.has(cacheKey)) {
    return programCache.get(cacheKey)!;
  }
  
  try {
    const url = `${PROGRAMS_BASE_URL}/${category}/${filename}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load program: ${response.statusText}`);
    }
    
    const code = await response.text();
    
    // Cache the result
    programCache.set(cacheKey, code);
    
    return code;
  } catch (error) {
    console.error(`Error loading program ${category}/${filename}:`, error);
    throw error;
  }
}

/**
 * Convert a catalog entry to a QuantumProgram
 */
export async function catalogEntryToQuantumProgram(
  entry: ProgramCatalogEntry,
  category: string,
  categoryName: string
): Promise<QuantumProgram> {
  const code = await loadProgramCode(category, entry.file);
  
  // Map difficulty levels
  const difficultyMap: Record<string, QuantumProgram['difficulty']> = {
    'beginner': 'beginner',
    'intermediate': 'intermediate',
    'advanced': 'advanced',
    'expert': 'expert'
  };
  
  // Determine OSH prediction based on category and content
  let oshPrediction: QuantumProgram['oshPrediction'] = 'neutral';
  let predictionStrength = 0.5;
  
  if (category === 'consciousness' || entry.osh_score !== undefined) {
    oshPrediction = 'supports';
    predictionStrength = entry.osh_score ? Math.min(entry.osh_score / 20, 1) : 0.8;
  }
  
  // Map category to program category
  const categoryMap: Record<string, QuantumProgram['category']> = {
    'basic': 'simulation',
    'intermediate': 'simulation',
    'advanced': 'consciousness',
    'consciousness': 'consciousness',
    'quantum_computing': 'simulation',
    'optimization': 'optimization',
    'experimental': 'simulation'
  };
  
  return {
    id: entry.id,
    name: entry.name,
    category: categoryMap[category] || 'simulation',
    difficulty: difficultyMap[entry.difficulty] || 'intermediate',
    description: entry.description,
    code,
    expectedOutcome: `Program demonstrates ${entry.concepts.join(', ')}`,
    oshPrediction,
    predictionStrength,
    scientificReferences: [],
    author: 'Johnie Waddell',
    dateCreated: new Date().toISOString().split('T')[0]
  };
}

/**
 * Load all programs from the catalog
 */
export async function loadAllPrograms(): Promise<QuantumProgram[]> {
  try {
    const catalog = await loadProgramCatalog();
    const programs: QuantumProgram[] = [];
    
    for (const [categoryKey, category] of Object.entries(catalog.categories)) {
      for (const entry of category.programs) {
        try {
          const program = await catalogEntryToQuantumProgram(
            entry,
            categoryKey,
            category.name
          );
          programs.push(program);
        } catch (error) {
          console.error(`Error loading program ${entry.id}:`, error);
          // Continue loading other programs
        }
      }
    }
    
    return programs;
  } catch (error) {
    console.error('Error loading programs:', error);
    // Return empty array on error
    return [];
  }
}

/**
 * Load programs for a specific category
 */
export async function loadProgramsByCategory(category: string): Promise<QuantumProgram[]> {
  try {
    const catalog = await loadProgramCatalog();
    const categoryData = catalog.categories[category];
    
    if (!categoryData) {
      throw new Error(`Category '${category}' not found`);
    }
    
    const programs: QuantumProgram[] = [];
    
    for (const entry of categoryData.programs) {
      try {
        const program = await catalogEntryToQuantumProgram(
          entry,
          category,
          categoryData.name
        );
        programs.push(program);
      } catch (error) {
        console.error(`Error loading program ${entry.id}:`, error);
      }
    }
    
    return programs;
  } catch (error) {
    console.error(`Error loading programs for category ${category}:`, error);
    return [];
  }
}

/**
 * Save a new program to the catalog
 * This would typically make an API call to the backend
 */
export async function saveProgramToCatalog(
  program: QuantumProgram,
  category: string
): Promise<void> {
  // In a real implementation, this would:
  // 1. Save the .recursia file to the appropriate directory
  // 2. Update the catalog.json
  // 3. Return success/failure
  
  // For now, we'll just log the intent
  console.log('Saving program:', {
    name: program.name,
    category,
    id: program.id
  });
  
  // In production:
  // const response = await fetch('/api/programs/save', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify({ program, category })
  // });
  
  throw new Error('Program saving not yet implemented - requires backend API');
}

/**
 * Clear the program cache
 */
export function clearProgramCache(): void {
  programCache.clear();
}