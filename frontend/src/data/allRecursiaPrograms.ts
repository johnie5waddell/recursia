/**
 * Dynamic Recursia program interface
 * Programs are loaded dynamically from the file system
 */

export interface RecursiaProgram {
  id: string;
  name: string;
  category: string;
  subcategory?: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  description: string;
  path: string; // Relative path from project root
  tags: string[];
  status: 'working' | 'experimental' | 'fixed' | 'unknown';
}

// This file is just for the type definition
// Actual programs are loaded dynamically from the API which scans the file system
export const allRecursiaPrograms: RecursiaProgram[] = [];