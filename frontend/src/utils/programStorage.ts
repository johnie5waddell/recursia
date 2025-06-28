/**
 * Program Storage Utility
 * Manages custom quantum programs in localStorage
 */

export interface CustomProgram {
  id: string;
  name: string;
  description: string;
  code: string;
  category: 'custom';
  difficulty: 'variable';
  createdAt: number;
  updatedAt: number;
  isReadOnly: boolean;
}

const CUSTOM_PROGRAMS_KEY = 'recursia_custom_programs';

/**
 * Load all custom programs from localStorage
 */
export const loadCustomPrograms = (): CustomProgram[] => {
  try {
    const stored = localStorage.getItem(CUSTOM_PROGRAMS_KEY);
    if (!stored) return [];
    
    const programs = JSON.parse(stored);
    // Ensure all programs have required fields
    return programs.map((p: any) => ({
      id: p.id || `custom-${Date.now()}-${Math.random()}`,
      name: p.name || 'Untitled Program',
      description: p.description || '',
      code: p.code || '',
      category: 'custom',
      difficulty: 'variable',
      createdAt: p.createdAt || Date.now(),
      updatedAt: p.updatedAt || Date.now(),
      isReadOnly: p.isReadOnly || false
    }));
  } catch (error) {
    console.error('Error loading custom programs:', error);
    return [];
  }
};

/**
 * Save a custom program
 */
export const saveCustomProgram = (
  program: Omit<CustomProgram, 'id' | 'createdAt' | 'updatedAt' | 'category' | 'difficulty' | 'isReadOnly'>
): CustomProgram => {
  const programs = loadCustomPrograms();
  
  const newProgram: CustomProgram = {
    ...program,
    id: `custom-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    category: 'custom',
    difficulty: 'variable',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    isReadOnly: false
  };
  
  programs.push(newProgram);
  localStorage.setItem(CUSTOM_PROGRAMS_KEY, JSON.stringify(programs));
  
  // Dispatch event for program library to update
  window.dispatchEvent(new CustomEvent('customProgramsUpdated', { 
    detail: { programs } 
  }));
  
  return newProgram;
};

/**
 * Update an existing custom program
 */
export const updateCustomProgram = (
  id: string,
  updates: Partial<Omit<CustomProgram, 'id' | 'createdAt' | 'category' | 'difficulty'>>
): CustomProgram | null => {
  const programs = loadCustomPrograms();
  const index = programs.findIndex(p => p.id === id);
  
  if (index === -1) return null;
  
  // Don't update read-only programs
  if (programs[index].isReadOnly) {
    console.warn('Cannot update read-only program');
    return null;
  }
  
  programs[index] = {
    ...programs[index],
    ...updates,
    updatedAt: Date.now()
  };
  
  localStorage.setItem(CUSTOM_PROGRAMS_KEY, JSON.stringify(programs));
  
  // Dispatch event for program library to update
  window.dispatchEvent(new CustomEvent('customProgramsUpdated', { 
    detail: { programs } 
  }));
  
  return programs[index];
};

/**
 * Delete a custom program
 */
export const deleteCustomProgram = (id: string): boolean => {
  const programs = loadCustomPrograms();
  const index = programs.findIndex(p => p.id === id);
  
  if (index === -1) return false;
  
  // Don't delete read-only programs
  if (programs[index].isReadOnly) {
    console.warn('Cannot delete read-only program');
    return false;
  }
  
  programs.splice(index, 1);
  localStorage.setItem(CUSTOM_PROGRAMS_KEY, JSON.stringify(programs));
  
  // Dispatch event for program library to update
  window.dispatchEvent(new CustomEvent('customProgramsUpdated', { 
    detail: { programs } 
  }));
  
  return true;
};

/**
 * Check if a program name already exists
 */
export const programNameExists = (name: string, excludeId?: string): boolean => {
  const programs = loadCustomPrograms();
  return programs.some(p => 
    p.name.toLowerCase() === name.toLowerCase() && 
    p.id !== excludeId
  );
};

/**
 * Generate a unique program name
 */
export const generateUniqueProgramName = (baseName: string): string => {
  let name = baseName;
  let counter = 1;
  
  while (programNameExists(name)) {
    name = `${baseName} ${counter}`;
    counter++;
  }
  
  return name;
};