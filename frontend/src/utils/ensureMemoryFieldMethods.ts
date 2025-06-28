/**
 * Utility to ensure MemoryFieldEngine has all required methods
 * Adds missing methods dynamically to maintain compatibility
 */

import { MemoryFieldEngine, MemoryFragment } from '../engines/MemoryFieldEngine';

/**
 * Ensure the MemoryFieldEngine instance has all required methods
 * @param engine The MemoryFieldEngine instance to enhance
 */
export function ensureMemoryFieldMethods(engine: MemoryFieldEngine): void {
  // Add calculateStrainAt if it doesn't exist
  if (typeof (engine as any).calculateStrainAt !== 'function') {
    console.log('[MemoryField] Adding calculateStrainAt method dynamically');
    
    (engine as any).calculateStrainAt = function(position: { x: number; y: number; z: number }): number {
      let totalStrain = 0;
      let contributingFragments = 0;
      
      const field = this.getCurrentField();
      const fragments = field.fragments || [];
      
      // Calculate strain based on nearby fragment density and coherence variance
      fragments.forEach((fragment: MemoryFragment) => {
        const fragPos = Array.isArray(fragment.position)
          ? { x: fragment.position[0], y: fragment.position[1], z: fragment.position[2] }
          : fragment.position;
        
        const dx = position.x - (fragPos.x || 0);
        const dy = position.y - (fragPos.y || 0);
        const dz = position.z - (fragPos.z || 0);
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        
        if (distance < 20) { // Consider fragments within radius
          const influence = Math.exp(-distance / 10);
          const fragmentStrain = fragment.strain || 0;
          totalStrain += fragmentStrain * influence;
          contributingFragments++;
        }
      });
      
      // Normalize by contributing fragments
      return contributingFragments > 0 ? totalStrain / contributingFragments : 0;
    };
  }
  
  // Ensure getCurrentField returns proper structure
  const originalGetCurrentField = engine.getCurrentField?.bind(engine);
  if (originalGetCurrentField) {
    engine.getCurrentField = function() {
      const field = originalGetCurrentField();
      
      // Ensure field has required properties
      if (field) {
        // Ensure fragments is an array
        if (!Array.isArray(field.fragments) && field.fragments instanceof Map) {
          field.fragments = Array.from(field.fragments.values());
        } else if (!field.fragments) {
          field.fragments = [];
        }
        
        // Ensure coherence values exist
        if (typeof field.averageCoherence !== 'number' || !isFinite(field.averageCoherence)) {
          const fragments = field.fragments;
          if (fragments.length > 0) {
            const totalCoherence = fragments.reduce((sum: number, f: MemoryFragment) => 
              sum + (f.coherence || 0), 0);
            field.totalCoherence = totalCoherence;
            field.averageCoherence = totalCoherence / fragments.length;
          } else {
            field.totalCoherence = 0;
            field.averageCoherence = 0;
          }
        }
        
        // Ensure other properties have defaults
        field.totalEntropy = field.totalEntropy || 0;
        field.lastDefragmentation = field.lastDefragmentation || Date.now();
      }
      
      return field;
    };
  }
  
  console.log('[MemoryField] Memory field methods ensured');
}