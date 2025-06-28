/**
 * Three.js Memory Management Utilities
 * Comprehensive resource tracking and disposal for Three.js objects
 */

import * as THREE from 'three';
import * as React from 'react';
import { getMemoryManager, ResourceType } from './memoryManager';

/**
 * Three.js resource types that need disposal
 */
type DisposableThreeObject = 
  | THREE.BufferGeometry
  | THREE.Material
  | THREE.Texture
  | THREE.RenderTarget
  | THREE.Object3D
  | THREE.BufferAttribute;

/**
 * Resource disposal mapping
 */
interface ResourceDisposal {
  dispose: () => void;
  type: string;
  size: number;
}

/**
 * Three.js specific memory manager
 */
export class ThreeMemoryManager {
  private memoryManager = getMemoryManager();
  private resources = new Map<string, ResourceDisposal>();
  private sceneResources = new WeakMap<THREE.Scene, Set<string>>();
  private componentName: string;

  constructor(componentName: string) {
    this.componentName = componentName;
  }

  /**
   * Track a Three.js resource
   */
  track(id: string, resource: DisposableThreeObject): void {
    const disposal = this.createDisposal(resource);
    if (disposal) {
      this.resources.set(id, disposal);
      this.memoryManager.track(
        `three-${id}`,
        ResourceType.WEBGL,
        () => this.dispose(id),
        {
          component: this.componentName,
          description: `Three.js ${disposal.type}`,
          size: disposal.size
        }
      );
    }
  }

  /**
   * Track an entire scene and its resources
   */
  trackScene(scene: THREE.Scene, sceneId: string): void {
    const resourceIds = new Set<string>();
    this.sceneResources.set(scene, resourceIds);

    // Traverse scene and track all resources
    scene.traverse((object) => {
      const objectId = `${sceneId}-object-${object.uuid}`;
      
      // Track the object itself
      if (object instanceof THREE.Mesh || object instanceof THREE.Line || object instanceof THREE.Points) {
        resourceIds.add(objectId);
        this.track(objectId, object);
        
        // Track geometry
        if (object.geometry) {
          const geoId = `${objectId}-geometry`;
          resourceIds.add(geoId);
          this.track(geoId, object.geometry);
        }
        
        // Track material(s)
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach((mat, idx) => {
              const matId = `${objectId}-material-${idx}`;
              resourceIds.add(matId);
              this.track(matId, mat);
              this.trackMaterialTextures(mat, matId, resourceIds);
            });
          } else {
            const matId = `${objectId}-material`;
            resourceIds.add(matId);
            this.track(matId, object.material);
            this.trackMaterialTextures(object.material, matId, resourceIds);
          }
        }
      }
    });
  }

  /**
   * Track textures in a material
   */
  private trackMaterialTextures(material: THREE.Material, materialId: string, resourceIds: Set<string>): void {
    const mat = material as any;
    const textureProperties = [
      'map', 'normalMap', 'bumpMap', 'roughnessMap', 'metalnessMap',
      'alphaMap', 'envMap', 'lightMap', 'aoMap', 'emissiveMap',
      'specularMap', 'displacementMap', 'gradientMap', 'matcap'
    ];

    textureProperties.forEach(prop => {
      if (mat[prop] && mat[prop] instanceof THREE.Texture) {
        const texId = `${materialId}-${prop}`;
        resourceIds.add(texId);
        this.track(texId, mat[prop]);
      }
    });
  }

  /**
   * Create disposal function for a resource
   */
  private createDisposal(resource: DisposableThreeObject): ResourceDisposal | null {
    if (resource instanceof THREE.BufferGeometry) {
      return {
        dispose: () => {
          resource.dispose();
          // Dispose attributes
          Object.values(resource.attributes).forEach(attr => {
            if (attr instanceof THREE.BufferAttribute || attr instanceof THREE.InterleavedBufferAttribute) {
              // Can't directly set array, but dispose() should handle cleanup
              if ('dispose' in attr && typeof attr.dispose === 'function') {
                attr.dispose();
              }
            }
          });
        },
        type: 'BufferGeometry',
        size: this.estimateGeometrySize(resource)
      };
    }

    if (resource instanceof THREE.Material) {
      return {
        dispose: () => {
          resource.dispose();
          // Dispose textures
          const mat = resource as any;
          const textureProperties = [
            'map', 'normalMap', 'bumpMap', 'roughnessMap', 'metalnessMap',
            'alphaMap', 'envMap', 'lightMap', 'aoMap', 'emissiveMap',
            'specularMap', 'displacementMap', 'gradientMap', 'matcap'
          ];
          
          textureProperties.forEach(prop => {
            if (mat[prop] && mat[prop].dispose) {
              mat[prop].dispose();
            }
          });
        },
        type: 'Material',
        size: 50000 // 50KB estimate
      };
    }

    if (resource instanceof THREE.Texture) {
      return {
        dispose: () => {
          resource.dispose();
          if (resource.image && resource.image instanceof ImageBitmap) {
            resource.image.close();
          }
          resource.image = null as any;
        },
        type: 'Texture',
        size: this.estimateTextureSize(resource)
      };
    }

    if (resource instanceof THREE.RenderTarget) {
      return {
        dispose: () => {
          resource.dispose();
        },
        type: 'RenderTarget',
        size: resource.width * resource.height * 4 * 4 // RGBA × float32
      };
    }

    if (resource instanceof THREE.Object3D) {
      return {
        dispose: () => {
          // Remove from parent
          if (resource.parent) {
            resource.parent.remove(resource);
          }
          
          // Clear children
          while (resource.children.length > 0) {
            resource.remove(resource.children[0]);
          }
          
          // Dispose geometry and material if it's a mesh
          if (resource instanceof THREE.Mesh || resource instanceof THREE.Line || resource instanceof THREE.Points) {
            if (resource.geometry) {
              resource.geometry.dispose();
            }
            if (resource.material) {
              if (Array.isArray(resource.material)) {
                resource.material.forEach(mat => mat.dispose());
              } else {
                resource.material.dispose();
              }
            }
          }
        },
        type: 'Object3D',
        size: 10000 // 10KB estimate
      };
    }

    return null;
  }

  /**
   * Estimate geometry size
   */
  private estimateGeometrySize(geometry: THREE.BufferGeometry): number {
    let size = 0;
    
    // Calculate size of all attributes
    Object.values(geometry.attributes).forEach(attribute => {
      if (attribute instanceof THREE.BufferAttribute || attribute instanceof THREE.InterleavedBufferAttribute) {
        const array = attribute.array;
        if (array) {
          size += array.byteLength || array.length * 4;
        }
      }
    });
    
    // Add index size
    if (geometry.index) {
      const indexArray = geometry.index.array;
      if (indexArray) {
        size += indexArray.byteLength || indexArray.length * 4;
      }
    }
    
    return size;
  }

  /**
   * Estimate texture size
   */
  private estimateTextureSize(texture: THREE.Texture): number {
    if (texture.image) {
      const img = texture.image;
      if (img.width && img.height) {
        // 4 bytes per pixel (RGBA)
        return img.width * img.height * 4;
      }
    }
    return 1048576; // Default 1MB
  }

  /**
   * Dispose a specific resource
   */
  dispose(id: string): void {
    const disposal = this.resources.get(id);
    if (disposal) {
      try {
        disposal.dispose();
        this.resources.delete(id);
      } catch (error) {
        console.error(`Failed to dispose Three.js resource ${id}:`, error);
      }
    }
  }

  /**
   * Dispose an entire scene
   */
  disposeScene(scene: THREE.Scene): void {
    const resourceIds = this.sceneResources.get(scene);
    if (resourceIds) {
      // Dispose all tracked resources
      for (const id of resourceIds) {
        this.dispose(id);
      }
      resourceIds.clear();
    }

    // Additional cleanup
    scene.traverse((object) => {
      if (object instanceof THREE.Mesh || object instanceof THREE.Line || object instanceof THREE.Points) {
        if (object.geometry) {
          object.geometry.dispose();
        }
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach(mat => mat.dispose());
          } else {
            object.material.dispose();
          }
        }
      }
    });

    // Clear the scene
    while (scene.children.length > 0) {
      scene.remove(scene.children[0]);
    }

    this.sceneResources.delete(scene);
  }

  /**
   * Dispose all resources
   */
  disposeAll(): void {
    for (const id of this.resources.keys()) {
      this.dispose(id);
    }
    this.resources.clear();
    this.sceneResources = new WeakMap();
  }

  /**
   * Get resource statistics
   */
  getStats(): {
    totalResources: number;
    totalSize: number;
    byType: Record<string, { count: number; size: number }>;
  } {
    const stats = {
      totalResources: this.resources.size,
      totalSize: 0,
      byType: {} as Record<string, { count: number; size: number }>
    };

    for (const resource of this.resources.values()) {
      stats.totalSize += resource.size;
      
      if (!stats.byType[resource.type]) {
        stats.byType[resource.type] = { count: 0, size: 0 };
      }
      
      stats.byType[resource.type].count++;
      stats.byType[resource.type].size += resource.size;
    }

    return stats;
  }
}

/**
 * React hook for Three.js memory management
 */
export function useThreeMemoryManager(componentName: string): ThreeMemoryManager {
  const managerRef = React.useRef<ThreeMemoryManager>();

  // Initialize immediately if not already done
  if (!managerRef.current) {
    managerRef.current = new ThreeMemoryManager(componentName);
  }

  React.useEffect(() => {
    // Ensure manager exists
    if (!managerRef.current) {
      managerRef.current = new ThreeMemoryManager(componentName);
    }
    
    return () => {
      if (managerRef.current) {
        managerRef.current.disposeAll();
      }
    };
  }, [componentName]);

  return managerRef.current;
}

/**
 * Utility to dispose Three.js renderer
 */
export function disposeRenderer(renderer: THREE.WebGLRenderer): void {
  renderer.dispose();
  renderer.forceContextLoss();
  
  // Clear render lists
  (renderer as any).renderLists?.dispose();
  
  // Clear info
  renderer.info.autoReset = false;
  renderer.info.reset();
  
  // Clear shadow map
  if (renderer.shadowMap.enabled) {
    renderer.shadowMap.enabled = false;
    // Force clear shadow map by toggling
    renderer.shadowMap.needsUpdate = true;
  }
  
  // Clear DOM element
  if (renderer.domElement.parentNode) {
    renderer.domElement.parentNode.removeChild(renderer.domElement);
  }
}

/**
 * Utility to optimize Three.js scene memory
 */
export function optimizeSceneMemory(scene: THREE.Scene): void {
  const geometryCache = new Map<string, THREE.BufferGeometry>();
  const materialCache = new Map<string, THREE.Material>();
  
  scene.traverse((object) => {
    if (object instanceof THREE.Mesh || object instanceof THREE.Line || object instanceof THREE.Points) {
      // Merge duplicate geometries
      if (object.geometry) {
        const geoKey = JSON.stringify({
          vertices: object.geometry.attributes.position?.count,
          indices: object.geometry.index?.count
        });
        
        const cached = geometryCache.get(geoKey);
        if (cached) {
          const oldGeo = object.geometry;
          object.geometry = cached;
          oldGeo.dispose();
        } else {
          geometryCache.set(geoKey, object.geometry);
        }
      }
      
      // Merge duplicate materials
      if (object.material && !Array.isArray(object.material)) {
        const matKey = JSON.stringify({
          type: object.material.type,
          color: (object.material as any).color?.getHex(),
          transparent: object.material.transparent,
          opacity: object.material.opacity
        });
        
        const cached = materialCache.get(matKey);
        if (cached) {
          const oldMat = object.material;
          object.material = cached;
          oldMat.dispose();
        } else {
          materialCache.set(matKey, object.material);
        }
      }
    }
  });
}

/**
 * Create a memory-efficient instanced mesh
 */
export function createInstancedMesh(
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  count: number,
  manager: ThreeMemoryManager
): THREE.InstancedMesh {
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  
  // Track the mesh
  manager.track(`instanced-mesh-${mesh.uuid}`, mesh);
  
  return mesh;
}

/**
 * Create a memory-efficient LOD object
 */
export function createLOD(
  levels: Array<{ geometry: THREE.BufferGeometry; material: THREE.Material; distance: number }>,
  manager: ThreeMemoryManager
): THREE.LOD {
  const lod = new THREE.LOD();
  
  levels.forEach((level, index) => {
    const mesh = new THREE.Mesh(level.geometry, level.material);
    lod.addLevel(mesh, level.distance);
    
    // Track resources
    manager.track(`lod-${lod.uuid}-level-${index}`, mesh);
  });
  
  return lod;
}