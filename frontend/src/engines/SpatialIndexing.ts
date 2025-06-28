/**
 * Spatial Indexing System
 * Octree-based spatial indexing for O(log n) neighbor queries
 */

export interface SpatialObject {
  id: string;
  position: [number, number, number];
  radius?: number;
}

export class OctreeNode {
  center: [number, number, number];
  halfSize: number;
  objects: SpatialObject[] = [];
  children: OctreeNode[] | null = null;
  maxObjects: number = 8;
  maxDepth: number = 8;
  depth: number;
  
  constructor(center: [number, number, number], halfSize: number, depth: number = 0) {
    this.center = center;
    this.halfSize = halfSize;
    this.depth = depth;
  }
  
  /**
   * Insert object into octree
   */
  insert(obj: SpatialObject): boolean {
    // Check if object is within bounds
    if (!this.contains(obj.position)) {
      return false;
    }
    
    // If no children and under capacity, add to this node
    if (this.children === null && this.objects.length < this.maxObjects) {
      this.objects.push(obj);
      return true;
    }
    
    // If at max depth, add to this node anyway
    if (this.depth >= this.maxDepth) {
      this.objects.push(obj);
      return true;
    }
    
    // Otherwise, subdivide if needed and insert into child
    if (this.children === null) {
      this.subdivide();
    }
    
    // Try to insert into children
    for (const child of this.children!) {
      if (child.insert(obj)) {
        return true;
      }
    }
    
    // If no child accepted it, keep it here
    this.objects.push(obj);
    return true;
  }
  
  /**
   * Remove object from octree
   */
  remove(obj: SpatialObject): boolean {
    const index = this.objects.findIndex(o => o.id === obj.id);
    if (index !== -1) {
      this.objects.splice(index, 1);
      return true;
    }
    
    if (this.children !== null) {
      for (const child of this.children) {
        if (child.remove(obj)) {
          return true;
        }
      }
    }
    
    return false;
  }
  
  /**
   * Query objects within radius
   */
  query(position: [number, number, number], radius: number): SpatialObject[] {
    const results: SpatialObject[] = [];
    
    // Check if search sphere intersects this node
    if (!this.intersectsSphere(position, radius)) {
      return results;
    }
    
    // Check objects in this node
    for (const obj of this.objects) {
      const dist = this.distance(position, obj.position);
      if (dist <= radius + (obj.radius || 0)) {
        results.push(obj);
      }
    }
    
    // Recursively check children
    if (this.children !== null) {
      for (const child of this.children) {
        results.push(...child.query(position, radius));
      }
    }
    
    return results;
  }
  
  /**
   * Get all objects in octree
   */
  getAllObjects(): SpatialObject[] {
    const results: SpatialObject[] = [...this.objects];
    
    if (this.children !== null) {
      for (const child of this.children) {
        results.push(...child.getAllObjects());
      }
    }
    
    return results;
  }
  
  /**
   * Check if position is within node bounds
   */
  private contains(position: [number, number, number]): boolean {
    return Math.abs(position[0] - this.center[0]) <= this.halfSize &&
           Math.abs(position[1] - this.center[1]) <= this.halfSize &&
           Math.abs(position[2] - this.center[2]) <= this.halfSize;
  }
  
  /**
   * Check if sphere intersects node
   */
  private intersectsSphere(center: [number, number, number], radius: number): boolean {
    let distSquared = 0;
    
    for (let i = 0; i < 3; i++) {
      const min = this.center[i] - this.halfSize;
      const max = this.center[i] + this.halfSize;
      
      if (center[i] < min) {
        distSquared += Math.pow(center[i] - min, 2);
      } else if (center[i] > max) {
        distSquared += Math.pow(center[i] - max, 2);
      }
    }
    
    return distSquared <= radius * radius;
  }
  
  /**
   * Calculate distance between positions
   */
  private distance(p1: [number, number, number], p2: [number, number, number]): number {
    return Math.sqrt(
      Math.pow(p1[0] - p2[0], 2) +
      Math.pow(p1[1] - p2[1], 2) +
      Math.pow(p1[2] - p2[2], 2)
    );
  }
  
  /**
   * Subdivide node into 8 children
   */
  private subdivide(): void {
    const childSize = this.halfSize / 2;
    this.children = [];
    
    for (let x = -1; x <= 1; x += 2) {
      for (let y = -1; y <= 1; y += 2) {
        for (let z = -1; z <= 1; z += 2) {
          const childCenter: [number, number, number] = [
            this.center[0] + x * childSize,
            this.center[1] + y * childSize,
            this.center[2] + z * childSize
          ];
          
          this.children.push(new OctreeNode(childCenter, childSize, this.depth + 1));
        }
      }
    }
    
    // Re-insert objects into children
    const objects = [...this.objects];
    this.objects = [];
    
    for (const obj of objects) {
      let inserted = false;
      for (const child of this.children) {
        if (child.insert(obj)) {
          inserted = true;
          break;
        }
      }
      if (!inserted) {
        this.objects.push(obj); // Keep in parent if no child accepts
      }
    }
  }
}

/**
 * Spatial Index wrapper for easy use
 */
export class SpatialIndex {
  private root: OctreeNode;
  private objects: Map<string, SpatialObject> = new Map();
  
  constructor(bounds: { center: [number, number, number]; size: number }) {
    this.root = new OctreeNode(bounds.center, bounds.size / 2);
  }
  
  /**
   * Add object to spatial index
   */
  add(obj: SpatialObject): void {
    this.objects.set(obj.id, obj);
    this.root.insert(obj);
  }
  
  /**
   * Remove object from spatial index
   */
  remove(id: string): void {
    const obj = this.objects.get(id);
    if (obj) {
      this.objects.delete(id);
      this.root.remove(obj);
    }
  }
  
  /**
   * Update object position
   */
  update(id: string, newPosition: [number, number, number]): void {
    const obj = this.objects.get(id);
    if (obj) {
      this.root.remove(obj);
      obj.position = newPosition;
      this.root.insert(obj);
    }
  }
  
  /**
   * Find neighbors within radius
   */
  findNeighbors(position: [number, number, number], radius: number): SpatialObject[] {
    return this.root.query(position, radius);
  }
  
  /**
   * Find k nearest neighbors
   */
  findKNearest(position: [number, number, number], k: number): SpatialObject[] {
    // Start with small radius and expand
    let radius = 1;
    let neighbors: SpatialObject[] = [];
    
    while (neighbors.length < k && radius < 1000) {
      neighbors = this.root.query(position, radius);
      radius *= 2;
    }
    
    // Sort by distance and return k nearest
    neighbors.sort((a, b) => {
      const distA = this.distance(position, a.position);
      const distB = this.distance(position, b.position);
      return distA - distB;
    });
    
    return neighbors.slice(0, k);
  }
  
  /**
   * Get all objects
   */
  getAll(): SpatialObject[] {
    return Array.from(this.objects.values());
  }
  
  /**
   * Clear the index
   */
  clear(): void {
    this.objects.clear();
    this.root = new OctreeNode(this.root.center, this.root.halfSize);
  }
  
  /**
   * Get statistics
   */
  getStats(): {
    objectCount: number;
    nodeCount: number;
    maxDepth: number;
  } {
    const stats = {
      objectCount: this.objects.size,
      nodeCount: 0,
      maxDepth: 0
    };
    
    const countNodes = (node: OctreeNode, depth: number) => {
      stats.nodeCount++;
      stats.maxDepth = Math.max(stats.maxDepth, depth);
      
      if (node.children !== null) {
        for (const child of node.children) {
          countNodes(child, depth + 1);
        }
      }
    };
    
    countNodes(this.root, 0);
    
    return stats;
  }
  
  private distance(p1: [number, number, number], p2: [number, number, number]): number {
    return Math.sqrt(
      Math.pow(p1[0] - p2[0], 2) +
      Math.pow(p1[1] - p2[1], 2) +
      Math.pow(p1[2] - p2[2], 2)
    );
  }
}