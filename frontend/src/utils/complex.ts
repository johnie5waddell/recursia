/**
 * Complex Number Implementation
 * Full mathematical operations for quantum computing
 */

export class Complex {
  real: number;
  imag: number;

  constructor(real: number = 0, imag: number = 0) {
    this.real = real;
    this.imag = imag;
  }
  
  // Getter aliases for convenience
  get re(): number {
    return this.real;
  }
  
  get im(): number {
    return this.imag;
  }

  // Basic arithmetic operations
  add(other: Complex): Complex {
    return new Complex(this.real + other.real, this.imag + other.imag);
  }

  subtract(other: Complex): Complex {
    return new Complex(this.real - other.real, this.imag - other.imag);
  }

  multiply(other: Complex): Complex {
    return new Complex(
      this.real * other.real - this.imag * other.imag,
      this.real * other.imag + this.imag * other.real
    );
  }

  divide(other: Complex): Complex {
    const denominator = other.real * other.real + other.imag * other.imag;
    if (Math.abs(denominator) < Number.EPSILON) {
      throw new Error('Division by zero in complex arithmetic');
    }
    return new Complex(
      (this.real * other.real + this.imag * other.imag) / denominator,
      (this.imag * other.real - this.real * other.imag) / denominator
    );
  }

  // Scalar operations
  scale(factor: number): Complex {
    return new Complex(this.real * factor, this.imag * factor);
  }

  // Complex conjugate
  conjugate(): Complex {
    return new Complex(this.real, -this.imag);
  }

  // Magnitude and phase
  magnitude(): number {
    return Math.sqrt(this.real * this.real + this.imag * this.imag);
  }

  phase(): number {
    return Math.atan2(this.imag, this.real);
  }

  // Power operations
  pow(exponent: number): Complex {
    const magnitude = Math.pow(this.magnitude(), exponent);
    const phase = this.phase() * exponent;
    return new Complex(
      magnitude * Math.cos(phase),
      magnitude * Math.sin(phase)
    );
  }

  // Exponential
  exp(): Complex {
    const expReal = Math.exp(this.real);
    return new Complex(
      expReal * Math.cos(this.imag),
      expReal * Math.sin(this.imag)
    );
  }

  // Natural logarithm
  log(): Complex {
    return new Complex(
      Math.log(this.magnitude()),
      this.phase()
    );
  }

  // Square root
  sqrt(): Complex {
    const magnitude = Math.sqrt(this.magnitude());
    const phase = this.phase() / 2;
    return new Complex(
      magnitude * Math.cos(phase),
      magnitude * Math.sin(phase)
    );
  }

  // Trigonometric functions
  sin(): Complex {
    return new Complex(
      Math.sin(this.real) * Math.cosh(this.imag),
      Math.cos(this.real) * Math.sinh(this.imag)
    );
  }

  cos(): Complex {
    return new Complex(
      Math.cos(this.real) * Math.cosh(this.imag),
      -Math.sin(this.real) * Math.sinh(this.imag)
    );
  }

  // Utility methods
  isZero(tolerance: number = Number.EPSILON): boolean {
    return Math.abs(this.real) < tolerance && Math.abs(this.imag) < tolerance;
  }

  equals(other: Complex, tolerance: number = Number.EPSILON): boolean {
    return Math.abs(this.real - other.real) < tolerance && 
           Math.abs(this.imag - other.imag) < tolerance;
  }

  clone(): Complex {
    return new Complex(this.real, this.imag);
  }

  toString(): string {
    if (this.imag >= 0) {
      return `${this.real.toFixed(6)} + ${this.imag.toFixed(6)}i`;
    } else {
      return `${this.real.toFixed(6)} - ${Math.abs(this.imag).toFixed(6)}i`;
    }
  }

  // Static factory methods
  static fromPolar(magnitude: number, phase: number): Complex {
    return new Complex(
      magnitude * Math.cos(phase),
      magnitude * Math.sin(phase)
    );
  }

  static zero(): Complex {
    return new Complex(0, 0);
  }

  static one(): Complex {
    return new Complex(1, 0);
  }

  static i(): Complex {
    return new Complex(0, 1);
  }

  // Array operations
  static sum(complexArray: Complex[]): Complex {
    return complexArray.reduce((sum, c) => sum.add(c), Complex.zero());
  }

  static normalize(complexArray: Complex[]): Complex[] {
    const normSquared = complexArray.reduce((sum, c) => sum + c.magnitude() * c.magnitude(), 0);
    const norm = Math.sqrt(normSquared);
    
    if (norm < Number.EPSILON) {
      throw new Error('Cannot normalize zero vector');
    }
    
    return complexArray.map(c => c.scale(1 / norm));
  }

  // Quantum state operations
  static innerProduct(a: Complex[], b: Complex[]): Complex {
    if (a.length !== b.length) {
      throw new Error('Vector dimensions must match for inner product');
    }
    
    return a.reduce((sum, aVal, i) => sum.add(aVal.conjugate().multiply(b[i])), Complex.zero());
  }

  static tensorProduct(a: Complex[], b: Complex[]): Complex[] {
    const result: Complex[] = [];
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < b.length; j++) {
        result.push(a[i].multiply(b[j]));
      }
    }
    return result;
  }

  // Matrix operations
  static matrixMultiply(matrix: Complex[][], vector: Complex[]): Complex[] {
    if (matrix[0].length !== vector.length) {
      throw new Error('Matrix and vector dimensions must be compatible');
    }
    
    return matrix.map(row => 
      row.reduce((sum, val, i) => sum.add(val.multiply(vector[i])), Complex.zero())
    );
  }

  // Serialization
  toJSON(): { real: number; imag: number; _type: string } {
    return {
      real: this.real,
      imag: this.imag,
      _type: 'Complex'
    };
  }

  static fromJSON(data: { real: number; imag: number; _type: string }): Complex {
    if (data._type !== 'Complex') {
      throw new Error('Invalid Complex number JSON data');
    }
    return new Complex(data.real, data.imag);
  }
}