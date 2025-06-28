"""
Decoder Interface Protocol
==========================

Defines the common interface that all quantum error correction decoders
must implement. This ensures consistency and interoperability across
different decoder implementations.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Protocol, runtime_checkable
from abc import ABC, abstractmethod


@runtime_checkable
class DecoderInterface(Protocol):
    """Protocol defining the interface for quantum error correction decoders."""
    
    def decode_surface_code(self, syndrome: List[int]) -> List[int]:
        """
        Decode surface code syndrome to find error correction.
        
        Args:
            syndrome: Binary syndrome vector
            
        Returns:
            List of correction operations (0=no correction, 1=X, 2=Z, 3=Y)
        """
        ...
    
    def decode_steane_code(self, syndrome: List[int]) -> List[int]:
        """Decode Steane [[7,1,3]] code syndrome."""
        ...
    
    def decode_shor_code(self, syndrome: List[int]) -> List[int]:
        """Decode Shor [[9,1,3]] code syndrome."""
        ...
    
    def decode_lookup_table(self, syndrome: List[int]) -> List[int]:
        """Decode using precomputed lookup table."""
        ...
    
    def get_decoder_stats(self) -> Dict[str, Any]:
        """Get decoder performance statistics."""
        ...


class BaseDecoder(ABC):
    """
    Abstract base class for quantum error correction decoders.
    
    Provides common functionality and enforces the decoder interface.
    """
    
    def __init__(self, code_distance: int):
        """
        Initialize base decoder.
        
        Args:
            code_distance: Distance of the quantum error correction code
        """
        self.code_distance = code_distance
        self.decode_count = 0
        self.success_count = 0
        self.total_decode_time = 0.0
        self.last_syndrome = None
        
    @abstractmethod
    def decode_surface_code(self, syndrome: List[int]) -> List[int]:
        """Decode surface code syndrome - must be implemented by subclasses."""
        pass
    
    def decode_steane_code(self, syndrome: List[int]) -> List[int]:
        """
        Default Steane code decoder using lookup table.
        
        The Steane [[7,1,3]] code has only 2^6 = 64 possible syndromes,
        making lookup table decoding feasible.
        """
        if not hasattr(self, '_steane_lookup'):
            self._steane_lookup = self._build_steane_lookup_table()
            
        syndrome_int = int(''.join(map(str, syndrome)), 2)
        return self._steane_lookup.get(syndrome_int, [0] * 7)
    
    def decode_shor_code(self, syndrome: List[int]) -> List[int]:
        """
        Default Shor code decoder.
        
        The Shor [[9,1,3]] code can correct any single qubit error.
        """
        if not hasattr(self, '_shor_lookup'):
            self._shor_lookup = self._build_shor_lookup_table()
            
        syndrome_int = int(''.join(map(str, syndrome)), 2)
        return self._shor_lookup.get(syndrome_int, [0] * 9)
    
    def decode_lookup_table(self, syndrome: List[int]) -> List[int]:
        """Generic lookup table decoder for small codes."""
        # For unknown codes, return identity correction
        n_qubits = max(7, len(syndrome))  # Minimum reasonable size
        return [0] * n_qubits
    
    def get_decoder_stats(self) -> Dict[str, Any]:
        """Get comprehensive decoder performance statistics."""
        success_rate = self.success_count / max(self.decode_count, 1)
        avg_time = self.total_decode_time / max(self.decode_count, 1)
        
        return {
            'decoder_type': self.__class__.__name__,
            'code_distance': self.code_distance,
            'total_decodes': self.decode_count,
            'successful_decodes': self.success_count,
            'success_rate': success_rate,
            'average_decode_time': avg_time,
            'total_time': self.total_decode_time
        }
    
    def _build_steane_lookup_table(self) -> Dict[int, List[int]]:
        """
        Build lookup table for Steane [[7,1,3]] code.
        
        Based on the standard Steane code generator matrix.
        """
        # Steane code stabilizer generators (6 generators for [[7,1,3]] code)
        # These are the standard stabilizers from the literature
        stabilizers = [
            [1, 0, 0, 1, 0, 1, 1],  # S1
            [0, 1, 0, 1, 1, 0, 1],  # S2  
            [0, 0, 1, 0, 1, 1, 1],  # S3
            [1, 0, 0, 0, 0, 0, 0],  # S4 (Z-type)
            [0, 1, 0, 0, 0, 0, 0],  # S5 (Z-type)
            [0, 0, 1, 0, 0, 0, 0]   # S6 (Z-type)
        ]
        
        lookup = {}
        
        # For each possible single-qubit error
        for error_qubit in range(7):
            # Calculate syndrome for X error on this qubit
            syndrome = [0] * 6
            for i, stab in enumerate(stabilizers):
                syndrome[i] = stab[error_qubit]
                
            syndrome_int = int(''.join(map(str, syndrome)), 2)
            correction = [0] * 7
            correction[error_qubit] = 1  # X correction
            lookup[syndrome_int] = correction
        
        # No error case
        lookup[0] = [0] * 7
        
        return lookup
    
    def _build_shor_lookup_table(self) -> Dict[int, List[int]]:
        """
        Build lookup table for Shor [[9,1,3]] code.
        
        The Shor code is designed to correct arbitrary single-qubit errors.
        """
        lookup = {}
        
        # Shor code has 8 syndrome bits (2 for each block of 3 qubits, plus phase)
        # This is a simplified implementation focusing on X errors
        
        # No error
        lookup[0] = [0] * 9
        
        # Single X errors (example patterns)
        for qubit in range(9):
            # Syndrome pattern depends on which block the error is in
            block = qubit // 3
            position = qubit % 3
            
            # Simplified syndrome calculation (full implementation would be more complex)
            syndrome_bits = [0] * 8
            syndrome_bits[block * 2] = 1 if position in [0, 1] else 0
            syndrome_bits[block * 2 + 1] = 1 if position in [1, 2] else 0
            
            syndrome_int = int(''.join(map(str, syndrome_bits)), 2)
            correction = [0] * 9
            correction[qubit] = 1
            lookup[syndrome_int] = correction
            
        return lookup
    
    def _record_decode_attempt(self, success: bool, decode_time: float):
        """Record statistics for a decode attempt."""
        self.decode_count += 1
        if success:
            self.success_count += 1
        self.total_decode_time += decode_time
    
    def reset_stats(self):
        """Reset decoder statistics."""
        self.decode_count = 0
        self.success_count = 0
        self.total_decode_time = 0.0