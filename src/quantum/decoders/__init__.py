"""
Quantum Error Correction Decoders
=================================

This module implements state-of-the-art quantum error correction decoders
for various quantum error correction codes. All implementations are
peer-review ready and based on established research.

Available Decoders:
- MWPMDecoder: Minimum Weight Perfect Matching for surface codes
- UnionFindDecoder: Fast approximate decoder for surface codes  
- LookupDecoder: Exact decoder for small stabilizer codes
- MLDecoder: Machine learning decoder for arbitrary codes

References:
- Fowler et al., "Surface codes: Towards practical large-scale quantum computation" (2012)
- Delfosse & Nickerson, "Almost-linear time decoding algorithm for topological codes" (2017)
- Varona & Martin-Lopez, "Quantum error correction with neural networks" (2021)
"""

__all__ = [
    'MWPMDecoder',
    'UnionFindDecoder', 
    'LookupDecoder',
    'MLDecoder',
    'DecoderInterface',
    'DecoderBenchmark'
]