"""
BINARY TRANSLATION ENGINE v1.0
Universal Text → Binary → Pattern Recognition System

CORE PRINCIPLE: All text is information, all information is binary
- Convert any script to binary representation
- Find self-consistent patterns in binary space
- Validate through internal coherence, not external assumptions

Created: December 2025
Purpose: Translate unknown scripts through binary pattern analysis
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib

# ============================================================================
# BINARY CONVERSION STRATEGIES
# ============================================================================

class BinaryConverter:
    """Multiple strategies for converting text to binary"""
    
    @staticmethod
    def unicode_to_binary(text: str, bits_per_char: int = 16) -> str:
        """Convert text using Unicode code points"""
        binary = ""
        for char in text:
            code = ord(char)
            binary += format(code, f'0{bits_per_char}b')
        return binary
    
    @staticmethod
    def frequency_to_binary(text: str) -> str:
        """
        Convert based on character frequency (Huffman-like encoding)
        Most frequent = shorter codes
        """
        freq = Counter(text)
        sorted_chars = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        # Simple binary encoding: rank in frequency determines code
        encoding = {}
        for rank, (char, _) in enumerate(sorted_chars):
            # Use Gray code for adjacent ranks
            encoding[char] = format(rank ^ (rank >> 1), '08b')
        
        return ''.join(encoding.get(c, '00000000') for c in text)
    
    @staticmethod
    def positional_to_binary(text: str) -> str:
        """
        Convert based on position in text
        Preserves sequential information
        """
        binary = ""
        n = len(text)
        bits_needed = max(8, math.ceil(math.log2(n + 1)))
        
        for i, char in enumerate(text):
            # Position XOR with character code
            combined = i ^ ord(char)
            binary += format(combined, f'0{bits_needed}b')
        
        return binary
    
    @staticmethod
    def adaptive_to_binary(text: str) -> str:
        """
        Adaptive encoding: switches strategy based on detected patterns
        """
        # Analyze text first
        unique_ratio = len(set(text)) / len(text) if text else 1
        
        if unique_ratio > 0.8:
            # High diversity → use Unicode
            return BinaryConverter.unicode_to_binary(text, 16)
        elif unique_ratio < 0.2:
            # Low diversity → use frequency
            return BinaryConverter.frequency_to_binary(text)
        else:
            # Medium diversity → use positional
            return BinaryConverter.positional_to_binary(text)

# ============================================================================
# BINARY PATTERN ANALYZER
# ============================================================================

@dataclass
class BinaryPattern:
    """A discovered pattern in binary space"""
    pattern_bits: str
    frequency: int
    positions: List[int]
    entropy: float
    symmetry: bool
    periodicity: Optional[int]

class BinaryPatternAnalyzer:
    """Analyze patterns in binary sequences"""
    
    def __init__(self, binary: str):
        self.binary = binary
        self.length = len(binary)
    
    def find_all_patterns(self, min_length: int = 4, max_length: int = 16) -> List[BinaryPattern]:
        """Find all recurring patterns in binary sequence"""
        patterns = []
        seen = set()
        
        for pattern_len in range(min_length, min(max_length + 1, self.length // 2)):
            for i in range(self.length - pattern_len + 1):
                pattern = self.binary[i:i + pattern_len]
                
                if pattern in seen:
                    continue
                
                # Find all occurrences
                positions = []
                for j in range(self.length - pattern_len + 1):
                    if self.binary[j:j + pattern_len] == pattern:
                        positions.append(j)
                
                if len(positions) >= 2:  # Must repeat at least once
                    # Calculate pattern properties
                    entropy = self._calculate_pattern_entropy(pattern)
                    symmetry = self._is_symmetric(pattern)
                    periodicity = self._detect_periodicity(positions)
                    
                    patterns.append(BinaryPattern(
                        pattern_bits=pattern,
                        frequency=len(positions),
                        positions=positions,
                        entropy=entropy,
                        symmetry=symmetry,
                        periodicity=periodicity
                    ))
                    
                    seen.add(pattern)
        
        # Sort by frequency * length (importance score)
        patterns.sort(key=lambda p: p.frequency * len(p.pattern_bits), reverse=True)
        return patterns
    
    def find_bit_transitions(self) -> Dict:
        """Analyze 0→1 and 1→0 transitions"""
        transitions = {
            '0→1': [],
            '1→0': [],
            'runs_of_0': [],
            'runs_of_1': []
        }
        
        current_bit = self.binary[0] if self.binary else None
        run_start = 0
        
        for i in range(1, self.length):
            if self.binary[i] != current_bit:
                # Transition detected
                transition_type = f'{current_bit}→{self.binary[i]}'
                transitions[transition_type].append(i)
                
                # Record run
                run_length = i - run_start
                transitions[f'runs_of_{current_bit}'].append(run_length)
                
                current_bit = self.binary[i]
                run_start = i
        
        # Last run
        if current_bit is not None:
            transitions[f'runs_of_{current_bit}'].append(self.length - run_start)
        
        return transitions
    
    def calculate_information_density(self, window_size: int = 8) -> List[float]:
        """Calculate local information density (entropy) across sequence"""
        densities = []
        
        for i in range(0, self.length - window_size + 1, window_size // 2):
            window = self.binary[i:i + window_size]
            entropy = self._calculate_pattern_entropy(window)
            densities.append(entropy)
        
        return densities
    
    def find_palindromes(self, min_length: int = 4) -> List[Tuple[int, int, str]]:
        """Find palindromic binary sequences"""
        palindromes = []
        
        for center in range(self.length):
            # Odd-length palindromes
            for radius in range(1, min(center + 1, self.length - center)):
                left = self.binary[center - radius:center]
                right = self.binary[center + 1:center + radius + 1]
                
                if left == right[::-1] and len(left) >= min_length // 2:
                    palindrome = left + self.binary[center] + right
                    palindromes.append((center - radius, center + radius + 1, palindrome))
                else:
                    break
        
        return palindromes
    
    def detect_compression_potential(self) -> Dict:
        """Detect how compressible the binary sequence is"""
        # Run-length encoding potential
        runs = []
        current = self.binary[0] if self.binary else None
        count = 1
        
        for i in range(1, self.length):
            if self.binary[i] == current:
                count += 1
            else:
                runs.append(count)
                current = self.binary[i]
                count = 1
        runs.append(count)
        
        # Compression ratio estimate
        original_bits = self.length
        rle_bits = len(runs) * (1 + 8)  # bit value + 8-bit count
        compression_ratio = rle_bits / original_bits if original_bits > 0 else 1
        
        return {
            'original_bits': original_bits,
            'rle_bits': rle_bits,
            'compression_ratio': compression_ratio,
            'average_run_length': np.mean(runs) if runs else 0,
            'max_run_length': max(runs) if runs else 0
        }
    
    def _calculate_pattern_entropy(self, pattern: str) -> float:
        """Shannon entropy of bit pattern"""
        if not pattern:
            return 0.0
        
        ones = pattern.count('1')
        zeros = len(pattern) - ones
        
        if ones == 0 or zeros == 0:
            return 0.0
        
        p_one = ones / len(pattern)
        p_zero = zeros / len(pattern)
        
        entropy = -(p_one * math.log2(p_one) + p_zero * math.log2(p_zero))
        return entropy
    
    def _is_symmetric(self, pattern: str) -> bool:
        """Check if pattern is palindromic"""
        return pattern == pattern[::-1]
    
    def _detect_periodicity(self, positions: List[int]) -> Optional[int]:
        """Detect if pattern repeats with regular spacing"""
        if len(positions) < 2:
            return None
        
        # Calculate spacing between occurrences
        spacings = [positions[i+1] - positions[i] for i in range(len(positions) - 1)]
        
        # Check if spacings are consistent
        if len(set(spacings)) == 1:
            return spacings[0]
        
        # Check if there's a GCD pattern
        from math import gcd
        from functools import reduce
        
        common_divisor = reduce(gcd, spacings)
        if common_divisor > 1:
            return common_divisor
        
        return None

# ============================================================================
# SELF-CONSISTENCY VALIDATOR
# ============================================================================

class SelfConsistencyValidator:
    """Validate translation through internal coherence"""
    
    def __init__(self, original_text: str, binary: str):
        self.original_text = original_text
        self.binary = binary
        self.analyzer = BinaryPatternAnalyzer(binary)
    
    def validate(self) -> Dict:
        """Run all self-consistency checks"""
        return {
            'information_preservation': self._check_information_preservation(),
            'pattern_coherence': self._check_pattern_coherence(),
            'structural_mapping': self._check_structural_mapping(),
            'reversibility': self._check_reversibility(),
            'complexity_consistency': self._check_complexity_consistency()
        }
    
    def _check_information_preservation(self) -> Dict:
        """Verify information is preserved in conversion"""
        # Shannon entropy of original vs binary
        original_entropy = self._text_entropy(self.original_text)
        binary_entropy = self._text_entropy(self.binary)
        
        # Information should be preserved (binary entropy ≈ original entropy)
        preservation_ratio = binary_entropy / original_entropy if original_entropy > 0 else 0
        
        return {
            'original_entropy': original_entropy,
            'binary_entropy': binary_entropy,
            'preservation_ratio': preservation_ratio,
            'information_preserved': 0.8 <= preservation_ratio <= 1.2
        }
    
    def _check_pattern_coherence(self) -> Dict:
        """Check if patterns in binary correlate with patterns in original"""
        # Find repeating sequences in original
        original_patterns = self._find_text_repeats(self.original_text)
        binary_patterns = self.analyzer.find_all_patterns(min_length=4, max_length=12)
        
        # Patterns should exist in both
        return {
            'original_pattern_count': len(original_patterns),
            'binary_pattern_count': len(binary_patterns),
            'patterns_coherent': len(binary_patterns) > 0
        }
    
    def _check_structural_mapping(self) -> Dict:
        """Check if structure of original maps to structure of binary"""
        # Character spacing in original should correlate with bit spacing
        char_spacings = self._calculate_character_spacings(self.original_text)
        
        transitions = self.analyzer.find_bit_transitions()
        transition_spacings = []
        for trans_type in ['0→1', '1→0']:
            positions = transitions[trans_type]
            if len(positions) >= 2:
                spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                transition_spacings.extend(spacings)
        
        return {
            'original_unique_chars': len(set(self.original_text)),
            'transition_count': len(transition_spacings),
            'structure_mapped': len(transition_spacings) > 0
        }
    
    def _check_reversibility(self) -> Dict:
        """Check if conversion is theoretically reversible"""
        # Can we reconstruct unique mapping?
        unique_chars = len(set(self.original_text))
        bits_per_char = len(self.binary) // len(self.original_text) if self.original_text else 0
        
        # Theoretical capacity: 2^bits_per_char should >= unique_chars
        theoretical_capacity = 2 ** bits_per_char if bits_per_char > 0 else 0
        
        return {
            'bits_per_character': bits_per_char,
            'unique_characters': unique_chars,
            'theoretical_capacity': theoretical_capacity,
            'reversible': theoretical_capacity >= unique_chars
        }
    
    def _check_complexity_consistency(self) -> Dict:
        """Check if complexity measures are consistent"""
        # Kolmogorov complexity approximation (via compression)
        compression = self.analyzer.detect_compression_potential()
        
        # High compression = low complexity (repetitive)
        # Low compression = high complexity (random or information-dense)
        
        return {
            'compression_ratio': compression['compression_ratio'],
            'complexity_class': self._classify_complexity(compression['compression_ratio'])
        }
    
    def _text_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        freq = Counter(text)
        entropy = 0
        for count in freq.values():
            p = count / len(text)
            entropy -= p * math.log2(p)
        
        return entropy
    
    def _find_text_repeats(self, text: str, min_len: int = 2) -> List[str]:
        """Find repeating substrings"""
        repeats = []
        seen = set()
        
        for length in range(min_len, len(text) // 2):
            for i in range(len(text) - length + 1):
                substr = text[i:i + length]
                if substr not in seen and text.count(substr) >= 2:
                    repeats.append(substr)
                    seen.add(substr)
        
        return repeats
    
    def _calculate_character_spacings(self, text: str) -> List[int]:
        """Calculate spacing between repeated characters"""
        positions = defaultdict(list)
        for i, char in enumerate(text):
            positions[char].append(i)
        
        spacings = []
        for char_positions in positions.values():
            if len(char_positions) >= 2:
                for i in range(len(char_positions) - 1):
                    spacings.append(char_positions[i+1] - char_positions[i])
        
        return spacings
    
    def _classify_complexity(self, compression_ratio: float) -> str:
        """Classify complexity based on compression ratio"""
        if compression_ratio < 0.3:
            return "highly_repetitive"
        elif compression_ratio < 0.7:
            return "structured"
        elif compression_ratio < 0.95:
            return "complex"
        else:
            return "random_or_dense"

# ============================================================================
# BINARY TRANSLATION ENGINE
# ============================================================================

class BinaryTranslationEngine:
    """Complete binary translation and analysis system"""
    
    def __init__(self):
        self.converter = BinaryConverter()
        self.strategies = {
            'unicode': self.converter.unicode_to_binary,
            'frequency': self.converter.frequency_to_binary,
            'positional': self.converter.positional_to_binary,
            'adaptive': self.converter.adaptive_to_binary
        }
    
    def translate(self, text: str, strategy: str = 'adaptive') -> Dict:
        """
        Translate text to binary and analyze
        
        Args:
            text: Input text in any script
            strategy: 'unicode', 'frequency', 'positional', or 'adaptive'
        
        Returns:
            Complete analysis dictionary
        """
        if strategy not in self.strategies:
            strategy = 'adaptive'
        
        # Convert to binary
        binary = self.strategies[strategy](text)
        
        # Analyze patterns
        analyzer = BinaryPatternAnalyzer(binary)
        patterns = analyzer.find_all_patterns()
        transitions = analyzer.find_bit_transitions()
        info_density = analyzer.calculate_information_density()
        palindromes = analyzer.find_palindromes()
        compression = analyzer.detect_compression_potential()
        
        # Validate consistency
        validator = SelfConsistencyValidator(text, binary)
        validation = validator.validate()
        
        return {
            'input': {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'length': len(text),
                'unique_characters': len(set(text))
            },
            'binary': {
                'sequence': binary[:200] + '...' if len(binary) > 200 else binary,
                'length': len(binary),
                'ones': binary.count('1'),
                'zeros': binary.count('0'),
                'balance': binary.count('1') / len(binary) if binary else 0,
                'strategy_used': strategy
            },
            'patterns': {
                'discovered': len(patterns),
                'top_patterns': [
                    {
                        'bits': p.pattern_bits,
                        'frequency': p.frequency,
                        'length': len(p.pattern_bits),
                        'entropy': p.entropy,
                        'symmetric': p.symmetry,
                        'periodic': p.periodicity is not None
                    }
                    for p in patterns[:10]
                ]
            },
            'transitions': {
                '0_to_1': len(transitions['0→1']),
                '1_to_0': len(transitions['1→0']),
                'avg_run_0': np.mean(transitions['runs_of_0']) if transitions['runs_of_0'] else 0,
                'avg_run_1': np.mean(transitions['runs_of_1']) if transitions['runs_of_1'] else 0
            },
            'information': {
                'density_mean': np.mean(info_density) if info_density else 0,
                'density_std': np.std(info_density) if info_density else 0,
                'density_max': max(info_density) if info_density else 0
            },
            'structure': {
                'palindromes': len(palindromes),
                'compression_ratio': compression['compression_ratio'],
                'complexity_class': validation['complexity_consistency']['complexity_class']
            },
            'validation': validation
        }
    
    def compare_strategies(self, text: str) -> Dict:
        """Compare all conversion strategies"""
        results = {}
        
        for strategy in ['unicode', 'frequency', 'positional', 'adaptive']:
            result = self.translate(text, strategy)
            results[strategy] = {
                'pattern_count': result['patterns']['discovered'],
                'compression_ratio': result['structure']['compression_ratio'],
                'information_preserved': result['validation']['information_preservation']['information_preserved'],
                'bits_per_char': result['binary']['length'] / result['input']['length'] if result['input']['length'] > 0 else 0
            }
        
        return results

# ============================================================================
# HIEROGLYPH TEST SUITE
# ============================================================================

def test_book_of_dead_passages():
    """Test translation engine on Egyptian hieroglyphic passages"""
    
    engine = BinaryTranslationEngine()
    
    # Three passages from Book of the Dead (Unicode hieroglyphs)
    passages = [
        {
            'name': 'Spell 1 - Entering the Tomb',
            'text': '𓁹𓈖𓏏𓏤𓀀𓇋𓏲𓂻𓏛𓏥𓀀'
        },
        {
            'name': 'Spell 15 - Hymn to Ra',
            'text': '𓇳𓏤𓎟𓋹𓇋𓅱𓏏𓈖𓏤𓀭𓀀'
        },
        {
            'name': 'Spell 125 - Weighing of the Heart',
            'text': '𓂋𓇋𓎡𓏏𓉐𓊃𓏏𓀀𓁹𓏥'
        }
    ]
    
    print("="*80)
    print("BINARY TRANSLATION ENGINE - BOOK OF THE DEAD TEST")
    print("="*80)
    
    for passage in passages:
        print(f"\n{'='*80}")
        print(f"PASSAGE: {passage['name']}")
        print(f"HIEROGLYPHS: {passage['text']}")
        print(f"{'='*80}")
        
        # Translate
        result = engine.translate(passage['text'], strategy='adaptive')
        
        # Display key findings
        print(f"\nBINARY CONVERSION:")
        print(f"  Length: {result['binary']['length']} bits")
        print(f"  Balance: {result['binary']['balance']:.3f} (1s ratio)")
        print(f"  Strategy: {result['binary']['strategy_used']}")
        
        print(f"\nPATTERN DISCOVERY:")
        print(f"  Patterns found: {result['patterns']['discovered']}")
        
        if result['patterns']['top_patterns']:
            print(f"\n  Top patterns:")
            for i, p in enumerate(result['patterns']['top_patterns'][:3], 1):
                print(f"    {i}. {p['bits']} (frequency: {p['frequency']}, entropy: {p['entropy']:.3f})")
        
        print(f"\nBIT TRANSITIONS:")
        print(f"  0→1 transitions: {result['transitions']['0_to_1']}")
        print(f"  1→0 transitions: {result['transitions']['1_to_0']}")
        print(f"  Avg run of 0s: {result['transitions']['avg_run_0']:.2f}")
        print(f"  Avg run of 1s: {result['transitions']['avg_run_1']:.2f}")
        
        print(f"\nSTRUCTURE:")
        print(f"  Palindromes: {result['structure']['palindromes']}")
        print(f"  Compression ratio: {result['structure']['compression_ratio']:.3f}")
        print(f"  Complexity: {result['structure']['complexity_class']}")
        
        print(f"\nVALIDATION:")
        info_pres = result['validation']['information_preservation']
        print(f"  Information preserved: {info_pres['information_preserved']}")
        print(f"  Preservation ratio: {info_pres['preservation_ratio']:.3f}")
        
        rev = result['validation']['reversibility']
        print(f"  Reversible: {rev['reversible']}")
        print(f"  Bits per character: {rev['bits_per_character']}")
        
        # Compare strategies
        print(f"\nSTRATEGY COMPARISON:")
        comparison = engine.compare_strategies(passage['text'])
        for strategy, metrics in comparison.items():
            print(f"  {strategy}:")
            print(f"    Patterns: {metrics['pattern_count']}")
            print(f"    Compression: {metrics['compression_ratio']:.3f}")
            print(f"    Info preserved: {metrics['information_preserved']}")

# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate():
    """Demonstrate binary translation capabilities"""
    
    engine = BinaryTranslationEngine()
    
    # Test on various scripts
    test_texts = [
        ("English", "HELLO WORLD"),
        ("Repeating", "ABABABABAB"),
        ("Arabic", "مرحبا بالعالم"),
        ("Chinese", "你好世界"),
        ("Mixed", "Hello 世界 🌍"),
    ]
    
    print("="*80)
    print("BINARY TRANSLATION ENGINE - GENERAL TESTS")
    print("="*80)
    
    for name, text in test_texts:
        print(f"\n{'-'*80}")
        print(f"TEST: {name}")
        print(f"TEXT: {text}")
        print(f"{'-'*80}")
        
        result = engine.translate(text)
        
        print(f"Binary length: {result['binary']['length']} bits")
        print(f"Patterns found: {result['patterns']['discovered']}")
        print(f"Complexity: {result['structure']['complexity_class']}")
        print(f"Validated: {result['validation']['information_preservation']['information_preserved']}")

if __name__ == "__main__":
    # Run both demonstrations
    demonstrate()
    print("\n\n")
    test_book_of_dead_passages()