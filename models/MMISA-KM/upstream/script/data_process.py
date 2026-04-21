# models/mmisa_km/upstream/script/data_process.py
"""
Feature extraction functions for MMISA-KM.
Matches original training-time feature specifications:
- Molecule: 78-dim atom features
- Protein: 33-dim residue features (21 one-hot + 12 physicochemical)
"""

from typing import List
import numpy as np


# ============================================================================
# Atomic Feature Definitions (78-dim total)
# ============================================================================

ELEMENT_LIST = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
    'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
    'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
    'Pt', 'Hg', 'Pb', 'UNK'  # 44 elements
]


def one_of_k_encoding(x: str, allowable_set: List[str]) -> List[int]:
    """One-hot encoding: returns 1 at matching index, 0 elsewhere."""
    if x not in allowable_set:
        x = allowable_set[-1]  # Map unknown to last element ('UNK')
    return [int(x == s) for s in allowable_set]


def one_of_k_encoding_unk(x: str, allowable_set: List[str]) -> List[int]:
    """One-hot encoding with unknown fallback to all-zeros."""
    if x not in allowable_set:
        return [0] * len(allowable_set)
    return [int(x == s) for s in allowable_set]


def atom_features(atom) -> np.ndarray:
    """
    Generate 78-dimensional feature vector for an RDKit atom.
    
    Feature breakdown:
    - [0:44]  : Atomic symbol one-hot (44 dims)
    - [44:55] : Degree (heavy atom neighbors) one-hot, 0-10 (11 dims)
    - [55:66] : Total hydrogen count one-hot, 0-10 (11 dims)
    - [66:77] : Implicit valence one-hot, 0-10 (11 dims)
    - [77]    : Is aromatic flag (1 dim)
    
    Total: 44 + 11 + 11 + 11 + 1 = 78
    """
    features = []
    
    # 44-dim: Atomic symbol
    features.extend(one_of_k_encoding_unk(atom.GetSymbol(), ELEMENT_LIST))
    
    # 11-dim: Degree (number of heavy atom neighbors)
    features.extend(one_of_k_encoding(atom.GetDegree(), list(range(11))))
    
    # 11-dim: Total hydrogen count (explicit + implicit)
    features.extend(one_of_k_encoding(atom.GetTotalNumHs(), list(range(11))))
    
    # 11-dim: Implicit valence
    features.extend(one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11))))
    
    # 1-dim: Aromaticity flag
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# Residue Feature Definitions (33-dim total: 21 one-hot + 12 properties)
# ============================================================================

# Amino acid classification sets
ALIPHATIC = set("AVILM")
AROMATIC = set("FWYH")
POLAR_NEUTRAL = set("STCNQ")
ACIDIC = set("DE")
BASIC = set("KR")

# Normalized physicochemical properties per residue
# Values scaled to [0, 1] based on literature ranges
# Format: [weight_norm, pKa_norm, pKb_norm, pKx_norm, pI_norm, hydro_ph2, hydro_ph7]
RES_PROPERTIES = {
    'A': [0.12, 0.00, 0.00, 0.00, 0.35, 0.62, 0.62],
    'C': [0.28, 0.00, 0.00, 0.52, 0.28, 0.38, 0.38],
    'D': [0.32, 0.95, 0.00, 0.00, 0.15, 0.15, 0.15],
    'E': [0.38, 0.88, 0.00, 0.00, 0.18, 0.18, 0.18],
    'F': [0.48, 0.00, 0.00, 0.00, 0.42, 0.88, 0.88],
    'G': [0.08, 0.00, 0.00, 0.00, 0.48, 0.48, 0.48],
    'H': [0.42, 0.00, 0.72, 0.00, 0.52, 0.45, 0.45],
    'I': [0.42, 0.00, 0.00, 0.00, 0.38, 0.92, 0.92],
    'K': [0.48, 0.00, 0.85, 0.00, 0.72, 0.28, 0.28],
    'L': [0.42, 0.00, 0.00, 0.00, 0.40, 0.90, 0.90],
    'M': [0.45, 0.00, 0.00, 0.00, 0.42, 0.75, 0.75],
    'N': [0.32, 0.00, 0.00, 0.00, 0.42, 0.35, 0.35],
    'P': [0.30, 0.00, 0.00, 0.00, 0.45, 0.55, 0.55],
    'Q': [0.38, 0.00, 0.00, 0.00, 0.42, 0.42, 0.42],
    'R': [0.52, 0.00, 0.92, 0.00, 0.78, 0.32, 0.32],
    'S': [0.22, 0.00, 0.00, 0.00, 0.42, 0.45, 0.45],
    'T': [0.30, 0.00, 0.00, 0.00, 0.42, 0.52, 0.52],
    'V': [0.35, 0.00, 0.00, 0.00, 0.40, 0.85, 0.85],
    'W': [0.62, 0.00, 0.00, 0.00, 0.48, 0.95, 0.95],
    'Y': [0.52, 0.00, 0.00, 0.68, 0.38, 0.72, 0.72],
    # Unknown/ambiguous residue fallback
    'X': [0.35, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
}


def residue_features(aa: str) -> np.ndarray:
    """
    Generate 12-dimensional physicochemical feature vector for an amino acid.
    
    Feature breakdown:
    - [0:5] : Class membership flags (aliphatic/aromatic/polar/acidic/basic)
    - [5:12]: Normalized properties (weight, pKa, pKb, pKx, pI, hydrophobicity@pH2/7)
    
    Total: 5 + 7 = 12
    """
    aa = aa.upper()
    
    # 5-dim: Class membership (mutually exclusive)
    class_flags = [
        1.0 if aa in ALIPHATIC else 0.0,
        1.0 if aa in AROMATIC else 0.0,
        1.0 if aa in POLAR_NEUTRAL else 0.0,
        1.0 if aa in ACIDIC else 0.0,
        1.0 if aa in BASIC else 0.0,
    ]
    
    # 7-dim: Physicochemical properties (fallback to 'X' for unknown)
    props = RES_PROPERTIES.get(aa, RES_PROPERTIES['X'])
    
    return np.array(class_flags + props, dtype=np.float32)


# ============================================================================
# Encoding Helpers for SMILES/Sequences
# ============================================================================

def encode_smiles(smiles: str, max_len: int = 100, char_set: dict = None) -> np.ndarray:
    """
    Encode SMILES string to fixed-length integer array.
    
    Args:
        smiles: Input SMILES string
        max_len: Padding/truncation length (default: 100)
        char_set: Character-to-index mapping (uses default if None)
    
    Returns:
        np.ndarray of shape (max_len,) with integer indices
    """
    if char_set is None:
        char_set = {
            "(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
            "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15,
            "R": 16, "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22,
            "f": 23, "h": 24, "l": 25, "n": 26, "r": 27, "t": 28, "#": 29,
            "%": 30, ")": 31, "+": 32, "-": 33, "/": 34, "1": 35, "3": 36,
            "5": 37, "7": 38, "9": 39, "=": 40, "A": 41, "C": 42, "E": 43,
            "G": 44, "I": 45, "K": 46, "M": 47, "O": 48, "S": 49, "U": 50,
            "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56, "e": 57,
            "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64,
        }
    
    encoded = np.zeros(max_len, dtype=np.int64)
    for idx, ch in enumerate(smiles[:max_len]):
        encoded[idx] = char_set.get(ch, 0)  # 0 = unknown/padding
    return encoded


def encode_sequence(sequence: str, max_len: int = 500, aa_to_id: dict = None) -> np.ndarray:
    """
    Encode protein sequence to fixed-length integer array.
    
    Args:
        sequence: Amino acid sequence (uppercase)
        max_len: Padding/truncation length (default: 500)
        aa_to_id: Amino acid-to-index mapping (uses default if None)
    
    Returns:
        np.ndarray of shape (max_len,) with integer indices
    """
    if aa_to_id is None:
        aa_to_id = {
            "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8,
            "K": 9, "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
            "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20, "X": 21,
        }
    
    encoded = np.zeros(max_len, dtype=np.int64)
    for idx, aa in enumerate(sequence[:max_len].upper()):
        encoded[idx] = aa_to_id.get(aa, 21)  # 21 = unknown
    return encoded