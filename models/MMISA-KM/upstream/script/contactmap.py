# models/mmisa_km/upstream/script/contactmap.py
"""
Contact map generation for protein graphs in MMISA-KM.

Supports:
- Precomputed contact maps (.npy format)
- Sequential adjacency fallback
- Optional integration with structure prediction tools (ESMFold/AlphaFold2)

Contact definition: Cβ atoms (Cα for Gly) within 8.0 Å threshold.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Default contact distance threshold (Angstroms)
DEFAULT_CONTACT_THRESHOLD = 8.0

# Cache directory environment variable
CACHE_DIR_ENV = "MMISA_KM_CONTACT_MAP_CACHE"


def compute_sequential_contacts(n_residues: int, window: int = 3) -> np.ndarray:
    """
    Generate sequential contact map as fallback.
    
    Args:
        n_residues: Number of residues in sequence
        window: Include contacts within ±window positions
    
    Returns:
        Binary adjacency matrix [n_residues, n_residues]
    """
    contacts = np.zeros((n_residues, n_residues), dtype=np.float32)
    for i in range(n_residues):
        for j in range(max(0, i - window), min(n_residues, i + window + 1)):
            contacts[i, j] = 1.0
    return contacts


def compute_contact_map_from_coords(
    coords: np.ndarray,
    threshold: float = DEFAULT_CONTACT_THRESHOLD,
    use_cbeta: bool = True
) -> np.ndarray:
    """
    Compute contact map from 3D coordinates.
    
    Args:
        coords: Array of shape [n_residues, 3] with Cα or Cβ coordinates
        threshold: Distance threshold in Angstroms (default: 8.0)
        use_cbeta: If True, expects Cβ coordinates; else uses Cα
    
    Returns:
        Binary adjacency matrix [n_residues, n_residues]
    """
    n = len(coords)
    contacts = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:
                contacts[i, j] = 1.0
                contacts[j, i] = 1.0  # Symmetric
    
    # Add self-contacts (important for GNN self-loops)
    np.fill_diagonal(contacts, 1.0)
    return contacts


def get_contact_map_cache_path(
    sequence: str,
    cache_dir: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """
    Generate deterministic cache path for a protein sequence.
    
    Args:
        sequence: Amino acid sequence
        cache_dir: Base cache directory (uses env var if None)
    
    Returns:
        Path to cache file, or None if cache_dir not configured
    """
    if cache_dir is None:
        cache_dir = os.getenv(CACHE_DIR_ENV)
        if not cache_dir:
            return None
        cache_dir = Path(cache_dir)
    else:
        cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        return None
    
    # Hash sequence for filename (first 16 chars of SHA256)
    seq_hash = hashlib.sha256(sequence.encode()).hexdigest()[:16]
    return cache_dir / f"{seq_hash}.npy"


def save_contact_map_to_cache(
    contacts: np.ndarray,
    sequence: str,
    cache_dir: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """Save contact map to cache and return path."""
    cache_path = get_contact_map_cache_path(sequence, cache_dir)
    if cache_path is None:
        return None
    
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, contacts)
        logger.debug(f"Saved contact map to {cache_path}")
        return cache_path
    except Exception as e:
        logger.warning(f"Failed to save contact map cache: {e}")
        return None


def load_contact_map(
    sequence: str,
    cache_dir: Optional[Union[str, Path]] = None,
    precomputed_dir: Optional[Union[str, Path]] = None
) -> Optional[np.ndarray]:
    """
    Load precomputed contact map from cache or precomputed directory.
    
    Search order:
    1. Runtime cache directory
    2. Precomputed directory (if provided)
    
    Args:
        sequence: Protein sequence
        cache_dir: Runtime cache directory path
        precomputed_dir: Directory with curated contact maps
    
    Returns:
        Contact map array [n, n] or None if not found
    """
    # Try runtime cache first
    cache_path = get_contact_map_cache_path(sequence, cache_dir)
    if cache_path and cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cached contact map {cache_path}: {e}")
    
    # Try precomputed directory
    if precomputed_dir:
        precomputed_dir = Path(precomputed_dir)
        seq_hash = hashlib.sha256(sequence.encode()).hexdigest()[:16]
        precomputed_path = precomputed_dir / f"{seq_hash}.npy"
        
        if precomputed_path.exists():
            try:
                return np.load(precomputed_path)
            except Exception as e:
                logger.warning(f"Failed to load precomputed contact map {precomputed_path}: {e}")
    
    return None


def generate_contact_map(
    sequence: str,
    structure_source: str = "sequential",
    threshold: float = DEFAULT_CONTACT_THRESHOLD,
    cache: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    precomputed_dir: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Main entry point: generate or retrieve contact map for a sequence.
    
    Args:
        sequence: Protein sequence (uppercase, standard AAs)
        structure_source: One of:
            - "sequential": Use sequential adjacency (fast, fallback)
            - "esmfold": Use ESMFold for structure prediction (slow, accurate)
            - "alphafold": Use AlphaFold2 (slowest, most accurate)
            - "template": Use template-based modeling
        threshold: Contact distance threshold in Angstroms
        cache: Whether to save/load from cache
        cache_dir: Directory for runtime cache
        precomputed_dir: Directory with precomputed maps
    
    Returns:
        Binary contact map array [n_residues, n_residues]
    """
    n_residues = len(sequence)
    
    # Try loading from cache/precomputed first
    if cache:
        contacts = load_contact_map(sequence, cache_dir, precomputed_dir)
        if contacts is not None:
            logger.debug(f"Loaded contact map from cache for sequence (len={n_residues})")
            return contacts
    
    # Generate based on source
    if structure_source == "sequential":
        logger.debug(f"Using sequential contacts for sequence (len={n_residues})")
        contacts = compute_sequential_contacts(n_residues)
    
    elif structure_source in ("esmfold", "alphafold", "template"):
        # Placeholder for structure prediction integration
        # In production, this would call external APIs or local tools
        logger.warning(
            f"Structure-based contact map requested ({structure_source}) but not configured. "
            f"Falling back to sequential contacts."
        )
        contacts = compute_sequential_contacts(n_residues)
    
    else:
        logger.warning(f"Unknown structure_source '{structure_source}', using sequential")
        contacts = compute_sequential_contacts(n_residues)
    
    # Save to cache if requested
    if cache and contacts is not None:
        save_contact_map_to_cache(contacts, sequence, cache_dir)
    
    return contacts