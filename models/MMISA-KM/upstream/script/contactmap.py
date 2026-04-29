# models/mmisa_km/upstream/script/contactmap.py
"""
Contact map generation for protein graphs in MMISA-KM.

Supports:
- Sequential adjacency fallback
- Optional integration with structure prediction tools (ESMFold/AlphaFold2)

Contact definition: Cβ atoms (Cα for Gly) within 8.0 Å threshold.
"""

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

# Default contact distance threshold (Angstroms)
DEFAULT_CONTACT_THRESHOLD = 8.0


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


def generate_contact_map(
    sequence: str,
    structure_source: str = "sequential",
    threshold: float = DEFAULT_CONTACT_THRESHOLD,
) -> np.ndarray:
    """
    Generate contact map for a sequence.

    Args:
        sequence: Protein sequence (uppercase, standard AAs)
        structure_source: One of:
            - "sequential": Use sequential adjacency (fast, fallback)
            - "esmfold": Use ESMFold for structure prediction (slow, accurate)
            - "alphafold": Use AlphaFold2 (slowest, most accurate)
            - "template": Use template-based modeling
        threshold: Contact distance threshold in Angstroms

    Returns:
        Binary contact map array [n_residues, n_residues]
    """
    n_residues = len(sequence)

    if structure_source == "sequential":
        logger.debug(f"Using sequential contacts for sequence (len={n_residues})")
        return compute_sequential_contacts(n_residues)

    if structure_source in ("esmfold", "alphafold", "template"):
        logger.warning(
            f"Structure-based contact map requested ({structure_source}) but not configured. "
            f"Falling back to sequential contacts."
        )
        return compute_sequential_contacts(n_residues)

    logger.warning(f"Unknown structure_source '{structure_source}', using sequential")
    return compute_sequential_contacts(n_residues)
