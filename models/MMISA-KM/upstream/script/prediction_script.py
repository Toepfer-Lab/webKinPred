# models/mmisa_km/upstream/script/prediction_script.py
"""
MMISA-KM batch prediction script for webKinPred integration.

Usage:
    python prediction_script.py --input input.json --output output.json

Input JSON format:
{
  "rows": [
    {"sequence": "MKTAY...", "substrates": "CC(=O)O"},
    ...
  ]
}

Output JSON format:
{
  "predictions": [12.5, null, 0.045, ...],
  "invalid_indices": [1, 4, ...]
}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

# Add upstream script directory to path for imports
_THIS_DIR = Path(__file__).resolve().parent
_UPSTREAM_SCRIPT_DIR = _THIS_DIR
if str(_UPSTREAM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_SCRIPT_DIR))

# Import model architecture and feature functions
from model import GNNNet  # noqa: E402
from data_process import (  # noqa: E402
    atom_features,
    residue_features,
    encode_smiles,
    encode_sequence,
    ELEMENT_LIST,
)
from contactmap import generate_contact_map, get_contact_map_cache_path  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants & Mappings
# ============================================================================

_VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")

_AA_TO_ID = {
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8,
    "K": 9, "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20, "X": 21,
}

_CHAR_SMI_SET = {
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

# Model configuration (must match training)
MODEL_CONFIG = {
    "embed_dim": 256,
    "n_output": 1,
    "num_features_pro": 33,  # 21 AA one-hot + 12 physicochemical
    "num_features_mol": 78,  # Full atom features
    "dropout": 0.2,
}

# Encoding limits
MAX_SMILES_LEN = 100
MAX_SEQ_LEN = 500

# ============================================================================
# Argument Parsing
# ============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMISA-KM webKinPred batch adapter")
    parser.add_argument("--input", required=True, help="Path to JSON input payload")
    parser.add_argument("--output", required=True, help="Path to JSON output payload")
    parser.add_argument(
        "--contact-map-source",
        default="sequential",
        choices=["sequential", "esmfold", "alphafold", "template"],
        help="Source for protein contact maps (default: sequential)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable contact map caching"
    )
    return parser.parse_args()

# ============================================================================
# Input Validation & Normalization
# ============================================================================

def _safe_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and extract rows from input payload."""
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("Input payload is malformed: 'rows' must be a list")
    
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Input payload is malformed: row {i} is not an object")
        out.append(row)
    return out


def _normalise_substrate(raw: Any) -> str | None:
    """
    Normalize substrate input to canonical SMILES.
    
    Supports:
    - Single SMILES string
    - List of SMILES (must contain exactly one valid entry)
    - InChI strings (converted to SMILES via RDKit)
    """
    if isinstance(raw, list):
        tokens = [str(item).strip() for item in raw if str(item).strip()]
    else:
        text = str(raw).strip()
        if not text:
            return None
        tokens = [tok.strip() for tok in text.split(";") if tok.strip()]

    if len(tokens) != 1:
        return None

    substrate = tokens[0]
    
    # Try parsing as SMILES first
    mol = Chem.MolFromSmiles(substrate)
    if mol is not None:
        return Chem.MolToSmiles(mol)  # Canonicalize
    
    # Fallback: try InChI
    mol = Chem.MolFromInchi(substrate)
    if mol is None:
        return None
    
    return Chem.MolToSmiles(mol)

# ============================================================================
# Graph Construction Functions (FIXED)
# ============================================================================

def _build_mol_graph(smiles: str) -> Data | None:
    """
    Build PyG Data object for molecule with full 78-dim atom features.
    
    Uses atom_features() from data_process.py to ensure training/inference parity.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Extract full 78-dim features for each atom
    x_rows: list[np.ndarray] = []
    for atom in mol.GetAtoms():
        x_rows.append(atom_features(atom))
    
    x = torch.tensor(np.stack(x_rows), dtype=torch.float32)
    
    # Edge construction: bidirectional bonds + self-loops
    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((a, b))
        edges.append((b, a))
    # Self-loops for message passing
    for i in range(mol.GetNumAtoms()):
        edges.append((i, i))
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def _build_pro_graph(
    sequence: str,
    contact_map_source: str = "sequential",
    cache: bool = True,
    cache_dir: Optional[Path] = None,
    precomputed_dir: Optional[Path] = None
) -> Data:
    """
    Build PyG Data object for protein with 33-dim residue features.
    
    Feature composition per residue:
    - [0:21] : One-hot amino acid encoding
    - [21:33]: Physicochemical properties (5 class flags + 7 normalized values)
    
    Edge construction:
    - Preferred: Contact map from structure (Cβ < 8.0 Å)
    - Fallback: Sequential adjacency (i ± 1, 2, 3) + self-loops
    """
    seq = sequence[:MAX_SEQ_LEN].upper()
    n_residues = len(seq)
    
    # Build node features: 21-dim one-hot + 12-dim properties = 33-dim
    features: list[np.ndarray] = []
    for aa in seq:
        # 21-dim one-hot encoding
        onehot = np.zeros(21, dtype=np.float32)
        aa_idx = _AA_TO_ID.get(aa, 21) - 1  # Map to 0-20, or 20 for unknown
        aa_idx = max(0, min(20, aa_idx))
        onehot[aa_idx] = 1.0
        
        # 12-dim physicochemical properties
        props = residue_features(aa)
        
        # Concatenate: 21 + 12 = 33
        features.append(np.concatenate([onehot, props]))
    
    # Handle empty sequence edge case
    if not features:
        features = [np.zeros(33, dtype=np.float32)]
        n_residues = 1
    
    x = torch.tensor(np.stack(features), dtype=torch.float32)
    
    # Edge construction via contact map
    contacts = generate_contact_map(
        sequence=seq,
        structure_source=contact_map_source,
        cache=cache,
        cache_dir=cache_dir,
        precomputed_dir=precomputed_dir,
    )
    
    # Convert contact matrix to edge list
    edges: list[tuple[int, int]] = []
    n = min(len(features), contacts.shape[0])  # Safety clamp
    for i in range(n):
        for j in range(n):
            if contacts[i, j] > 0.5:  # Threshold for binary contact
                edges.append((i, j))
    
    # Ensure self-loops exist (important for GNN)
    for i in range(n):
        if not any(e[0] == i and e[1] == i for e in edges):
            edges.append((i, i))
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# ============================================================================
# Model Loading
# ============================================================================

def _load_model(device: torch.device, weights_path: Optional[Path] = None) -> GNNNet:
    """Load pretrained MMISA-KM model."""
    if weights_path is None:
        weights_path = _UPSTREAM_SCRIPT_DIR / "trained_model.pt"
    
    if not weights_path.exists():
        raise RuntimeError(f"Missing MMISA-KM model weights at: {weights_path}")
    
    model = GNNNet(**MODEL_CONFIG)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model

# ============================================================================
# Batch Prediction Logic
# ============================================================================

def _predict_rows(
    rows: list[dict[str, Any]],
    contact_map_source: str = "sequential",
    cache: bool = True
) -> tuple[list[Any], list[int]]:
    """
    Process batch of input rows and generate predictions.
    
    Returns:
        predictions: List of float Km values (mM) or None for invalid rows
        invalid_indices: List of row indices that failed validation
    """
    predictions: list[Any] = [None] * len(rows)
    invalid_indices: list[int] = []

    # Pre-filter and prepare valid inputs
    valid_global_indices: list[int] = []
    smiles_encoded: list[np.ndarray] = []
    seq_encoded: list[np.ndarray] = []
    mol_graphs: list[Data] = []
    pro_graphs: list[Data] = []

    # Get cache directories from environment
    cache_dir = None
    precomputed_dir = None
    mmisa_root = os.getenv("MMISA_KM_ROOT")
    if mmisa_root:
        root = Path(mmisa_root)
        cache_dir = root / "contact_maps" / "cache"
        precomputed_dir = root / "contact_maps" / "precomputed"

    for idx, row in enumerate(rows):
        # Validate sequence
        sequence = str(row.get("sequence", "")).strip().upper()
        if not sequence or any(ch not in _VALID_AAS for ch in sequence):
            invalid_indices.append(idx)
            continue

        # Normalize substrate
        substrate = _normalise_substrate(row.get("substrates", row.get("substrate")))
        if not substrate:
            invalid_indices.append(idx)
            continue

        # Build molecule graph
        mol_graph = _build_mol_graph(substrate)
        if mol_graph is None:
            invalid_indices.append(idx)
            continue

        # All validations passed - queue for batch inference
        valid_global_indices.append(idx)
        smiles_encoded.append(encode_smiles(substrate, MAX_SMILES_LEN, _CHAR_SMI_SET))
        seq_encoded.append(encode_sequence(sequence, MAX_SEQ_LEN, _AA_TO_ID))
        mol_graphs.append(mol_graph)
        pro_graphs.append(
            _build_pro_graph(
                sequence,
                contact_map_source=contact_map_source,
                cache=cache,
                cache_dir=cache_dir,
                precomputed_dir=precomputed_dir,
            )
        )

    # Handle case where no valid rows remain
    if not valid_global_indices:
        return predictions, sorted(set(invalid_indices))

    # Setup device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running inference on {device}")
    model = _load_model(device)

    # Prepare batch tensors
    smile_tensor = torch.tensor(
        np.stack(smiles_encoded), dtype=torch.long, device=device
    )
    seq_tensor = torch.tensor(
        np.stack(seq_encoded), dtype=torch.long, device=device
    )
    mol_batch = Batch.from_data_list(mol_graphs).to(device)
    pro_batch = Batch.from_data_list(pro_graphs).to(device)

    # Forward pass
    with torch.no_grad():
        # Model output: [batch, 1] with log10(Km)
        raw_output = model(smile_tensor, seq_tensor, mol_batch, pro_batch)
        raw_values = raw_output.view(-1).detach().cpu().tolist()

    # Post-process: convert log10(Km) -> Km (mM)
    for local_idx, pred_log in enumerate(raw_values):
        global_idx = valid_global_indices[local_idx]
        
        # Validate prediction
        if pred_log is None or (isinstance(pred_log, float) and (math.isnan(pred_log) or math.isinf(pred_log))):
            invalid_indices.append(global_idx)
            predictions[global_idx] = None
            continue
        
        # Transform: log10(Km) -> Km
        km_value = float(10.0 ** float(pred_log))
        predictions[global_idx] = km_value

    return predictions, sorted(set(invalid_indices))

# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load input payload
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Validate and extract rows
    rows = _safe_rows(payload)
    logger.info(f"Processing {len(rows)} input rows")

    # Run predictions
    predictions, invalid_indices = _predict_rows(
        rows,
        contact_map_source=args.contact_map_source,
        cache=not args.no_cache,
    )

    # Prepare output
    output_data = {
        "predictions": predictions,
        "invalid_indices": invalid_indices,
        "metadata": {
            "processed": len(rows),
            "valid": len(rows) - len(invalid_indices),
            "invalid": len(invalid_indices),
        }
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(
        f"Complete: {len(rows) - len(invalid_indices)}/{len(rows)} valid predictions. "
        f"Output written to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())