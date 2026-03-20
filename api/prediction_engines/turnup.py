# api/prediction_engines/turnup.py
#
# Prediction engine for the TurNup model.
#
# Wraps the TurNup prediction script in a subprocess call.  Handles
# multi-substrate/multi-product molecule validation, InChI conversion,
# progress reporting, and user-friendly error messages.

import os
import subprocess

import numpy as np
import pandas as pd
from rdkit import Chem

from api.methods.base import PredictionError
from api.models import Job
from api.prediction_engines.subprocess_runner import run_prediction_subprocess
from api.prediction_engines.runtime_paths import (
    DATA_PATHS,
    PREDICTION_SCRIPTS,
    PYTHON_PATHS,
)
from api.utils.convert_to_mol import convert_to_mol
from webKinPred.settings import MEDIA_ROOT

_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def turnup_predictions(
    sequences: list[str],
    public_id: str,
    substrates: list[str],
    products: list[str],
    **kwargs,
) -> tuple[list, dict[int, str]]:
    """
    Run the TurNup model on the given protein sequences, substrates, and
    products.

    Parameters
    ----------
    sequences : list[str]
        Pre-filtered protein sequences.
    public_id : str
        Job identifier for progress tracking.
    substrates : list[str]
        Semicolon-separated SMILES/InChI strings per row
        (e.g. ``"CC(=O)O;OCC"``).
    products : list[str]
        Semicolon-separated SMILES/InChI strings per row.

    Returns
    -------
    predictions : list
        Predicted kcat values (float) or None for invalid rows.
    invalid_reasons : dict[int, str]
        Maps row index to a human-readable reason for rows that could not
        be processed.

    Raises
    ------
    PredictionError
        On subprocess failure or any unrecoverable error.
    """
    print("Running TurNup model...")

    job = Job.objects.get(public_id=public_id)
    job.molecules_processed = 0
    job.invalid_rows = 0
    job.predictions_made = 0
    job.total_molecules = len(sequences)
    job.save(update_fields=[
        "molecules_processed", "invalid_rows", "predictions_made", "total_molecules"
    ])

    python_path = PYTHON_PATHS.get("TurNup", "")
    prediction_script = PREDICTION_SCRIPTS.get("TurNup", "")
    job_dir = os.path.join(MEDIA_ROOT, "jobs", str(public_id))
    input_file = os.path.join(job_dir, f"input_{public_id}.csv")
    output_file = os.path.join(job_dir, f"output_{public_id}.csv")

    env = os.environ.copy()
    if DATA_PATHS.get("media"):
        env["TURNUP_MEDIA_PATH"] = DATA_PATHS["media"]
    if DATA_PATHS.get("tools"):
        env["TURNUP_TOOLS_PATH"] = DATA_PATHS["tools"]

    valid_indices: list[int] = []
    invalid_reasons: dict[int, str] = {}
    valid_sub_inchis: list[str] = []
    valid_prod_inchis: list[str] = []
    valid_sequences: list[str] = []
    predictions: list = [None] * len(sequences)

    # ── Validate inputs reaction by reaction ──────────────────────────────────
    for idx, (seq, sub_str, prod_str) in enumerate(zip(sequences, substrates, products)):
        job.molecules_processed += 1

        sub_tokens = [s.strip() for s in str(sub_str).split(";") if s.strip()]
        prod_tokens = [p.strip() for p in str(prod_str).split(";") if p.strip()]
        sub_mols = [convert_to_mol(s) for s in sub_tokens]
        prod_mols = [convert_to_mol(p) for p in prod_tokens]
        seq_valid = all(c in _AMINO_ACIDS for c in seq)

        if not seq_valid:
            reason = "Invalid protein sequence (unsupported amino acid characters)"
        elif None in sub_mols:
            reason = "Invalid substrate (not a valid SMILES or InChI)"
        elif None in prod_mols:
            reason = "Invalid product (not a valid SMILES or InChI)"
        else:
            reason = ""

        if reason:
            print(f"  Row {idx + 1}: {reason}")
            invalid_reasons[idx] = reason
            job.invalid_rows += 1
        else:
            sub_inchi = ";".join(Chem.MolToInchi(mol) for mol in sub_mols)
            prod_inchi = ";".join(Chem.MolToInchi(mol) for mol in prod_mols)
            valid_sub_inchis.append(sub_inchi)
            valid_prod_inchis.append(prod_inchi)
            valid_sequences.append(seq)
            valid_indices.append(idx)

        job.save(update_fields=["molecules_processed", "invalid_rows"])

    job.total_predictions = len(valid_indices)
    job.save(update_fields=["total_predictions"])

    if not valid_indices:
        return predictions, invalid_reasons

    # ── Write CSV input file ──────────────────────────────────────────────────
    try:
        df_input = pd.DataFrame({
            "Substrates": valid_sub_inchis,
            "Products": valid_prod_inchis,
            "Protein Sequence": valid_sequences,
        })
        df_input.to_csv(input_file, index=False)
    except OSError as e:
        raise PredictionError(
            "TurNup could not write its input file. "
            "Please contact support if this persists."
        ) from e

    # ── Run prediction subprocess ─────────────────────────────────────────────
    try:
        run_prediction_subprocess(
            command=[python_path, prediction_script, input_file, output_file],
            job=job,
            env=env,
            label="TurNup",
        )
    except subprocess.CalledProcessError as e:
        _cleanup(input_file, output_file)
        if e.returncode in (-9, 137):
            raise PredictionError(
                "TurNup ran out of memory. "
                "Try reducing the number of rows or the sequence lengths."
            ) from e
        raise PredictionError(
            "TurNup encountered an internal error and could not complete. "
            "Please verify your input and try again."
        ) from e
    except Exception as e:
        _cleanup(input_file, output_file)
        if isinstance(e, PredictionError):
            raise
        raise PredictionError(
            "TurNup encountered an unexpected error. "
            "Please verify your input and try again."
        ) from e

    # ── Read output CSV ───────────────────────────────────────────────────────
    try:
        df_output = pd.read_csv(output_file)
        raw_values = df_output["kcat [s^(-1)]"].tolist()
    except Exception as e:
        _cleanup(input_file, output_file)
        raise PredictionError(
            "TurNup completed but its output file could not be read. "
            "Please contact support if this persists."
        ) from e

    for local_idx, pred in enumerate(raw_values):
        global_idx = valid_indices[local_idx]
        predictions[global_idx] = None if pred in ("None", "", np.nan, "nan") else pred

    _cleanup(input_file, output_file)
    return predictions, invalid_reasons


def _cleanup(*paths: str) -> None:
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
