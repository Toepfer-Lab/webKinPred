# api/prediction_engines/unikp.py
#
# Prediction engine for the UniKP model.
#
# Wraps the UniKP prediction script in a subprocess call.  Handles molecule
# validation, progress reporting, and user-friendly error messages.

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


def unikp_predictions(
    sequences: list[str],
    public_id: str,
    substrates: list[str],
    kinetics_type: str = "KCAT",
    **kwargs,
) -> tuple[list, dict[int, str]]:
    """
    Run the UniKP model on the given protein sequences and substrates.

    Parameters
    ----------
    sequences : list[str]
        Pre-filtered protein sequences.
    public_id : str
        Job identifier for progress tracking.
    substrates : list[str]
        Substrate SMILES or InChI strings, one per sequence.
    kinetics_type : str
        ``"KCAT"`` (default) or ``"KM"``.

    Returns
    -------
    predictions : list
        Predicted values (float) or None for invalid rows.
    invalid_reasons : dict[int, str]
        Maps row index to a human-readable reason for rows that could not
        be processed.

    Raises
    ------
    PredictionError
        On subprocess failure or any unrecoverable error.
    """
    print(f"Running UniKP model (kinetics_type={kinetics_type})...")

    job = Job.objects.get(public_id=public_id)
    job.molecules_processed = 0
    job.invalid_rows = 0
    job.predictions_made = 0
    job.total_molecules = len(sequences)
    job.save(update_fields=[
        "molecules_processed", "invalid_rows", "predictions_made", "total_molecules"
    ])

    python_path = PYTHON_PATHS.get("UniKP", "")
    prediction_script = PREDICTION_SCRIPTS.get("UniKP", "")
    job_dir = os.path.join(MEDIA_ROOT, "jobs", str(public_id))
    input_file = os.path.join(job_dir, f"input_{public_id}.csv")
    output_file = os.path.join(job_dir, f"output_{public_id}.csv")

    env = os.environ.copy()
    if DATA_PATHS.get("media"):
        env["UNIKP_MEDIA_PATH"] = DATA_PATHS["media"]
    if DATA_PATHS.get("tools"):
        env["UNIKP_TOOLS_PATH"] = DATA_PATHS["tools"]

    valid_indices: list[int] = []
    invalid_reasons: dict[int, str] = {}
    valid_smiles: list[str] = []
    valid_sequences: list[str] = []
    predictions: list = [None] * len(sequences)

    # ── Validate inputs molecule by molecule ──────────────────────────────────
    for idx, (seq, substrate) in enumerate(zip(sequences, substrates)):
        job.molecules_processed += 1
        seq_valid = all(c in _AMINO_ACIDS for c in seq)
        mol = convert_to_mol(substrate)

        if mol and seq_valid:
            valid_smiles.append(Chem.MolToSmiles(mol))
            valid_sequences.append(seq)
            valid_indices.append(idx)
        else:
            reason = (
                "Invalid protein sequence (unsupported amino acid characters)"
                if not seq_valid
                else "Invalid substrate (not a valid SMILES or InChI)"
            )
            print(f"  Row {idx + 1}: {reason}")
            invalid_reasons[idx] = reason
            job.invalid_rows += 1

        job.save(update_fields=["molecules_processed", "invalid_rows"])

    job.total_predictions = len(valid_indices)
    job.save(update_fields=["total_predictions"])

    if not valid_indices:
        return predictions, invalid_reasons

    # ── Write CSV input file ──────────────────────────────────────────────────
    try:
        df_input = pd.DataFrame({
            "Substrate SMILES": valid_smiles,
            "Protein Sequence": valid_sequences,
        })
        df_input.to_csv(input_file, index=False)
    except OSError as e:
        raise PredictionError(
            "UniKP could not write its input file. "
            "Please contact support if this persists."
        ) from e

    # ── Run prediction subprocess ─────────────────────────────────────────────
    try:
        run_prediction_subprocess(
            command=[python_path, prediction_script, input_file, output_file, kinetics_type],
            job=job,
            env=env,
            label="UniKP",
        )
    except subprocess.CalledProcessError as e:
        _cleanup(input_file, output_file)
        if e.returncode in (-9, 137):
            raise PredictionError(
                "UniKP ran out of memory. "
                "Try reducing the number of rows or the sequence lengths."
            ) from e
        raise PredictionError(
            "UniKP encountered an internal error and could not complete. "
            "Please verify your input and try again."
        ) from e
    except Exception as e:
        _cleanup(input_file, output_file)
        if isinstance(e, PredictionError):
            raise
        raise PredictionError(
            "UniKP encountered an unexpected error. "
            "Please verify your input and try again."
        ) from e

    # ── Read output CSV ───────────────────────────────────────────────────────
    try:
        df_output = pd.read_csv(output_file)
        raw_values = df_output["Predicted Value"].tolist()
    except Exception as e:
        _cleanup(input_file, output_file)
        raise PredictionError(
            "UniKP completed but its output file could not be read. "
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
