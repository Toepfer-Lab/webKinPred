import os
import sys
import pandas as pd
import numpy as np
import argparse
import pickle
import subprocess
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# gen_features.py
#
# Produces the Pseq2Sites input pickle from a TSV of protein sequences by
# delegating T5 embedding to the prot_t5 conda env (t5_embeddings.py).
# No T5 model is loaded here — the pseq2sites env only runs the Pseq2Sites
# neural-network predictor (test.py).
#
# Output pickle format (unchanged from original):
#   labels=False:  (IDs, seqs, feats)
#   labels=True:   (IDs, seqs, binding_sites, feats)
# where feats is a list of float32 numpy arrays of shape [L, 1024].
# ---------------------------------------------------------------------------

_T5_SCRIPT = str(
    Path(__file__).resolve().parents[2] / "protein_embeddings" / "t5_embeddings.py"
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _embed_via_prot_t5(chain_dict: dict, residue_dir: Path) -> None:
    """Call t5_embeddings.py in the prot_t5 env to generate residue embeddings
    for any chain IDs not already on disk."""
    missing = [cid for cid in chain_dict if not (residue_dir / f"{cid}.npy").exists()]
    if not missing:
        return

    t5_bin = os.environ.get("KINFORM_T5_PATH", "/opt/conda/envs/prot_t5/bin/python")

    with tempfile.TemporaryDirectory() as tmpdir:
        seq_file = Path(tmpdir) / "seq_ids.txt"
        id_to_seq_file = Path(tmpdir) / "id_to_seq.pkl"

        with open(seq_file, "w") as f:
            for cid in missing:
                f.write(f"{cid}\n")
        with open(id_to_seq_file, "wb") as f:
            pickle.dump(chain_dict, f)

        cmd = [
            t5_bin, _T5_SCRIPT,
            "--seq_file", str(seq_file),
            "--id_to_seq_file", str(id_to_seq_file),
            "--setting", "residue",
            "--batch_size", "1",
        ]
        subprocess.run(cmd, check=True, env=os.environ.copy())


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pseq2Sites input features via prot_t5 env embeddings."
    )
    parser.add_argument("--input", "-i", required=True, type=str,
                        help="TSV file with protein IDs and sequences.")
    parser.add_argument("--output", "-o", required=True, type=str,
                        help="Output pickle path.")
    parser.add_argument("--labels", "-l", required=True, type=str2bool,
                        help="True if binding-site labels are present in input.")
    args = parser.parse_args()

    input_abspath = os.path.abspath(args.input)
    if not os.path.isfile(input_abspath):
        raise IOError(f"Input file not found: {input_abspath}")
    output_abspath = os.path.abspath(args.output)
    if not os.path.isdir(os.path.abspath(os.path.dirname(args.output))):
        raise IOError(f"Output directory does not exist: {os.path.dirname(output_abspath)}")

    media_path = os.environ.get("KINFORM_MEDIA_PATH")
    if not media_path:
        raise RuntimeError("KINFORM_MEDIA_PATH environment variable is not set.")
    residue_dir = Path(media_path) / "sequence_info" / "prot_t5_last" / "residue_vecs"

    print("1. Load data ...")
    prots_df = pd.read_csv(input_abspath, sep="\t")
    if args.labels:
        IDs = prots_df.iloc[:, 0].values
        seqs = prots_df.iloc[:, 1].values
        binding_sites = prots_df.iloc[:, 2].values
    else:
        IDs = prots_df.iloc[:, 0].values
        seqs = prots_df.iloc[:, 1].values

    # Build per-chain mapping (handles multi-chain sequences joined with ",")
    # Single-chain proteins use the protein ID directly so embeddings are
    # cached under the protein ID (matches the rest of the pipeline).
    chain_dict = {}       # chain_id  -> chain_sequence
    id_to_chains = {}     # protein_id -> [chain_id, ...]

    for prot_id, seq in zip(IDs, seqs):
        chains = str(seq).split(",")
        chain_ids = []
        for ci, chain_seq in enumerate(chains):
            cid = prot_id if len(chains) == 1 else f"{prot_id}__c{ci}"
            chain_dict[cid] = chain_seq
            chain_ids.append(cid)
        id_to_chains[prot_id] = chain_ids

    print("2. Generate T5 residue embeddings (via prot_t5 env) ...")
    _embed_via_prot_t5(chain_dict, residue_dir)

    print("3. Assemble features ...")
    prots_feat_list = []
    for prot_id in IDs:
        chain_ids = id_to_chains[prot_id]
        chain_feats = []
        ok = True
        for cid in chain_ids:
            npy_path = residue_dir / f"{cid}.npy"
            if npy_path.exists():
                chain_feats.append(np.load(npy_path))
                # Do NOT unlink — t5_embeddings.py will reuse these residue files
                # to derive mean/weighted embeddings without reloading T5, then
                # delete them itself.
            else:
                ok = False
                break
        if ok and chain_feats:
            prots_feat_list.append(np.concatenate(chain_feats, axis=0).astype(np.float32))
        else:
            # Mirror original OOM-fallback: empty array signals downstream failure
            prots_feat_list.append(np.zeros((0, 1024), dtype=np.float32))

    with open(output_abspath, "wb") as f:
        if args.labels:
            pickle.dump((IDs, seqs, binding_sites, prots_feat_list), f)
        else:
            pickle.dump((IDs, seqs, prots_feat_list), f)

    print(f"Features saved to {output_abspath}")


if __name__ == "__main__":
    main()
