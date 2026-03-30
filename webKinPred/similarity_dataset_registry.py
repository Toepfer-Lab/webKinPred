"""
Central registry for MMseqs similarity datasets.

Add new datasets here once, then both local and Docker configs will pick them up.
"""

from __future__ import annotations


SIMILARITY_DATASET_REGISTRY: dict[str, dict[str, str]] = {
    # label -> metadata
    "DLKcat/UniKP": {
        "fasta_filename": "dlkcat_sequences.fasta",
        "db_name": "targetdb_dlkcat",
    },
    "EITLEM/KinForm": {
        "fasta_filename": "EITLEM_sequences.fasta",
        "db_name": "targetdb_EITLEM",
    },
    "TurNup": {
        "fasta_filename": "turnup_sequences.fasta",
        "db_name": "targetdb_turnup",
    },
    "CataPro": {
        "fasta_filename": "catapro_sequences.fasta",
        "db_name": "targetdb_catapro",
    },
    "CatPred": {
        "fasta_filename": "catpred_kcat_sequences.fasta",
        "db_name": "targetdb_catpred_kcat",
    },
}
