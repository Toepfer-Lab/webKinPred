from __future__ import annotations

from pathlib import Path

from webKinPred.similarity_dataset_registry import SIMILARITY_DATASET_REGISTRY


SERVER_LIMIT = 10000
DEFAULT_ALLOWED_FRONTEND_IPS = ["127.0.0.1", "localhost"]


_DATA_PATH_REL = {
    "CatPred": "models/CatPred",
    "CatPred_production_checkpoints": "models/CatPred/data/pretrained/production",
    "DLKcat": "models/DLKcat/DeeplearningApproach/Data",
    "DLKcat_Results": "models/DLKcat/DeeplearningApproach/Results",
    "EITLEM": "models/EITLEM",
    "TurNup": "models/TurNup/data",
    "UniKP": "models/UniKP-main",
    "CataPro": "models/CataPro",
    "KinForm": "models/KinForm/results",
    "media": "media",
    "tools": "tools",
}


_PREDICTION_SCRIPT_REL = {
    "CatPred": "models/CatPred/catpred/integration/webkinpred_adapter.py",
    "DLKcat": "models/DLKcat/DeeplearningApproach/Code/example/prediction_for_input.py",
    "EITLEM": "models/EITLEM/Code/eitlem_prediction_script_batch.py",
    "TurNup": "models/TurNup/code/kcat_prediction_batch.py",
    "UniKP": "models/UniKP-main/run_unikp_batch.py",
    "CataPro": "models/CataPro/inference/custom_predict.py",
    "KinForm": "models/KinForm/code/main.py",
}


def _join(base_path: str | Path, rel_path: str) -> str:
    return str((Path(base_path) / rel_path).resolve())


def build_data_paths(base_path: str | Path) -> dict[str, str]:
    return {key: _join(base_path, rel) for key, rel in _DATA_PATH_REL.items()}


def build_prediction_scripts(base_path: str | Path) -> dict[str, str]:
    return {key: _join(base_path, rel) for key, rel in _PREDICTION_SCRIPT_REL.items()}


def build_similarity_datasets(fastas_dir: str | Path) -> dict[str, dict]:
    fastas_dir = str(Path(fastas_dir).resolve())
    return {
        label: {
            "label": label,
            "fasta": f"{fastas_dir}/{meta['fasta_filename']}",
            "target_db": f"{fastas_dir}/dbs/{meta['db_name']}",
            "method_keys": list(meta.get("method_keys", [])),
        }
        for label, meta in SIMILARITY_DATASET_REGISTRY.items()
    }


def build_experimental_paths(media_dir: str | Path) -> tuple[str, str]:
    media_dir = Path(media_dir).resolve()
    km_csv = str(media_dir / "experimental" / "km_experimental.csv")
    kcat_csv = str(media_dir / "experimental" / "kcat_experimental.csv")
    return km_csv, kcat_csv
