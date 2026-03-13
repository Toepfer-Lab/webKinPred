from webKinPred.config_base import (
    DEFAULT_ALLOWED_FRONTEND_IPS,
    SERVER_LIMIT,
    build_data_paths,
    build_experimental_paths,
    build_prediction_scripts,
    build_similarity_datasets,
)


BASE_PATH = "/app"
DATA_PATH = BASE_PATH
FASTAS_DIR = f"{BASE_PATH}/fastas"

PYTHON_PATHS = {
    "DLKcat": "/opt/conda/envs/dlkcat_env/bin/python",
    "EITLEM": "/opt/conda/envs/eitlem_env/bin/python",
    "TurNup": "/opt/conda/envs/turnup_env/bin/python",
    "UniKP": "/opt/conda/envs/unikp/bin/python",
    "KinForm": "/opt/conda/envs/kinform_env/bin/python",
    "esm2": "/opt/conda/envs/esm/bin/python",
    "esmc": "/opt/conda/envs/esmc/bin/python",
    "t5": "/opt/conda/envs/prot_t5/bin/python",
    "pseq2sites": "/opt/conda/envs/pseq2sites/bin/python",
}

DATA_PATHS = build_data_paths(BASE_PATH)
PREDICTION_SCRIPTS = build_prediction_scripts(BASE_PATH)

SIMILARITY_DATASETS = build_similarity_datasets(FASTAS_DIR)
TARGET_DBS = {label: item["target_db"] for label, item in SIMILARITY_DATASETS.items()}

CONDA_PATH = None
DEBUG = True
ALLOWED_FRONTEND_IPS = [*DEFAULT_ALLOWED_FRONTEND_IPS, "frontend", "backend"]

KM_CSV, KCAT_CSV = build_experimental_paths(f"{BASE_PATH}/media")
