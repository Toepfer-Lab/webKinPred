# api/methods/catapro.py
#
# Method descriptor for CataPro.

from api.methods.base import MethodDescriptor


def _catapro_predictions_lazy(*args, **kwargs):
    # Import prediction engine only when a job is actually executed.
    from api.prediction_engines.catapro import catapro_predictions

    return catapro_predictions(*args, **kwargs)

descriptor = MethodDescriptor(
    key="CataPro",
    display_name="CataPro",
    description=(
        "Predicting kcat, Km, and kcat/Km based on protein language models, "
        "small-molecule language models, and molecular fingerprint features"
    ),
    authors=(
        "Zechen Wang, Dongqi Xie, Dong Wu, Xiaozhou Luo, Sheng Wang,"
        "Yangyang Li, Yanmei Yang, Weifeng Li, Liangzhen Zheng"
    ),
    publication_title=(
        "Robust enzyme discovery and engineering with deep learning using CataPro "
    ),
    citation_url="https://www.nature.com/articles/s41467-025-58038-4",
    repo_url="https://github.com/zchwang/CataPro",
    more_info="",

    # ── Capabilities ──────────────────────────────────────────────────────────
    supports=["kcat", "Km", "kcat/Km"],
    input_format="single",
    output_cols={
        "kcat": "kcat (1/s)",
        "Km": "Km (mM)",
        "kcat/Km": "kcat/Km (1/(s*mM))"
    },

    # ── Sequence length ───────────────────────────────────────────────────────
    max_seq_len=1000,

    # ── Input mapping ─────────────────────────────────────────────────────────
    col_to_kwarg={"Substrate": "substrates"},

    # ── Per-target extra kwargs ───────────────────────────────────────────────
    # catapro_predictions() accepts a `kinetics_type` argument to switch between
    # kCAT, KM and KCAT/KM prediction modes.
    target_kwargs={
        "kcat": {"kinetics_type": "KCAT"},
        "Km":   {"kinetics_type": "KM"},
    },

    pred_func=_catapro_predictions_lazy,

    # ── Embeddings ────────────────────────────────────────────────────────────
    # CataPro uses ProtT5 embeddings to represent proteins, 
    # and MolT5 along with MACCS key fingerprints to encode substrate information.
    embeddings_used=["prot_t5"],
)
