# syntax=docker/dockerfile:1.4
# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — full worker image (celery only)
#
# Multi-stage build: BuildKit builds all env stages in PARALLEL, then the
# final stage assembles them. Cold build time is determined by the slowest
# single env, not the sum of all envs.
#
# Rebuild strategy (fastest path after git pull):
#   • Code only changed         → only COPY . . layer rebuilds   (~seconds)
#   • requirements.txt changed  → pip install layer rebuilds      (~2 min)
#   • One env's requirements    → only that env stage rebuilds    (parallel)
#   • Dockerfile itself changed → all env stages rebuild in parallel
#
# BuildKit is required (enabled by default on Docker 23+, or set
# DOCKER_BUILDKIT=1). The deploy.sh script handles this automatically.
# ─────────────────────────────────────────────────────────────────────────────

# ── Base: system packages + miniconda + mamba ─────────────────────────────────
# All env stages inherit from here. Built once, reused by every env stage.
FROM ubuntu:22.04 AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        wget \
        libgomp1 \
        python3 \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda config --set always_yes yes \
    && /opt/conda/bin/conda config --add channels conda-forge \
    && /opt/conda/bin/conda config --set channel_priority strict \
    && /opt/conda/bin/conda tos accept --channel https://repo.anaconda.com/pkgs/main \
    && /opt/conda/bin/conda tos accept --channel https://repo.anaconda.com/pkgs/r \
    && /opt/conda/bin/conda install -n base -c conda-forge mamba -y \
    && /opt/conda/bin/conda clean -afy

# ─────────────────────────────────────────────────────────────────────────────
# Env stages — all FROM base, so BuildKit builds them in PARALLEL.
# Each stage only contains the one conda env it creates.
# The final stage copies each env directory into the assembled image.
# ─────────────────────────────────────────────────────────────────────────────

# ── KinForm ───────────────────────────────────────────────────────────────────
FROM base AS env-kinform
COPY docker-requirements/kinform_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py312,target=/root/.cache/pip,sharing=locked \
    mamba create -n kinform_env python=3.12 -c conda-forge -y \
    && conda run -n kinform_env pip install -r docker-requirements/kinform_requirements.txt

# ── DLKcat ────────────────────────────────────────────────────────────────────
FROM base AS env-dlkcat
COPY docker-requirements/dlkcat_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py37,target=/root/.cache/pip,sharing=locked \
    mamba create -n dlkcat_env python=3.7.12 -c conda-forge -y \
    && mamba install -n dlkcat_env -c conda-forge --override-channels rdkit=2020.09.1 -y \
    && conda run -n dlkcat_env pip install -r docker-requirements/dlkcat_requirements.txt

# ── EITLEM ────────────────────────────────────────────────────────────────────
FROM base AS env-eitlem
COPY docker-requirements/eitlem_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py310,target=/root/.cache/pip,sharing=locked \
    mamba create -n eitlem_env python=3.10.15 -c conda-forge -y \
    && conda run -n eitlem_env pip install -r docker-requirements/eitlem_requirements.txt

# ── TurNup ────────────────────────────────────────────────────────────────────
FROM base AS env-turnup
COPY docker-requirements/turnup_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py37,target=/root/.cache/pip,sharing=locked \
    mamba create -n turnup_env python=3.7 -c conda-forge -y \
    && mamba install -n turnup_env -c conda-forge py-xgboost=1.6.1 -y \
    && conda run -n turnup_env pip install -r docker-requirements/turnup_requirements.txt

# ── UniKP ─────────────────────────────────────────────────────────────────────
FROM base AS env-unikp
COPY docker-requirements/unikp_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py37,target=/root/.cache/pip,sharing=locked \
    mamba create -n unikp python=3.7.12 -c conda-forge -y \
    && conda run -n unikp pip install -r docker-requirements/unikp_requirements.txt \
    && conda run -n unikp pip install accelerate

# ── CataPro ───────────────────────────────────────────────────────────────────
FROM base AS env-catapro
COPY docker-requirements/catapro_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py310,target=/root/.cache/pip,sharing=locked \
    mamba create -n catapro_env python=3.10.15 -c conda-forge -y \
    && mamba install -n catapro_env -c conda-forge rdkit=2024.03.6 -y \
    && conda run -n catapro_env pip install -r docker-requirements/catapro_requirements.txt

# ── CatPred ───────────────────────────────────────────────────────────────────
FROM base AS env-catpred
COPY docker-requirements/catpred_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py310,target=/root/.cache/pip,sharing=locked \
    mamba create -n catpred_env python=3.10.15 -c conda-forge -y \
    && mamba install -n catpred_env -c conda-forge rdkit=2024.03.6 -y \
    && conda run -n catpred_env pip install -r docker-requirements/catpred_requirements.txt

# ── MMISA-KM ──────────────────────────────────────────────────────────────────
FROM base AS env-mmisakm
COPY docker-requirements/mmisakm_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py39,target=/root/.cache/pip,sharing=locked \
    mamba create -n mmisakm_env python=3.9 -c conda-forge -y \
    && mamba install -n mmisakm_env -c conda-forge rdkit=2022.9.5 -y \
    && conda run -n mmisakm_env pip install -r docker-requirements/mmisakm_requirements.txt

    # ── OmniESI ──────────────────────────────────────────────────────────────────
FROM base AS env-omniesi
COPY docker-requirements/omniesi_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py38,target=/root/.cache/pip,sharing=locked \
    mamba create -n omniesi_env python=3.8 -c conda-forge -y \
    && mamba install -n omniesi_env -c conda-forge rdkit=2024.03.6 -y \
    && conda run -n omniesi_env pip install -r docker-requirements/omniesi_requirements.txt

# ── pseq2sites ────────────────────────────────────────────────────────────────
FROM base AS env-pseq2sites
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py37,target=/root/.cache/pip,sharing=locked \
    mamba create -n pseq2sites python=3.7.12 -c conda-forge -y \
    && conda run -n pseq2sites pip install --prefer-binary \
        torch==1.7.1 numpy==1.20.0 transformers==4.30.2 sentencepiece==0.2.0 \
        biopython==1.79 rdkit-pypi==2021.3.1 openbabel-wheel pandas tqdm

# ── ESM ───────────────────────────────────────────────────────────────────────
FROM base AS env-esm
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py37,target=/root/.cache/pip,sharing=locked \
    mamba create -n esm python=3.7 -c conda-forge -y \
    && conda run -n esm pip install torch fair-esm pandas tqdm

# ── ESMC ──────────────────────────────────────────────────────────────────────
FROM base AS env-esmc
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py312,target=/root/.cache/pip,sharing=locked \
    mamba create -n esmc python=3.12 -c conda-forge -y \
    && conda run -n esmc pip install esm pandas tqdm

# ── ProtT5 ────────────────────────────────────────────────────────────────────
FROM base AS env-prot-t5
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=webkinpred-pip-py39,target=/root/.cache/pip,sharing=locked \
    mamba create -n prot_t5 python=3.9 -c conda-forge -y \
    && conda run -n prot_t5 pip install \
        torch transformers sentencepiece pandas tqdm accelerate

# ── mmseqs2 ───────────────────────────────────────────────────────────────────
FROM base AS env-mmseqs2
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    mamba create -n mmseqs2_env python=3.10 -c conda-forge -y \
    && mamba install -n mmseqs2_env -c bioconda mmseqs2=13.45111 -y

# ─────────────────────────────────────────────────────────────────────────────
# Final image — assembles base + all envs + app code.
# The COPY --from lines are independent so BuildKit can fetch them in parallel.
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS final

COPY --from=env-kinform   /opt/conda/envs/kinform_env  /opt/conda/envs/kinform_env
COPY --from=env-dlkcat    /opt/conda/envs/dlkcat_env   /opt/conda/envs/dlkcat_env
COPY --from=env-eitlem    /opt/conda/envs/eitlem_env   /opt/conda/envs/eitlem_env
COPY --from=env-turnup    /opt/conda/envs/turnup_env   /opt/conda/envs/turnup_env
COPY --from=env-unikp     /opt/conda/envs/unikp        /opt/conda/envs/unikp
COPY --from=env-catapro   /opt/conda/envs/catapro_env  /opt/conda/envs/catapro_env
COPY --from=env-catpred   /opt/conda/envs/catpred_env  /opt/conda/envs/catpred_env
COPY --from=env-mmisakm   /opt/conda/envs/mmisakm_env  /opt/conda/envs/mmisakm_env
COPY --from=env-omniesi   /opt/conda/envs/omniesi_env  /opt/conda/envs/omniesi_env
COPY --from=env-pseq2sites /opt/conda/envs/pseq2sites  /opt/conda/envs/pseq2sites
COPY --from=env-esm       /opt/conda/envs/esm          /opt/conda/envs/esm
COPY --from=env-esmc      /opt/conda/envs/esmc         /opt/conda/envs/esmc
COPY --from=env-prot-t5   /opt/conda/envs/prot_t5      /opt/conda/envs/prot_t5
COPY --from=env-mmseqs2   /opt/conda/envs/mmseqs2_env  /opt/conda/envs/mmseqs2_env

# ── Django / app-level dependencies ──────────────────────────────────────────
COPY requirements.txt ./
RUN --mount=type=cache,id=webkinpred-pip-py313,target=/root/.cache/pip,sharing=locked \
    pip install -r requirements.txt

# ── Final bytecode cleanup ────────────────────────────────────────────────────
RUN find /opt/conda -name "*.pyc" -delete \
    && find /opt/conda -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

RUN mkdir -p /app/models/EITLEM/Weights \
             /app/models/CatPred \
             /app/models/TurNup/data/saved_models \
             /app/models/UniKP-main/models \
             /app/media/sequence_info \
             /app/staticfiles \
             /app/mmseqs_tmp \
    && chmod 777 /app/mmseqs_tmp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

CMD ["celery", "-A", "webKinPred", "worker", \
     "--loglevel=info", "--queues=webkinpred", "--concurrency=1"]