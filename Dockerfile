# syntax=docker/dockerfile:1.4
# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — full worker image (celery only)
#
# Contains all conda model environments.  Used exclusively by the celery
# worker; backend and celery-beat use the much lighter Dockerfile.web.
#
# Rebuild strategy (fastest path after git pull):
#   • Code only changed         → only COPY . . layer rebuilds   (~seconds)
#   • requirements.txt changed  → pip install layer rebuilds      (~2 min)
#   • One env's requirements    → only THAT env layer rebuilds    (~3–8 min)
#   • Dockerfile itself changed → full rebuild                    (~20 min,
#                                  but conda/pip download cache speeds it up)
#
# To enable BuildKit (required for --mount=type=cache):
#   export DOCKER_BUILDKIT=1
#   export COMPOSE_DOCKER_CLI_BUILD=1
# ─────────────────────────────────────────────────────────────────────────────

FROM ubuntu:22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:$PATH"

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        wget \
        libgomp1 \
        python3 \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda + mamba ─────────────────────────────────────────────────────────
# mamba replaces conda for environment creation: parallel downloads,
# faster dependency solver — typically 3-5x quicker than conda.
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

# ── Django / app-level dependencies ──────────────────────────────────────────
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Model conda environments
# Each env is its own RUN layer so Docker can cache them independently.
# If only kinform_requirements.txt changes, only the kinform_env layer
# is invalidated; all others stay cached.
# The --mount=type=cache mounts persist the conda package cache and pip
# download cache on the host between builds, making reinstalls fast.
# ─────────────────────────────────────────────────────────────────────────────

# ── KinForm (H + L) ──────────────────────────────────────────────────────────
COPY docker-requirements/kinform_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    mamba create -n kinform_env python=3.12 -c conda-forge -y \
    && conda run -n kinform_env pip install -r docker-requirements/kinform_requirements.txt \
    && conda clean -afy

# ── DLKcat ────────────────────────────────────────────────────────────────────
COPY docker-requirements/dlkcat_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    mamba create -n dlkcat_env python=3.7.12 -c conda-forge -y \
    && mamba install -n dlkcat_env -c conda-forge --override-channels rdkit=2020.09.1 -y \
    && conda run -n dlkcat_env pip install -r docker-requirements/dlkcat_requirements.txt \
    && conda clean -afy

# ── EITLEM ────────────────────────────────────────────────────────────────────
COPY docker-requirements/eitlem_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    mamba create -n eitlem_env python=3.10.15 -c conda-forge -y \
    && conda run -n eitlem_env pip install -r docker-requirements/eitlem_requirements.txt \
    && conda clean -afy

# ── TurNup ────────────────────────────────────────────────────────────────────
COPY docker-requirements/turnup_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    mamba create -n turnup_env python=3.7 -c conda-forge -y \
    && mamba install -n turnup_env -c conda-forge py-xgboost=1.6.1 -y \
    && conda run -n turnup_env pip install -r docker-requirements/turnup_requirements.txt \
    && conda clean -afy

# ── UniKP ─────────────────────────────────────────────────────────────────────
COPY docker-requirements/unikp_requirements.txt ./docker-requirements/
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    mamba create -n unikp python=3.7.12 -c conda-forge -y \
    && conda run -n unikp pip install -r docker-requirements/unikp_requirements.txt \
    && conda clean -afy

# ── Embedding envs (inline deps — rarely change) ──────────────────────────────
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    mamba create -n pseq2sites python=3.7.12 -c conda-forge -y \
    && conda run -n pseq2sites pip install --prefer-binary \
        torch==1.7.1 numpy==1.20.0 transformers==4.30.2 sentencepiece==0.2.0 \
        biopython==1.79 rdkit-pypi==2021.3.1 openbabel-wheel pandas tqdm \
    && conda clean -afy

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    mamba create -n esm python=3.7 -c conda-forge -y \
    && conda run -n esm pip install torch fair-esm pandas tqdm \
    && conda clean -afy

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    mamba create -n esmc python=3.12 -c conda-forge -y \
    && conda run -n esmc pip install esm pandas tqdm \
    && conda clean -afy

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    mamba create -n prot_t5 python=3.9 -c conda-forge -y \
    && conda run -n prot_t5 pip install \
        torch transformers sentencepiece pandas tqdm accelerate \
    && conda clean -afy

# ── mmseqs2 ───────────────────────────────────────────────────────────────────
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    mamba create -n mmseqs2_env python=3.10 -c conda-forge -y \
    && mamba install -n mmseqs2_env -c bioconda mmseqs2=13.45111 -y \
    && conda clean -afy

# ── Final bytecode cleanup ────────────────────────────────────────────────────
RUN find /opt/conda -name "*.pyc" -delete \
    && find /opt/conda -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ── Application code (only this layer + below rebuild on git pull) ────────────
COPY . .

RUN mkdir -p /app/api/EITLEM/Weights \
             /app/api/TurNup/data/saved_models \
             /app/api/UniKP-main/models \
             /app/media/sequence_info \
             /app/staticfiles \
             /app/mmseqs_tmp \
    && chmod 777 /app/mmseqs_tmp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

CMD ["celery", "-A", "webKinPred", "worker", \
     "--loglevel=info", "--queues=webkinpred", "--concurrency=1"]
