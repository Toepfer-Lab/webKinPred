#!/usr/bin/env bash
# deploy.sh — rebuild changed services and reclaim space from old image layers
#
# Usage:
#   ./deploy.sh                     # prod (default)
#   ./deploy.sh dev                 # dev compose
#   ./deploy.sh prod celery         # rebuild only the celery worker
#
# How space is reclaimed:
#   Each `--build` tags the new image (e.g. webkinpred-worker:latest) and
#   leaves the old image untagged ("dangling"). Docker never auto-removes
#   these. `docker image prune -f` at the end cleans them up.
#
# How the cache is preserved:
#   BuildKit reuses unchanged stages (each conda env is its own stage).
#   Only stages whose inputs changed are rebuilt — the rest are skipped
#   entirely. The --mount=type=cache mounts in the Dockerfile keep download
#   caches (conda pkgs, pip wheels) across full rebuilds so you never
#   re-download the same file twice.

set -euo pipefail

ENV="${1:-prod}"
shift 2>/dev/null || true   # remaining args = specific services to rebuild

case "$ENV" in
  prod) COMPOSE_FILE="docker-compose.prod.yml" ;;
  dev)  COMPOSE_FILE="docker-compose.yml" ;;
  *)    echo "Unknown env '$ENV'. Use 'prod' or 'dev'."; exit 1 ;;
esac

echo "==> Building and starting services (compose: $COMPOSE_FILE) ..."
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 \
  sudo docker compose -f "$COMPOSE_FILE" up -d --build --remove-orphans "$@"

echo "==> Pruning dangling images to reclaim disk space ..."
sudo docker image prune -f

echo "==> Done."
