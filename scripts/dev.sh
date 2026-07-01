#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-dev}"
RUN_INDEX="${RUN_INDEX:-auto}"
RUN_WARMUP="${RUN_WARMUP:-0}"
RELOAD="${RELOAD:-1}"

usage() {
  cat <<'EOF'
Usage: ./scripts/dev.sh [options]

Options:
  --host HOST       Bind host, default 127.0.0.1
  --port PORT       Bind port, default 8000
  --skip-index      Do not build or check the local index
  --reindex         Rebuild the local index before starting
  --warmup          Run model warmup before starting
  --no-reload       Start uvicorn without --reload
  -h, --help        Show this help

Environment:
  HOST, PORT, PYTHON_BIN, VENV_DIR, INSTALL_EXTRAS, RUN_INDEX, RUN_WARMUP, RELOAD
EOF
}

log() {
  printf '==> %s\n' "$*"
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      [ -n "$HOST" ] || fail "--host requires a value"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      [ -n "$PORT" ] || fail "--port requires a value"
      shift 2
      ;;
    --skip-index)
      RUN_INDEX="never"
      shift
      ;;
    --reindex)
      RUN_INDEX="always"
      shift
      ;;
    --warmup)
      RUN_WARMUP="1"
      shift
      ;;
    --no-reload)
      RELOAD="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

ensure_venv() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    log "Creating virtual environment in $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
}

install_dependencies() {
  local marker="$VENV_DIR/.rag-dev-installed"
  if [ ! -f "$marker" ] || [ pyproject.toml -nt "$marker" ]; then
    log "Installing Python dependencies"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
    "$VENV_DIR/bin/python" -m pip install -e ".[$INSTALL_EXTRAS]"
    touch "$marker"
  fi
}

ensure_env_file() {
  if [ ! -f .env ]; then
    cp .env.example .env
    cat <<'EOF'
Created .env from .env.example.
Edit .env and set OPENAI_API_KEY before starting the service.
EOF
    exit 1
  fi
}

env_value() {
  local key="$1"
  local default="$2"
  local value
  value="$(awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' .env \
    | sed 's/^["'\'']//;s/["'\'']$//;s/^[[:space:]]*//;s/[[:space:]]*$//')"
  printf '%s' "${value:-$default}"
}

check_api_key() {
  local api_key
  api_key="$(env_value OPENAI_API_KEY "")"
  case "$api_key" in
    ""|"replace-with-your-api-key")
      fail "OPENAI_API_KEY is not configured in .env"
      ;;
  esac
}

index_ready() {
  local chroma_dir bm25_path parent_path index_root active_path active_version version_dir
  chroma_dir="$(env_value CHROMA_DIR "data/chroma")"
  bm25_path="$(env_value BM25_CORPUS_PATH "data/chroma/bm25_corpus.jsonl")"
  parent_path="$(env_value PARENT_CORPUS_PATH "data/chroma/parent_corpus.jsonl")"
  index_root="$(env_value INDEX_ROOT_DIR "data/indexes")"
  active_path="$(env_value ACTIVE_INDEX_VERSION_PATH "$index_root/active_version.txt")"

  if [ -s "$active_path" ]; then
    active_version="$(tr -d '[:space:]' < "$active_path")"
    version_dir="$index_root/$active_version"
    [ -n "$active_version" ] \
      && [ -s "$version_dir/chroma/chroma.sqlite3" ] \
      && [ -s "$version_dir/bm25_corpus.jsonl" ] \
      && [ -s "$version_dir/parent_corpus.jsonl" ] \
      && return 0
  fi

  [ -s "$chroma_dir/chroma.sqlite3" ] && [ -s "$bm25_path" ] && [ -s "$parent_path" ]
}

maybe_build_index() {
  case "$RUN_INDEX" in
    never)
      log "Skipping index build"
      ;;
    always)
      log "Rebuilding local index"
      "$VENV_DIR/bin/python" -m scripts.ingest
      ;;
    auto)
      if index_ready; then
        log "Existing local index detected"
      else
        log "Local index missing; building it now"
        "$VENV_DIR/bin/python" -m scripts.ingest
      fi
      ;;
    *)
      fail "RUN_INDEX must be auto, always, or never"
      ;;
  esac
}

maybe_warmup() {
  if [ "$RUN_WARMUP" = "1" ]; then
    log "Warming up retrieval models"
    "$VENV_DIR/bin/python" -m scripts.warmup
  fi
}

start_server() {
  local cmd=("$VENV_DIR/bin/uvicorn" app.main:app --host "$HOST" --port "$PORT")
  if [ "$RELOAD" = "1" ]; then
    cmd+=(--reload)
  fi
  log "Starting http://$HOST:$PORT/"
  exec "${cmd[@]}"
}

ensure_venv
install_dependencies
ensure_env_file
check_api_key
maybe_build_index
maybe_warmup
start_server
