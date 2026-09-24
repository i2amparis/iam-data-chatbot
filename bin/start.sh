#!/usr/bin/env bash
# Boot the web process.
#
# When a local model is configured (LOCAL_LLM_MODEL) the container runs Ollama
# itself, so the app can reach it at http://localhost:11434/api/chat without any
# host-level install. The Ollama binary and models live on a Dokku storage
# volume mounted at /app/ollama-data, so they survive restarts and deploys.
#
# OLLAMA_PREWARM=1 downloads the binary and pulls the models in the background
# while the app keeps serving with OpenAI; use it to warm the volume before
# switching LOCAL_LLM_* on, so the switchover restarts in seconds.
set -euo pipefail

OLLAMA_DATA_DIR="${OLLAMA_DATA_DIR:-/app/ollama-data}"
OLLAMA_VERSION="${OLLAMA_VERSION:-v0.34.4}"
CHAT_MODEL="${LOCAL_LLM_MODEL:-qwen3:0.6b-q4_K_M}"
EMBED_MODEL="${IAM_EMBEDDING_MODEL:-nomic-embed-text}"
OLLAMA_BIN="$OLLAMA_DATA_DIR/bin/ollama"

export OLLAMA_MODELS="${OLLAMA_MODELS:-$OLLAMA_DATA_DIR/models}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-10m}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"

log() { echo "[start] $*"; }

ensure_storage() {
  mkdir -p "$OLLAMA_DATA_DIR" 2>/dev/null || true
  if ( : > "$OLLAMA_DATA_DIR/.writable" ) 2>/dev/null; then
    rm -f "$OLLAMA_DATA_DIR/.writable"
  else
    log "WARNING: $OLLAMA_DATA_DIR is not writable; using \$HOME (not persistent)"
    OLLAMA_DATA_DIR="${HOME:-/tmp}/ollama-data"
    OLLAMA_BIN="$OLLAMA_DATA_DIR/bin/ollama"
    export OLLAMA_MODELS="$OLLAMA_DATA_DIR/models"
    mkdir -p "$OLLAMA_DATA_DIR"
  fi
}

install_ollama() {
  [ -x "$OLLAMA_BIN" ] && return 0
  log "installing Ollama $OLLAMA_VERSION into $OLLAMA_DATA_DIR"
  local url="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
  curl -fL --retry 3 --retry-delay 5 "$url" | zstd -d | tar -xf - -C "$OLLAMA_DATA_DIR"
  log "Ollama binary installed"
}

start_ollama() {
  log "starting ollama serve on $OLLAMA_HOST"
  "$OLLAMA_BIN" serve >"$OLLAMA_DATA_DIR/ollama-serve.log" 2>&1 &
  local i
  for i in $(seq 1 60); do
    if curl -fsS "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1; then
      log "ollama is ready"
      return 0
    fi
    sleep 1
  done
  log "WARNING: ollama did not become ready within 60s"
  return 1
}

pull_models() {
  local model
  for model in "$CHAT_MODEL" "$EMBED_MODEL"; do
    [ -n "$model" ] || continue
    if "$OLLAMA_BIN" list 2>/dev/null | awk '{print $1}' | grep -q "^${model}"; then
      log "model $model is present"
    else
      log "pulling model $model"
      "$OLLAMA_BIN" pull "$model" || log "WARNING: pull failed for $model"
    fi
  done
}

ensure_storage

if [ -n "${LOCAL_LLM_MODEL:-}" ]; then
  install_ollama
  start_ollama || true
  pull_models
elif [ "${OLLAMA_PREWARM:-0}" = "1" ]; then
  log "prewarming Ollama in the background (app keeps serving with its default provider)"
  (install_ollama && start_ollama && pull_models) &
fi

log "starting web process"
exec uvicorn fastapi_app:app --host 0.0.0.0 --port "${PORT:-5000}"
