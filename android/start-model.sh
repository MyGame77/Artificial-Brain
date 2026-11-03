#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
MODEL="$HOME/storage/shared/Music/Projects/Artificial-Brain-Android-ready/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
BIN="$(command -v llama-server || true)"
if [[ -z "$BIN" ]]; then echo "llama-server is missing. Build llama.cpp in Termux first." >&2; exit 1; fi
if [[ ! -f "$MODEL" ]]; then echo "Model not found: $MODEL" >&2; exit 1; fi
exec "$BIN" -m "$MODEL" --alias artificial-brain-qwen2.5-1.5b:latest --host 127.0.0.1 --port 8080 -c 1024 -t 2 --parallel 1