#!/usr/bin/env bash
# Protocol parity: the unmodified Python and JS SDKs against the real `von serve`.
# Starts the release binary with Bearer auth on, then runs
#   - tools/cross_sdk_client.py: all golden requests through VonClient(local=False),
#     compared with the Python engine's responses, plus ports of test_server.py and
#     test_client.py over HTTP;
#   - tools/cross_sdk_client.ts: the JS SDK's systemOne/decide/judge/rate and a 401.
#
# Usage (from anywhere):  bash von-rs/tools/cross_sdk_check.sh
# Env: VON_DEVICE (default cpu), PORT (default 8765), VON_PY_SRC (default: the fork).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RS="$ROOT/von-rs"
. "$ROOT/von-rs/tools/model_paths.sh"  # VON_WEIGHTS, HF_HOME
export VON_PY_SRC="${VON_PY_SRC:-$ROOT/bug-fix-fork-von/src}"
SDK_JS="${SDK_JS:-$ROOT/bug-fix-fork-von/js}"
PORT="${PORT:-8765}"
KEY="cross-sdk-$$"
BASE="http://127.0.0.1:$PORT"

cd "$RS"
cargo build --release -q --bin von
VON_API_KEY="$KEY" VON_CHECKPOINT_DIR="$VON_WEIGHTS" VON_DEVICE="${VON_DEVICE:-cpu}" \
  target/release/von serve --host 127.0.0.1 --port "$PORT" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  curl -sf "$BASE/health" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server exited during startup"; exit 1; }
  sleep 0.5
done
curl -sf "$BASE/health" >/dev/null || { echo "server did not come up"; exit 1; }

echo "== Python SDK =="
(cd "$ROOT" && uv run python "$RS/tools/cross_sdk_client.py" "$BASE" "$KEY")
echo "== JS SDK =="
bun run "$RS/tools/cross_sdk_client.ts" "$BASE" "$KEY" "$SDK_JS"
echo "== cross-SDK check passed =="
