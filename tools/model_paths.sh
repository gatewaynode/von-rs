# Model storage for the dev tools. Source it from bash: `. tools/model_paths.sh`.
#
# One knob, VON_MODELS_DIR, read from the environment or from `von-rs/.env`
# (gitignored; one `VON_MODELS_DIR=/path` line). When it is set:
#   VON_WEIGHTS = $VON_MODELS_DIR/von-1.1       (converted checkpoint)
#   HF_HOME     = $VON_MODELS_DIR/huggingface   (Hub cache, shared with Python)
# When it is unset, both stay inside the checkout:
#   VON_WEIGHTS = von-rs/checkpoints/von-1.1
#   HF_HOME     = <repo root>/.hf-cache
# An explicit VON_WEIGHTS or HF_HOME always wins. Both are exported as absolute paths.

_vmp_rs="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_vmp_root="$(cd "$_vmp_rs/.." && pwd)"

if [ -z "${VON_MODELS_DIR:-}" ] && [ -f "$_vmp_rs/.env" ]; then
  VON_MODELS_DIR="$(sed -n 's/^[[:space:]]*VON_MODELS_DIR[[:space:]]*=[[:space:]]*//p' "$_vmp_rs/.env" | tail -n 1)"
  VON_MODELS_DIR="${VON_MODELS_DIR%\"}"
  VON_MODELS_DIR="${VON_MODELS_DIR#\"}"
fi
case "${VON_MODELS_DIR:-}" in
  "~") VON_MODELS_DIR="$HOME" ;;
  "~/"*) VON_MODELS_DIR="$HOME/${VON_MODELS_DIR#"~/"}" ;;
esac

if [ -n "${VON_MODELS_DIR:-}" ]; then
  export VON_MODELS_DIR
  export VON_WEIGHTS="${VON_WEIGHTS:-$VON_MODELS_DIR/von-1.1}"
  export HF_HOME="${HF_HOME:-$VON_MODELS_DIR/huggingface}"
else
  export VON_WEIGHTS="${VON_WEIGHTS:-$_vmp_rs/checkpoints/von-1.1}"
  export HF_HOME="${HF_HOME:-$_vmp_root/.hf-cache}"
fi
case "$VON_WEIGHTS" in /*) ;; *) VON_WEIGHTS="$PWD/$VON_WEIGHTS" ;; esac
case "$HF_HOME" in /*) ;; *) HF_HOME="$PWD/$HF_HOME" ;; esac
unset _vmp_rs _vmp_root
