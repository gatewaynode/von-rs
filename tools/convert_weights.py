"""Convert Von's option_marker.pt into a self-contained safetensors checkpoint dir.

option_marker.pt holds the full fine-tuned state dict (encoder.* + scorer.*), so
it supersedes the Hub's model.safetensors backbone. The Rust runtime never reads
pickle; it loads the directory this script writes:

    <out>/option_marker.safetensors   same keys as the .pt
    <out>/config.json
    <out>/tokenizer.json
    <out>/tokenizer_config.json
    <out>/marker_calibration.json

Usage (from the repo root, so the von env is used):
    uv run python von-rs/tools/convert_weights.py --out von-rs/checkpoints/von-1.1
    uv run python von-rs/tools/convert_weights.py --src <local ckpt dir> --out ...
or `just convert` from von-rs/, which writes to VON_WEIGHTS (see tools/model_paths.sh).
"""

import argparse
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

HF_REPO = "wfzyx/von"
SIDE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "marker_calibration.json")


def resolve(src: str | None, name: str) -> str | None:
    if src:
        path = os.path.join(src, name)
        return path if os.path.exists(path) else None
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=HF_REPO, filename=name)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help=f"local checkpoint dir (default: Hub '{HF_REPO}')")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pt_path = resolve(args.src, "option_marker.pt")
    if not pt_path:
        raise SystemExit(f"option_marker.pt not found in {args.src or HF_REPO}")

    os.makedirs(args.out, exist_ok=True)
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    state = {k: v.contiguous() for k, v in state.items()}
    out_path = os.path.join(args.out, "option_marker.safetensors")
    save_file(state, out_path, metadata={"format": "pt", "source": "option_marker.pt"})

    # Round-trip check: bit-identical tensors, same key set.
    back = load_file(out_path)
    assert back.keys() == state.keys(), "key mismatch after conversion"
    for k, v in state.items():
        assert torch.equal(back[k], v), f"tensor mismatch: {k}"
    print(f"wrote {out_path} ({len(state)} tensors, verified)")

    for name in SIDE_FILES:
        path = resolve(args.src, name)
        if path is None:
            if name == "marker_calibration.json":
                print(f"skip {name} (not present; runtime falls back to T=1.0)")
                continue
            raise SystemExit(f"{name} not found in {args.src or HF_REPO}")
        shutil.copyfile(path, os.path.join(args.out, name))
        print(f"copied {name}")


if __name__ == "__main__":
    main()
