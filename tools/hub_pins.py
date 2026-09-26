"""The Hub commits of wfzyx/von that von-rs is checked against.

The repo's main branch moves when a new model ships, so every tool pins a commit:
the converter, the golden exporters and the Python benchmarks. Keep HF_REVISION in
step with `HF_REVISION` in src/weights.rs.

Run it to print a local snapshot dir for a version, downloading what is missing:
    uv run python von-rs/tools/hub_pins.py 1.2
"""

import sys

HF_REPO = "wfzyx/von"
# Von 1.2, the model von-rs implements.
HF_REVISION = "5df8185a4f2327ad0a7cd117cc4f701ac557b9ae"
# Von 1.1: default attention; kept as a regression set.
HF_REVISION_V11 = "d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd"
VERSIONS = {"1.2": HF_REVISION, "1.1": HF_REVISION_V11}
# What the Python backend reads from a checkpoint dir.
SNAPSHOT_FILES = ["option_marker.pt", "model.safetensors", "config.json", "tokenizer.json",
                  "tokenizer_config.json", "marker_calibration.json"]


def revision(name: str) -> str:
    """A version ("1.2", "1.1") or a commit hash."""
    return VERSIONS.get(name, name)


def snapshot(name: str = "1.2") -> str:
    """Local snapshot dir of a pinned revision, usable as a Python checkpoint_dir.

    File by file rather than snapshot_download, which needs a cached listing of the
    commit to work offline (HF_HUB_OFFLINE=1); pinned files resolve from the cache.
    """
    import os

    from huggingface_hub import hf_hub_download

    paths = [hf_hub_download(repo_id=HF_REPO, filename=f, revision=revision(name)) for f in SNAPSHOT_FILES]
    return os.path.dirname(paths[0])


if __name__ == "__main__":
    print(snapshot(sys.argv[1] if len(sys.argv) > 1 else "1.2"))
