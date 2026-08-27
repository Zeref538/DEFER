"""The checks every Kaggle run does before it spends a GPU hour.

Three things have already killed a run here, so all three are asserted up front
where they cost seconds instead of hours:

- the notebook was scheduled with no accelerator at all
- the card is a P100 (sm_60) and this PyTorch has no compiled code for it, which
  surfaces at the *first generate()* -- on the far side of a 6 GB download
- the mounted checkpoint is not the model the study is about

Shared by every phase so a fix lands once. Copy-pasting these into each phase is
how a bug that was fixed comes back.

Run the self-check:  python ml/kaggle_env.py
"""
from __future__ import annotations

import hashlib
import os
import sys

MODEL_HINT = "llama-3.2"
FALLBACK_MODEL = "microsoft/Phi-3.5-mini-instruct"


def line(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68, flush=True)


def die(what, why, fix):
    print()
    print(f"  FAILED: {what}")
    print(f"  what it means: {why}")
    print(f"  what to do:    {fix}", flush=True)
    sys.exit(1)


def code_fingerprint(directory):
    """A short hash of the code that is actually running.

    An attached dataset can serve an older version than the one just published,
    and stale code producing plausible-looking results is the worst failure mode
    there is. Printing this makes it visible instead of silent.
    """
    digest = hashlib.sha256()
    for name in sorted(os.listdir(directory)):
        if name.endswith(".py"):
            digest.update(name.encode())
            digest.update(open(os.path.join(directory, name), "rb").read())
    return digest.hexdigest()[:12]


def check_gpu():
    """Fail now if this card cannot run this PyTorch. Returns the device name."""
    import torch

    print(f"  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        die("no GPU", "This notebook was scheduled without an accelerator.",
            'Set "machine_shape": "NvidiaTeslaT4" in kernel-metadata.json, or '
            "pick Settings -> Accelerator -> GPU T4 x2 in the browser.")

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    this = f"sm_{major}{minor}"
    built = torch.cuda.get_arch_list()
    print(f"  device: {name}  ({this})")
    print(f"  this PyTorch was built for: {' '.join(built)}")

    if this not in built:
        die(f"this PyTorch cannot run on a {name}",
            f"The card is {this}; this build only has kernels for "
            f"{' '.join(built)}. Nothing is wrong with the code -- there is simply "
            "no compiled GPU code for this chip, so the first generate() would die "
            "with 'no kernel image is available for execution on the device'.",
            'Set "machine_shape": "NvidiaTeslaT4" in kernel-metadata.json and push '
            "again. The default P100 cannot run this.")
    print("  usable.")
    return name


def find_model(hint: str = MODEL_HINT, fallback: str = FALLBACK_MODEL):
    """Locate mounted weights under /kaggle/input, or name a fallback.

    Kaggle hosts Meta's Llama itself, so attaching it as a `model_sources` entry
    sidesteps the Hugging Face gate and the token that goes with it entirely.
    """
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "config.json" in files and any(f.endswith(".safetensors") for f in files):
            if hint in root.lower():
                print(f"  weights at: {root}")
                return root

    print("  no mounted checkpoint found. /kaggle/input holds:")
    for root, _dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) - "/kaggle/input".count(os.sep) > 4:
            continue
        print(f"    {root}  ->  {files[:5]}")
    print()
    print(f"  FALLING BACK to {fallback} (ungated, MIT).")
    print("  For Llama, accept Meta's terms at")
    print("  https://www.kaggle.com/models/metaresearch/llama-3.2 and re-run.")
    return fallback


def assert_size(model, low=2.5e9, high=4.5e9):
    """The cheapest possible catch for a wrong-model launch."""
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params / 1e9:.2f}B parameters")
    if not low < params < high:
        die("that is not the model this study is designed around",
            f"Counted {params / 1e9:.2f}B parameters. The previous study found "
            "1.5B too small to learn the behaviour, so size is not cosmetic.",
            "Check MODEL_HINT / FALLBACK_MODEL in ml/kaggle_env.py.")
    return params


def demo():
    """Only the parts that run without a GPU. The rest is Kaggle-only by nature."""
    import tempfile
    from pathlib import Path

    work = Path(tempfile.mkdtemp())
    (work / "a.py").write_text("print(1)", encoding="utf-8")
    first = code_fingerprint(work)
    assert len(first) == 12
    assert code_fingerprint(work) == first, "fingerprint must be stable"
    (work / "a.py").write_text("print(2)", encoding="utf-8")
    assert code_fingerprint(work) != first, "a code change must change the hash"
    (work / "notes.txt").write_text("ignored", encoding="utf-8")
    changed = code_fingerprint(work)
    (work / "notes.txt").write_text("still ignored", encoding="utf-8")
    assert code_fingerprint(work) == changed, "non-python files must not count"
    print("kaggle_env self-check passed")


if __name__ == "__main__":
    demo()
