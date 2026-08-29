"""Stage and publish the `defer-code` dataset that every Kaggle run imports.

Why a script instead of a few `cp` commands: the staging folder used to live in
a temp directory, so what actually reached Kaggle depended on which files had
been copied there most recently. A run once imported code that no longer existed
on the laptop. This builds the folder from scratch every time, from an explicit
list, and prints the fingerprint the notebook will print back -- so "is the
cloud running my latest code" becomes a string comparison instead of a hope.

    python ml/kernels/publish.py               # stage, show what would go
    python ml/kernels/publish.py --push        # stage and upload a new version

Everything is uploaded flat -- code and eval side by side. Kaggle's CLI can only
skip subfolders or zip them, and neither leaves a readable `data/eval.jsonl` on
the far side, so `load_eval` looks in both places instead.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET = "johnandreimartinez/defer-code"

# Everything the notebooks import, listed rather than globbed. A glob would have
# quietly started shipping whatever new file appeared in ml/, including scratch
# ones, and a dataset that grows by accident is a dataset nobody trusts.
CODE = [
    "ml/analyse_drift.py",
    "ml/build.py",
    "ml/build_replay.py",
    "ml/conflict.py",
    "ml/generate.py",
    "ml/kaggle_env.py",
    "ml/metrics.py",
    "ml/phase0.py",
    "ml/phase1.py",
    "ml/phase2.py",
    "ml/phase3.py",
    "ml/probe.py",
    "ml/rules.py",
    "ml/runner.py",
    "ml/score.py",
    "ml/squad.py",
    "ml/tests.py",
    "ml/train.py",
]
DATA = [
    "data/eval.jsonl",
    "data/eval.lock",
    "data/train_mix.jsonl",
]


def fingerprint(directory: Path) -> str:
    """Must match kaggle_env.code_fingerprint exactly, or the check is theatre."""
    digest = hashlib.sha256()
    for name in sorted(os.listdir(directory)):
        if name.endswith(".py"):
            digest.update(name.encode())
            digest.update((directory / name).read_bytes())
    return digest.hexdigest()[:12]


def stage(into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    for rel in CODE:
        source = ROOT / rel
        if not source.exists():
            raise SystemExit(f"missing {source} -- listed in CODE but not on disk")
        shutil.copy2(source, into / Path(rel).name)
    for rel in DATA:
        source = ROOT / rel
        if not source.exists():
            raise SystemExit(
                f"missing {source}. Run `python ml/build.py` before publishing.")
        shutil.copy2(source, into / Path(rel).name)

    (into / "dataset-metadata.json").write_text(json.dumps({
        "title": "defer-code",
        "id": DATASET,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2) + "\n", encoding="utf-8", newline="\n")

    # The eval must survive the copy byte for byte. A truncated eval on Kaggle
    # would score as a smaller, easier study rather than failing.
    locked = (ROOT / "data" / "eval.lock").read_text(encoding="utf-8").strip()
    copied = hashlib.sha256((into / "eval.jsonl").read_bytes()).hexdigest()
    if copied != locked:
        raise SystemExit(
            f"the staged eval does not match data/eval.lock\n"
            f"  staged: {copied}\n  locked: {locked}")
    return into


def main():
    work = Path(tempfile.mkdtemp(prefix="defer-code-"))
    stage(work)

    print(f"staged {len(CODE)} python files + {len(DATA)} data files in {work}")
    print(f"code fingerprint: {fingerprint(work)}")
    print("  the notebook prints this back at the top of its log. If they differ,")
    print("  the run is using an older dataset version and its results are stale.")

    if "--push" not in sys.argv:
        print()
        print("dry run. Add --push to upload a new version.")
        return

    message = " ".join(a for a in sys.argv[1:] if not a.startswith("-")) or "update code"
    result = subprocess.run(
        # `python -m kaggle` rather than `kaggle`: subprocess does not go
        # through a shell, so the console shim that only PATH knows about is
        # invisible to it. The module is always importable from this interpreter.
        [sys.executable, "-m", "kaggle", "datasets", "version",
         "-p", str(work), "-m", message],
        capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print()
    print("Kaggle takes a minute or two to finish processing the new version.")
    print("Re-run the notebook only after it does, or it mounts the old one.")


if __name__ == "__main__":
    main()
