"""Loading SQuAD 2.0 into the flat records the rest of the pipeline expects.

SQuAD 2.0 is a reading-comprehension dataset: a paragraph, questions about it,
and short answers quoted from the paragraph. Half the questions are deliberately
*unanswerable* -- they look answerable but the paragraph never says -- which is
exactly the abstention slice this study needs, already built and public.

It is downloaded as plain JSON over HTTPS rather than through a dataset library.
One file, no dependency, works the same on a laptop and inside a Kaggle kernel.

Run the self-check:  python ml/squad.py
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

BASE = "https://rajpurkar.github.io/SQuAD-explorer/dataset"
SPLITS = {"train": "train-v2.0.json", "dev": "dev-v2.0.json"}

# The dev split is the one nobody trains on, so it is where the frozen
# evaluation comes from. Training mixes are drawn from train, and the two never
# share a qid -- checked in ml/tests.py rather than assumed.


def default_dir() -> Path:
    """Where to cache the downloaded split.

    Beside the repo when the code is checked out normally, which keeps the data
    next to the project and out of git. But the code does not always live
    somewhere writable: on Kaggle it is mounted at /kaggle/input/defer-code,
    read-only, and writing beside it fails with

        OSError: [Errno 30] Read-only file system: '/kaggle/input/data'

    That only ever appears once the code is mounted rather than checked out, so
    it passes every local test. Fall back to the working directory, which is
    writable in both places.
    """
    beside = Path(__file__).resolve().parent.parent / "data"
    if beside.exists() and os.access(beside, os.W_OK):
        return beside
    if os.access(beside.parent, os.W_OK):
        return beside
    return Path.cwd() / "data"


def ensure(split: str = "dev", cache_dir=None) -> Path:
    """Download the split if it is not already on disk. Returns its path."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
    cache_dir = Path(cache_dir or default_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"squad2_{split}.json"
    if target.exists() and target.stat().st_size > 1000:
        return target
    tmp = target.with_suffix(".part")
    urllib.request.urlretrieve(f"{BASE}/{SPLITS[split]}", tmp)
    tmp.replace(target)      # only becomes the real name once complete
    return target


def load(split: str = "dev", cache_dir=None) -> dict:
    return json.loads(ensure(split, cache_dir).read_text(encoding="utf-8"))


def items(split: str = "dev", kind: str = "answerable", cache_dir=None) -> list:
    """Flatten the nested dataset into records.

    kind="answerable"   -> qid, question, passage, gold
    kind="unanswerable" -> qid, question, passage, gold=None

    Only the first listed answer is taken as `gold`. SQuAD gives several human
    answers for dev questions, but they are near-identical spans and the extra
    ones would only make the substitute pools noisier.
    """
    if kind not in ("answerable", "unanswerable"):
        raise ValueError(f"kind must be answerable or unanswerable, got {kind!r}")
    want_impossible = kind == "unanswerable"

    out = []
    for article in load(split, cache_dir)["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if bool(qa["is_impossible"]) != want_impossible:
                    continue
                if not want_impossible and not qa["answers"]:
                    continue
                out.append({
                    "qid": qa["id"],
                    "question": qa["question"],
                    "passage": paragraph["context"],
                    "gold": None if want_impossible else qa["answers"][0]["text"],
                    "title": article.get("title"),
                    "source": f"squad2:{split}:{qa['id']}",
                })
    return out


def demo():
    """Self-check. Downloads the dev split (4.4 MB) the first time only."""
    chosen = default_dir()
    assert os.access(chosen if chosen.exists() else chosen.parent, os.W_OK), (
        f"cache dir {chosen} is not writable -- this is the Kaggle read-only "
        "mount failure, and it only shows up where the code is mounted")
    path = ensure("dev")
    assert path.exists() and path.stat().st_size > 1_000_000, path

    answerable = items("dev", "answerable")
    unanswerable = items("dev", "unanswerable")

    assert len(answerable) > 5000, len(answerable)
    assert len(unanswerable) > 5000, len(unanswerable)

    ids = {r["qid"] for r in answerable} | {r["qid"] for r in unanswerable}
    assert len(ids) == len(answerable) + len(unanswerable), "qids must be unique"

    for record in answerable[:200]:
        assert record["gold"], "an answerable item needs an answer"
        assert record["gold"] in record["passage"], "SQuAD answers are spans"
    for record in unanswerable[:200]:
        assert record["gold"] is None, "unanswerable must be None, not empty string"

    print(f"squad self-check passed  "
          f"(dev: {len(answerable)} answerable, {len(unanswerable)} unanswerable)")


if __name__ == "__main__":
    demo()
