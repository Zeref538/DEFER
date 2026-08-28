"""Assemble the frozen evaluation set and the training mixes.

Everything the study ever reports is measured on `data/eval.jsonl`, which is
written once here and then never touched. `data/eval.lock` holds its sha256, and
every scoring run checks it. That guard exists because the most tempting failure
in this genre is nudging the evaluation set until the number improves.

Three slices, not four. Over-abstention is not its own slice -- it is the same
answerable items counted a second way, across `grounded` and `conflict` both.

Where the conflict items come from is worth stating plainly. Dev alone yields 68
balanced conflict items, and a bootstrap on 68 items puts a 22-point interval
around the headline number -- wider than the 18-point spread the previous study
measured between two seeds, so the effect could never be ranked against noise.
A slice of train is therefore *reserved* for the evaluation and kept out of
training entirely, enforced by qid and asserted here rather than assumed.

Run it:  python ml/build.py
Self-check only, no files written:  python ml/build.py --check
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import conflict  # noqa: E402
import squad  # noqa: E402
from runner import atomic_write  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROBE = ROOT / "runs" / "probe"

# Sizes. The conflict slice carries the headline number, so it is sized by the
# width of its error bars rather than by taste. Measured with the bootstrap at a
# rate of 0.30:
#
#     n= 68  ->  22.1 point interval     (dev alone: unusable)
#     n=238  ->  11.8 points
#     n=500  ->   8.0 points
#
# The previous study measured an 18-point spread between two seeds, so anything
# near 238 could not be ranked against seed noise. 500 can.
#
# This trades training data for statistical power, deliberately. The eval is
# frozen once and can never be enlarged afterwards; the training mix can always
# be regenerated.
N_CONFLICT = 800
N_GROUNDED = 300
N_UNANSWERABLE = 300

# Cap per (type, position) cell when levelling. The default -- the median cell --
# is dragged down by `number`, which the model almost never knows (75 of 1831 in
# train). Measured: per_cell=60 keeps 508 items at a 39% worst-position share,
# against 544 items at 40% with no levelling at all. Nearly free.
PER_CELL = 60

# How many "refuse" rows to include, as a multiple of the "answer" rows.
#
# The first trained pair used 0.5 -- 1,308 rows teaching "answer from the
# passage" against 327 teaching "refuse", four to one. It worked exactly as that
# ratio predicts: conflict following hit 97.9% while abstention on unanswerable
# items FELL to 20.3%, below the 33.3% a plain prompt gets for free. The model
# learned the refusal sentence verbatim and then almost never used it, because
# four times out of five the lesson was "extract something".
#
# 1.0 is the direct test of that diagnosis. Over-abstention is the guard: if the
# model starts refusing questions the passage plainly answers, this overshot,
# and that column is what makes it visible.
REFUSE_RATIO = 1.0

# One answer type is kept out of training completely and reported as its own row.
# `year` is the choice because it is structurally unlike the others -- a number
# rather than a proper noun -- so it is the sharpest test of the trap this whole
# project is built around: a model can learn "follow the passage when the answer
# is a name" and score well while having learned a pattern, not the behaviour.
HELD_OUT_TYPE = "year"

# The instruction every arm shares. Kept here rather than in the training data so
# the same wording is used at build time and at generation time.
SYSTEM_PROMPT = (
    "Answer the question using only the passage provided. If the passage does "
    "not contain the answer, say that it is not in the passage."
)


def frozen_eval_qids() -> set:
    """The qids in data/eval.jsonl as it exists on disk, or an empty set.

    Deliberately reads the file rather than the eval this run just built. Those
    are the same thing only while nothing upstream has changed, and the whole
    point of a frozen eval is that it stays put while everything else moves.
    """
    path = DATA / "eval.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line)["qid"]
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_known(split: str) -> set:
    """qids the base model reliably answered closed-book.

    Only these can become conflict items. A conflict item built from a fact the
    model never knew tests reading, not memory -- it would have read the passage
    for that one anyway, so catching it proves nothing.
    """
    path = PROBE / f"probe_{split}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"missing {path}. Run the probe kernel first -- see docs/APP_FLOW.md."
        )
    known = set()
    for record in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()):
        if record["knows"]:
            known.add(record["qid"])
    return known


def conflict_items(split: str, seed: int = 0):
    known = load_known(split)
    items = [i for i in squad.items(split, "answerable") if i["qid"] in known]
    records, drops = conflict.build(items, seed=seed)
    return records, drops


def as_eval_record(item, slice_name):
    """Flatten a SQuAD item into the eval shape. See docs/SCHEMA.md."""
    return {
        "qid": item["qid"],
        "slice": slice_name,
        "passage": item["passage"],
        "question": item["question"],
        "answer": item["gold"],
        "memorised": None,
        "edit_type": None,
        "edit_pos": None,
        "construction": None,
        "n_replacements": None,
        "source": item["source"],
    }


def fill(record):
    """Give every record every key, so readers never guess absent from null."""
    keys = ("qid", "slice", "passage", "question", "answer", "memorised",
            "edit_type", "edit_pos", "construction", "n_replacements", "source")
    return {k: record.get(k) for k in keys}


def build(seed: int = 0, write: bool = True, log=print, mix_only: bool = False):
    rng = random.Random(seed)

    # ---------------------------------------------------------- conflict
    dev_conflict, dev_drops = conflict_items("dev", seed)
    train_conflict, train_drops = conflict_items("train", seed)
    log(f"  conflict built: dev {len(dev_conflict)}, train {len(train_conflict)}")

    # Dev goes to the eval whole. Train is split: enough reserved to reach the
    # target, the rest available for training.
    need = max(0, N_CONFLICT - len(dev_conflict))
    pool = list(train_conflict)
    rng.shuffle(pool)
    reserved = pool[:need]
    trainable = pool[need:]

    eval_conflict = conflict.balance(dev_conflict + reserved, seed=seed,
                                     per_cell=PER_CELL)
    log(f"  conflict eval: {len(dev_conflict)} dev + {len(reserved)} reserved "
        f"from train -> {len(eval_conflict)} after levelling")

    # ---------------------------------------------------------- other slices
    conflict_qids = {r["qid"] for r in eval_conflict}
    # Exclude any item sharing a passage with a conflict item. Not a leak as
    # such -- each prompt is scored on its own -- but an eval holding the same
    # paragraph both edited and unedited is a confusing artefact to defend.
    conflict_passages = {r["passage"][:200] for r in dev_conflict + reserved}

    dev_answerable = [
        i for i in squad.items("dev", "answerable")
        if i["qid"] not in conflict_qids and i["passage"][:200] not in conflict_passages
    ]
    dev_unanswerable = [
        i for i in squad.items("dev", "unanswerable")
        if i["passage"][:200] not in conflict_passages
    ]
    rng.shuffle(dev_answerable)
    rng.shuffle(dev_unanswerable)

    grounded = [as_eval_record(i, "grounded") for i in dev_answerable[:N_GROUNDED]]
    unanswerable = [as_eval_record(i, "unanswerable") for i in dev_unanswerable[:N_UNANSWERABLE]]
    for record in unanswerable:
        record["answer"] = None      # by construction: the passage does not say

    eval_records = [fill(r) for r in eval_conflict + grounded + unanswerable]
    rng.shuffle(eval_records)

    # ---------------------------------------------------------- training mix
    train_records, held = conflict.split_holdout(trainable, HELD_OUT_TYPE)
    log(f"  training conflict: {len(train_records)} usable, "
        f"{len(held)} dropped as held-out '{HELD_OUT_TYPE}'")

    # Every qid already spoken for: used as a training conflict item, reserved
    # for the eval, or dropped as the held-out type. Excluding the reserved ones
    # is the part that is easy to miss -- they live in the train split, so a
    # naive "anything from train" pool pulls an eval item straight back into
    # training. The overlap assert in check() caught exactly that.
    # The frozen eval on disk is the authority, not the one just recomputed.
    # Those two can differ -- a change anywhere in normalisation or conflict
    # construction shifts which items survive -- and when they do, filtering
    # against the recomputed set would let an item that IS in the frozen eval
    # walk straight into training. Read the file.
    frozen = frozen_eval_qids()
    if frozen:
        log(f"  frozen eval on disk holds {len(frozen)} qids, all excluded from training")

    spoken_for = ({r["qid"] for r in train_records}
                  | {r["qid"] for r in reserved}
                  | {r["qid"] for r in held}
                  | frozen)
    train_records = [r for r in train_records if r["qid"] not in frozen]

    train_answerable = [
        i for i in squad.items("train", "answerable") if i["qid"] not in spoken_for
    ]
    rng.shuffle(train_answerable)
    train_grounded = [as_eval_record(i, "grounded")
                      for i in train_answerable[:len(train_records)]]
    n_answer = len(train_records) + len(train_grounded)
    n_refuse = int(n_answer * REFUSE_RATIO)
    train_unans = [as_eval_record(i, "unanswerable")
                   for i in squad.items("train", "unanswerable")
                   if i["qid"] not in spoken_for][:n_refuse]
    if len(train_unans) < n_refuse:
        raise SystemExit(
            f"only {len(train_unans)} unanswerable train items available, "
            f"need {n_refuse} at REFUSE_RATIO={REFUSE_RATIO}")
    log(f"  mix balance: {n_answer} answer rows, {len(train_unans)} refuse rows "
        f"({n_answer / len(train_unans):.1f} : 1)")
    for record in train_unans:
        record["answer"] = None

    training = [fill(r) for r in train_records + train_grounded + train_unans]
    rng.shuffle(training)

    check(eval_records, training, log=log)

    if write:
        DATA.mkdir(parents=True, exist_ok=True)
        eval_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eval_records)
        digest = hashlib.sha256(eval_text.encode("utf-8")).hexdigest()

        # Rebuilding to change the TRAINING mix must never disturb the eval.
        # This guard was added *after* a rebuild silently replaced a 1,083-item
        # eval with a 1,093-item one. Nothing about that looked wrong: the build
        # printed a clean summary and re-locked the new file to itself. The four
        # arms already scored would simply have been measured against a
        # different exam, and no later output would have said so.
        #
        # The cause was not the mix change at all. `conflict.py` calls
        # `metrics.contains`, so fixing the possessive and en-dash bugs in
        # normalisation changed which conflict items pass their own checks --
        # a scoring fix quietly reshaped the dataset. That coupling is exactly
        # why the lock compares bytes rather than trusting that inputs held
        # still.
        lock_path = DATA / "eval.lock"
        if mix_only:
            # Rebuilding the mix alone. The eval file is not written at all --
            # not even to an identical value -- so there is no window in which
            # a crash could leave it half-replaced.
            atomic_write(DATA / "train_mix.jsonl",
                         "".join(json.dumps(r, ensure_ascii=False) + "\n"
                                 for r in training))
            log(f"  wrote data/train_mix.jsonl ({len(training)} items)")
            log("  data/eval.jsonl untouched")
            return eval_records, training
        if lock_path.exists():
            locked = lock_path.read_text(encoding="utf-8").strip()
            if digest != locked:
                raise SystemExit(
                    "this build would change the frozen eval.\n"
                    f"  would write: {digest}\n"
                    f"  locked:      {locked}\n"
                    f"  items:       {len(eval_records)}\n"
                    "Every arm already scored is measured against the locked "
                    "one, so the two could never be compared. Use "
                    "`--mix-only` to rebuild just the training mix. If the eval "
                    "genuinely must change, delete data/eval.lock, rebuild, and "
                    "re-run EVERY arm from scratch.")
            log(f"  eval unchanged, still {digest[:16]}...")
        atomic_write(DATA / "eval.jsonl", eval_text)
        atomic_write(lock_path, digest)
        atomic_write(DATA / "train_mix.jsonl",
                     "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in training))
        log(f"  wrote data/eval.jsonl ({len(eval_records)} items)")
        log(f"  wrote data/eval.lock  {digest[:16]}...")
        log(f"  wrote data/train_mix.jsonl ({len(training)} items)")

    return eval_records, training


def check(eval_records, training, log=print):
    """The invariants the whole study rests on. Asserted, never assumed."""
    eval_ids = {r["qid"] for r in eval_records}
    train_ids = {r["qid"] for r in training}

    assert len(eval_ids) == len(eval_records), "duplicate qid in the eval"
    overlap = eval_ids & train_ids
    assert not overlap, f"{len(overlap)} qids are in BOTH eval and training"

    # And against the eval actually on disk, which is what every arm was scored
    # on. The check above only compares training to the eval this run rebuilt;
    # if those two drifted apart, it would pass while the real leak went by.
    frozen = frozen_eval_qids()
    leaked = frozen & train_ids
    assert not leaked, (
        f"{len(leaked)} training rows are in the FROZEN eval on disk. "
        "Training on them would score the model on questions it was taught.")

    for record in eval_records:
        is_unans = record["slice"] == "unanswerable"
        assert (record["answer"] is None) == is_unans, (
            f"{record['qid']}: answer is None iff slice is unanswerable")
        if record["slice"] != "conflict":
            continue
        prompt = record["passage"] + " " + record["question"]
        # The invariant the headline number rests on. If the memorised answer
        # survives anywhere in the prompt, the item cannot catch anything.
        from metrics import contains
        assert not contains(prompt, record["memorised"]), (
            f"{record['qid']}: memorised answer {record['memorised']!r} leaked")
        assert contains(record["passage"], record["answer"]), (
            f"{record['qid']}: the passage does not state its own answer")

    held_in_training = [r for r in training if r.get("edit_type") == HELD_OUT_TYPE]
    assert not held_in_training, (
        f"{len(held_in_training)} held-out '{HELD_OUT_TYPE}' items reached training")

    types = Counter(r["edit_type"] for r in eval_records if r["slice"] == "conflict")
    positions = Counter(r["edit_pos"] for r in eval_records if r["slice"] == "conflict")
    assert len(types) >= 2, f"conflict slice is one type only: {dict(types)}"
    assert len(positions) == 3, f"conflict slice misses a position: {dict(positions)}"
    # A slice dominated by one cell measures pattern-spotting, not reading.
    biggest = max(positions.values()) / sum(positions.values())
    assert biggest < 0.55, f"positions too concentrated: {dict(positions)}"

    slices = Counter(r["slice"] for r in eval_records)
    log(f"  eval slices: {dict(slices)}")
    log(f"  conflict types: {dict(types)}   positions: {dict(positions)}")
    log(f"  checks passed: {len(eval_records)} eval, {len(training)} train, no overlap")


def demo():
    """Runs the real build without writing, so the invariants are exercised."""
    build(write=False, log=lambda *a: None)
    print("build self-check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        demo()
    elif "--mix-only" in sys.argv:
        print("rebuilding the training mix only; the frozen eval is not touched")
        build(write=True, mix_only=True)
    else:
        print("building the frozen eval and the training mix")
        build(write=True)
