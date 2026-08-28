"""Score the logged generations. Runs on a laptop, needs no GPU and no network.

This is the file that produces every number the study publishes, and it works
from `runs/*/generations.jsonl` alone -- committed text files anyone can read.
That is the difference between a result and a claim: someone who doubts the
headline can re-run this and get the same figure, or find where it breaks.

Three guards run before anything is counted:

1. the eval on disk still matches `data/eval.lock`
2. the arm answered *that* eval, not an older one (its `run.json` carries the
   hash it saw)
3. every item was answered -- a partial run is reported as partial, never
   silently scored as though the missing items did not exist

Run it:            python ml/score.py
One arm only:      python ml/score.py base
Self-check:        python ml/score.py --check
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics  # noqa: E402
from generate import load_eval  # noqa: E402
from runner import atomic_write  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

# Printed names. The stored keys stay machine-friendly; these are for humans.
LABELS = {
    "grounded_accuracy": "grounded accuracy",
    "conflict_following": "conflict following  <- headline",
    "abstention_unanswerable": "abstention (unanswerable)",
    "over_abstention": "over-abstention",
}

# Read as "higher is better" unless listed here. Over-abstention is the one
# metric where a bigger number is worse, and forgetting that is exactly how a
# model that learned to say "not in the documents" gets published as an
# improvement.
LOWER_IS_BETTER = {"over_abstention"}


def load_arm(name: str, eval_sha: str, n_expected: int, log=print):
    """Read one arm's generations, refusing anything that cannot be compared."""
    directory = RUNS / name
    path = directory / "generations.jsonl"
    if not path.exists():
        raise SystemExit(f"no generations for arm {name!r} at {path}")

    generations = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        generations[record["qid"]] = record["generation"]

    manifest_path = directory / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    saw = manifest.get("eval_sha256")
    if saw and saw != eval_sha:
        raise SystemExit(
            f"arm {name!r} answered a different evaluation set.\n"
            f"  it saw:  {saw}\n"
            f"  current: {eval_sha}\n"
            "Ranking these against each other would compare two studies. "
            "Re-run this arm against the current eval."
        )
    if not saw:
        log(f"  {name}: no run.json, so which eval it answered is unverifiable")

    if len(generations) < n_expected:
        # Reported, not fatal. A partial arm is still worth looking at while a
        # run is in flight -- it just must never be presented as a full result.
        log(f"  {name}: PARTIAL -- {len(generations)} of {n_expected} answered")
    return generations, manifest


def score_arm(items, generations):
    """Attach each generation to its item and label it."""
    scored = []
    for item in items:
        text = generations.get(item["qid"])
        if text is None:
            continue
        record = dict(item)
        record["generation"] = text
        record["verdict"] = metrics.verdict(item, text)
        scored.append(record)
    return scored


def trained_types(root: Path = ROOT):
    """Which conflict edit types the training mix actually contained.

    Read from the mix rather than hardcoded, so the held-out row stays honest if
    the mix is ever rebuilt with a different holdout. A constant here would
    quietly keep claiming "never seen in training" about a type that now is.
    """
    path = root / "data" / "train_mix.jsonl"
    if not path.exists():
        return None
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("slice") == "conflict" and record.get("edit_type"):
            seen.add(record["edit_type"])
    return seen


def held_out_split(scored, trained):
    """Conflict following, split by whether the edit type was ever trained on.

    This is the sharpest test in the study. A model can score well on conflict
    items by learning the *pattern* -- "when a city name looks swapped, use the
    one on the page" -- rather than the behaviour. An edit type it has never
    seen cannot be answered by pattern. If the two columns match, it learned the
    behaviour; if the held-out column collapses, it learned the trick.
    """
    if not trained:
        return None
    seen, unseen, kinds = [], [], set()
    for record in scored:
        if record["slice"] != "conflict":
            continue
        hit = 1 if record["verdict"] == "followed" else 0
        if record.get("edit_type") in trained:
            seen.append(hit)
        else:
            unseen.append(hit)
            kinds.add(record.get("edit_type"))
    if not unseen or not seen:
        return None
    out = {}
    for key, flags in (("trained_types", seen), ("held_out", unseen)):
        lo, hi = metrics.bootstrap_ci(flags)
        out[key] = {"rate": sum(flags) / len(flags), "lo": lo, "hi": hi,
                    "n": len(flags)}
    out["held_out_types"] = sorted(k for k in kinds if k)
    out["gap"] = out["held_out"]["rate"] - out["trained_types"]["rate"]
    return out


def format_arm(name, summary, scored, manifest, split=None):
    lines = [f"arm: {name}   ({len(scored)} items scored)"]
    model = manifest.get("model")
    if model:
        lines.append(f"  model: {model}   decoding: {manifest.get('decoding', '?')}")
    for key, label in LABELS.items():
        stat = summary[key]
        if stat["rate"] is None:
            lines.append(f"  {label:28} --      (no items)")
            continue
        arrow = "v" if key in LOWER_IS_BETTER else "^"
        lines.append(
            f"  {label:28} {stat['rate']:6.1%}  "
            f"[{stat['lo']:.1%}, {stat['hi']:.1%}]  n={stat['n']:<5} {arrow}")

    conflict = [r for r in scored if r["slice"] == "conflict"]
    if conflict:
        counts = Counter(r["verdict"] for r in conflict)
        parts = "  ".join(f"{v}={counts.get(v, 0)}" for v in metrics.VERDICTS)
        lines.append(f"  conflict slice breakdown:  {parts}")

    if split:
        kinds = ", ".join(split["held_out_types"])
        a, b = split["trained_types"], split["held_out"]
        lines.append("  generalisation -- edit types seen in training, "
                     f"against '{kinds}' which was never seen:")
        lines.append(f"    trained types        {a['rate']:6.1%}  "
                     f"[{a['lo']:.1%}, {a['hi']:.1%}]  n={a['n']}")
        lines.append(f"    HELD OUT             {b['rate']:6.1%}  "
                     f"[{b['lo']:.1%}, {b['hi']:.1%}]  n={b['n']}   "
                     f"gap {split['gap'] * 100:+.1f}pt")
    return "\n".join(lines)


def compare(summaries):
    """One row per arm, so the gap between them is readable at a glance."""
    if len(summaries) < 2:
        return ""
    width = max(len(a) for a in summaries)
    header = f"{'arm':<{width}}  " + "  ".join(f"{k[:11]:>11}" for k in LABELS)
    rows = [header, "-" * len(header)]
    for arm, summary in summaries.items():
        cells = []
        for key in LABELS:
            rate = summary[key]["rate"]
            cells.append(f"{rate:>10.1%} " if rate is not None else f"{'--':>11}")
        rows.append(f"{arm:<{width}}  " + "  ".join(cells))
    rows.append("")
    rows.append("over-abstention is the only column where lower is better.")
    return "\n".join(rows)


def main(arms=None, log=print):
    items, eval_sha = load_eval(ROOT)
    log(f"eval: {len(items)} items, sha256 {eval_sha[:16]}... matches the lock")

    if not arms:
        arms = sorted(d.name for d in RUNS.iterdir()
                      if (d / "generations.jsonl").exists()) if RUNS.exists() else []
    if not arms:
        raise SystemExit(
            "no arms to score. Run an arm first -- see docs/APP_FLOW.md.")

    trained = trained_types(ROOT)
    if trained:
        log(f"training mix covered conflict types: {sorted(trained)}")

    blocks, summaries, splits = [], {}, {}
    for arm in arms:
        generations, manifest = load_arm(arm, eval_sha, len(items), log=log)
        scored = score_arm(items, generations)
        summary = metrics.summarise(scored)
        split = held_out_split(scored, trained)
        if split:
            splits[arm] = split
            summary["generalisation"] = split
        summaries[arm] = summary
        blocks.append(format_arm(arm, summary, scored, manifest, split))

    text = "\n\n".join(blocks)
    table = compare(summaries)
    if table:
        text += "\n\n" + table
    if len(splits) > 1:
        width = max(len(a) for a in splits)
        kinds = ", ".join(next(iter(splits.values()))["held_out_types"])
        rows = ["", f"conflict following, split by whether the edit type was "
                    f"ever trained on ('{kinds}' never was)",
                f"{'arm':<{width}}  {'trained':>9}  {'held out':>9}  {'gap':>8}",
                "-" * (width + 32)]
        for arm, s in splits.items():
            rows.append(f"{arm:<{width}}  {s['trained_types']['rate']:>8.1%}  "
                        f"{s['held_out']['rate']:>8.1%}  "
                        f"{s['gap'] * 100:>+7.1f}pt")
        text += "\n" + "\n".join(rows)
    log("")
    log(text)

    RESULTS.mkdir(parents=True, exist_ok=True)
    atomic_write(RESULTS / "scores.txt", text + "\n")
    atomic_write(RESULTS / "scores.json", json.dumps(
        {"eval_sha256": eval_sha, "n_eval": len(items), "arms": summaries},
        indent=2) + "\n")
    log("")
    log("wrote results/scores.txt and results/scores.json")
    return summaries


def demo():
    """Self-check on a hand-built eval where every right answer is known."""
    import hashlib
    import shutil
    import tempfile

    work = Path(tempfile.mkdtemp())
    try:
        items = [
            {"qid": "c1", "slice": "conflict", "answer": "Lyon", "memorised": "Paris",
             "passage": "The capital, Lyon.", "question": "Capital?"},
            {"qid": "c2", "slice": "conflict", "answer": "Lyon", "memorised": "Paris",
             "passage": "The capital, Lyon.", "question": "Capital?"},
            {"qid": "g1", "slice": "grounded", "answer": "Tokyo", "memorised": None,
             "passage": "Tokyo is the capital.", "question": "Capital?"},
            {"qid": "u1", "slice": "unanswerable", "answer": None, "memorised": None,
             "passage": "Nothing relevant.", "question": "Who?"},
        ]
        raw = "".join(json.dumps(i) + "\n" for i in items).encode("utf-8")
        (work / "data").mkdir()
        (work / "data" / "eval.jsonl").write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        (work / "data" / "eval.lock").write_text(sha, encoding="utf-8")

        answers = {
            "c1": "Lyon",                       # followed
            "c2": "Paris",                      # from memory
            "g1": "Tokyo",                      # grounded hit
            "u1": "That is not in the passage.",  # correct abstention
        }
        scored = score_arm(items, answers)
        summary = metrics.summarise(scored)
        assert summary["conflict_following"]["rate"] == 0.5, summary
        assert summary["grounded_accuracy"]["rate"] == 1.0
        assert summary["abstention_unanswerable"]["rate"] == 1.0
        assert summary["over_abstention"]["rate"] == 0.0
        assert summary["over_abstention"]["n"] == 3, "conflict items are answerable"

        # a model that refuses everything must look good on one metric and
        # terrible on the one built to catch exactly that
        refuser = {q: "That is not in the passage." for q in answers}
        bad = metrics.summarise(score_arm(items, refuser))
        assert bad["abstention_unanswerable"]["rate"] == 1.0
        assert bad["over_abstention"]["rate"] == 1.0, (
            "a total refuser must max out over-abstention, or the metric is "
            "not doing its job")
        assert bad["conflict_following"]["rate"] == 0.0

        # a partial arm scores only what it answered, and says so
        partial = metrics.summarise(score_arm(items, {"c1": "Lyon"}))
        assert partial["conflict_following"]["n"] == 1
        assert partial["grounded_accuracy"]["rate"] is None, "absent is not zero"

        # an arm that answered a different eval must be refused, not ranked
        runs = work / "runs" / "old"
        runs.mkdir(parents=True)
        (runs / "generations.jsonl").write_text('{"qid": "c1", "generation": "Lyon"}\n',
                                                encoding="utf-8", newline="\n")
        (runs / "run.json").write_text(json.dumps({"eval_sha256": "0" * 64}),
                                       encoding="utf-8", newline="\n")
        global RUNS
        saved, RUNS = RUNS, work / "runs"
        try:
            load_arm("old", sha, 4, log=lambda *a: None)
            raise AssertionError("a stale eval hash must stop the score")
        except SystemExit as exc:
            assert "different evaluation set" in str(exc), exc
        finally:
            RUNS = saved

        print("score self-check passed")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        demo()
    else:
        main([a for a in sys.argv[1:] if not a.startswith("-")])
