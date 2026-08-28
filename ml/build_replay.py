"""Pick the cases the demo replays, straight from the committed generation logs.

The demo never runs a model. Every answer it shows is a real logged generation,
which is the entire reason it cannot drift from the study -- a live demo can
disagree with the paper, and then one of them is lying.

Selection is deterministic and it is *not* a highlight reel. Three of the four
categories are wins and the fourth is a failure, chosen by the same rule as the
rest and shown with the same prominence. A demo that only shows wins tells the
reader nothing about when to distrust the thing.

Run it:  python ml/build_replay.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics  # noqa: E402
from generate import load_eval  # noqa: E402
from runner import atomic_write  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "data"

# The arms the page shows, in the order it shows them, with the labels a reader
# who has never heard of an "arm" can follow.
SHOWN = [
    ("base", "Llama 3.2 3B, plain prompt"),
    ("prompt", "the best prompt, no training"),
    ("defer_s0", "fine-tuned, 4:1 mix"),
    ("deferb_s0", "fine-tuned, balanced mix"),
]
PER_CATEGORY = 4
MAX_PASSAGE = 900       # characters; longer passages are trimmed around the edit


def load_arm(name):
    path = ROOT / "runs" / name / "generations.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}. Score the arms first.")
    return {json.loads(l)["qid"]: json.loads(l)["generation"]
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def trim(passage: str, keep: str) -> str:
    """Shorten a long passage without cutting out the sentence that matters."""
    if len(passage) <= MAX_PASSAGE:
        return passage
    at = passage.find(keep)
    if at < 0:
        return passage[:MAX_PASSAGE] + "..."
    start = max(0, at - MAX_PASSAGE // 2)
    end = min(len(passage), at + MAX_PASSAGE // 2)
    out = passage[start:end]
    return ("..." if start else "") + out + ("..." if end < len(passage) else "")


def categorise(item, verdicts):
    """Which story does this item tell? None means it tells none of them."""
    base, prompt, old, new = (verdicts[a] for a, _ in SHOWN)

    if item["slice"] == "conflict":
        # The headline: the untrained model reached for its memory, the trained
        # one read the page.
        if base == "from_memory" and new == "followed":
            return "caught"
    if item["slice"] == "unanswerable":
        # The regression the four-metric rule exposed, and its fix, in one item.
        if old != "abstained" and new == "abstained":
            return "admitted"
    if item["slice"] == "grounded":
        # An ordinary question, to show nothing was broken to get the rest.
        if base == "followed" and new == "followed":
            return "unchanged"
    # The honest failures. Shown, not hidden.
    if new not in ("followed", "abstained"):
        return "missed"
    return None


def build(log=print):
    items, eval_sha = load_eval(ROOT)
    arms = {name: load_arm(name) for name, _ in SHOWN}

    buckets = {"caught": [], "admitted": [], "unchanged": [], "missed": []}
    for item in items:
        if not all(item["qid"] in arms[name] for name, _ in SHOWN):
            continue
        answers = {name: arms[name][item["qid"]] for name, _ in SHOWN}
        verdicts = {name: metrics.verdict(item, text)
                    for name, text in answers.items()}
        kind = categorise(item, verdicts)
        if kind is None:
            continue
        keep = item["answer"] or item["question"]
        buckets[kind].append({
            "qid": item["qid"],
            "slice": item["slice"],
            "kind": kind,
            "passage": trim(item["passage"], keep),
            "question": item["question"],
            "answer": item["answer"],
            "memorised": item["memorised"],
            "edit_type": item["edit_type"],
            "answers": [
                {"arm": name, "label": label, "text": answers[name],
                 "verdict": verdicts[name]}
                for name, label in SHOWN
            ],
        })

    # A real failure teaches more than a near miss. The scorer needs the whole
    # gold answer present, and SQuAD spans carry filler -- gold "in protest
    # against the occupation" against a model saying "protest against the
    # occupation" is a scoring limit, not a model that got it wrong. Show the
    # genuine failures first: memory beating the page, or an invented answer to
    # a question the passage cannot answer.
    # There are no `from_memory` failures left to show -- the trained model made
    # zero of them -- so the ranking falls through to slice. An invented answer
    # to an unanswerable question is a real failure a reader should see. A
    # grounded near-miss is usually the scorer, not the model.
    def failure_first(case):
        final = case["answers"][-1]["verdict"]
        return (
            {"from_memory": 0, "other": 1}.get(final, 2),
            {"unanswerable": 0, "conflict": 1, "grounded": 2}[case["slice"]],
        )

    buckets["missed"].sort(key=failure_first)

    cases = []
    for kind in ("caught", "admitted", "unchanged", "missed"):
        picked = buckets[kind][:PER_CATEGORY]
        cases.extend(picked)
        log(f"  {kind:10} {len(picked)} of {PER_CATEGORY} "
            f"(from {len(buckets[kind])} candidates)")
        if not picked:
            log(f"  WARNING: no '{kind}' case found. The page will be missing a "
                "category it is written to explain.")

    scores = json.loads((ROOT / "results" / "scores.json").read_text(encoding="utf-8"))
    payload = {
        "eval_sha256": eval_sha,
        "n_eval": len(items),
        "arms": [{"arm": a, "label": l} for a, l in SHOWN],
        "scores": scores["arms"],
        "cases": cases,
    }
    WEB.mkdir(parents=True, exist_ok=True)
    atomic_write(WEB / "replay.json", json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    log(f"  wrote web/data/replay.json ({len(cases)} cases, "
        f"{(WEB / 'replay.json').stat().st_size / 1024:.0f} KB)")
    return payload


def demo():
    """Self-check on the categoriser. No files needed."""
    conflict = {"slice": "conflict", "answer": "Lyon", "memorised": "Paris"}
    caught = {"base": "from_memory", "prompt": "from_memory",
              "defer_s0": "followed", "deferb_s0": "followed"}
    assert categorise(conflict, caught) == "caught"

    unans = {"slice": "unanswerable", "answer": None, "memorised": None}
    fixed = {"base": "other", "prompt": "other",
             "defer_s0": "other", "deferb_s0": "abstained"}
    assert categorise(unans, fixed) == "admitted"

    # a still-wrong item must be reported as a miss, never dropped
    still_wrong = {"base": "from_memory", "prompt": "from_memory",
                   "defer_s0": "other", "deferb_s0": "from_memory"}
    assert categorise(conflict, still_wrong) == "missed"

    grounded = {"slice": "grounded", "answer": "Tokyo", "memorised": None}
    fine = {"base": "followed", "prompt": "followed",
            "defer_s0": "followed", "deferb_s0": "followed"}
    assert categorise(grounded, fine) == "unchanged"

    # trimming must never remove the sentence the answer lives in
    long_passage = ("filler. " * 200) + "The capital is Lyon today. " + ("more. " * 200)
    assert "Lyon" in trim(long_passage, "Lyon")
    assert len(trim(long_passage, "Lyon")) < len(long_passage)
    assert trim("short", "short") == "short"
    print("build_replay self-check passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        demo()
    else:
        print("selecting replay cases from the committed logs")
        build()
