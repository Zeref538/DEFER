"""Read the Gate D log and say whether the drift it measured is real.

Written as a file rather than a shell one-liner because the first attempt at
this analysis was run through PowerShell, which interpolated the `$` in the
regex and turned an end-of-string anchor into a literal dollar sign. Every reply
then looked truncated, including 40-word ones that plainly were not.

Run it:  python ml/analyse_drift.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics  # noqa: E402
import rules  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "runs" / "rule_drift.jsonl"

# A reply ending without any sentence-closing punctuation was almost certainly
# cut off by max_new_tokens rather than finished. That distinction decides
# whether "did not end with a question mark" is disobedience or a measurement
# artifact -- and the whole Arm B decision rests on it.
_FINISHED = re.compile(r"[.!?\"'’”)\]]\s*$")


def truncated(text: str) -> bool:
    return not _FINISHED.search(text.strip())


def load():
    if not LOG.exists():
        raise SystemExit(f"missing {LOG}. Run the Gate D kernel first.")
    records = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [{"rule": r["rule"], "condition": r["condition"], "turn": t["turn"],
             "ok": t["ok"], "answer": t["answer"]}
            for r in records for t in r["turns"]]


def main():
    flat = load()
    print(f"{len(flat)} generations from {len(flat) // 10} conversations\n")

    print("truncation, per rule -- a cut-off reply cannot end in a question mark")
    for rule in rules.RULES:
        rows = [x for x in flat if x["rule"] == rule]
        cut = sum(truncated(x["answer"]) for x in rows)
        words = sorted(len(x["answer"].split()) for x in rows)
        print(f"  {rule:14} {cut:3}/{len(rows)} cut off   median {words[len(words) // 2]:3} words")

    print("\ncompliance curve, rule stated once")
    header = "  " + f"{'rule':14}" + "".join(f"T{t:<6}" for t in (1, 2, 3, 5, 8, 10))
    print(header)
    for rule in rules.RULES:
        cells = []
        for t in (1, 2, 3, 5, 8, 10):
            flags = [x["ok"] for x in flat
                     if x["rule"] == rule and x["condition"] == "once" and x["turn"] == t]
            cells.append(f"{sum(flags) / len(flags):.0%}")
        print("  " + f"{rule:14}" + "".join(f"{c:<7}" for c in cells))

    print("\nend_question failures, split by cause")
    fails = [x for x in flat if x["rule"] == "end_question" and not x["ok"]]
    cut = [x for x in fails if truncated(x["answer"])]
    print(f"  {len(fails)} failures")
    print(f"    cut off by the token limit: {len(cut)} ({len(cut) / len(fails):.0%})")
    print(f"    finished, but no '?':       {len(fails) - len(cut)}")

    print("\nend_question again, counting only replies that actually finished")
    print("  (a reply cut off mid-sentence cannot end in '?' no matter how "
          "obedient the model is)")
    for condition in ("once", "reinjected"):
        cells = []
        for t in (1, 2, 3, 5, 8, 10):
            rows = [x for x in flat
                    if x["rule"] == "end_question" and x["condition"] == condition
                    and x["turn"] == t and not truncated(x["answer"])]
            cells.append(f"{sum(x['ok'] for x in rows) / len(rows):.0%}({len(rows)})"
                         if rows else "--")
        print(f"  {condition:11}" + "".join(f"{c:<10}" for c in cells))

    print("\nthe two rules that a truncated reply cannot fake:")
    for rule in ("no_bullets", "word_cap"):
        for condition in ("once", "reinjected"):
            first = [x["ok"] for x in flat
                     if x["rule"] == rule and x["condition"] == condition and x["turn"] == 1]
            last = [x["ok"] for x in flat
                    if x["rule"] == rule and x["condition"] == condition and x["turn"] == 10]
            a, b = sum(first) / len(first), sum(last) / len(last)
            lo, hi = metrics.bootstrap_ci([1 if x else 0 for x in last])
            print(f"  {rule:12} {condition:11} turn 1 {a:6.1%} -> turn 10 {b:6.1%} "
                  f"[{lo:.0%}-{hi:.0%}]   drift {(b - a) * 100:+.1f}pt")


if __name__ == "__main__":
    main()
