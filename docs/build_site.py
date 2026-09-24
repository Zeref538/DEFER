"""Generate docs/index.html from the real artifacts.

    python docs/build_site.py

Every rate, count and table on the page is read from results/ and runs/ at build
time. Nothing is transcribed by hand, so the page cannot drift from what actually
ran. Re-run this after any new result and the site updates itself.

Editorial choices (which cases the widget replays, what the prose says) live in
build_replay.py and the template. Measurements do not.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "index.html"
sys.path.insert(0, str(ROOT / "ml"))

# The arms the page shows, in the order it shows them, with labels a reader who
# has never heard of an "arm" can follow.
LABELS = {
    "base": "Llama 3.2 3B, plain prompt",
    "prompt": "the best prompt, no training",
    "defer_s0": "fine-tuned, 4:1 mix (seed 0)",
    "defer_s1": "fine-tuned, 4:1 mix (seed 1)",
    "deferb_s0": "fine-tuned, balanced (seed 0)",
    "deferb_s1": "fine-tuned, balanced (seed 1)",
    "deferb_s2": "fine-tuned, balanced (seed 2)",
    "deferb_s3": "fine-tuned, balanced (seed 3)",
}
# The published adapter, and so the arm every headline on the page refers to.
HEADLINE = "deferb_s0"
BALANCED = ["deferb_s0", "deferb_s1", "deferb_s2", "deferb_s3"]


def breakdowns(scores: dict) -> dict:
    """Count each arm's conflict verdicts, recomputed rather than transcribed.

    scores.json keeps rates, not the underlying verdict counts, and the page
    leads with a count ("0 of 483"), so the verdicts are recomputed here from the
    same generations the scorer read.
    """
    import metrics
    from generate import load_eval
    from score import load_arm, score_arm

    items, eval_sha = load_eval(ROOT)
    assert eval_sha == scores["eval_sha256"], (
        "results/scores.json was produced against a different evaluation set than "
        "the one on disk. Re-run ml/score.py before building the page."
    )
    out = {}
    for arm in LABELS:
        generations, _ = load_arm(arm, eval_sha, len(items), log=lambda *a: None)
        scored = score_arm(items, generations)
        counts = Counter(r["verdict"] for r in scored if r["slice"] == "conflict")
        out[arm] = {v: counts.get(v, 0) for v in metrics.VERDICTS}
    return out


def main():
    scores = json.loads((ROOT / "results" / "scores.json").read_text(encoding="utf-8"))
    probe = json.loads((ROOT / "results" / "probe_summary.json").read_text(encoding="utf-8"))
    replay = json.loads((ROOT / "docs" / "data" / "replay.json").read_text(encoding="utf-8"))

    data = {
        "sha": scores["eval_sha256"],
        "nEval": scores["n_eval"],
        "arms": scores["arms"],
        "breakdown": breakdowns(scores),
        "probe": {name: {"typed": s["typed"], "known": s["known"],
                         "known_rate": s["known_rate"]}
                  for name, s in probe["splits"].items()},
        "cases": replay["cases"],
        "labels": LABELS,
        "order": list(LABELS),
        "headline": HEADLINE,
    }

    html = (ROOT / "docs" / "template.html").read_text(encoding="utf-8")
    assert "/*%%DATA%%*/" in html, "template has no data slot"
    html = html.replace("/*%%DATA%%*/", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html, encoding="utf-8", newline="")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")

    check(html, data)


def check(html: str, data: dict):
    """Fail loudly rather than publish a number the results do not support.

    The prose quotes figures a reader cannot recompute while reading. Each one is
    recomputed here from the artifacts and matched against the string actually on
    the page, so a re-run with new results breaks the build instead of leaving
    stale numbers in the copy. This is the check that caught a wrong ratio on the
    LiitLLM page, which is why it is here too.
    """
    arms, bd = data["arms"], data["breakdown"]

    def r(arm, metric):
        return f"{arms[arm][metric]['rate'] * 100:.1f}%"

    claims = [
        (r("base", "conflict_following"), "base conflict following"),
        (r("prompt", "conflict_following"), "prompt conflict following"),
        (r(HEADLINE, "conflict_following"), "headline conflict following"),
        (r("defer_s0", "conflict_following"), "4:1 conflict following"),
        (r("defer_s0", "abstention_unanswerable"), "4:1 abstention"),
        (r("defer_s0", "over_abstention"), "4:1 over-abstention"),
        (r("defer_s1", "abstention_unanswerable"), "4:1 seed 1 abstention"),
        (r("defer_s1", "over_abstention"), "4:1 seed 1 over-abstention"),
        (r("prompt", "abstention_unanswerable"), "prompt abstention"),
        (r("prompt", "over_abstention"), "prompt over-abstention"),
        (r("deferb_s1", "abstention_unanswerable"), "balanced seed 1 abstention"),
        (r("deferb_s1", "over_abstention"), "balanced seed 1 over-abstention"),
        (r("prompt", "grounded_accuracy"), "prompt grounded accuracy"),
        (r(HEADLINE, "grounded_accuracy"), "headline grounded accuracy"),
    ]

    # Abstention is quoted as a range because four seeds of one recipe disagree by
    # more than the effect most of this page is about.
    absts = [arms[a]["abstention_unanswerable"]["rate"] for a in BALANCED]
    follows = [arms[a]["conflict_following"]["rate"] for a in BALANCED]
    claims += [
        (f"{(max(absts) - min(absts)) * 100:.1f} points", "abstention spread"),
        (f"{(max(follows) - min(follows)) * 100:.1f} points", "conflict spread"),
    ]
    for arm in BALANCED:
        claims.append((r(arm, "abstention_unanswerable"), f"{arm} abstention"))

    gen = arms["base"]["generalisation"]
    claims += [
        (f"{gen['held_out']['rate'] * 100:.1f}%", "base held-out rate"),
        (f"{gen['trained_types']['rate'] * 100:.1f}%", "base trained-types rate"),
    ]

    for split in ("dev", "train"):
        p = data["probe"][split]
        claims.append((f"{p['known_rate'] * 100:.1f}%", f"probe {split} known rate"))

    # The headline count: zero answers taken from memory, on every trained arm.
    claims.append((str(bd["base"]["from_memory"]), "base from-memory count"))
    assert all(bd[a]["from_memory"] == 0 for a in BALANCED), (
        "a balanced seed now answers from memory, so the page's 'zero on all four "
        "seeds' claim is false. Fix the copy before publishing."
    )

    n_conflict = arms[HEADLINE]["conflict_following"]["n"]
    claims.append((str(n_conflict), "conflict item count"))
    claims.append((f"{data['nEval']:,}", "evaluation set size"))
    claims.append((data["sha"], "eval sha256"))

    missing = [(t, why) for t, why in claims if t not in html]
    for text, why in missing:
        print(f"  MISSING: page does not state {why} = {text}")
    assert not missing, f"{len(missing)} claim(s) on the page do not match the results"

    # A rail number per section, and the copy rule John set for every case study.
    n_sections = html.count('<section class="panel-sec"')
    n_steps = html.count('<span class="step-no">')
    assert n_steps == n_sections + 1, (
        f"{n_sections} sections but {n_steps} step numbers (the hero has one too). "
        "The contents rail numbers each section, so these must line up."
    )
    # Checked against the template, not the built page: the template holds the
    # copy and the script, while the built page also carries quoted SQuAD
    # passages, and rewriting a source passage to satisfy a house style rule
    # would be falsifying evidence.
    template = (ROOT / "docs" / "template.html").read_text(encoding="utf-8")
    assert "—" not in template, "em dash in the copy, which the brand forbids"

    print(f"  checked: {len(claims)} claims, {n_sections} sections, no em dashes")
    print(f"  from memory: base {bd['base']['from_memory']}, "
          f"prompt {bd['prompt']['from_memory']}, "
          f"balanced {[bd[a]['from_memory'] for a in BALANCED]}")


if __name__ == "__main__":
    main()
