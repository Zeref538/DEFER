"""Metrics for DEFER, plus the text normalisation the rest of the project shares.

No GPU, no model, no network. Every number this project publishes comes out of
this file, so it has to be runnable by anyone holding the committed generation
logs and a plain Python install. That is the whole difference between a result
and a claim.

Run the self-check:  python ml/metrics.py
"""
from __future__ import annotations

import random
import re
import string
import unicodedata

# --------------------------------------------------------------- normalising

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WS = re.compile(r"\s+")

# Apostrophes are deleted; every other punctuation mark becomes a space.
# That split matters. Deleting hyphens the way the plain SQuAD rule does turns
# "New-York" into "newyork", which then never matches "New York" and silently
# scores a correct answer as wrong. Deleting apostrophes keeps "don't" as one
# token instead of splitting it into "don t".
_DROP = str.maketrans("", "", "'’`")
_TO_SPACE = str.maketrans({c: " " for c in string.punctuation if c not in "'`"})

# Dotted acronyms collapse to one token *before* the dots become spaces.
# Without this, "U.S.A." splits to "u s a", the article rule then eats the lone
# "a", and the answer can never match "USA". Acronyms are common answers, so
# this was worth a line.
_ACRONYM = re.compile(r"\b(?:[a-z]\.){2,}")


def normalize(text: str) -> str:
    """Lowercase, drop accents and articles, split on punctuation, squash space.

    Crude on purpose: a crude rule a reader can re-run themselves beats a judge
    model they cannot.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = _ACRONYM.sub(lambda m: m.group(0).replace(".", ""), text)
    text = text.translate(_DROP).translate(_TO_SPACE)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def contains(haystack: str, needle: str) -> bool:
    """Whole-token containment.

    Token-wise rather than substring, so the answer "Lyon" does not count as
    found inside "Lyons Banking Group". Substring matching here would quietly
    inflate every number in the study.
    """
    if not needle:
        return False
    h = normalize(haystack).split()
    n = normalize(needle).split()
    if not n or len(n) > len(h):
        return False
    return any(h[i:i + len(n)] == n for i in range(len(h) - len(n) + 1))


# --------------------------------------------------------------- abstention

# Phrases that mean "the passage does not answer this". Deliberately a short,
# explicit list rather than anything clever: every entry is auditable, and a
# reader can see exactly what was counted as a refusal.
_ABSTAIN = re.compile(
    r"not (?:in|mentioned|stated|provided|specified|given|present|found|available)"
    r"|no (?:information|mention|answer|indication|reference)"
    r"|does\s?n\W?o?\W?t (?:say|state|mention|specify|provide|contain|include|indicate)"
    r"|do\s?n\W?o?\W?t (?:know|have)"
    r"|can\s?n\W?o?\W?t be (?:determined|answered|found|established)"
    r"|unable to (?:determine|answer|find)"
    r"|unanswerable"
    r"|insufficient (?:information|context|detail)",
    re.IGNORECASE,
)


def abstained(generation: str) -> bool:
    return bool(_ABSTAIN.search(generation))


# ------------------------------------------------------------------ verdicts

VERDICTS = ("followed", "from_memory", "abstained", "other")


def verdict(item: dict, generation: str) -> str:
    """Label one generation against its evaluation item.

    Order matters, and this order is a decision worth stating. Producing the
    passage's answer counts as *followed* even if the model hedged on the way
    there -- "the text doesn't really say, but it gives Lyon" read the passage,
    and scoring that as a refusal would manufacture over-abstention that is not
    there. Abstention is therefore checked last, not first.

    Saying both answers is scored `other`, not a win. It is evidence of neither
    behaviour, and counting it as followed would be the flattering choice.
    """
    answer = item.get("answer")
    memorised = item.get("memorised")

    hit_answer = answer is not None and contains(generation, answer)
    hit_memory = memorised is not None and contains(generation, memorised)

    if hit_answer and hit_memory:
        return "other"
    if hit_answer:
        return "followed"
    if hit_memory:
        return "from_memory"
    if abstained(generation):
        return "abstained"
    return "other"


# ------------------------------------------------------------------- scoring

# The four numbers. Reported together, always. A table carrying fewer than all
# four is a bug in whatever produced it, not a stylistic choice -- see PRD s6.
METRICS = {
    # name: (which items it is measured on, what counts as a hit)
    "grounded_accuracy": (
        lambda r: r["slice"] == "grounded",
        lambda r: r["verdict"] == "followed",
    ),
    "conflict_following": (
        lambda r: r["slice"] == "conflict",
        lambda r: r["verdict"] == "followed",
    ),
    "abstention_unanswerable": (
        lambda r: r["slice"] == "unanswerable",
        lambda r: r["verdict"] == "abstained",
    ),
    # Measured on every answerable item, conflict ones included. That is where a
    # naive fine-tune breaks first: taught to say "not in the documents", it
    # starts saying it about documents that plainly answer the question.
    "over_abstention": (
        lambda r: r["slice"] in ("grounded", "conflict"),
        lambda r: r["verdict"] == "abstained",
    ),
}


def bootstrap_ci(flags, iters=2000, seed=0, alpha=0.05):
    """Percentile bootstrap interval for the mean of a list of 0/1 flags.

    Resample the results you already have, with replacement, a couple of
    thousand times, and watch how far the average moves. That spread is how much
    the number would have wobbled had you drawn a different sample of the same
    size. Without it, "31%" reads as precise when it might really be 24-38%.
    """
    # ponytail: pure-python, ~2s at n=800 x 2000 iters. numpy only if it matters.
    if not flags:
        return (None, None)
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(
        sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    lo = means[int(alpha / 2 * iters)]
    hi = means[min(int((1 - alpha / 2) * iters), iters - 1)]
    return (lo, hi)


def summarise(records, seed=0):
    """records: eval items merged with their generation and verdict.

    Returns {metric: {"rate", "lo", "hi", "n"}}. A metric with no items gets
    rate None rather than 0.0 -- absent is not the same as zero, and a 0.0 here
    would be read as a measured floor that was never measured.
    """
    out = {}
    for name, (among, is_hit) in METRICS.items():
        picked = [r for r in records if among(r)]
        if not picked:
            out[name] = {"rate": None, "lo": None, "hi": None, "n": 0}
            continue
        flags = [1 if is_hit(r) else 0 for r in picked]
        lo, hi = bootstrap_ci(flags, seed=seed)
        out[name] = {
            "rate": sum(flags) / len(flags),
            "lo": lo,
            "hi": hi,
            "n": len(flags),
        }
    return out


def demo():
    """Self-check. Fails loudly if the labelling rules drift."""
    assert normalize("  The  Cafe-Noir! ") == "cafe noir"
    assert contains("the capital is Lyon today", "Lyon")
    assert not contains("Lyons Banking Group failed", "Lyon"), "substring leak"
    assert contains("Answer: New York City", "new york city")
    assert contains("founded in the U.S.A. in 1801", "USA"), "dotted acronym"
    assert normalize("New-York") == "new york", "hyphen must split, not delete"

    conflict = {"slice": "conflict", "answer": "Lyon", "memorised": "Paris"}
    assert verdict(conflict, "The capital is Lyon.") == "followed"
    assert verdict(conflict, "The capital is Paris.") == "from_memory"
    assert verdict(conflict, "Lyon, though usually Paris.") == "other", "said both"
    assert verdict(conflict, "That is not mentioned in the passage.") == "abstained"
    # hedged but correct still counts as reading the passage
    assert verdict(conflict, "The text does not really say, but Lyon.") == "followed"

    unans = {"slice": "unanswerable", "answer": None, "memorised": None}
    assert verdict(unans, "There is no information about that.") == "abstained"
    assert verdict(unans, "It was 1996.") == "other"
    assert abstained("The passage doesn't mention it."), "contraction form"

    recs = [
        {"slice": "conflict", "verdict": "followed"},
        {"slice": "conflict", "verdict": "from_memory"},
        {"slice": "grounded", "verdict": "abstained"},
        {"slice": "unanswerable", "verdict": "abstained"},
    ]
    s = summarise(recs)
    assert s["conflict_following"]["rate"] == 0.5, s
    assert s["conflict_following"]["n"] == 2
    assert s["over_abstention"]["n"] == 3, "conflict items count as answerable"
    assert abs(s["over_abstention"]["rate"] - 1 / 3) < 1e-9
    assert s["abstention_unanswerable"]["rate"] == 1.0
    lo, hi = s["conflict_following"]["lo"], s["conflict_following"]["hi"]
    assert lo is not None and lo <= 0.5 <= hi

    empty = summarise([])
    assert empty["grounded_accuracy"]["rate"] is None, "absent must not read as 0"
    print("metrics self-check passed")


if __name__ == "__main__":
    demo()
