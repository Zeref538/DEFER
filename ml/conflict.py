"""Build the conflict slice: passages edited so the true answer is no longer the
one the model memorised.

The trick, in one picture. The model knows the capital of France is Paris. Hand
it a passage saying the capital is Lyon and ask the question. There is now
exactly one right answer -- Lyon, the one in front of it -- and any model saying
Paris has been caught reading its own memory instead of the page. No judge
model, no human rater, no argument.

Two things make or break this:

1. **It only works on facts the model actually memorised.** An item built from a
   fact it never knew proves nothing, because it would have read the passage for
   that one anyway. So input here is already filtered by the closed-book probe.

2. **The replacement has to be plausible and same-type.** Swapping a person's
   name for a year produces a passage that reads as broken, and a model may
   refuse it for reasons that have nothing to do with trusting its memory. So
   substitutes are drawn from real answers to other questions of the same kind.

No GPU, no network. Run the self-check:  python ml/conflict.py
"""
from __future__ import annotations

import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import contains, normalize  # noqa: E402

# ------------------------------------------------------- typing the question

# The question word tells you what kind of thing the answer is, for free. This
# is why there is no named-entity model here: SQuAD questions already announce
# their own type, and a dependency that adds nothing is a permanent tax.
_TYPE_PATTERNS = (
    ("year", re.compile(r"\b(?:what|which)\s+year\b|^\s*(?:in\s+)?what\s+year\b", re.I)),
    ("number", re.compile(r"^\s*how\s+(?:many|much|long|old|far|tall|deep)\b", re.I)),
    ("person", re.compile(r"^\s*(?:who|whom|whose)\b", re.I)),
    # "what/which <place noun>" is included because "what is the capital of X"
    # is one of the most common shapes in the data and the noun removes all
    # ambiguity about the answer's type. Bare "what is X" stays untyped on
    # purpose -- guessing between a person, a place and a thing needs an
    # entity model, and a wrong guess produces a passage that reads as broken.
    ("place", re.compile(
        r"^\s*where\b"
        r"|\b(?:what|which)\s+(?:\w+\s+){0,2}"
        r"(?:capital|city|country|state|province|town|region|county|river|island|"
        r"mountain|continent|village|district)\b",
        re.I,
    )),
    ("year", re.compile(r"^\s*when\b", re.I)),  # last: "when" is only a year if it looks like one
)

_YEAR = re.compile(r"^\s*(?:1[0-9]{3}|20[0-9]{2})\s*$")
_NUMBER = re.compile(r"^\s*[\d][\d,]*(?:\.\d+)?\s*$")


def classify(question: str, gold: str) -> str | None:
    """Return person | place | year | number, or None when we should not guess.

    The shape of the answer has to agree with the question word. "When was it
    built? -- During the reign of Henry VIII" is a `when` question with a
    non-year answer, and forcing it into the year bucket would hand it a year
    substitute and produce nonsense. Dropping it is free; we have plenty.
    """
    gold = gold.strip()
    if not gold:
        return None
    for kind, pattern in _TYPE_PATTERNS:
        if not pattern.search(question):
            continue
        if kind == "year":
            return "year" if _YEAR.match(gold) else None
        if kind == "number":
            return "number" if _NUMBER.match(gold) else None
        # person / place must not be bare numbers
        return None if _NUMBER.match(gold) else kind
    return None


# ------------------------------------------------------------------ position

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def edit_position(passage: str, index: int) -> str:
    """Which third of the passage the edit landed in: first, middle or last.

    Recorded because position has to be *varied*. If every conflict fact sits in
    the last sentence, a model can score well by learning "trust the end of the
    passage" and the headline number stops measuring reading.
    """
    starts, cursor = [], 0
    for sentence in _SENTENCE.split(passage):
        starts.append(cursor)
        cursor += len(sentence) + 1
    if len(starts) < 2:
        return "first"
    hit = sum(1 for s in starts if s <= index) - 1
    third = len(starts) / 3
    if hit < third:
        return "first"
    return "middle" if hit < 2 * third else "last"


# -------------------------------------------------------------- construction

def _word_pattern(text: str) -> re.Pattern:
    """Match `text` only as a whole word, case-insensitively."""
    return re.compile(r"(?<![\w])" + re.escape(text) + r"(?![\w])", re.IGNORECASE)


def build(items, seed: int = 0, construction: str = "swap"):
    """items: dicts with qid, question, passage, gold. All already probe-confirmed
    as facts the base model knows closed-book.

    Returns (records, drops) where drops is a Counter of why items were rejected.
    The drop counts are returned rather than logged and forgotten: a generator
    quietly rejecting 90% of its input would otherwise be invisible, and the
    survivors would be a weird unrepresentative slice nobody chose.
    """
    rng = random.Random(seed)

    typed = []
    drops: Counter = Counter()
    for item in items:
        kind = classify(item["question"], item["gold"])
        if kind is None:
            drops["untyped_question"] += 1
            continue
        typed.append((kind, item))

    # The substitute pool for each type is the real answers to other questions of
    # that type. Free, plausible, and in the same register as what it replaces.
    pools: dict[str, list[str]] = {}
    for kind, item in typed:
        pools.setdefault(kind, []).append(item["gold"].strip())

    records = []
    for kind, item in typed:
        gold = item["gold"].strip()
        passage, question = item["passage"], item["question"]

        if not _word_pattern(gold).search(passage):
            drops["answer_not_in_passage"] += 1
            continue
        if contains(question, gold):
            # The question already gives the answer away, so the passage is not
            # what the model would be reading. Unwinnable as evidence.
            drops["answer_leaks_in_question"] += 1
            continue

        candidates = [
            c for c in pools[kind]
            if normalize(c) != normalize(gold)
            and not contains(passage, c)
            and not contains(question, c)
        ]
        if not candidates:
            drops["no_usable_substitute"] += 1
            continue
        substitute = rng.choice(candidates)

        match = _word_pattern(gold).search(passage)
        position = edit_position(passage, match.start())
        edited, n = _word_pattern(gold).subn(substitute, passage)

        # The guards that make the item mean something. Checked here, per record,
        # rather than trusted -- the whole headline number rests on them holding.
        if contains(edited, gold):
            drops["original_survived_edit"] += 1
            continue
        if not contains(edited, substitute):
            drops["substitute_missing"] += 1
            continue

        records.append({
            "qid": item["qid"],
            "slice": "conflict",
            "passage": edited,
            "question": question,
            "answer": substitute,     # what the passage now says
            "memorised": gold,        # what the model will want to say
            "edit_type": kind,
            "edit_pos": position,
            "construction": construction,
            "n_replacements": n,
            "source": item.get("source", "constructed"),
        })

    return records, drops


def variation_report(records):
    """How evenly the edits are spread. A concentrated slice is a broken slice."""
    return {
        "edit_type": Counter(r["edit_type"] for r in records),
        "edit_pos": Counter(r["edit_pos"] for r in records),
    }


def split_holdout(records, held_out_type: str = "year"):
    """Split into (trainable, held_out).

    One answer type is kept out of training entirely and reported as its own row.
    That is the test for the trap this whole file is designed around: a model can
    learn "follow the passage when it is a proper noun" and look like it learned
    "follow the passage". If the held-out type scores far worse, it learned the
    pattern, not the behaviour -- and that gap is a finding, not a footnote.
    """
    train = [r for r in records if r["edit_type"] != held_out_type]
    held = [r for r in records if r["edit_type"] == held_out_type]
    return train, held


def demo():
    """Self-check on hand-built items. No network, no dataset download."""
    items = [
        {"qid": "a1", "question": "What is the capital?",
         "passage": "France is large. The capital, Paris, sits on the Seine. Paris grew fast.",
         "gold": "Paris"},
        {"qid": "a2", "question": "Where was the treaty signed?",
         "passage": "Delegates gathered. The treaty was signed in Vienna that autumn.",
         "gold": "Vienna"},
        {"qid": "a3", "question": "Who wrote the report?",
         "passage": "The committee met. Hoover wrote the report. It was filed later.",
         "gold": "Hoover"},
        {"qid": "a4", "question": "Who led the delegation?",
         "passage": "Marshall led the delegation through a long winter of talks.",
         "gold": "Marshall"},
        {"qid": "a5", "question": "In what year was it founded?",
         "passage": "The society formed slowly. It was founded in 1834 by local traders.",
         "gold": "1834"},
        {"qid": "a6", "question": "What year did it close?",
         "passage": "Trade declined. The mill closed in 1902 after a long dispute.",
         "gold": "1902"},
        # dropped: 'when' question whose answer is not a year
        {"qid": "d1", "question": "When was it built?",
         "passage": "It was built during the reign of Henry VIII.",
         "gold": "during the reign of Henry VIII"},
        # dropped: the answer is not in the passage at all
        {"qid": "d2", "question": "Who signed it?",
         "passage": "The document was signed at dawn.", "gold": "Adams"},
        # dropped: the question hands over the answer
        {"qid": "d3", "question": "Was Berlin the capital, and which city was it?",
         "passage": "Berlin served as the capital.", "gold": "Berlin"},
    ]

    records, drops = build(items, seed=1)
    by_id = {r["qid"]: r for r in records}

    assert "d1" not in by_id and drops["untyped_question"] >= 1
    assert "d2" not in by_id and drops["answer_not_in_passage"] == 1
    assert "d3" not in by_id, "question leaking the answer must be dropped"

    paris = by_id["a1"]
    assert paris["memorised"] == "Paris"
    assert not contains(paris["passage"], "Paris"), "original answer survived"
    assert contains(paris["passage"], paris["answer"])
    assert paris["n_replacements"] == 2, "every mention must be replaced, not just the first"
    assert paris["edit_type"] == "place"

    year = by_id["a5"]
    assert year["edit_type"] == "year"
    assert _YEAR.match(year["answer"]), "a year must be replaced by a year"

    for r in records:
        assert normalize(r["answer"]) != normalize(r["memorised"])
        assert not contains(r["question"], r["memorised"])
        assert r["edit_pos"] in ("first", "middle", "last")

    # the whole point of the holdout
    train, held = split_holdout(records, held_out_type="year")
    assert held and all(r["edit_type"] == "year" for r in held)
    assert all(r["edit_type"] != "year" for r in train)
    assert not (set(r["qid"] for r in train) & set(r["qid"] for r in held))

    report = variation_report(records)
    assert len(report["edit_type"]) >= 2, "a single-type slice measures nothing"

    print(f"conflict self-check passed  ({len(records)} built, drops: {dict(drops)})")
    print(f"  types: {dict(report['edit_type'])}   positions: {dict(report['edit_pos'])}")


if __name__ == "__main__":
    demo()
