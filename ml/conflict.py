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
_PLACE_NOUN = re.compile(
    r"\b(?:what|which)\s+(?:\w+\s+){0,2}"
    r"(capital|city|country|state|province|town|region|county|river|island|"
    r"mountain|continent|village|district)\b",
    re.I,
)

_TYPE_PATTERNS = (
    ("year", re.compile(r"\b(?:what|which)\s+year\b|^\s*(?:in\s+)?what\s+year\b", re.I)),
    ("number", re.compile(r"^\s*how\s+(?:many|much|long|old|far|tall|deep)\b", re.I)),
    ("person", re.compile(r"^\s*(?:who|whom|whose)\b", re.I)),
    # Only "what/which <place noun>", never a bare "where".
    #
    # Measured on SQuAD 2.0 dev: bare "where" produced 224 of 384 place-typed
    # items, and its answers included "third", "Battle of Hastings" and
    # "between P and PSPACE". Those are not places. They went into the substitute
    # pool and came back out as replacements -- one run rewrote "Normandy, a
    # region in France" to "a region in Baldwin", which is not a conflict item,
    # it is a broken sentence a model might refuse for reasons that have nothing
    # to do with trusting its memory.
    #
    # Bare "what is X" stays untyped for the same reason: telling a person from a
    # place from a thing needs an entity model, and a wrong guess is worse than
    # a dropped item. We keep 20% of the data and that is plenty.
    ("place", _PLACE_NOUN),
    ("year", re.compile(r"^\s*when\b", re.I)),  # last: "when" is only a year if it looks like one
)

_YEAR = re.compile(r"^\s*(?:1[0-9]{3}|20[0-9]{2})\s*$")

# Smallest substitute pool we will draw from. Below this the same handful of
# replacements recur across many items, which is itself a learnable pattern.
MIN_POOL = 8
_NUMBER = re.compile(r"^\s*[\d][\d,]*(?:\.\d+)?\s*$")


def _looks_like_a_name(gold: str) -> bool:
    """Cheap proper-noun test: starts with a capital, and is short.

    This is what strains out "southern", "free", "third", "business" and
    "between P and PSPACE" -- all real SQuAD answers to questions whose wording
    promised a person or a place. Without it they end up in the substitute pool
    and get swapped into passages where they read as gibberish.
    """
    gold = gold.strip()
    return bool(gold) and gold[0].isupper() and len(gold.split()) <= 5


def analyse(question: str, gold: str):
    """Return (edit_type, pool_key), or None when we should not guess.

    `edit_type` is the coarse bucket that gets reported. `pool_key` is finer, and
    decides what a given answer may be replaced *by*. They differ for places on
    purpose: swapping a country for a city ("a region in Los Angeles") is as
    broken as swapping it for a person, so countries only ever become other
    countries. The question already names the noun, so this costs nothing.
    """
    gold = (gold or "").strip()
    if not gold:
        return None
    for kind, pattern in _TYPE_PATTERNS:
        match = pattern.search(question)
        if not match:
            continue
        if kind == "year":
            return ("year", "year") if _YEAR.match(gold) else None
        if kind == "number":
            return ("number", "number") if _NUMBER.match(gold) else None
        if _NUMBER.match(gold) or not _looks_like_a_name(gold):
            return None
        if kind == "place":
            noun = _PLACE_NOUN.search(question)
            return ("place", f"place:{noun.group(1).lower()}") if noun else None
        return ("person", "person")
    return None


def classify(question: str, gold: str) -> str | None:
    """The coarse type only. Thin wrapper so callers that do not care about the
    substitute pool stay readable."""
    found = analyse(question, gold)
    return found[0] if found else None


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

def _plausible(key: str, gold: str, candidate: str) -> bool:
    """Reject substitutes that are the right *type* but the wrong size or era.

    Same-type alone is not enough. Real output before this existed:

        "he led an army of 30,000 men"  ->  "an army of 2 men"
        "the city fell in 1082"         ->  "the city fell in 1916"

    The first is nonsense and the second is centuries out of period. A passage a
    reader would call obviously broken is a bad test item, because a model may
    balk at it for reasons that have nothing to do with trusting its memory.

    Nothing here understands the world -- it is digit counting and subtraction.
    Anachronistic *names* survive this (a Norse leader can still be renamed to a
    20th-century one) and that is a known limitation, written up in the README
    rather than papered over.
    """
    if key == "year":
        try:
            return abs(int(gold.strip()) - int(candidate.strip())) <= 60
        except ValueError:
            return True
    if key == "number":
        # A year-shaped candidate reads as a date, not a count. Real output:
        # "he led an army of 30,000 men" -> "an army of 1957 men". Digit length
        # alone let that through, because 1957 and 30000 are one digit apart.
        if _YEAR.match(candidate.strip()) and not _YEAR.match(gold.strip()):
            return False
        g = re.sub(r"\D", "", gold)
        c = re.sub(r"\D", "", candidate)
        # within one order of magnitude: 30,000 may become 4-, 5- or 6-digit
        return bool(g) and bool(c) and abs(len(g) - len(c)) <= 1
    return True


def balance(records, seed: int = 0, per_cell: int = None):
    """Level the slice so no (edit_type, edit_pos) cell dominates it.

    Measured on SQuAD 2.0 dev: 57% of edits land in the first third of the
    passage, 27% middle, 16% last -- because that is simply where SQuAD answers
    live. An evaluation set with that shape quietly rewards a model that skims
    the opening and stops, which is not the behaviour under test.

    Default cap is the median cell size, which flattens the big cells without
    shrinking everything to the size of the smallest.
    """
    rng = random.Random(seed)
    cells: dict = {}
    for r in records:
        cells.setdefault((r["edit_type"], r["edit_pos"]), []).append(r)
    if per_cell is None:
        sizes = sorted(len(v) for v in cells.values())
        per_cell = sizes[len(sizes) // 2]
    out = []
    for key in sorted(cells):
        group = list(cells[key])
        rng.shuffle(group)
        out.extend(group[:per_cell])
    return out


def _word_pattern(text: str) -> re.Pattern:
    """Match `text` only as a whole word, case-insensitively."""
    return re.compile(r"(?<![\w])" + re.escape(text) + r"(?![\w])", re.IGNORECASE)


def build(items, seed: int = 0, construction: str = "swap", min_pool: int = None):
    """items: dicts with qid, question, passage, gold. All already probe-confirmed
    as facts the base model knows closed-book.

    Returns (records, drops) where drops is a Counter of why items were rejected.
    The drop counts are returned rather than logged and forgotten: a generator
    quietly rejecting 90% of its input would otherwise be invisible, and the
    survivors would be a weird unrepresentative slice nobody chose.
    """
    rng = random.Random(seed)
    min_pool = MIN_POOL if min_pool is None else min_pool

    typed = []
    drops: Counter = Counter()
    for item in items:
        found = analyse(item["question"], item["gold"])
        if found is None:
            drops["untyped_question"] += 1
            continue
        typed.append((found[0], found[1], item))

    # The substitute pool for each key is the real answers to other questions of
    # that exact kind. Free, plausible, and in the same register as what it
    # replaces -- a country is only ever replaced by another country.
    pools: dict[str, set] = {}
    for _, key, item in typed:
        pools.setdefault(key, set()).add(item["gold"].strip())

    records = []
    for kind, key, item in typed:
        # A pool of two or three makes the same substitute appear over and over,
        # which is a pattern a model can learn instead of the behaviour.
        if len(pools[key]) < min_pool:
            drops[f"pool_too_small:{key}"] += 1
            continue
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
            c for c in pools[key]
            if normalize(c) != normalize(gold)
            and _plausible(key, gold, c)
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
        {"qid": "a2", "question": "What was the capital of the empire?",
         "passage": "Trade flourished. The capital, Vienna, drew merchants each spring.",
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
         "passage": "Trade declined. The mill closed in 1867 after a long dispute.",
         "gold": "1867"},
        # dropped: 'when' question whose answer is not a year
        {"qid": "d1", "question": "When was it built?",
         "passage": "It was built during the reign of Henry VIII.",
         "gold": "during the reign of Henry VIII"},
        # dropped: the answer is not in the passage at all
        {"qid": "d2", "question": "Who signed it?",
         "passage": "The document was signed at dawn.", "gold": "Adams"},
        # dropped: the question hands over the answer
        {"qid": "d3", "question": "What was the capital of Prussia, given Berlin held that role?",
         "passage": "Berlin served as the capital for many decades.", "gold": "Berlin"},
        # dropped: a bare 'where' is no longer typed at all -- see _TYPE_PATTERNS
        {"qid": "d4", "question": "Where did the monks flee to?",
         "passage": "The monks fled to southern Italy that winter.",
         "gold": "southern Italy"},
        # dropped: promised a person, delivered a lowercase fragment
        {"qid": "d5", "question": "Who benefits most?",
         "passage": "The scheme favoured the wealthy above all.", "gold": "the wealthy"},
    ]

    # min_pool is relaxed here only because this is nine hand-written items. The
    # real default (MIN_POOL) stays strict; see the constant for why.
    records, drops = build(items, seed=1, min_pool=2)
    by_id = {r["qid"]: r for r in records}

    assert "d1" not in by_id, "a 'when' with a non-year answer must be dropped"
    assert "d2" not in by_id and drops["answer_not_in_passage"] == 1
    assert "d3" not in by_id and drops["answer_leaks_in_question"] == 1
    assert "d4" not in by_id, "a bare 'where' is not reliably a place question"
    assert "d5" not in by_id, "'the wealthy' is not a name"
    assert drops["untyped_question"] >= 3

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

    # A capital must only ever be replaced by another capital. This is the bug
    # real SQuAD exposed: "Normandy, a region in France" became "a region in
    # Baldwin". Note d3 is dropped as a test item (its question leaks the answer)
    # yet "Berlin" still belongs in the pool -- it is a real capital, and a
    # rejected item can still contribute a good substitute.
    capitals = {r["answer"] for r in records if r["edit_type"] == "place"}
    assert capitals <= {"Paris", "Vienna", "Berlin"}, f"place pool leaked: {capitals}"
    people = {r["answer"] for r in records if r["edit_type"] == "person"}
    assert not (people & capitals), f"a person and a place shared a pool: {people & capitals}"

    report = variation_report(records)
    assert len(report["edit_type"]) >= 2, "a single-type slice measures nothing"

    print(f"conflict self-check passed  ({len(records)} built, drops: {dict(drops)})")
    print(f"  types: {dict(report['edit_type'])}   positions: {dict(report['edit_pos'])}")


if __name__ == "__main__":
    demo()
