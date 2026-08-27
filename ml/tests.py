"""The runnable checks. No GPU, no network, no fixtures.

    python -m pytest ml/tests.py -q

Most of the logic is guarded by the `demo()` self-check inside each module, so
those are called here rather than copied -- one place to fix when a rule
changes. What lives directly in this file is the stuff a single module cannot
check about itself: whether a *built slice* is shaped the way the study needs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import conflict  # noqa: E402
import metrics  # noqa: E402


def test_metrics_self_check():
    metrics.demo()


def test_conflict_self_check():
    conflict.demo()


def test_edit_position_reaches_every_third():
    """A position bucket that can never be produced would silently bias the slice."""
    passage = "One alpha here. Two beta here. Three gamma here."
    seen = {conflict.edit_position(passage, passage.index(w))
            for w in ("alpha", "beta", "gamma")}
    assert seen == {"first", "middle", "last"}, seen


def test_short_passage_does_not_crash():
    assert conflict.edit_position("Only one sentence here.", 5) == "first"


def test_holdout_never_leaks_into_training():
    """The whole point of the held-out type: it must not be trainable on."""
    items = [
        {"qid": f"p{i}", "question": "Where was it signed?",
         "passage": f"Talks ran long. It was signed in {city} that year.",
         "gold": city}
        for i, city in enumerate(["Vienna", "Lisbon", "Oslo", "Prague"])
    ] + [
        {"qid": f"y{i}", "question": "In what year was it founded?",
         "passage": f"It grew slowly. The body was founded in {yr} by traders.",
         "gold": yr}
        for i, yr in enumerate(["1834", "1902", "1755"])
    ]
    records, _ = conflict.build(items, seed=3)
    train, held = conflict.split_holdout(records, held_out_type="year")

    assert held, "holdout must not be empty"
    train_ids = {r["qid"] for r in train}
    held_ids = {r["qid"] for r in held}
    assert not (train_ids & held_ids)
    assert all(r["edit_type"] != "year" for r in train)


def test_substitute_is_same_type_as_the_answer_it_replaces():
    """A year swapped for a person's name reads as broken text, and a model may
    refuse it for reasons that have nothing to do with trusting its memory."""
    items = [
        {"qid": f"y{i}", "question": "In what year did it open?",
         "passage": f"Work finished early. The hall opened in {yr} to great noise.",
         "gold": yr}
        for i, yr in enumerate(["1834", "1902", "1755", "1990"])
    ] + [
        {"qid": f"w{i}", "question": "Who led it?",
         "passage": f"The group met often. {name} led it for a decade.",
         "gold": name}
        for i, name in enumerate(["Hoover", "Marshall", "Attlee"])
    ]
    records, _ = conflict.build(items, seed=5)
    for r in records:
        if r["edit_type"] == "year":
            assert conflict._YEAR.match(r["answer"]), r
        if r["edit_type"] == "person":
            assert not conflict._YEAR.match(r["answer"]), r


def test_no_conflict_item_leaks_its_memorised_answer():
    """The invariant the headline number rests on: if the memorised answer is
    still somewhere in the prompt, the item cannot catch anything."""
    items = [
        {"qid": f"c{i}", "question": "Where is the capital?",
         "passage": f"The region is old. The capital, {city}, sits by the river. "
                    f"{city} has grown since.",
         "gold": city}
        for i, city in enumerate(["Paris", "Vienna", "Lisbon", "Oslo"])
    ]
    records, drops = conflict.build(items, seed=7)
    assert records, f"nothing built, drops={dict(drops)}"
    for r in records:
        prompt = r["passage"] + " " + r["question"]
        assert not metrics.contains(prompt, r["memorised"]), r
        assert metrics.contains(r["passage"], r["answer"]), r
        assert r["n_replacements"] >= 2, "both mentions should be replaced"


def test_absent_is_not_zero():
    """`if x:` treating a valid 0 as missing has cost time before."""
    s = metrics.summarise([{"slice": "grounded", "verdict": "followed"}])
    assert s["conflict_following"]["rate"] is None
    assert s["conflict_following"]["n"] == 0
    assert s["grounded_accuracy"]["rate"] == 1.0

    zero = metrics.summarise([{"slice": "conflict", "verdict": "from_memory"}])
    assert zero["conflict_following"]["rate"] == 0.0, "a measured 0 is a result"
    assert zero["conflict_following"]["rate"] is not None
