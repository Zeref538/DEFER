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

import build  # noqa: E402
import conflict  # noqa: E402
import generate  # noqa: E402
import metrics  # noqa: E402
import kaggle_env  # noqa: E402
import probe  # noqa: E402
import runner  # noqa: E402
import score  # noqa: E402
import squad  # noqa: E402


def test_metrics_self_check():
    metrics.demo()


def test_conflict_self_check():
    conflict.demo()


def test_build_self_check():
    """Exercises the real assembly, including the eval/training overlap check
    that caught a reserved eval item being pulled back into training."""
    cached = Path(__file__).resolve().parent.parent / "runs" / "probe" / "probe_dev.jsonl"
    if not cached.exists():
        import pytest
        pytest.skip("probe output not present; run the phase 0 kernel first")
    build.demo()


def test_eval_lock_matches_the_eval_file():
    """The guard against quietly nudging the eval until the number improves."""
    import hashlib
    data = Path(__file__).resolve().parent.parent / "data"
    if not (data / "eval.lock").exists():
        import pytest
        pytest.skip("eval not built yet; run `python ml/build.py`")
    digest = hashlib.sha256((data / "eval.jsonl").read_bytes()).hexdigest()
    assert digest == (data / "eval.lock").read_text(encoding="utf-8").strip(), (
        "data/eval.jsonl does not match data/eval.lock -- the frozen eval "
        "changed after it was locked, so no result measured on it is comparable")


def test_runner_self_check():
    runner.demo()


def test_probe_self_check():
    probe.demo()


def test_squad_self_check():
    """Skipped rather than failed when the split is not cached -- a test that
    needs a 4 MB download to pass is a test people stop running."""
    cached = Path(__file__).resolve().parent.parent / "data" / "squad2_dev.json"
    if not cached.exists():
        import pytest
        pytest.skip("dev split not cached; run `python ml/squad.py` once")
    squad.demo()


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
        {"qid": f"p{i}", "question": "What city was it signed in?",
         "passage": f"Talks ran long. It was signed in {city} that year.",
         "gold": city}
        for i, city in enumerate(["Vienna", "Lisbon", "Oslo", "Prague"])
    ] + [
        {"qid": f"y{i}", "question": "In what year was it founded?",
         "passage": f"It grew slowly. The body was founded in {yr} by traders.",
         "gold": yr}
        for i, yr in enumerate(["1834", "1867", "1812"])
    ]
    records, _ = conflict.build(items, seed=3, min_pool=2)
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
        for i, yr in enumerate(["1834", "1867", "1812", "1880"])
    ] + [
        {"qid": f"w{i}", "question": "Who led it?",
         "passage": f"The group met often. {name} led it for a decade.",
         "gold": name}
        for i, name in enumerate(["Hoover", "Marshall", "Attlee"])
    ]
    records, _ = conflict.build(items, seed=5, min_pool=2)
    assert records, "fixture built nothing - the loop below would pass vacuously"
    for r in records:
        if r["edit_type"] == "year":
            assert conflict._YEAR.match(r["answer"]), r
        if r["edit_type"] == "person":
            assert not conflict._YEAR.match(r["answer"]), r


def test_no_conflict_item_leaks_its_memorised_answer():
    """The invariant the headline number rests on: if the memorised answer is
    still somewhere in the prompt, the item cannot catch anything."""
    items = [
        {"qid": f"c{i}", "question": "What is the capital?",
         "passage": f"The region is old. The capital, {city}, sits by the river. "
                    f"{city} has grown since.",
         "gold": city}
        for i, city in enumerate(["Paris", "Vienna", "Lisbon", "Oslo"])
    ]
    records, drops = conflict.build(items, seed=7, min_pool=2)
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


def test_generate_self_check():
    generate.demo()


def test_score_self_check():
    score.demo()


def test_kaggle_env_self_check():
    kaggle_env.demo()


def test_publish_ships_every_module_a_kaggle_run_imports():
    """The dataset list is hand-written, so it can silently fall behind ml/.

    A missing module does not fail on the laptop -- it fails nine minutes into a
    GPU session with an ImportError, which is the expensive way to find out.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent / "kernels"))
    import publish

    shipped = {Path(rel).name for rel in publish.CODE}
    on_disk = {p.name for p in (Path(__file__).resolve().parent).glob("*.py")}
    missing = on_disk - shipped
    assert not missing, f"ml/ has modules the dataset never ships: {sorted(missing)}"


def test_the_two_baseline_arms_differ_only_in_grounding():
    """The `prompt` arm is the control the fine-tune has to beat.

    If the base arm quietly picked up a grounding hint, the gap between them
    would shrink for a reason that has nothing to do with the model, and the
    fine-tune would look better than it is by comparison.
    """
    assert "passage" not in generate.ARMS["base"].lower()
    assert "only the passage" in generate.ARMS["prompt"].lower()
    for name, system in generate.ARMS.items():
        assert "say so" in system, f"{name} must permit abstention"
        assert "explanation" in system, f"{name} must ask for a short answer"


def test_scoring_a_stale_arm_is_refused_not_ranked():
    """Two arms answering two different evals must never end up in one table."""
    import json
    import tempfile

    work = Path(tempfile.mkdtemp())
    arm = work / "runs" / "ghost"
    arm.mkdir(parents=True)
    runner.atomic_write(arm / "generations.jsonl",
                        '{"qid": "a", "generation": "x"}' + "\n")
    runner.atomic_write(arm / "run.json",
                        json.dumps({"eval_sha256": "f" * 64}))
    saved, score.RUNS = score.RUNS, work / "runs"
    try:
        score.load_arm("ghost", "a" * 64, 1, log=lambda *a: None)
        raise AssertionError("a stale eval hash must stop the score")
    except SystemExit as exc:
        assert "different evaluation set" in str(exc)
    finally:
        score.RUNS = saved
