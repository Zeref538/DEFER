"""Arm B's gate: does a standing instruction survive ten turns of conversation?

The second failure this project is about. You tell a model "answer in under 40
words, never use bullet points", it obeys for a while, and by turn eight it is
writing bulleted essays. Same shape as the document bug -- something you supplied
loses to something the model would rather do.

**Every rule here is checkable by a script.** No judge model, no rater, no
"be more helpful". A rule that needs an opinion to score cannot produce a number
anyone can re-derive, and this whole project is built on numbers anyone can
re-derive.

The gate exists because there is a free fix that might already win. Real chat
software re-sends the system prompt on every turn, so a large part of what people
call rule drift is fixed before any training happens. If re-injection closes the
gap, Arm B is cut and these numbers become its section -- see ADR 0003, written
before any of this was measured.

Run the self-check:  python ml/rules.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# A bullet is a line that opens with a marker. Checked per line rather than
# anywhere in the text, so a sentence containing a hyphen is not a violation.
_BULLET = re.compile(r"^\s*(?:[-*+•‣◦⁃]|\d+[.)])\s+", re.MULTILINE)
_WORD = re.compile(r"\S+")

WORD_CAP = 40


def no_bullets(text: str) -> bool:
    return not _BULLET.search(text)


def under_word_cap(text: str) -> bool:
    return len(_WORD.findall(text)) <= WORD_CAP


def ends_with_question(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped.endswith("?")


# name -> (instruction the model is given, checker)
#
# Deliberately three rules of different kinds: a formatting ban, a length cap and
# a required ending. A single rule would measure one habit; three measure whether
# instructions in general survive, which is the actual question.
RULES = {
    "no_bullets": (
        "Never use bullet points, numbered lists, or any kind of list formatting. "
        "Write in plain prose sentences only.",
        no_bullets,
    ),
    "word_cap": (
        f"Every reply must be {WORD_CAP} words or fewer. Never exceed it.",
        under_word_cap,
    ),
    "end_question": (
        "End every single reply with a question mark. The last character of your "
        "reply must always be '?'.",
        ends_with_question,
    ),
}

# Ten user turns. Fixed, ordinary, and deliberately the sort of thing that
# invites a long bulleted answer -- a rule is only tested when something pulls
# against it.
TURNS = [
    "What is photosynthesis?",
    "Can you explain how a bicycle stays upright?",
    "What are the main causes of inflation?",
    "How does a refrigerator work?",
    "What should I consider when adopting a dog?",
    "Explain the difference between weather and climate.",
    "What makes sourdough bread different from regular bread?",
    "How do noise-cancelling headphones work?",
    "What are the steps to change a flat tyre?",
    "Summarise how vaccines train the immune system.",
]

CHECKPOINTS = (1, 3, 5, 10)      # turns reported; the shape is the finding


def system_message(rule_name: str) -> str:
    instruction, _ = RULES[rule_name]
    return "You are a helpful assistant. " + instruction


def build_messages(rule_name: str, history, reinject: bool):
    """The conversation so far, ready for the chat template.

    history is a list of (user, assistant) pairs already exchanged.

    `reinject=True` is the free fix being tested: the rule is restated as a
    system message before every user turn, which is what real chat frameworks do
    without anyone asking. `reinject=False` states it once at the top, which is
    what a naive script does.
    """
    system = system_message(rule_name)
    messages = [{"role": "system", "content": system}]
    for i, (user, assistant) in enumerate(history):
        if reinject and i > 0:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    if reinject and history:
        messages.append({"role": "system", "content": system})
    return messages


def check(rule_name: str, text: str) -> bool:
    _, checker = RULES[rule_name]
    return checker(text)


def compliance(records):
    """records: {rule, condition, turn, ok}. Returns rate per condition per turn."""
    out = {}
    for record in records:
        key = (record["condition"], record["turn"])
        hit, total = out.get(key, (0, 0))
        out[key] = (hit + (1 if record["ok"] else 0), total + 1)
    return {k: (h / t if t else None, t) for k, (h, t) in out.items()}


def gate(rates, log=print):
    """The kill rule from ADR 0003, applied to numbers rather than to a feeling.

    Cut Arm B if drift is already small, or if re-injecting the rule closes it.
    """
    once = rates.get(("once", 10), (None, 0))[0]
    again = rates.get(("reinjected", 10), (None, 0))[0]
    first = rates.get(("once", 1), (None, 0))[0]
    if once is None or again is None or first is None:
        log("  gate cannot be decided: missing turn 1 or turn 10 numbers")
        return None

    drift = first - once
    closed = again - once
    log(f"  turn 1 compliance:          {first:.1%}")
    log(f"  turn 10, rule stated once:  {once:.1%}   (drift {drift * 100:+.1f}pt)")
    log(f"  turn 10, rule re-injected:  {again:.1%}   (recovers {closed * 100:+.1f}pt)")

    if drift < 0.10:
        log("  GATE: drift is under 10 points. There is little to fix, so Arm B")
        log("  is cut and these numbers are its section.")
        return "cut_small_drift"
    if again >= 0.90 or closed >= drift * 0.8:
        log("  GATE: re-injecting the rule -- which real chat software already")
        log("  does for free -- recovers most of the loss. Arm B is cut and")
        log("  these numbers are its section.")
        return "cut_free_fix_wins"
    log("  GATE PASSED: drift is real and re-injection does not close it.")
    log("  Arm B has headroom a fine-tune could claim.")
    return "passed"


def demo():
    """Self-check. Every rule must be decidable with no model and no opinion."""
    assert no_bullets("Plain prose, with a hyphen - like this.")
    assert not no_bullets("Here you go:\n- first\n- second")
    assert not no_bullets("Steps:\n1. boil water\n2. wait")
    assert not no_bullets("• a bullet")

    assert under_word_cap("short answer")
    assert not under_word_cap(" ".join(["word"] * (WORD_CAP + 1)))
    assert under_word_cap(" ".join(["word"] * WORD_CAP)), "the cap is inclusive"

    assert ends_with_question("Does that help?")
    assert not ends_with_question("That helps.")
    assert not ends_with_question("")
    assert ends_with_question("  Really?  "), "trailing space must not fail it"

    # the rule text has to actually describe what the checker tests
    for name, (instruction, checker) in RULES.items():
        assert instruction and callable(checker), name
    assert "bullet" in RULES["no_bullets"][0].lower()
    assert str(WORD_CAP) in RULES["word_cap"][0]
    assert "?" in RULES["end_question"][0]

    # one statement at the top, versus a statement before every turn
    history = [("q1", "a1"), ("q2", "a2")]
    once = build_messages("word_cap", history, reinject=False)
    again = build_messages("word_cap", history, reinject=True)
    assert sum(1 for m in once if m["role"] == "system") == 1
    assert sum(1 for m in again if m["role"] == "system") > 1
    assert once[0]["role"] == "system"
    assert again[-1]["role"] == "system", "the rule must be the last thing it reads"

    rates = compliance([
        {"condition": "once", "turn": 1, "ok": True},
        {"condition": "once", "turn": 10, "ok": False},
        {"condition": "reinjected", "turn": 10, "ok": True},
    ])
    assert rates[("once", 1)][0] == 1.0
    assert rates[("once", 10)][0] == 0.0

    # the kill rule has to fire on numbers that should kill it
    quiet = lambda *a: None
    assert gate({("once", 1): (1.0, 10), ("once", 10): (0.95, 10),
                 ("reinjected", 10): (0.97, 10)}, log=quiet) == "cut_small_drift"
    assert gate({("once", 1): (1.0, 10), ("once", 10): (0.40, 10),
                 ("reinjected", 10): (0.95, 10)}, log=quiet) == "cut_free_fix_wins"
    assert gate({("once", 1): (1.0, 10), ("once", 10): (0.40, 10),
                 ("reinjected", 10): (0.50, 10)}, log=quiet) == "passed"
    print("rules self-check passed")


if __name__ == "__main__":
    demo()
