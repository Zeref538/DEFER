"""DEFER - Gate D. Does a standing rule survive ten turns, and is it free to fix?

No training. This is the cheap measurement ADR 0003 put in front of Arm B, and
it is being run now so that arm is either built or killed on evidence rather than
left unresolved.

Three machine-checkable rules, ten turns each, two conditions:

    once        the rule is stated in the system message and never repeated
    reinjected  the rule is restated before every turn -- what real chat
                software already does for nobody's benefit in particular

If drift is small, or if re-injection closes it, Arm B is cut and these numbers
become its section in the write-up. That is a result, not an apology: finding it
here costs one evaluation pass, and finding it after twenty GPU-hours of
training would have cost twenty GPU-hours.

Output: /kaggle/working/rule_drift.jsonl and rule_drift_summary.json
"""
import json
import os
import time
import traceback
from pathlib import Path

import metrics                # noqa: E402  (kept for the self-check)
import probe as probe_mod     # noqa: E402
import rules                  # noqa: E402
from kaggle_env import (assert_size, check_gpu, code_fingerprint, die,  # noqa: E402
                        find_model, line)
from runner import JsonlSink  # noqa: E402

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK = Path("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")

# How many conversations per (rule, condition). Each is 10 sequential turns and
# cannot be batched across turns, so this is the expensive axis. 12 x 3 rules x
# 2 conditions x 10 turns = 720 generations.
CONVERSATIONS = 12
MAX_NEW_TOKENS = 220      # long enough that a model CAN break the word cap
TIME_BUDGET_S = 7.0 * 3600


def main():
    line("0. code fingerprint")
    print(f"  running code from: {CODE_DIR}")
    print(f"  fingerprint:       {code_fingerprint(CODE_DIR)}")
    rules.demo()
    metrics.demo()
    print("  module self-checks passed")

    line("1. the rules being tested")
    for name, (instruction, _) in rules.RULES.items():
        print(f"  {name:14} {instruction}")
    print(f"  {len(rules.TURNS)} turns, {CONVERSATIONS} conversations per cell")

    line("2. checking the GPU is one this PyTorch can actually run on")
    device_name = check_gpu()

    line("3. locating the Llama weights")
    model_ref = find_model()
    print(f"  MODEL IN USE: {model_ref}")

    line("4. loading the model")
    try:
        started = time.time()
        model, tokenizer = probe_mod.load_model(model_ref)
        print(f"  loaded in {time.time() - started:.0f}s")
        assert_size(model)
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        die("the model would not load", f"{type(exc).__name__}: {exc}",
            "Check the transformers version against the checkpoint format.")

    import torch

    def reply(messages):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**encoded, do_sample=False,
                                 max_new_tokens=MAX_NEW_TOKENS,
                                 pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][encoded["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    # ------------------------------------------------------------ 5. the runs
    line("5. running the conversations")
    sink = JsonlSink(WORK / "rule_drift.jsonl", key="cell")
    deadline = time.time() + TIME_BUDGET_S
    shown = False

    cells = [(rule, condition, c)
             for rule in rules.RULES
             for condition in ("once", "reinjected")
             for c in range(CONVERSATIONS)]
    todo = [c for c in cells if f"{c[0]}|{c[1]}|{c[2]}" not in sink]
    print(f"  {len(todo)} conversations to run of {len(cells)}")

    t0 = time.time()
    for done, (rule, condition, index) in enumerate(todo, start=1):
        if time.time() > deadline:
            print("  stopped on the time budget - partial results are on disk "
                  "and a re-run resumes from here.")
            break

        # Each conversation starts at a different point in the turn list so the
        # rule is not always tested against the same question in the same slot.
        order = rules.TURNS[index % len(rules.TURNS):] + \
            rules.TURNS[:index % len(rules.TURNS)]

        history, results = [], []
        for turn, question in enumerate(order, start=1):
            messages = rules.build_messages(
                rule, history, reinject=(condition == "reinjected"))
            messages.append({"role": "user", "content": question})
            answer = reply(messages)
            ok = rules.check(rule, answer)
            history.append((question, answer))
            results.append({"turn": turn, "ok": ok, "answer": answer})

            if not shown and turn == 1:
                print()
                print(f"  --- first reply, rule '{rule}' ---")
                print(f"  {answer[:300]}")
                print(f"  passes the rule: {ok}")
                print("  --- end ---", flush=True)
                shown = True

        sink.write([{
            "cell": f"{rule}|{condition}|{index}",
            "rule": rule, "condition": condition, "conversation": index,
            "turns": results,
        }])
        if done % 6 == 0 or done == len(todo):
            print(f"    {done}/{len(todo)} conversations, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

    # --------------------------------------------------------- 6. the verdict
    line("6. compliance by turn")

    flat = []
    for record in sink.read():
        for turn in record["turns"]:
            flat.append({"rule": record["rule"], "condition": record["condition"],
                         "turn": turn["turn"], "ok": turn["ok"]})
    rates = rules.compliance(flat)

    print(f"  {'condition':<12}" + "".join(f"turn {t:<5}" for t in rules.CHECKPOINTS))
    for condition in ("once", "reinjected"):
        cells_out = []
        for t in rules.CHECKPOINTS:
            rate, n = rates.get((condition, t), (None, 0))
            cells_out.append(f"{rate:6.1%}    " if rate is not None else "   --     ")
        print(f"  {condition:<12}" + "".join(cells_out))

    print()
    print("  per rule, turn 10:")
    for rule in rules.RULES:
        row = []
        for condition in ("once", "reinjected"):
            hits = [f["ok"] for f in flat
                    if f["rule"] == rule and f["condition"] == condition and f["turn"] == 10]
            row.append(f"{condition} {sum(hits) / len(hits):.0%}" if hits else f"{condition} --")
        print(f"    {rule:14} " + "   ".join(row))

    line("7. the gate")
    verdict = rules.gate(rates)

    summary = {
        "model": model_ref, "device": device_name,
        "conversations_per_cell": CONVERSATIONS,
        "turns": len(rules.TURNS),
        "rules": {k: v[0] for k, v in rules.RULES.items()},
        "rates": {f"{c}|{t}": {"rate": r, "n": n} for (c, t), (r, n) in rates.items()},
        "verdict": verdict,
    }
    with open(WORK / "rule_drift_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    line("DONE")
    print("  rule_drift.jsonl          every conversation, every turn")
    print("  rule_drift_summary.json   the table above and the gate verdict")


if __name__ == "__main__":
    main()
