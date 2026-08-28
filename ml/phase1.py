"""DEFER - Phase 0.5, the free baselines. The gate that can end the study.

Two arms, no training, one GPU session:

    base    an ordinary prompt -- passage, question, answer it
    prompt  the same, plus "the passage is the only authority"

Both questions this answers are ones that make the fine-tune pointless if they
come back the wrong way:

1. **Does the bug exist?** If the untrained model already follows a contradicting
   passage most of the time, there is nothing to fix and that is the finding.
2. **Does the free fix already close it?** Asking politely costs nothing and
   ships today. Whatever prompting fixes is not the fine-tune's to claim, so it
   has to be measured *before* training rather than discovered afterwards.

Running this first is the whole reason it is cheap to be wrong here. The
alternative -- train for twenty GPU-hours, then find out the baseline was
already there -- is the expensive version of the same discovery.

Output: /kaggle/working/<arm>__generations.jsonl and <arm>__run.json, plus the
scores printed in full. Scoring also happens on the laptop from the downloaded
generations; it is printed here only so a session that is never downloaded still
told you something.
"""
import json
import os
import time
import traceback
from pathlib import Path

ARMS = ["base", "prompt"]
TIME_BUDGET_S = 7.5 * 3600

import generate as gen        # noqa: E402
import metrics                # noqa: E402
import probe as probe_mod     # noqa: E402
from kaggle_env import (assert_size, check_gpu, code_fingerprint, die,  # noqa: E402
                        find_model, line)
from runner import JsonlSink  # noqa: E402

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."


def main():
    line("0. code fingerprint")
    print(f"  running code from: {CODE_DIR}")
    print(f"  fingerprint:       {code_fingerprint(CODE_DIR)}")
    print("  if that hash is not the one you just published, the notebook is "
          "serving an older dataset version and the results are stale.")

    gen.demo()
    metrics.demo()
    print("  module self-checks passed")

    # --------------------------------------------------------------- 1. the eval
    line("1. loading the frozen evaluation set")
    # The eval travels inside the code dataset, so `check_lock` here is checking
    # that the copy which arrived is the copy that was frozen -- an upload can
    # truncate, and a truncated eval would score as a smaller, easier study.
    # The dataset is staged with the eval in a data/ subfolder, so the code
    # directory doubles as the project root and this line works unchanged on the
    # laptop and on Kaggle.
    items, eval_sha = gen.load_eval(root=Path(CODE_DIR))
    print(f"  {len(items)} items, sha256 {eval_sha[:16]}... matches data/eval.lock")
    slices = {}
    for item in items:
        slices[item["slice"]] = slices.get(item["slice"], 0) + 1
    print(f"  slices: {slices}")

    # ---------------------------------------------------------------- 2. the GPU
    line("2. checking the GPU is one this PyTorch can actually run on")
    device_name = check_gpu()

    line("3. locating the Llama weights")
    model_ref = find_model()
    print()
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
            "The files are mounted, so this is not an access problem. Check the "
            "transformers version against the checkpoint format.")

    # --------------------------------------------------------------- 3. the arms
    deadline = time.time() + TIME_BUDGET_S
    summaries = {}

    for arm in ARMS:
        line(f"5. arm '{arm}'")
        system = gen.ARMS[arm]
        print(f"  system prompt: {system}")
        raw = gen.hf_generator(model, tokenizer, system)

        def budgeted(batch):
            if time.time() > deadline:
                raise TimeoutError("time budget reached")
            return raw(batch)

        # Time one warm batch before committing to the whole arm. The first batch
        # pays a one-off cost for compiling CUDA kernels, so it is timed twice and
        # only the second number is used for the projection.
        warm = items[:gen.BATCH_SIZE]
        t0 = time.time()
        sample_answers = raw(warm)
        first = time.time() - t0
        t0 = time.time()
        raw(warm)
        per_item = (time.time() - t0) / len(warm)
        print(f"  first batch {first:.1f}s (warm-up), steady {per_item:.3f}s/item")
        print(f"  this arm projects to {per_item * len(items) / 60:.1f} min")

        if arm == ARMS[0]:
            print()
            print("  --- the prompt, exactly as the model sees it ---")
            shown = gen.build_prompt(tokenizer, items[0], system)
            print(shown[:1200] + ("..." if len(shown) > 1200 else ""))
            print("  --- end ---", flush=True)

        print()
        print("  sample answers from the warm-up batch:")
        for item, answer in list(zip(warm, sample_answers))[:4]:
            label = metrics.verdict(item, answer)
            print(f"    [{item['slice']:12}] {item['question'][:56]}")
            print(f"       gold {str(item['answer'])[:28]!r}  "
                  f"memorised {str(item['memorised'])[:20]!r}")
            print(f"       said {answer[:70]!r}  -> {label}")

        sink = JsonlSink(os.path.join(WORK, f"{arm}__generations.jsonl"))
        before = len(sink)
        t0 = time.time()

        def progress(done, total, _arm=arm, _t0=t0):
            if done % (gen.BATCH_SIZE * 10) == 0 or done == total:
                elapsed = time.time() - _t0
                print(f"    {_arm}: {done}/{total}  {elapsed / 60:.1f} min", flush=True)

        try:
            gen.run(items, budgeted, sink=sink, on_batch=progress)
        except TimeoutError:
            print("  stopped on the time budget - partial results are on disk and "
                  "a re-run resumes from here.")
        elapsed = time.time() - t0
        print(f"  wrote {len(sink) - before} answers in {elapsed / 60:.1f} min")

        gen.write_manifest(Path(WORK) / f"{arm}__run.json", arm, eval_sha,
                           len(sink), model_ref, device_name, per_item)

        answers = {r["qid"]: r["generation"] for r in sink.read()}
        scored = []
        for item in items:
            if item["qid"] not in answers:
                continue
            record = dict(item)
            record["generation"] = answers[item["qid"]]
            record["verdict"] = metrics.verdict(item, record["generation"])
            scored.append(record)
        summaries[arm] = metrics.summarise(scored)

        counts = {}
        for record in scored:
            if record["slice"] == "conflict":
                counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
        print(f"  conflict breakdown: {counts}")

    # ------------------------------------------------------------ 4. the verdict
    line("6. the four numbers, both arms")

    names = {
        "grounded_accuracy": "grounded accuracy",
        "conflict_following": "conflict following <- headline",
        "abstention_unanswerable": "abstention (unanswerable)",
        "over_abstention": "over-abstention (lower better)",
    }
    for arm, summary in summaries.items():
        print(f"  {arm}")
        for key, label in names.items():
            stat = summary[key]
            if stat["rate"] is None:
                print(f"    {label:32} --")
                continue
            print(f"    {label:32} {stat['rate']:6.1%}  "
                  f"[{stat['lo']:.1%}, {stat['hi']:.1%}]  n={stat['n']}")

    with open(os.path.join(WORK, "baseline_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"eval_sha256": eval_sha, "model": model_ref,
                   "device": device_name, "arms": summaries}, handle, indent=2)

    # The gate, read out loud rather than left for someone to eyeball.
    line("7. what this means for the study")
    base = summaries.get("base", {}).get("conflict_following", {})
    prompted = summaries.get("prompt", {}).get("conflict_following", {})
    if base.get("rate") is not None and prompted.get("rate") is not None:
        print(f"  conflict following: base {base['rate']:.1%} "
              f"-> prompt {prompted['rate']:.1%}")
        print(f"  the free fix is worth {(prompted['rate'] - base['rate']) * 100:+.1f} points")
        print()
        if prompted["rate"] > 0.90:
            print("  GATE: prompting alone nearly solves it. A fine-tune has almost")
            print("  no room left, and that result gets published rather than buried.")
        elif base["rate"] > 0.90:
            print("  GATE: the untrained model already follows the passage. There is")
            print("  no bug here to fix, and that is the finding.")
        else:
            print("  GATE PASSED: room remains above the free baseline. Training is")
            print(f"  aiming at the {(1 - prompted['rate']) * 100:.0f} points prompting left on the table.")
        print()
        print("  Whatever training scores has to beat the PROMPT arm, not the base")
        print("  arm. Beating an unprompted model is beating a strawman.")

    line("DONE")
    print("  <arm>__generations.jsonl   every answer, one per line")
    print("  <arm>__run.json            what produced them")
    print("  baseline_summary.json      the numbers above")
    print()
    print("  Download these into runs/<arm>/ and re-score on the laptop with")
    print("  `python ml/score.py`. The scorer is the authority, not this printout.")


if __name__ == "__main__":
    main()
