"""DEFER - Phase 0.2, the closed-book probe. The whole run lives here.

This is imported by a three-line stub notebook rather than being the notebook,
and that split is deliberate: a `kaggle kernels push` replaces the notebook and
starts a fresh version, while a `kaggle datasets version` swaps the code under a
notebook that stays put. Shipping logic through the dataset means the notebook,
its settings and its run history are never disturbed by a code change.

Asks each question with no passage attached and records whether the model
already knows the answer. Only reliably-known questions can become conflict
items, because you cannot catch a model trusting its memory over the page if it
had no memory of that fact to begin with.

Both SQuAD splits in one run: about 50 minutes at the measured 0.186 s/question.

Output: /kaggle/working/probe_{dev,train}.jsonl, written batch by batch so a
hard kill loses one batch rather than the run.
"""
import json
import os
import time
import traceback

SPLITS = ["dev", "train"]
TIME_BUDGET_S = 7.5 * 3600

import conflict            # noqa: E402
import probe as probe_mod  # noqa: E402
import squad               # noqa: E402
from kaggle_env import (assert_size, check_gpu, code_fingerprint, die,  # noqa: E402
                        find_model, line)
from runner import JsonlSink  # noqa: E402

CODE_DIR = os.path.dirname(os.path.abspath(__file__))



def main():
    line("0. code fingerprint")
    print(f"  running code from: {CODE_DIR}")
    print(f"  fingerprint:       {code_fingerprint(CODE_DIR)}")
    print("  if that hash is not the one you just published, the notebook is "
          "serving an older dataset version and the results are stale.")

    # Cheap guard in front of an expensive job: these prove the code that travelled
    # here is the code that passed on the laptop. squad.demo() is included because
    # it touches the filesystem -- an earlier run died 80 seconds in trying to cache
    # the dataset next to the code, which is a read-only mount here.
    conflict.demo()
    probe_mod.demo()
    squad.demo()
    print("  module self-checks passed")

    # ------------------------------------------------------- 2. is the GPU usable
    line("1. checking the GPU is one this PyTorch can actually run on")
    device_name = check_gpu()

    # ----------------------------------------------------------- 3. find the model
    line("2. locating the Llama weights")
    model_ref = find_model()
    print()
    print(f"  MODEL IN USE: {model_ref}")

    # ----------------------------------------------------------------- 4. load it
    line("3. loading the model")
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

    generate = probe_mod.hf_generator(model, tokenizer)

    # -------------------------------------------------------------- 5. the splits
    summaries = {}
    deadline = time.time() + TIME_BUDGET_S
    per_question = None


    def budgeted(batch_questions):
        """Wrap the generator so the loop stops cleanly when time runs out."""
        if time.time() > deadline:
            raise TimeoutError("time budget reached")
        return generate(batch_questions)


    for split in SPLITS:
        line(f"4. {split}: loading SQuAD 2.0 and keeping the typeable questions")

        answerable = squad.items(split, "answerable")
        items = [i for i in answerable if conflict.classify(i["question"], i["gold"])]
        print(f"  answerable: {len(answerable)}   typed: {len(items)}  "
              f"({len(items) / len(answerable):.1%})")
        print("  the rest are skipped on purpose -- probing questions that can never "
              "become conflict items would spend GPU time no later stage reads.")
        if not items:
            die("no questions survived typing",
                "The classifier rejected everything, so there is nothing to ask.",
                "Run `python ml/conflict.py` locally; something in the typing broke.")

        if per_question is None:
            print()
            print("  --- the prompt, exactly as the model sees it (no passage) ---")
            print(probe_mod.closed_book_prompt(tokenizer, items[0]["question"]))
            print("  --- end ---", flush=True)
            warm = items[:probe_mod.BATCH_SIZE]
            t0 = time.time()
            sampled = generate([w["question"] for w in warm])
            first = time.time() - t0
            # The first batch pays for CUDA kernel compilation; time a second one
            # for the steady-state number the projection is built from.
            t0 = time.time()
            generate([w["question"] for w in warm])
            per_question = (time.time() - t0) / len(warm)
            print(f"  first batch {first:.1f}s (one-off warm-up), steady "
                  f"{per_question:.3f}s per question at k={probe_mod.K}")
            print(f"  both splits project to {per_question * 15944 / 60:.0f} min "
                  f"against a {TIME_BUDGET_S / 3600:.1f} h budget")
            print()
            print("  sample answers from the warm-up batch:")
            for item, samples in list(zip(warm, sampled))[:3]:
                hits = probe_mod.judge(samples, item["gold"])
                print(f"    Q: {item['question'][:64]}")
                print(f"       gold {item['gold']!r} | {hits}/{probe_mod.K} | {samples[:3]}")

        sink = JsonlSink(f"/kaggle/working/probe_{split}.jsonl")
        before = len(sink)
        t0 = time.time()
        try:
            probe_mod.run(items, budgeted, sink=sink)
        except TimeoutError:
            print("  stopped on the time budget - partial results are on disk and a "
                  "re-run resumes from here.")

        print(f"  wrote {len(sink) - before} records in {(time.time() - t0) / 60:.1f} min")

        records = sink.read()
        stats = probe_mod.summarise(records)

        by_type = {}
        for record in records:
            kind = conflict.classify(record["question"], record["gold"]) or "?"
            seen, known = by_type.get(kind, (0, 0))
            by_type[kind] = (seen + 1, known + (1 if record["knows"] else 0))
        print("  by answer type:")
        for kind, (seen, known) in sorted(by_type.items()):
            print(f"    {kind:8} {known:6} / {seen:6}  ({known / seen:.1%})")

        summaries[split] = {
            "typed": len(items),
            "probed": stats["n"],
            "known": stats["known"],
            "known_rate": stats["rate"],
            "by_type": {k: {"seen": v[0], "known": v[1]} for k, v in by_type.items()},
        }

    # ------------------------------------------------------------- 6. the summary
    line("5. what the model actually knows")

    summary = {
        "model": model_ref,
        "device": device_name,
        "splits": summaries,
        "seconds_per_question": per_question,
        "k": probe_mod.K,
        "threshold": probe_mod.THRESHOLD,
    }
    with open("/kaggle/working/probe_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    for split, stat in summaries.items():
        print(f"  {split:6} {stat['known']:6} known of {stat['probed']:6} probed "
              f"({stat['known_rate']:.1%})")

    line("DONE")
    print("  probe_dev.jsonl / probe_train.jsonl   one record per question")
    print("  probe_summary.json                    the numbers above, as JSON")
    print()
    print("  Only reliably-known questions can become conflict items. The rest")
    print("  would test reading rather than memory, which is a different study.")


if __name__ == "__main__":
    main()
