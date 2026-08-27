"""DEFER - Phase 0.2, the closed-book probe.

Asks each question with no passage attached and records whether the model
already knows the answer. Only questions it reliably knows can become conflict
items later, because you cannot catch a model trusting its memory over the page
if it had no memory of that fact to begin with.

Both SQuAD splits in one run. Measured on the first green run: 0.186 seconds per
question on a T4, so dev (870 typed questions) takes under three minutes and
train (15,074) about forty-seven. They go together because pushing a new kernel
version resets the notebook's accelerator back to a P100, and a P100 cannot run
this PyTorch at all -- so every extra push costs a manual fix in the browser.

Output: /kaggle/working/probe_{dev,train}.jsonl, written batch by batch so a
hard kill loses one batch rather than the run.
"""
import json
import os
import sys
import time
import traceback

# The weights come from Kaggle's own copy of Meta's model, attached through
# `model_sources` in kernel-metadata.json, not downloaded from Hugging Face.
# That removes the access token, the Kaggle secret and the gate check from this
# run entirely -- three things that can fail, replaced by a mounted directory.
MODEL_HINT = "llama-3.2"

# Used only if the Llama mount is missing. Phi-3.5-mini is ungated and MIT, so
# it needs no token, no secret and no approval queue.
FALLBACK_MODEL = "microsoft/Phi-3.5-mini-instruct"

SPLITS = ["dev", "train"]

# Stop cleanly before the platform kills us. A session is cut at the wall clock
# with no warning and no `finally`, so the run stops itself with time to spare.
TIME_BUDGET_S = 7.5 * 3600


def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68, flush=True)


def die(what, why, fix):
    print(f"\n  FAILED: {what}")
    print(f"  what it means: {why}")
    print(f"  what to do:    {fix}", flush=True)
    sys.exit(1)


# ------------------------------------------------------------ 1. find the code
line("1. locating the defer-code dataset")

code_dir = None
for root, _dirs, files in os.walk("/kaggle/input"):
    if "probe.py" in files and "metrics.py" in files:
        code_dir = root
        break

if code_dir is None:
    # Printing the real tree is the one line that identifies this instantly.
    print("  /kaggle/input actually contains:")
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) - "/kaggle/input".count(os.sep) > 3:
            continue
        print(f"    {root}  ->  {files[:6]}")
    die("the code dataset is not attached",
        "Nothing under /kaggle/input has probe.py and metrics.py in it.",
        "Attach johnandreimartinez/defer-code, or push a new dataset version.")

sys.path.insert(0, code_dir)
print(f"  code found at: {code_dir}")

import conflict            # noqa: E402
import probe as probe_mod  # noqa: E402
import squad               # noqa: E402
from runner import JsonlSink  # noqa: E402

# Cheap guard in front of an expensive job: these prove the code that travelled
# here is the code that passed on the laptop. squad.demo() is included because
# it touches the filesystem -- an earlier run died 80 seconds in trying to cache
# the dataset next to the code, which is a read-only mount here.
conflict.demo()
probe_mod.demo()
squad.demo()
print("  module self-checks passed")

# ------------------------------------------------------- 2. is the GPU usable
line("2. checking the GPU is one this PyTorch can actually run on")

import torch  # noqa: E402

print(f"  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    die("no GPU", "This notebook was scheduled without an accelerator.",
        "Settings -> Accelerator -> GPU T4 x2.")

_name = torch.cuda.get_device_name(0)
_major, _minor = torch.cuda.get_device_capability(0)
_this = f"sm_{_major}{_minor}"
_built = torch.cuda.get_arch_list()
print(f"  device: {_name}  ({_this})")
print(f"  this PyTorch was built for: {' '.join(_built)}")

if _this not in _built:
    # Caught here rather than at the first generate(), which is on the far side
    # of a multi-gigabyte model download. The card loads weights happily and
    # only fails when asked to run compiled code that does not exist for it.
    die(f"this PyTorch cannot run on a {_name}",
        f"The card is {_this}; this build only has kernels for "
        f"{' '.join(_built)}. Nothing is wrong with the code -- there is simply "
        "no compiled GPU code for this chip, so the first generate() would die "
        "with 'no kernel image is available for execution on the device'.",
        "Settings -> Accelerator -> GPU T4 x2, then Save & Run All. This cannot "
        "be set from the CLI, and a `kernels push` resets it.")
print("  usable.")

# ----------------------------------------------------------- 3. find the model
line("3. locating the Llama weights")

model_dir = None
for root, _dirs, files in os.walk("/kaggle/input"):
    if "config.json" in files and any(f.endswith(".safetensors") for f in files):
        if MODEL_HINT in root.lower():
            model_dir = root
            break

if model_dir is None:
    print("  no mounted Llama checkpoint found. /kaggle/input holds:")
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) - "/kaggle/input".count(os.sep) > 4:
            continue
        print(f"    {root}  ->  {files[:5]}")
    model_ref = FALLBACK_MODEL
    print()
    print(f"  FALLING BACK to {FALLBACK_MODEL} (ungated, MIT).")
    print("  For Llama, accept Meta's terms at")
    print("  https://www.kaggle.com/models/metaresearch/llama-3.2 and re-run.")
else:
    model_ref = model_dir
    print(f"  weights at: {model_dir}")

print()
print(f"  MODEL IN USE: {model_ref}")

# ----------------------------------------------------------------- 4. load it
line("4. loading the model")
try:
    started = time.time()
    model, tokenizer = probe_mod.load_model(model_ref)
    params = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {time.time() - started:.0f}s, {params / 1e9:.2f}B parameters")
    if not 2.5e9 < params < 4.5e9:
        die("that is not the model this study is designed around",
            f"Counted {params / 1e9:.2f}B parameters. The previous study found "
            "1.5B too small to learn the behaviour, so size is not cosmetic.",
            "Check MODEL_HINT / FALLBACK_MODEL at the top of this script.")
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
    line(f"5. {split}: loading SQuAD 2.0 and keeping the typeable questions")

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
line("6. what the model actually knows")

summary = {
    "model": model_ref,
    "device": _name,
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
