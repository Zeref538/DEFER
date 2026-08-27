"""DEFER - Phase 0.2, the closed-book probe on the SQuAD 2.0 dev split.

Asks each question with no passage attached and records whether the model
already knows the answer. Only questions it reliably knows can become conflict
items later, because you cannot catch a model trusting its memory over the page
if it had no memory of that fact to begin with.

This run deliberately covers the dev split only -- around 870 questions after
typing. Dev is where the frozen evaluation comes from, so it is the half that
matters most, and it is small enough to finish inside one session. It also
measures its own generation speed on a warm-up batch and prints a projection,
which is what sizes the much larger train-split run rather than a guess copied
from some other project.

Output: /kaggle/working/probe_dev.jsonl, written batch by batch so a hard kill
loses one batch rather than the run.
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
SPLIT = "dev"
OUT = "/kaggle/working/probe_dev.jsonl"

# Stop cleanly before the platform kills us. A batch CPU/GPU session is cut at
# the wall clock with no warning and no `finally`, so the run stops itself with
# time to spare and leaves a summary behind.
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
    # Attached datasets do NOT mount at a flat /kaggle/input/<slug>/.
    print("  /kaggle/input actually contains:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
        if depth > 3:
            continue
        print(f"    {root}  ->  {files[:6]}")
    die("the code dataset is not attached",
        "Nothing under /kaggle/input has probe.py and metrics.py in it.",
        "Attach johnandreimartinez/defer-code to this notebook, or push a new "
        "dataset version if the code changed.")

sys.path.insert(0, code_dir)
print(f"  code found at: {code_dir}")

import conflict          # noqa: E402
import probe as probe_mod  # noqa: E402
import squad             # noqa: E402
from runner import JsonlSink  # noqa: E402

# The self-checks run in seconds and prove the code that travelled here is the
# code that passed on the laptop. Cheap guard in front of an expensive job.
conflict.demo()
probe_mod.demo()
print("  module self-checks passed")

# --------------------------------------------------------------- 2. the model
line("2. locating the attached Llama weights")

model_dir = None
for root, _dirs, files in os.walk("/kaggle/input"):
    if "config.json" in files and any(f.endswith(".safetensors") for f in files):
        if MODEL_HINT in root.lower():
            model_dir = root
            break

if model_dir is None:
    print("  /kaggle/input actually contains:")
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) - "/kaggle/input".count(os.sep) > 4:
            continue
        print(f"    {root}  ->  {files[:5]}")
    die("the Llama model is not attached",
        "Nothing under /kaggle/input looks like a transformers checkpoint whose "
        f"path contains {MODEL_HINT!r}.",
        "Attach metaresearch/llama-3.2/transformers/3b-instruct to this "
        "notebook. Accepting Meta's terms on the Kaggle model page is a "
        "separate click from accepting them on Hugging Face.")

print(f"  weights at: {model_dir}")
print(f"  files: {sorted(os.listdir(model_dir))[:8]}")

# ----------------------------------------------------------- 3. the questions
line(f"3. loading SQuAD 2.0 {SPLIT} and keeping the typeable questions")

answerable = squad.items(SPLIT, "answerable")
items = [i for i in answerable if conflict.classify(i["question"], i["gold"])]
print(f"  answerable: {len(answerable)}")
print(f"  typed:      {len(items)}  ({len(items) / len(answerable):.1%})")
print("  the rest are skipped on purpose -- probing questions that can never "
      "become conflict items would spend GPU time no later stage reads.")
if not items:
    die("no questions survived typing",
        "The classifier rejected everything, so the probe has nothing to ask.",
        "Run `python ml/conflict.py` locally; something in the typing broke.")

# --------------------------------------------------------------- 4. the model
line("4. loading the model")
try:
    import torch

    print(f"  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
    started = time.time()
    model, tokenizer = probe_mod.load_model(model_dir)
    params = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {time.time() - started:.0f}s, {params / 1e9:.2f}B parameters")
    if not 2.5e9 < params < 4.5e9:
        die("that is not the model this study is designed around",
            f"Counted {params/1e9:.2f}B parameters. The previous study found "
            "1.5B too small to learn the behaviour, so size is not cosmetic.",
            "Check MODEL at the top of this script.")
except SystemExit:
    raise
except Exception as exc:
    traceback.print_exc()
    die("the model would not load",
        f"{type(exc).__name__}: {exc}",
        "The files are mounted, so this is not an access problem. Check the "
        "transformers version against the checkpoint format.")

generate = probe_mod.hf_generator(model, tokenizer)

# ------------------------------------------------------------- 5. the warm-up
line("5. timing one batch before committing to the whole run")

print("  --- the prompt, exactly as the model sees it (no passage) ---")
print(probe_mod.closed_book_prompt(tokenizer, items[0]["question"]))
print("  --- end ---", flush=True)

warm = items[:probe_mod.BATCH_SIZE]
started = time.time()
sampled = generate([w["question"] for w in warm])
first = time.time() - started
# The first batch pays for CUDA kernel compilation, so time a second one too.
started = time.time()
generate([w["question"] for w in warm])
steady = time.time() - started

per_question = steady / len(warm)
projected = per_question * len(items)
print(f"  first batch {first:.1f}s (includes one-off warm-up), steady {steady:.1f}s")
print(f"  {per_question:.2f}s per question at k={probe_mod.K}")
print(f"  projected for {len(items)} questions: {projected / 60:.0f} min")
print(f"  time budget: {TIME_BUDGET_S / 3600:.1f} h")
print("\n  sample answers from the warm-up batch:")
for item, samples in list(zip(warm, sampled))[:3]:
    hits = probe_mod.judge(samples, item["gold"])
    print(f"    Q: {item['question'][:66]}")
    print(f"       gold {item['gold']!r} | {hits}/{probe_mod.K} | {samples[:3]}")

if projected > TIME_BUDGET_S:
    keep = int(TIME_BUDGET_S / per_question)
    print(f"\n  projection exceeds the budget; probing the first {keep} only.")
    items = items[:keep]

# ----------------------------------------------------------------- 6. the run
line("6. probing")

sink = JsonlSink(OUT)
deadline = time.time() + TIME_BUDGET_S
done_at_start = len(sink)
started = time.time()


def budgeted(batch_items):
    """Wrap the generator so the loop stops cleanly when time runs out."""
    if time.time() > deadline:
        raise TimeoutError("time budget reached")
    return generate(batch_items)


try:
    probe_mod.run(items, budgeted, sink=sink)
except TimeoutError:
    print("  stopped on the time budget - partial results are on disk and a "
          "re-run resumes from here.")
except KeyboardInterrupt:
    print("  interrupted - partial results are on disk.")

elapsed = time.time() - started
written = len(sink) - done_at_start
print(f"  wrote {written} records in {elapsed / 60:.1f} min "
      f"({elapsed / max(written, 1):.2f}s each)")

# ------------------------------------------------------------- 7. the summary
line("7. what the model actually knows")

records = sink.read()
stats = probe_mod.summarise(records)

by_type = {}
for record in records:
    kind = conflict.classify(record["question"], record["gold"]) or "?"
    seen, known = by_type.get(kind, (0, 0))
    by_type[kind] = (seen + 1, known + (1 if record["knows"] else 0))
print("\n  by answer type:")
for kind, (seen, known) in sorted(by_type.items()):
    print(f"    {kind:8} {known:5} / {seen:5}  ({known / seen:.1%})")

summary = {
    "model": model_dir,
    "split": SPLIT,
    "typed_questions": len(items),
    "probed": stats["n"],
    "known": stats["known"],
    "known_rate": stats["rate"],
    "by_type": {k: {"seen": v[0], "known": v[1]} for k, v in by_type.items()},
    "seconds_per_question": per_question,
    "k": probe_mod.K,
    "threshold": probe_mod.THRESHOLD,
}
with open("/kaggle/working/probe_summary.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

line("DONE")
print(f"""
  probe_dev.jsonl      {stats['n']} records
  probe_summary.json   the numbers above, machine-readable

  {stats['known']} questions are reliably known closed-book. Those are the only
  ones that can become conflict items -- the rest would test reading, not memory.

  Measured cost: {per_question:.2f}s per question. The train split has ~15,000
  typed questions, so that run is roughly {per_question * 15074 / 3600:.1f} h at this rate.
""")
