"""DEFER - Phase 2, training. Two seeds, then their arms, in one session.

Two seeds is not caution, it is the minimum that makes a claim rankable. The
predecessor study measured an 18.5-point spread between two seeds on one axis;
a single run cannot tell a real effect from that wobble, so a one-seed result is
reported INCONCLUSIVE by construction rather than ranked.

Each seed does the whole loop -- train, save the adapter, generate its arm over
the frozen eval -- before the next one starts. If the session dies halfway, a
re-run skips the seed whose adapter is already on disk and resumes the arm whose
generations are partly written. Nothing restarts from zero.

The bar these arms have to clear is the `prompt` arm, not the `base` arm, and
they are trained and evaluated under that same system prompt so the comparison
is on identical inputs. Beating an unprompted model would be beating a strawman.

Output, per seed, in /kaggle/working:
    defer_s<seed>/                    the LoRA adapter, ~50 MB
    defer_s<seed>__generations.jsonl  its answers on the frozen eval
    defer_s<seed>__run.json           what produced them
"""
import json
import os
import time
import traceback
from pathlib import Path

# Seeds 0 and 1 are done. 2 and 3 exist because the abstention result rested on
# two numbers 10.4 points apart, and two points cannot tell "this behaviour is
# noisy" from "one run was unlucky". Four can. Same mix, same settings -- only
# the shuffle differs, so anything that moves is seed noise by definition.
SEEDS = [2, 3]
TIME_BUDGET_S = 8.0 * 3600

# Arm names are prefixed by which training mix produced them, because the first
# pair is not being thrown away -- the comparison between the two mixes IS the
# result. `defer` was the 4:1 answer-to-refuse mix; `deferb` is the balanced 1:1
# rebuild. Overwriting the first pair would delete the evidence for why the
# second one exists.
ARM_PREFIX = "deferb"

import generate as gen        # noqa: E402
import metrics                # noqa: E402
import train as train_mod     # noqa: E402
from kaggle_env import (assert_size, check_gpu, code_fingerprint, die,  # noqa: E402
                        find_model, line)
from runner import JsonlSink  # noqa: E402

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK = Path("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")


def free(*objects):
    """Hand the GPU memory back before the next seed loads its own copy."""
    import gc

    import torch
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


def main():
    line("0. code fingerprint")
    print(f"  running code from: {CODE_DIR}")
    print(f"  fingerprint:       {code_fingerprint(CODE_DIR)}")

    gen.demo()
    metrics.demo()
    train_mod.demo()
    print("  module self-checks passed")

    line("1. the frozen evaluation set and the training mix")
    items, eval_sha = gen.load_eval(root=Path(CODE_DIR))
    print(f"  eval: {len(items)} items, sha256 {eval_sha[:16]}...")
    mix = train_mod.load_mix(Path(CODE_DIR) / "train_mix.jsonl"
                             if (Path(CODE_DIR) / "train_mix.jsonl").exists()
                             else None)
    counts = {}
    for record in mix:
        counts[record["slice"]] = counts.get(record["slice"], 0) + 1
    print(f"  training mix: {len(mix)} rows {counts}")

    # The invariant the whole study rests on, re-checked on the far side of an
    # upload rather than trusted. A leak here would inflate every trained number.
    eval_ids = {i["qid"] for i in items}
    leaked = eval_ids & {r["qid"] for r in mix}
    if leaked:
        die(f"{len(leaked)} training rows are also in the evaluation set",
            "The trained model would be scored on questions it was taught. Every "
            "number from this run would be meaningless.",
            "Re-run `python ml/build.py` locally and publish the dataset again.")
    print("  no overlap between the eval and the training mix")

    line("2. checking the GPU is one this PyTorch can actually run on")
    device_name = check_gpu()

    line("3. locating the Llama weights")
    model_ref = find_model()
    print(f"  MODEL IN USE: {model_ref}")

    deadline = time.time() + TIME_BUDGET_S

    for seed in SEEDS:
        line(f"4. seed {seed}")
        arm = f"{ARM_PREFIX}_s{seed}"
        out_dir = WORK / arm
        gen_path = WORK / f"{arm}__generations.jsonl"

        if (out_dir / "adapter_model.safetensors").exists():
            print(f"  adapter already on disk at {out_dir}, skipping training")
            model, tokenizer = None, None
        else:
            if time.time() > deadline:
                print("  out of time budget before this seed started. Re-run to "
                      "continue -- finished seeds are skipped.")
                break
            print("  loading the base model in 4-bit and attaching a LoRA adapter")
            try:
                model, tokenizer = train_mod.load_for_training(model_ref)
            except ImportError as exc:
                die("a training library is missing", f"{exc}",
                    "This image needs `peft` and `bitsandbytes`. Add "
                    "`!pip install -q peft bitsandbytes` to the stub notebook.")
            except Exception as exc:
                traceback.print_exc()
                die("the model would not load for training", f"{type(exc).__name__}: {exc}",
                    "Check the transformers/peft versions against the checkpoint.")

            rows = train_mod.build_dataset(tokenizer, mix, seed=seed)
            print(f"  {train_mod.EPOCHS} epochs, effective batch "
                  f"{train_mod.BATCH_SIZE * train_mod.GRAD_ACCUM}, lr {train_mod.LR}")
            started = time.time()
            train_mod.train(model, tokenizer, rows, out_dir, seed=seed)
            print(f"  trained in {(time.time() - started) / 60:.1f} min")
            print(f"  adapter saved to {out_dir}")

        # ------------------------------------------------------- generate its arm
        print(f"  generating arm '{arm}' over the frozen eval")
        if model is None:
            model, tokenizer = load_for_inference(model_ref, out_dir)
        else:
            model.config.use_cache = True     # turned off for gradient checkpointing
            model.eval()
        tokenizer.padding_side = "left"       # decoder-only batched generation

        raw = gen.hf_generator(model, tokenizer, gen.GROUNDED_SYSTEM)

        def budgeted(batch):
            if time.time() > deadline:
                raise TimeoutError("time budget reached")
            return raw(batch)

        sink = JsonlSink(gen_path)
        t0 = time.time()

        def progress(done, total, _arm=arm, _t0=t0):
            if done % (gen.BATCH_SIZE * 10) == 0 or done == total:
                print(f"    {_arm}: {done}/{total}  "
                      f"{(time.time() - _t0) / 60:.1f} min", flush=True)

        try:
            gen.run(items, budgeted, sink=sink, on_batch=progress)
        except TimeoutError:
            print("  stopped on the time budget - partial results are on disk.")

        gen.write_manifest(WORK / f"{arm}__run.json", arm, eval_sha, len(sink),
                           model_ref, device_name, (time.time() - t0) / max(len(sink), 1))
        # The manifest's system_prompt comes from gen.ARMS, which has no entry
        # for a trained arm. Record the one actually used instead of a null.
        manifest_path = WORK / f"{arm}__run.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["system_prompt"] = gen.GROUNDED_SYSTEM
        manifest["seed"] = seed
        manifest["lora"] = {"r": train_mod.LORA_R, "alpha": train_mod.LORA_ALPHA,
                            "targets": train_mod.TARGET_MODULES,
                            "epochs": train_mod.EPOCHS, "lr": train_mod.LR}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8", newline="\n")

        answers = {r["qid"]: r["generation"] for r in sink.read()}
        scored = []
        for item in items:
            if item["qid"] in answers:
                record = dict(item)
                record["generation"] = answers[item["qid"]]
                record["verdict"] = metrics.verdict(item, record["generation"])
                scored.append(record)
        summary = metrics.summarise(scored)
        print(f"  --- {arm} ---")
        for key in ("grounded_accuracy", "conflict_following",
                    "abstention_unanswerable", "over_abstention"):
            stat = summary[key]
            if stat["rate"] is not None:
                print(f"    {key:26} {stat['rate']:6.1%}  "
                      f"[{stat['lo']:.1%}, {stat['hi']:.1%}]  n={stat['n']}")

        with open(WORK / f"{arm}__summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        free(model, tokenizer)
        model = tokenizer = None

    line("DONE")
    print(f"  Download {ARM_PREFIX}_s*/ and {ARM_PREFIX}_s*__generations.jsonl into runs/,")
    print("  then `python ml/score.py` on the laptop. The scorer is the authority.")
    print()
    print("  Two bars, not one. The PROMPT arm is the free baseline: 87.2% conflict")
    print("  following, 33.3% abstention. The first trained pair is the thing this")
    print("  rebuild is trying to beat on abstention WITHOUT losing its 97.9%:")
    print("  defer_s0/s1 scored 97.5/97.9 conflict, 20.7/19.7 abstention.")


def load_for_inference(model_ref: str, adapter_dir: Path):
    """Reload a saved adapter for generation, without touching the optimiser."""
    import torch
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_ref, quantization_config=quant, device_map="auto",
        torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    main()
