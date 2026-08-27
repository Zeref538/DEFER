"""Run one arm over the frozen evaluation set and log every answer.

An *arm* is one thing being measured: the untrained model with a plain prompt,
the untrained model with a grounding instruction, a fine-tuned checkpoint. Each
arm sees the exact same 1,083 items in the exact same order, so any difference
between two arms is the arm and not the data.

Two rules this file exists to enforce:

**Greedy decoding, never sampling.** The probe sampled on purpose -- it was
asking "does this model know the fact", and one lucky hit in eight is not
knowing. Here the question is "what does this model answer", and a sampled
answer would change on every re-run, so the published demo could show a
generation nobody can reproduce. Greedy means the same weights and the same
prompt always give the same text.

**The eval's hash is recorded next to the generations.** Not checked here --
recorded. If someone rebuilds the eval halfway through a study, the scorer
notices that arm A and arm B answered different question sets, instead of
quietly ranking them against each other.

Run the self-check:  python ml/generate.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runner import JsonlSink, atomic_write, chunks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# How long an answer may run. Short answers are the norm, but a refusal is a
# sentence and a hedge is two, and a truncated refusal would be scored as
# "other" -- an invented failure. 64 tokens is well past both.
MAX_NEW_TOKENS = 64

# One sequence per item now, not eight, so the batch can be wider than the
# probe's. The passage makes each prompt roughly ten times longer, which is why
# this is 16 rather than 64.
BATCH_SIZE = 16

# ---------------------------------------------------------------------- arms
#
# Both arms are told they may say the answer is missing. Only the *grounding*
# sentence differs. That is the whole point of having a `prompt` arm: if the two
# system messages differed in two ways, a gap between them could not be
# attributed to either one.
#
# `base` is the prompt a person writes without having thought about this bug --
# the one sitting inside most retrieval apps, including four on this portfolio.

BASE_SYSTEM = (
    "Answer the question. If the answer is not available, say so. "
    "Give the answer only, with no explanation."
)

GROUNDED_SYSTEM = (
    "Answer the question using only the passage provided. The passage is the "
    "only authority: where it disagrees with what you already believe, follow "
    "the passage. If the passage does not contain the answer, say so. "
    "Give the answer only, with no explanation."
)

ARMS = {
    "base": BASE_SYSTEM,
    "prompt": GROUNDED_SYSTEM,
}


def user_message(item: dict) -> str:
    """Passage first, question last.

    Deliberate: the question sits closest to where the model starts writing, so
    a model that ignores the passage cannot blame prompt layout for it.
    """
    return f"Passage:\n{item['passage']}\n\nQuestion: {item['question']}"


def build_prompt(tokenizer, item: dict, system: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message(item)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


# ------------------------------------------------------------------ the eval

def load_eval(root: Path = ROOT, check_lock: bool = True):
    """Read data/eval.jsonl and refuse to proceed if it is not the locked one.

    The guard is the point. The most tempting failure in this genre is nudging
    the evaluation set until the number improves, and it never looks like
    cheating from the inside -- it looks like fixing a bad item. So the hash is
    written once, and everything downstream compares against it.
    """
    # Repo layout puts the eval in data/; a Kaggle dataset mounts flat, because
    # the CLI's only options for subfolders are "skip them" or "zip them", and
    # neither leaves a readable data/eval.jsonl on the far side. So look in both.
    directory = root / "data" if (root / "data" / "eval.jsonl").exists() else root
    path = directory / "eval.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}. Run `python ml/build.py` first.")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    lock_path = directory / "eval.lock"
    if check_lock:
        if not lock_path.exists():
            raise SystemExit(f"missing {lock_path}. The eval is not frozen.")
        locked = lock_path.read_text(encoding="utf-8").strip()
        if digest != locked:
            raise SystemExit(
                "data/eval.jsonl does not match data/eval.lock.\n"
                f"  on disk: {digest}\n"
                f"  locked:  {locked}\n"
                "The evaluation set changed after it was frozen. Either restore "
                "the locked file, or -- if the change was intended -- rebuild, "
                "re-lock, and re-run EVERY arm. Comparing arms across two "
                "different evals is the one thing this lock exists to stop."
            )
    items = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    return items, digest


# ------------------------------------------------------------------- the run

def run(items, generate, sink, batch_size: int = BATCH_SIZE, log=print,
        on_batch=None):
    """generate: list[item] -> list[str], one answer per item, same order.

    Appends as it goes, so a killed session loses one batch and re-running the
    same command finishes the job.
    """
    if sink.repaired:
        log(f"  repaired {sink.repaired} truncated line(s) from a previous kill")
    todo = sink.pending(items)
    if len(todo) < len(items):
        log(f"  resuming: {len(items) - len(todo)} of {len(items)} already done")

    done = len(items) - len(todo)
    for batch in chunks(todo, batch_size):
        answers = generate(batch)
        assert len(answers) == len(batch), (
            f"generator returned {len(answers)} answers for {len(batch)} items")
        sink.write([{"qid": i["qid"], "generation": a}
                    for i, a in zip(batch, answers)])
        done += len(batch)
        if on_batch:
            on_batch(done, len(items))
    return sink


def hf_generator(model, tokenizer, system: str,
                 max_new_tokens: int = MAX_NEW_TOKENS):
    """The greedy generator the run loop calls."""
    import torch

    def generate(items):
        prompts = [build_prompt(tokenizer, i, system) for i in items]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                do_sample=False,          # greedy: same input, same output, always
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        fresh = output[:, encoded["input_ids"].shape[1]:]
        return [t.strip() for t in
                tokenizer.batch_decode(fresh, skip_special_tokens=True)]

    return generate


def write_manifest(path, arm: str, eval_sha: str, n: int,
                   model: str, device: str, seconds_per_item: float):
    """A run is not its generations alone -- it is also what produced them."""
    atomic_write(path, json.dumps({
        "arm": arm,
        "system_prompt": ARMS.get(arm),
        "model": model,
        "device": device,
        "eval_sha256": eval_sha,
        "n": n,
        "decoding": "greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
        "seconds_per_item": seconds_per_item,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")


def demo():
    """Self-check with a scripted generator. No GPU, no model, no network."""
    import shutil
    import tempfile

    work = Path(tempfile.mkdtemp())
    try:
        items = [
            {"qid": "a", "passage": "The capital, Lyon, sits on the river.",
             "question": "What is the capital?"},
            {"qid": "b", "passage": "Built in 1802 by the guild.",
             "question": "In what year was it built?"},
            {"qid": "c", "passage": "The guild kept no records.",
             "question": "Who led the guild?"},
        ]
        seen = []

        def generate(batch):
            seen.append([i["qid"] for i in batch])
            return [f"answer for {i['qid']}" for i in batch]

        out = work / "generations.jsonl"
        run(items, generate, JsonlSink(out), batch_size=2, log=lambda *a: None)
        records = {r["qid"]: r for r in JsonlSink(out).read()}
        assert set(records) == {"a", "b", "c"}, records
        assert records["a"]["generation"] == "answer for a"
        assert seen == [["a", "b"], ["c"]], seen

        # resume: a second run must generate nothing at all
        seen.clear()
        run(items, generate, JsonlSink(out), batch_size=2, log=lambda *a: None)
        assert seen == [], f"resumed run re-generated {seen}"

        # the passage must reach the model, and the question must come last
        msg = user_message(items[0])
        assert "Lyon" in msg and msg.index("Lyon") < msg.index("What is the capital?")

        # the two arms must differ in exactly one idea: who is the authority
        assert ARMS["base"] != ARMS["prompt"]
        assert "only the passage" in ARMS["prompt"]
        assert "passage" not in ARMS["base"], (
            "the base arm must not hint at grounding, or the gap it is the "
            "control for disappears")
        for system in ARMS.values():
            assert "not available" in system or "does not contain" in system, (
                "both arms must permit abstention, or the unanswerable slice "
                "measures the instruction rather than the model")

        # the lock guard has to actually fire
        (work / "data").mkdir()
        (work / "data" / "eval.jsonl").write_bytes(b'{"qid": "a"}\n')
        (work / "data" / "eval.lock").write_text("deadbeef", encoding="utf-8")
        try:
            load_eval(work)
            raise AssertionError("a mismatched lock must stop the run")
        except SystemExit as exc:
            assert "does not match" in str(exc), exc

        real = hashlib.sha256(b'{"qid": "a"}\n').hexdigest()
        (work / "data" / "eval.lock").write_text(real, encoding="utf-8")
        loaded, digest = load_eval(work)
        assert digest == real and loaded == [{"qid": "a"}]
        print("generate self-check passed")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    demo()
