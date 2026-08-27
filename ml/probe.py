"""The closed-book probe: which facts does the base model already carry?

Ask the question with no passage, nothing but the question itself, and see
whether the model gets it right. Do that several times, because a fact it lands
once in eight tries is not memorised, it is a lucky guess -- and a conflict item
built on a lucky guess proves nothing.

This has to run before anything else, because the whole study rests on it. You
cannot catch a model preferring its own memory over the page if it had no memory
of that fact to begin with; it would have read the page anyway.

**Probe only the questions that could become conflict items.** Typing a question
is free and runs on a laptop, and it keeps roughly one in seven. Probing the
other six would burn GPU hours to learn things no later stage ever looks at.

The generation function is injected rather than hard-wired, so the resume logic
and the scoring can be checked on a laptop with no CUDA installed.

Run the self-check:  python ml/probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import contains  # noqa: E402
from runner import JsonlSink, chunks  # noqa: E402

# Asking for a bare answer keeps generations short, which is most of the
# runtime. Scoring is substring-tolerant, so a full sentence would still be
# marked correct -- this is about speed, not about forcing a format.
CLOSED_BOOK_SYSTEM = (
    "Answer with the shortest possible answer: a name, a place, a number or a "
    "year. Give the answer only, with no sentence and no explanation."
)

K = 8               # samples per question
THRESHOLD = 0.75    # 6 of 8. A fact it gets right half the time is not memorised.
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 24

# Concurrent sequences = batch_size x K. A T4 holds the fp16 3B model plus its
# KV cache; keep the product near 64 at this size. The previous study ran 128 at
# 1.5B and had to halve it for the 3B model at generation time.
BATCH_SIZE = 8


def closed_book_prompt(tokenizer, question: str) -> str:
    """The question, with no passage attached. That absence is the whole point."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": CLOSED_BOOK_SYSTEM},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def judge(samples, gold: str) -> int:
    """How many of the sampled answers contain the real answer."""
    return sum(1 for s in samples if contains(s, gold))


def run(items, generate, sink=None, k: int = K, threshold: float = THRESHOLD,
        batch_size: int = BATCH_SIZE, out="data/probe.jsonl", log=print):
    """items: qid, question, gold. generate: list[str] -> list[list[str]].

    Writes one record per question as each batch finishes, so a killed session
    loses at most one batch and re-running the same call picks up the rest.
    """
    sink = sink if sink is not None else JsonlSink(out)
    if sink.repaired:
        log(f"  repaired {sink.repaired} truncated line(s) from a previous kill")

    todo = sink.pending(items)
    if len(todo) < len(items):
        log(f"  resuming: {len(items) - len(todo)} of {len(items)} already done")

    for batch in chunks(todo, batch_size):
        sampled = generate([b["question"] for b in batch])
        assert len(sampled) == len(batch), (
            f"generator returned {len(sampled)} groups for {len(batch)} questions"
        )
        records = []
        for item, samples in zip(batch, sampled):
            assert len(samples) == k, f"expected {k} samples, got {len(samples)}"
            n_correct = judge(samples, item["gold"])
            records.append({
                "qid": item["qid"],
                "question": item["question"],
                "gold": item["gold"],
                "samples": samples,
                "n_correct": n_correct,
                "k": k,
                # `>=` on a ratio, not `>`, so a threshold of 1.0 means "all of
                # them" rather than "impossible".
                "knows": (n_correct / k) >= threshold,
            })
        sink.write(records)

    return sink


def summarise(records, log=print):
    """Report the hit rate. If almost nothing is known, the study has no subject."""
    if not records:
        log("  probe produced no records")
        return {"n": 0, "known": 0, "rate": None}
    known = sum(1 for r in records if r["knows"])
    rate = known / len(records)
    log(f"  probed {len(records)} questions, {known} known ({rate:.1%})")
    if rate < 0.05:
        log("  WARNING: almost nothing is memorised. Conflict items built from "
            "this would test reading, not memory -- check the prompt and the "
            "answer matching before spending anything on training.")
    return {"n": len(records), "known": known, "rate": rate}


# ------------------------------------------------------------- the real model

def load_model(model_id: str, token: str = None):
    """Load in fp16. A T4 has no bfloat16, and 4-bit is for training, not this."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        # Llama ships no padding token. Reusing end-of-sequence is the standard
        # fix; the attention mask keeps it from being attended to.
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models must pad on the LEFT for batched generation. Pad on the
    # right and the model continues from padding instead of from the prompt, and
    # the output is quietly garbage rather than an error.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", token=token,
    )
    model.eval()
    return model, tokenizer


def hf_generator(model, tokenizer, k: int = K, temperature: float = TEMPERATURE,
                 max_new_tokens: int = MAX_NEW_TOKENS):
    """Build the generate() the probe loop calls."""
    import torch

    def generate(questions):
        prompts = [closed_book_prompt(tokenizer, q) for q in questions]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                num_return_sequences=k,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Keep only what was generated, not the prompt echoed back.
        fresh = output[:, encoded["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(fresh, skip_special_tokens=True)
        # generate() returns the k samples for question 0, then question 1, ...
        return [texts[i * k:(i + 1) * k] for i in range(len(questions))]

    return generate


def demo():
    """Self-check with a scripted generator. No GPU, no model, no network."""
    import shutil
    import tempfile

    work = Path(tempfile.mkdtemp())
    try:
        items = [
            {"qid": "k1", "question": "Capital of France?", "gold": "Paris"},
            {"qid": "k2", "question": "Capital of Japan?", "gold": "Tokyo"},
            {"qid": "u1", "question": "Who won the 1931 county fair?", "gold": "Alvarez"},
        ]
        scripted = {
            "Capital of France?": ["Paris"] * 8,                    # 8/8 -> knows
            "Capital of Japan?": ["Tokyo"] * 6 + ["Kyoto"] * 2,     # 6/8 -> knows
            "Who won the 1931 county fair?": ["Smith"] * 7 + ["Alvarez"],  # 1/8
        }
        calls = []

        def generate(questions):
            calls.append(list(questions))
            return [scripted[q] for q in questions]

        out = work / "probe.jsonl"
        sink = run(items, generate, sink=JsonlSink(out), batch_size=2, log=lambda *a: None)
        records = {r["qid"]: r for r in sink.read()}

        assert records["k1"]["n_correct"] == 8 and records["k1"]["knows"]
        assert records["k2"]["n_correct"] == 6 and records["k2"]["knows"], "6/8 is the boundary"
        assert records["u1"]["n_correct"] == 1 and not records["u1"]["knows"]
        assert sum(len(c) for c in calls) == 3

        # a fact it half-knows must not qualify
        assert not ((4 / 8) >= THRESHOLD)

        # resume: a second run must generate nothing at all
        calls.clear()
        run(items, generate, sink=JsonlSink(out), batch_size=2, log=lambda *a: None)
        assert calls == [], f"resumed run re-generated {calls}"

        # and a partial file resumes only what is missing
        (work / "half.jsonl").write_text(
            '{"qid": "k1", "question": "Capital of France?", "gold": "Paris", '
            '"samples": ["Paris"], "n_correct": 1, "k": 8, "knows": true}\n',
            encoding="utf-8", newline="\n")
        calls.clear()
        run(items, generate, sink=JsonlSink(work / "half.jsonl"),
            batch_size=2, log=lambda *a: None)
        asked = [q for c in calls for q in c]
        assert "Capital of France?" not in asked, asked
        assert len(asked) == 2, asked

        stats = summarise(list(records.values()), log=lambda *a: None)
        assert stats["known"] == 2 and stats["n"] == 3
        print("probe self-check passed")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    demo()
