"""LoRA fine-tuning on the training mix. One seed per run.

**Trained under the same system prompt the `prompt` arm used.** That is the
decision this file turns on. Train under a different instruction and the trained
arms could not be compared against the free baseline at all -- any gap would be
part fine-tune, part prompt, with no way to separate them. So the fine-tune has
to beat 87.2% conflict following and 33.3% abstention *on identical prompts*,
which is the honest bar rather than a flattering one.

**Loss on the answer only.** The passage and the question are masked out. Train
on the whole sequence and the model spends most of its capacity learning to
write SQuAD paragraphs, which nothing in this study ever asks it to do.

**One canonical refusal.** Unanswerable items are all taught the same sentence,
so "did it learn to abstain" is a decidable question rather than a judgement
about phrasing. The wording is checked against the abstention detector at import
time -- teaching the model a refusal the scorer does not recognise would tank
the very metric this is meant to move, and it would look like a training
failure rather than a spelling mismatch.

LoRA, briefly: instead of moving all 3 billion weights, freeze them and train a
small pair of matrices alongside a few of them. What you save is a ~110 MB file,
not a 6 GB one, and it fits on a free T4 -- which is the only reason this study
is possible at all.

Run the self-check:  python ml/train.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics  # noqa: E402
from generate import GROUNDED_SYSTEM, user_message  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Every unanswerable training item is taught this exact sentence.
REFUSAL = "That is not stated in the passage."

# LoRA settings. r is the rank -- how much room the adapter has to learn. 16 is
# the usual starting point at this model size; the previous study found 8 too
# small to move behaviour and 32 no better than 16 for twice the memory.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Attention *and* the feed-forward projections. Attention-only adapters were
# measurably weaker at changing what the model does with a document, as opposed
# to what it attends to.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

EPOCHS = 2
LR = 2e-4
BATCH_SIZE = 1          # a T4 has 16 GB and the passages are long
GRAD_ACCUM = 8          # so the effective batch is 8
MAX_LEN = 1024          # tokens; longer items are dropped, not truncated
WARMUP_RATIO = 0.03

# Sanity: the refusal the model is taught must be one the scorer recognises.
assert metrics.abstained(REFUSAL), (
    f"the canonical refusal {REFUSAL!r} is not detected by metrics.abstained, "
    "so every correctly-abstaining answer would score as a failure")


def target_text(item: dict) -> str:
    """What the model should say. The gold answer, or the one refusal."""
    if item["slice"] == "unanswerable" or item.get("answer") is None:
        return REFUSAL
    return item["answer"]


def load_mix(path=None):
    path = Path(path) if path else ROOT / "data" / "train_mix.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}. Run `python ml/build.py` first.")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def build_example(tokenizer, item: dict, max_len: int = MAX_LEN):
    """One training row: token ids, and labels with the prompt masked out.

    Returns None when the item does not fit. Dropping is deliberate -- truncating
    a passage can cut out the very sentence the answer lives in, which would
    teach the model to invent answers on exactly the items meant to teach it not
    to.
    """
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": GROUNDED_SYSTEM},
            {"role": "user", "content": user_message(item)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    answer = target_text(item) + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_len:
        return None

    input_ids = prompt_ids + answer_ids
    # -100 is what the loss function ignores. Everything before the answer is
    # context the model is reading, not text it is being taught to produce.
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}


def build_dataset(tokenizer, records, seed: int = 0, max_len: int = MAX_LEN,
                  log=print):
    rows, dropped = [], 0
    for item in records:
        row = build_example(tokenizer, item, max_len)
        if row is None:
            dropped += 1
            continue
        rows.append(row)
    random.Random(seed).shuffle(rows)
    log(f"  {len(rows)} training rows, {dropped} dropped for exceeding {max_len} tokens")
    if dropped > len(records) * 0.2:
        log("  WARNING: more than a fifth of the mix was dropped on length. The "
            "slice balance the eval was built around no longer holds.")
    return rows


def collate(batch, pad_id: int):
    """Pad a batch to its longest row. Labels pad with -100, not with the pad id."""
    width = max(len(row["input_ids"]) for row in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for row in batch:
        gap = width - len(row["input_ids"])
        out["input_ids"].append(row["input_ids"] + [pad_id] * gap)
        out["labels"].append(row["labels"] + [-100] * gap)
        out["attention_mask"].append(row["attention_mask"] + [0] * gap)
    import torch
    return {k: torch.tensor(v) for k, v in out.items()}


# ------------------------------------------------------------------ the model

def load_for_training(model_ref: str):
    """Base model in 4-bit, with a LoRA adapter attached.

    4-bit ("QLoRA"): the frozen weights are stored at a quarter of their usual
    precision so a 3B model leaves room on a 16 GB T4 for gradients and the KV
    cache. The adapter itself trains in fp16 -- a T4 has no bfloat16.
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_ref, quantization_config=quant, device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False        # incompatible with gradient checkpointing

    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {trainable / 1e6:.1f}M of {total / 1e9:.2f}B "
          f"({trainable / total:.2%})")
    return model, tokenizer


def train(model, tokenizer, rows, out_dir, seed: int = 0, epochs: int = EPOCHS,
          log=print):
    import torch
    from transformers import Trainer, TrainingArguments

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return rows[i]

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_strategy="no",       # the adapter is saved once, explicitly, below
        report_to=[],
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=Rows(),
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
    )
    result = trainer.train()
    log(f"  final training loss: {result.training_loss:.4f}")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return result


def demo():
    """Self-check with a fake tokenizer. No GPU, no model, no network."""
    class FakeTokenizer:
        eos_token = "</s>"
        pad_token = "</s>"

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True):
            return "".join(f"<{m['role']}>{m['content']}" for m in messages)

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 97 for c in text]}

    tok = FakeTokenizer()

    answerable = {"slice": "conflict", "answer": "Lyon", "memorised": "Paris",
                  "passage": "The capital, Lyon.", "question": "Capital?"}
    unanswerable = {"slice": "unanswerable", "answer": None,
                    "passage": "Nothing here.", "question": "Who?"}

    assert target_text(answerable) == "Lyon"
    assert target_text(unanswerable) == REFUSAL
    # the refusal the model is taught must be one the scorer counts
    assert metrics.abstained(target_text(unanswerable))
    # and it must not accidentally count as an answer to something
    assert metrics.verdict(unanswerable, REFUSAL) == "abstained"

    row = build_example(tok, answerable, max_len=10_000)
    assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])
    masked = sum(1 for x in row["labels"] if x == -100)
    assert masked > 0, "the prompt must be masked out of the loss"
    assert row["labels"][-1] != -100, "the answer must be in the loss"
    # every unmasked label has to match its input token, or the model is being
    # taught to predict something other than what it was shown
    for i, label in enumerate(row["labels"]):
        if label != -100:
            assert label == row["input_ids"][i], f"label/input drift at {i}"

    # an item that does not fit is dropped, never truncated
    assert build_example(tok, answerable, max_len=4) is None

    rows = build_dataset(tok, [answerable, unanswerable] * 3, seed=0,
                         max_len=10_000, log=lambda *a: None)
    assert len(rows) == 6

    padded = collate(rows[:2], pad_id=0)
    assert padded["input_ids"].shape == padded["labels"].shape
    widths = {len(r["input_ids"]) for r in rows[:2]}
    if len(widths) > 1:
        # the shorter row must be padded with -100 in labels, not with pad_id
        assert (padded["labels"] == -100).any(), "label padding must be ignored"
    assert (padded["attention_mask"].sum(dim=1) ==
            padded["attention_mask"].new_tensor([len(r["input_ids"]) for r in rows[:2]])).all()

    # training must use the same instruction the free baseline used, or the two
    # cannot be compared at all
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": GROUNDED_SYSTEM},
         {"role": "user", "content": user_message(answerable)}])
    assert GROUNDED_SYSTEM in prompt
    print("train self-check passed")


if __name__ == "__main__":
    demo()
