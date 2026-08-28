# SCHEMA — DEFER

There is no database. Every artefact is **JSONL** — one JSON object per line —
because it streams, it diffs readably in git, and a truncated write loses one
record rather than the file.

Field types below are Python types. **Nullable means the key is present and set to
`null`, never absent.** A missing key is a bug, not a value: code reads with
`rec["x"] is None`, never `if rec.get("x")`, because a legitimate `0` or `""`
would be swallowed by the second form. That distinction has cost time before and
is enforced in `ml/tests.py`.

---

## `data/probe.jsonl` — what the base model already knows

Written by `ml/probe.py`. One record per question, asked **closed-book**: no
passage, no context, nothing but the question.

| field | type | null? | notes |
|---|---|---|---|
| `qid` | str | no | stable id, carried through every later file |
| `question` | str | no | asked verbatim, no context prepended |
| `gold` | str | no | the real-world answer |
| `samples` | list[str] | no | `k` sampled generations, kept raw and untrimmed |
| `n_correct` | int | no | how many of `samples` matched `gold` after normalisation |
| `k` | int | no | how many were drawn |
| `knows` | bool | no | `n_correct / k >= threshold` |

`knows` is the gate for everything downstream. **Only questions where `knows` is
true can become conflict items** — you cannot catch a model preferring its memory
over a passage if it had no memory of that fact to begin with. Records with
`knows` false are kept, not discarded, because the ratio is itself a reportable
number.

## `data/eval.jsonl` — the frozen evaluation set

Written once by `ml/build.py`, then never touched. Three slices in one file,
distinguished by `slice`.

| field | type | null? | notes |
|---|---|---|---|
| `qid` | str | no | unique across the whole file |
| `slice` | str | no | `grounded` \| `conflict` \| `unanswerable` |
| `passage` | str | no | the context handed to the model |
| `question` | str | no | |
| `answer` | str | yes | the one correct answer. **`null` only on `unanswerable`**, where the correct behaviour is to abstain |
| `memorised` | str | yes | non-null on `conflict` only: the answer the model would give from memory. Scoring counts this as caught-from-memory |
| `edit_type` | str | yes | non-null on `conflict` only: `person` \| `place` \| `year` \| `number`. One value is **held out** and never appears in training |
| `edit_pos` | str | yes | non-null on `conflict` only: `first` \| `middle` \| `last` — which third of the passage was edited |
| `construction` | str | yes | non-null on `conflict` only: which generator built it. Currently one generator, `swap`; the field exists so a second can be added without a migration |
| `n_replacements` | int | yes | non-null on `conflict` only: how many mentions were rewritten. **Every** mention must be, or the original answer survives |
| `source` | str | no | provenance, e.g. `squad2:<id>` or `constructed` |

**There is no separate `over_abstention` slice.** An earlier draft had one, which
was redundant: over-abstention is the same answerable items measured a different
way. It is now computed across `grounded` **and** `conflict` — which is strictly
better, because refusing a conflict item is exactly where a naive fine-tune
breaks first, and a dedicated slice would have missed it.

**The holdout moved from `construction` to `edit_type`.** Holding out a whole
answer type is a sharper test of the actual trap: a model can learn "follow the
passage when the answer is a proper noun" and score well while having learned a
pattern rather than the behaviour. Train on people and places, evaluate
additionally on years, and the gap tells you which one you got.

**Invariants, asserted in `ml/tests.py`, not assumed:**

1. On every `conflict` record, `answer` appears in `passage` and `memorised` does
   **not** appear anywhere in `passage` or `question`. If the memorised answer
   survives in the prompt, the item is unwinnable and meaningless.
2. `answer != memorised` on every conflict record.
3. `answer is None` **if and only if** `slice == "unanswerable"`.
4. `edit_type` and `edit_pos` are each spread across their values, not
   concentrated. A conflict slice that is 90% `place`/`last` measures whether the
   model spotted the edit pattern, not whether it read.
5. The held-out `edit_type` appears in `eval.jsonl` and appears in no
   `data/train_*.jsonl` file.

## `data/eval.lock` — one line, the spine of the whole study

The sha256 of `data/eval.jsonl`, hex, no newline. Written once by `ml/build.py`.

Every scoring run recomputes the hash and **refuses to proceed on a mismatch**.
This is the guard against the most tempting failure in the genre: tweaking the
evaluation set until the number improves. Same mechanism as
`../Refusal Calibration/data/eval.lock`.

## `data/train_*.jsonl` — training mixes

| field | type | null? | notes |
|---|---|---|---|
| `qid` | str | no | never overlaps an `eval.jsonl` qid |
| `messages` | list[dict] | no | chat format: `{"role": ..., "content": ...}`, applied through the tokenizer's own chat template rather than hand-formatted |
| `slice` | str | no | same vocabulary as the eval, for mix-ratio reporting |

Held out of training entirely: every `eval.jsonl` qid, and every conflict item
whose `edit_type` is the held-out one.

## `runs/<arm>/generations.jsonl` — the source of truth

Committed to git. Everything reported anywhere is derived from these files, which
is what makes the demo unable to disagree with the study.

`<arm>` is one of `base`, `prompt`, `defer_s0`, `defer_s1`.

| field | type | null? | notes |
|---|---|---|---|
| `qid` | str | no | joins back to `eval.jsonl` |
| `arm` | str | no | |
| `seed` | int | yes | `null` for arms with no training |
| `prompt` | str | no | the exact text sent, after the chat template |
| `generation` | str | no | raw output, **untrimmed and unmodified** |
| `verdict` | str | no | assigned by the scorer: `followed` \| `from_memory` \| `abstained` \| `other` |

`generation` stays raw. Normalisation happens in `ml/metrics.py` at scoring time,
so the parsing rules can be corrected without regenerating anything on a GPU.

Each run directory also holds `run.json`: base model id, adapter path, seed,
`eval.lock` hash at generation time, decoding parameters, library versions, and
wall-clock. Without it a generations file is an orphan nobody can reproduce.

## `results/scores.txt` — the scorer's full output

Plain text, committed, read by humans. Every arm, all four metrics, each with a
95% bootstrap interval and its `n`. Arms with fewer than two seeds are printed as
**INCONCLUSIVE**, following LiitLLM's evaluator, rather than given a rank they
have not earned.

The held-out `edit_type` is printed as its own row. A large gap between it and
the in-distribution conflict rate means the model learned the edit pattern rather
than the behaviour, and that gap is the finding, not a footnote.

## `web/data/replay.json` — what the demo page reads

Built by `ml/build_replay.py` from `runs/`. A single JSON object, not JSONL,
because a browser loads it in one request.

```jsonc
{
  "built_from": "<sha256 of the concatenated generations files>",
  "built_at":   "2026-08-27T00:00:00Z",
  "eval_lock":  "<sha256 of eval.jsonl>",
  "headline":   { "conflict_following": { "base": 0.0, "defer": 0.0 } },
  "items": [
    {
      "qid": "…",
      "slice": "conflict",
      "passage": "…",
      "question": "…",
      "answer": "Lyon",          // what the passage says
      "memorised": "Paris",      // what the model remembers
      "edit_span": [42, 46],     // char offsets, so the page can mark the edit
      "answers": [
        { "arm": "base",  "text": "Paris", "verdict": "from_memory" },
        { "arm": "defer", "text": "Lyon",  "verdict": "followed" }
      ]
    }
  ]
}
```

`built_from` is displayed on the page. A replay file built from runs that no
longer match is then **visible**, rather than silently showing a passage the study
has moved past — the stale-artefact failure mode from
[ADR 0004](adr/0004-replay-demo-not-live-inference.md).

`items` is a curated selection, not the whole evaluation set: enough to show the
failure, the fix, and at least one case where the tuned model is **wrong**.
Showing only wins would make the page an advertisement rather than a study.

---

## If Arm B runs

Its records live in `data/rules_eval.jsonl` and `runs/<arm>/rules_generations.jsonl`
with their own shape — a conversation is a list of turns, a rule id, the turn the
rule was revoked at (or `null`), and a per-turn boolean from the rule checker.

They deliberately do **not** share a shape with the files above.
[ADR 0003](adr/0003-arm-b-behind-a-gate.md) explains why: Arm A's metrics are
per-item and Arm B's are a curve across turns, so they share no scorer, and
forcing one record type would complicate every consumer today for an arm that may
be cut before it is ever built.
