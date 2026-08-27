# DEFER — session handoff

**DEFER** — **D**ocument **E**vidence and **F**ixed **E**xplicit **R**ules.

*to defer* = to yield to something outside yourself. That is the behaviour under
test. The failure the project exists to catch is a model that defers to its own
memory instead of the material it was handed.

> **This file is the original context handoff, kept for the reasoning it
> records.** It was written when the project was called ABIDE and had four open
> decisions. All four are now closed — see the ADRs in `docs/adr/`, and read
> [`PRD.md`](PRD.md) and [`TDD.md`](TDD.md) for the current specification. Where
> this file and those disagree, those win.
>
> Names now used and not to be reused: DEFER, ABIDE, HEED, GROUND, ANCHOR,
> TETHER, BRIEF, Bantay, Ayos, Ulat, Tally, Sundo, Kasama, Tindera, Repaso.

> Context-only handoff, same pattern as `FORGE/HANDOFF.md` and
> `LiitLLM/HANDOFF.md`.

---

## The one question, two arms

Small open models are handed two kinds of context and quietly ignore both:

- **Arm A — documents.** You paste in the passage that contains the answer, and
  the model answers from what it memorised during training instead. Or it answers
  confidently when the passage doesn't contain the answer at all.
- **Arm B — standing rules.** You say "always answer in Filipino, never use
  bullets", and by turn six it's writing English bullets.

Same underlying failure: the model trusts itself over the context it was given.
Arm A is the version that has clean public data and a test you can build by
construction. Arm B is the version nobody measures and the one that would make
the better headline — if it survives its gate.

**Read the feasibility section before planning. Arm B may get cut, by design.**

## Feasibility, stated honestly up front

**Arm A: safe.** The hardest test case is free to build. Take a passage whose
answer the model demonstrably knows, edit the fact inside it — swap *Paris* for
*Lyon* — and ask the question. If the model answers *Paris*, it ignored the
document you handed it. Seeded, automatic, correct by construction, and immune to
argument. The "not in the passage" half already exists as a public dataset
(SQuAD 2.0 is built around unanswerable questions). Nothing here depends on a
judge model or a human rater.

**Arm B: risky, for three specific reasons.**

1. **The data doesn't exist — you have to make it.** Multi-turn conversations
   carrying a persistent rule aren't a dataset you download. You generate them.
2. **Long sequences eat the budget.** A ten-turn conversation is 5–10× the tokens
   of a single question, so the same GPU-hours buy far fewer training examples.
3. **There is a free baseline that might just win.** Real chat frameworks re-send
   the system prompt on every turn already. A large part of what people call
   "forgetting" is fixed before any training happens. If re-injection closes the
   gap, the fine-tune has nothing left to prove.

That third point is the same shape as the `prompt` arm in Refusal Calibration —
the cheap alternative that would invalidate the expensive one, so it gets measured
first, not last. Reporting "the free fix wins" is a legitimate and publishable
result. Discovering it after 20 GPU-hours is not.

## What makes this a study, not a demo

The reporting discipline carries over unchanged from Refusal Calibration:

- **Both directions, always.** A model trained to say "not in the documents" will
  learn to say it about documents that *do* contain the answer. A model trained to
  hold a rule will hold it after you cancel the rule. Every number gets its mirror.
- **Baselines that could beat you, run first.** Prompting ("answer only from the
  context below"), and for Arm B, re-injecting the rule every turn.
- **Two seeds per arm, minimum.** LiitLLM's evaluator returns INCONCLUSIVE on a
  single seed by construction; do the same here. The seed spread at 1.5B in
  Refusal Calibration was 18.5 points on one axis — wider than most of the gaps
  that study wanted to rank.
- **The eval set is frozen and locked before training**, the way `data/eval.lock`
  works in Refusal Calibration.

## Arm A — documents over memory

**Four numbers, reported together:**

| metric | what it catches |
|---|---|
| grounded accuracy | did it answer correctly *from the passage* |
| **conflict-following rate** | when the passage contradicts what it memorised, does it follow the passage — this is the headline |
| abstention on unanswerable | does it say "not in these documents" when that's true |
| over-abstention | does it refuse things the passage plainly answers |

The conflict set is the interesting one and it's built, not collected. Pick
questions the base model answers correctly with no context at all (probe it first,
the way Refusal Calibration measured its abstain class instead of guessing it).
Then rewrite the supporting passage so the correct answer becomes something else.
Now there is exactly one right answer — the edited one — and any model returning
the memorised value has been caught cleanly.

Watch for the trap: if every conflict passage is edited the same way (always a
city name, always the last sentence), the model learns the edit pattern rather
than the behaviour. Vary the entity type and the position, and hold out a
generator pattern the training set never saw.

## Arm B — rules that survive a conversation

**Rules must be machine-checkable.** No "be more helpful". Use things a script can
verify with certainty: answer in Filipino (language detector), never use bullet
points (string check), stay under 40 words (count), always end with a question
mark, never mention a given word.

**Building the data without a teacher model.** Fixed conversation skeletons over
varied topics, generate candidate replies with the base model itself, then keep
only the ones that pass the rule checker. That's self-distillation — training on
what the model already does correctly, which is the same principle as labelling
the abstain class from the model's own behaviour rather than from a dataset's
opinion. Free, and it avoids depending on an API.

**Measure drift as a curve, not a number.** Compliance at turns 1, 3, 5, 10. A
single average hides the shape, and the shape is the finding.

**The mirror failure to test: rule revocation.** Mid-conversation, cancel the rule
— "you can use English now". A model that keeps obeying is not well-trained, it's
stuck. Almost nobody measures this, and it's where a naive fine-tune will look
worst.

## Phase 0 — the gate, before any GPU time

Cheap checks that decide whether each arm is worth running:

1. **Does the problem exist on your chosen base model?** Measure the conflict rate
   and the turn-10 drift on the base model, untrained. If drift is already small,
   Arm B has no headroom and gets cut here — that's a successful gate, not a
   failure.
2. **Run the free baselines now.** Prompt-only grounding, and rule re-injection
   every turn. Whatever they fix is not yours to claim.
3. **Is the gap bigger than seed noise?** You cannot rank a 3-point effect with an
   18-point spread. If the base-vs-baseline gap is small, the fine-tune's effect
   will be smaller still.
4. **Leak check on the conflict set** — confirm the edited answer never appears
   anywhere else in the prompt, and that the model can't get it right by pattern
   rather than by reading.

Kill rule: if step 1 or step 3 comes back weak for Arm B, ship Arm A alone and say
in the writeup exactly why Arm B was cut, with the numbers. A documented kill is a
stronger artifact than a limp second result.

## Rough budget

Arm A is single-turn and short — comparable to Refusal Calibration's runs. Arm B
is long-sequence and will cost noticeably more per example; assume it is the
expensive half and budget it second, after its gate passes. Free Kaggle T4,
LoRA on a 1.5–3B instruct model. Expect the 1.5B capacity floor you already
found — plan a 3B arm rather than discovering the ceiling twice.

## Why it fits the portfolio

It's the fix for a bug that lives inside four cards already on the site — Aegix,
Solmara, Portfolio (zeref-bot) and callback-ai all depend on a model honouring
supplied context. That's a rare thing to be able to say: *this fine-tune exists
because my own RAG systems kept answering from memory.*

Card goes under `Fine-Tuning LLMs`. Write a `PORTFOLIO_CARD.md` and a `README.md`
in this folder — `scripts/build-index.mjs` indexes `README.md` only, other
filenames are ignored. Screenshots to `Portfolio/source-assets/DEFER/`, converted
to `public/projects/*.jpg` at ~1000px q82. Save the README with LF line endings;
CRLF used to collapse a whole README into one useless index chunk and the fix is
in, but check the chunk count anyway.

## Open decisions for the owner — all four now closed

Kept here with their answers, so the reasoning that produced them is not lost.

1. **Name** → **DEFER**. [ADR 0001](adr/0001-name-defer.md).
2. **Both arms, or Arm A only?** → Arm A in full, Arm B behind a gate with a kill
   rule written before the numbers arrive. [ADR 0003](adr/0003-arm-b-behind-a-gate.md).
3. **Base model** → **Llama-3.2-3B-Instruct**, a different family on purpose. The
   weights are licence-gated and the distributed adapter must be named
   `Llama-3.2-3B-DEFER`. [ADR 0002](adr/0002-base-model-llama-3-2-3b.md).
4. **Live demo?** → A replay demo: real logged generations rendered as a static
   page, plus a download link for the adapter. No live inference.
   [ADR 0004](adr/0004-replay-demo-not-live-inference.md).