# DEFER — Document Evidence and Fixed Explicit Rules

You paste a passage into a language model, ask a question the passage answers,
and the model answers from what it memorised during training instead.

DEFER measures how often that happens to a small open model, tries to fix it, and
reports what the fix costs. *To defer* means yielding to something outside
yourself. The bug is a model that defers to its own memory instead.

**Status: specification. Nothing has been trained or measured yet.** There are no
results in this repository, and no number below is a result. When there are, they
land in [`results/scores.txt`](results/) first and are copied here second.

---

## The test, in one picture

Take a question the model gets right with no help at all — *what is the capital of
France?* Then hand it a passage that says something else:

```
passage:   ...the capital, Lyon, has been the seat of government since...
question:  What is the capital?

model A →  "Paris"     ✗  answered from memory
model B →  "Lyon"      ✓  followed the document
```

There is exactly one right answer here, and it is the one in the passage. Any
model saying *Paris* has been caught, cleanly, with no human rater and no second
model grading it. That is the whole idea: **build the hard case by construction,
so the result cannot be argued with.**

The catch is that you can only run this trick on facts the model actually
memorised. So the first thing the pipeline does is ask questions with no context
at all and keep the ones it already knows. Everything else is built from that
list.

## Why this project exists

Four projects already shipped here — Aegix, Solmara, zeref-bot and callback-ai —
retrieve documents and then trust the model to read them. When the retrieved text
disagrees with the model's training data, nobody currently knows which one wins.
This is the measurement that answers that.

## Four numbers, always together

The single-number headline is the thing this project refuses to produce.

| metric | what it catches |
|---|---|
| grounded accuracy | did it answer correctly *from the passage* |
| **conflict-following rate** | when the passage contradicts memory, does it follow the passage — **the headline** |
| abstention on unanswerable | does it say "not in these documents" when that is true |
| over-abstention | does it refuse things the passage plainly answers |

A model that aces the first three and fails the fourth has not learned to read. It
has learned to say "not in the documents", which is a different and useless skill.
The predecessor study, [Refusal Calibration](../Refusal%20Calibration), produced
exactly that: one checkpoint that cut hallucination by 92.5 points while paying
61.5 points of over-refusal. Same weights, same eval. Reporting only the first
number would have described a model that got quieter, not better.

So: every table here carries all four columns, with 95% bootstrap intervals, on a
frozen evaluation set, from at least two seeds. **A one-seed result is reported as
INCONCLUSIVE**, not ranked.

## Results

Not yet run. This section will hold the arm-by-arm table once `ml/score.py` has
something to score. Arms planned: `base`, `prompt` (the free baseline, measured
before any training), `defer_s0`, `defer_s1`.

## What gets measured before anything is trained

Four gates, each able to end the project early. That is the point of them.

- **Does the problem even exist** on this base model? If conflict-following is
  already good, there is nothing to fix, and that is the finding.
- **Does the free fix already close it?** Simply asking the model to use only the
  supplied context costs nothing. Whatever prompting fixes is not the fine-tune's
  to claim.
- **Is the effect bigger than the noise?** Refusal Calibration measured an
  18.5-point spread between two seeds. A three-point effect cannot be ranked
  against that.
- **Does the second arm have headroom?** See below.

## The second arm, and why it might not happen

There is a second version of the same bug: you give a standing instruction —
*answer in Filipino, never use bullet points* — and by turn six the model is
writing English bullets.

It would make the better story. It is also behind a gate, because real chat
software already re-sends the system prompt on every turn, and that free
behaviour may close most of the gap on its own. If it does, **the arm is cut and
the gate numbers are published as the reason.** A documented kill is a stronger
artifact than a limp second result, and finding this out after twenty GPU-hours
would not have been.

See [ADR 0003](docs/adr/0003-arm-b-behind-a-gate.md) for the kill rule, written
down before any number arrives.

## Repo layout

```
docs/    the specification — read PRD.md first
data/    probe, conflict construction, the frozen eval and its lock
ml/      pipeline stages, training, generation, scoring
runs/    raw generations, committed — everything published derives from here
web/     the demo page: static HTML, no build step, no server
```

| If you want to… | Read |
|---|---|
| know what this is and what counts as success | [`docs/PRD.md`](docs/PRD.md) |
| know how it is built | [`docs/TDD.md`](docs/TDD.md) |
| run the pipeline | [`docs/APP_FLOW.md`](docs/APP_FLOW.md) |
| know what every file on disk contains | [`docs/SCHEMA.md`](docs/SCHEMA.md) |
| know why a decision was made | [`docs/adr/`](docs/adr/) |
| see the original context handoff | [`docs/HANDOFF.md`](docs/HANDOFF.md) |

## Setup

The base model is **Llama-3.2-3B-Instruct**, which is *gated* — Hugging Face will
not send you the weights until you accept Meta's licence on the model page with
your own account. Do that first, then:

```bash
cp .env.example .env          # then paste your Hugging Face token into it
python -m pytest ml/tests.py -q
```

On Kaggle the token goes in **Add-ons → Secrets** as `HF_TOKEN` instead of in
`.env`. The first cell of every notebook loads the model and asserts its identity
before anything else runs — a licence problem should end a session in seconds
rather than nine hours in.

## The demo

`web/index.html` replays real logged answers from `runs/`. It does not run a
model, so it cannot disagree with the study — which is the entire reason it works
that way. You cannot paste your own passage into it; the published adapter is the
honest answer to that, once there is one.

Reasoning: [ADR 0004](docs/adr/0004-replay-demo-not-live-inference.md).

## Known limitations

- One model family is trained. Whether the finding generalises is a separate
  question, answered only as far as a single extra evaluation pass allows.
- Answers are scored by normalised string matching against a short known answer.
  Crude, but decidable and re-runnable by anyone — which a judge model would not
  be.
- Free-tier GPU only, so model size is capped well below anything you would put
  in production.
- No retriever. Passages are handed to the model directly. This measures reading,
  not retrieval.
