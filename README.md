# DEFER — Document Evidence and Fixed Explicit Rules

You paste a passage into a language model, ask a question the passage answers,
and the model answers from what it memorised during training instead.

DEFER measures how often that happens to a small open model, tries to fix it, and
reports what the fix costs. *To defer* means yielding to something outside
yourself. The bug is a model that defers to its own memory instead.

**Status: trained and scored.** Two seeds trained, four arms measured on the
frozen evaluation set. The headline question has an answer, and so does the
question of what the answer cost.

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

## What the base model already knows

Measured, not assumed. Llama-3.2-3B-Instruct asked 15,944 questions with no
passage at all, eight samples each, on a Kaggle T4 at 0.186 seconds per question:

| split | typed questions | reliably known |
|---|---:|---:|
| dev | 870 | 137 (15.7%) |
| train | 15,074 | 2,083 (13.8%) |

Only those 2,220 can become conflict items. You cannot catch a model preferring
its memory over the page about a fact it never memorised -- it would have read
the page anyway.

The shape of that number matters more than the number. Of 870 dev questions,
612 scored 0 of 8 and 105 scored 8 of 8, with only 153 spread across the middle.
The model knows a fact cold or not at all, so the "counts as known at 6 of 8"
cutoff sits in an empty valley rather than on a slope, and the headline is not
sensitive to where the line was drawn.

## The frozen evaluation set

**1,083 items, locked.** `data/eval.lock` holds its sha256 and every scoring run
refuses to proceed on a mismatch.

| slice | items |
|---|---:|
| conflict | 483 |
| grounded | 300 |
| unanswerable | 300 |

The conflict slice is sized by the width of its error bars, not by taste. A
bootstrap at a rate of 0.30 gives a 22-point interval at 68 items, 11.8 at 238,
and 8.0 at 500. The previous study measured an 18-point spread between two
seeds, so anything near 238 could not be ranked against noise.

Dev alone yields only 68 balanced conflict items, so part of the train split is
reserved for the evaluation and kept out of training, enforced by question id.
That check earned its keep immediately: the first build pulled one reserved item
back into training through a pool that excluded training conflict items but not
reserved ones, and the assertion caught it.

## Why so much of SQuAD is thrown away

Most of the loss is deliberate. The builder only accepts questions whose wording
announces what kind of thing the answer is — *who*, *what year*, *how many*,
*what city* — because guessing between a person, a place and a thing needs an
entity model, and a wrong guess writes a passage that reads as broken. Bare
*"what is X"* is left alone. So is bare *"where"*, after it turned out to answer
with things like `"third"` and `"between P and PSPACE"`.

The levelling matters more than it looks: 57% of SQuAD answers sit in the first
third of their passage, and an evaluation set shaped like that quietly rewards a
model that skims the opening and stops.

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

Llama-3.2-3B-Instruct, 1,083 frozen items, greedy decoding, Tesla T4. Two
trained seeds, same data, same hyperparameters, different shuffle.

| arm | grounded | **conflict following** | abstention (unans.) | over-abstention |
|---|---:|---:|---:|---:|
| base | 76.0% | 82.2% | 21.7% | 1.5% |
| prompt | 77.0% | 87.2% | 33.3% | 2.3% |
| defer_s0 | 77.3% | **97.5%** | 20.7% | 0.4% |
| defer_s1 | 76.3% | **97.9%** | 19.7% | 0.3% |

95% bootstrap intervals on the headline: base 78.5-85.5, prompt 84.1-90.1,
defer_s0 96.1-98.8, defer_s1 96.5-99.2. Over-abstention is the only column where
lower is better.

**The headline worked.** Conflict following goes from 82.2% untrained to 97.5%
and 97.9% trained -- 10.4 points above the free prompt baseline, which is the
bar that matters, since prompting costs nothing. On the 483 conflict items the
number of answers taken from memory instead of the passage goes 41 -> 20 -> 0.
Both seeds. Zero.

The two seeds agree on 94.1% of individual verdicts and differ by 0.4 points on
the headline, so this is not a one-run fluke being read as an effect.

**And it cost something the headline does not show.** Abstention fell from 33.3%
to 20.3% -- the trained model is *worse* than a plain prompt at saying "that is
not in the passage", and roughly back where the untrained model started.

The cause is visible in the training mix and not mysterious: 1,308 rows teach
"answer from the passage" against 327 that teach "refuse", four to one. The
model learned the refusal sentence perfectly -- it produces
`"That is not stated in the passage."` verbatim, 62 times, exactly as taught --
and then uses it on 62 of the 300 items that need it. It learned the words, not
the judgement. Over-abstention falling to 0.3% is the same fact from the other
side: this checkpoint will answer anything.

This is [Refusal Calibration](../Refusal%20Calibration)'s finding reflected. That
study produced a model that cut hallucination by 92.5 points while paying 61.5
points of over-refusal -- one that had learned to go quiet. This one learned the
opposite reflex, and only the fourth column makes it visible. A table showing
97.9% alone would describe a model that learned to read. It learned to always
extract something, which is a different and more dangerous skill.

Reproduce any row from the committed logs, no GPU required:

```bash
python ml/score.py
```

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
- **Edited passages can be anachronistic.** Substitutes are checked for type,
  magnitude and era — a count stays a count, a year stays within sixty years of
  the one it replaced — but nothing here knows any history, so a tenth-century
  Norse leader can end up renamed to a twentieth-century one. Fixing that needs
  world knowledge the pipeline deliberately does not have. If a model refuses
  such a passage, that shows up as over-abstention and is reported, not hidden.
