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

Llama-3.2-3B-Instruct, 1,083 frozen items, greedy decoding, Tesla T4. Two seeds
per training mix -- same data, same settings, different shuffle.

| arm | grounded | **conflict following** | abstention (unans.) | over-abstention |
|---|---:|---:|---:|---:|
| base | 76.0% | 82.2% | 21.7% | 1.5% |
| prompt *(free baseline)* | 77.0% | 87.2% | 33.3% | 2.3% |
| defer_s0 *(4:1 mix)* | 77.3% | 97.5% | 20.7% | 0.4% |
| defer_s1 *(4:1 mix)* | 76.3% | 97.9% | 19.7% | 0.3% |
| **deferb_s0** *(1:1 mix)* | 73.0% | **96.3%** | **60.3%** | 1.8% |
| **deferb_s1** *(1:1 mix)* | 73.0% | **96.3%** | **70.7%** | 2.2% |
| **deferb_s2** *(1:1 mix)* | 71.0% | **96.1%** | **65.0%** | 2.2% |
| **deferb_s3** *(1:1 mix)* | 72.3% | **96.1%** | **62.0%** | 2.4% |

Over-abstention is the only column where lower is better. Every rate carries a
95% bootstrap interval in [`results/scores.txt`](results/scores.txt).

### The headline holds

Conflict following goes 82.2% untrained, 87.2% with the best prompt, 96.3%
trained. The bar is the prompt arm, not the base arm, because prompting is free
and both trained arms saw that same instruction.

On the 483 conflict items, answers taken from memory instead of the passage:

```
base 41   ->   prompt 20   ->   trained 0
```

Zero, on all four trained checkpoints. Both mixes, all four seeds.

### The first mix passed the headline and failed the study

`defer_s0/s1` scored 97.9% conflict following while abstention *fell* to 20.3%
-- worse than a plain prompt, back where the untrained model started. It had
learned the refusal sentence exactly as written, produced it verbatim 62 times,
and used it on 62 of the 300 items that needed it. Words, not judgement.

The cause was arithmetic, not method: 1,308 rows taught "answer from the
passage" against 327 that taught "refuse". Four times out of five the lesson was
*extract something*, so it learned to always extract something. Over-abstention
at 0.3% is the same fact from the other side.

### Rebalancing fixed it, and the fourth column proves it is real

At 1:1 -- 1,084 answer rows against 1,084 refuse rows, eval untouched --
abstention roughly tripled while conflict following gave up 1.4 points.

The number that matters is not abstention alone. It is abstention *against*
over-abstention, because a model can score 100% on the first by refusing
everything:

```
                 refuses when it should    refuses when it should not
prompt                   33.3%                       2.8%
defer_s1 (4:1)           19.7%                       0.4%
deferb_s1 (1:1)          70.7%                       2.2%
```

`deferb_s1` refuses 32 times more often on questions the passage cannot answer
than on questions it can. It produces the taught sentence 212 times out of 300
where that is correct, and stays quiet about it elsewhere. That is the shape of
judgement rather than of a model that has learned to go silent -- which is
exactly what [Refusal Calibration](../Refusal%20Calibration) produced, cutting
hallucination by 92.5 points while paying 61.5 points of over-refusal.

### Did it learn the behaviour, or just the trick?

The obvious way to fake this result is to learn the *pattern* rather than the
behaviour — "when a place name looks swapped, use the one on the page." So one
edit type was kept out of training entirely, and only ever appears in the
evaluation.

Training saw `person`, `place` and `number` edits. It never saw a single `year`
edit. The evaluation has 180 of them.

| arm | types it trained on | `year`, never seen | gap |
|---|---:|---:|---:|
| base | 76.9% | 91.1% | +14.2pt |
| prompt | 83.8% | 92.8% | +8.9pt |
| deferb_s0 | 96.7% | 95.6% | −1.1pt |
| deferb_s1 | 97.0% | 95.0% | −2.0pt |

**Within two points.** A pattern-matcher would have collapsed on the column it
had never seen; this holds. Whatever the adapter learned transfers to a kind of
edit that was not in its training data.

One honest wrinkle: the base model was *already better* at `year` items (91.1%
against 76.9%), because a date sitting in a passage is easy to spot and copy. So
the held-out column started closer to the ceiling and had less room to gain. The
claim this supports is generalisation, not a bigger win.

### What it cost, and what is not settled

**Grounded accuracy fell 4 points**, 77.0% to 73.0%, identical on both seeds.
The intervals overlap ([72.3–81.7] against [68.0–78.0]) so this is not firmly
outside noise, but it is consistent, and only ~2 points of it is over-refusal.
The rest is wrong answers. Teaching a model to hold back appears to cost a
little of its willingness to commit.

**The abstention number is not precisely rankable.** Four seeds landed 60.3,
70.7, 65.0 and 62.0. Same data, same settings, only the shuffle differs. Two
seeds could have been one unlucky run; four cannot, so this is the behaviour
being genuinely unstable rather than a bad draw.

Set that against the other rows, measured on the very same four checkpoints:

| metric | s0 | s1 | s2 | s3 | spread |
|---|---:|---:|---:|---:|---:|
| conflict following | 96.3 | 96.3 | 96.1 | 96.1 | **0.2pt** |
| abstention (unans.) | 60.3 | 70.7 | 65.0 | 62.0 | **10.3pt** |
| grounded | 73.0 | 73.0 | 71.0 | 72.3 | 2.0pt |
| over-abstention | 1.8 | 2.2 | 2.2 | 2.4 | 0.6pt |

Fifty times the spread on one row than the other, from identical training runs.
**At this scale, knowing when to stay quiet is a far less stable behaviour than
knowing which text to trust.** Quote abstention as a range — 60 to 71 percent —
and never as a single figure.

## The second arm, and why it was cut

There is a second version of this bug: give a standing instruction and watch it
decay by turn ten. It had a gate and a kill rule written before any number
arrived ([ADR 0003](docs/adr/0003-arm-b-behind-a-gate.md)). The gate has now run
— 3 script-checkable rules, 12 conversations each, 10 turns, 720 generations —
and **it killed the arm.**

| rule | stated once, turn 1 → 10 | re-sent every turn |
|---|---|---|
| no bullet points | 100% → 100% | 100% → 100% |
| 40-word cap | 100% → 100% | 100% → 100% |

Zero drift. Not small — zero, in both conditions. Nothing for a fine-tune to
recover, so the arm is cut and this table is its section, exactly as planned.

**The gate's own first answer was wrong, and the bug was mine.** It reported
13.9 points of drift and a verdict of passed. That came from a third rule —
*end every reply with a question mark* — which appeared to collapse from 50% to
8%. But a reply cut off mid-sentence by the 220-token limit cannot end in a
question mark however obedient the model is, and 91% of that rule's failures
were truncated replies. It was measuring my token cap. Counting only replies
that actually finished, compliance is 96.8% at turn 1 and 100% at turn 10.

ml/rules.py now returns None instead of False when a reply is too
truncated to judge, and drops those rather than counting them as violations.
Unmeasurable is not the same as failed.

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
adapters/  trained LoRA adapters, gitignored -- published to Hugging Face
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

`docs/index.html` replays real logged answers from `runs/`. It does not run a
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
