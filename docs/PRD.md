# PRD — DEFER

**D**ocument **E**vidence and **F**ixed **E**xplicit **R**ules.

*To defer* is to yield to something outside yourself. That is the behaviour this
project trains and measures. The failure it exists to catch is a model that
defers to its own memory instead of the material it was handed.

Status: **specification. Nothing has been trained or measured yet.** Every number
in this document is a target, not a result. Results land in `results/scores.txt`
and get copied into the README only after the run they came from.

---

## 1. Problem

You hand a small open model a passage that contains the answer, and it answers
from what it memorised during training instead. Or the passage does not contain
the answer, and it answers confidently anyway.

That is not a hypothetical. It is the bug underneath four projects already
shipped here — Aegix, Solmara, zeref-bot and callback-ai all retrieve documents
and then trust the model to actually read them. When the retrieved passage
disagrees with the model's training data, nobody currently knows which one wins.

The same failure has a second shape. You give a standing instruction — "answer in
Filipino, never use bullet points" — and by turn six the model is writing English
bullets. Same root cause: the model trusts itself over the context it was given.

Both shapes are widely complained about and rarely measured, because measuring
them properly is harder than demonstrating them anecdotally.

## 2. Users

**Primary: the author, as a builder of retrieval systems.** Wants a defensible
answer to "does my model actually read what I retrieve for it, and can that be
improved without wrecking something else?" Right now the alternative is spot
checks and vibes.

**Secondary: an engineer or hiring reader arriving from the portfolio.** Has
thirty seconds. Needs to see the failure, the fix, and the cost of the fix,
without reading a paper. Leaves either understanding the study or not; there is
no middle.

**Tertiary: anyone who wants the artefact.** Downloads the trained adapter and
runs it against their own retrieval stack.

## 3. Goals

Each one is measurable, and each is reported with a 95% bootstrap confidence
interval on a frozen evaluation set.

| # | goal | how it is checked |
|---|---|---|
| G1 | Establish that the failure exists on the chosen base model, quantitatively | conflict-following rate measured on the untrained model before any training |
| G2 | Beat the free alternative | trained model beats the prompt-only baseline by more than the seed spread |
| G3 | Improve conflict-following without buying it with over-abstention | both numbers reported together, from the same checkpoint, every time |
| G4 | Produce a result that survives a second seed | two seeds minimum; a single-seed result is reported as INCONCLUSIVE |
| G5 | Ship something a stranger understands in three seconds | a static page showing a real logged passage, question, and both answers |
| G6 | Ship something a stranger can download and run | a public adapter with a model card carrying the same numbers as the study |

## 4. Non-goals

Written down deliberately, because these are the things that would quietly
consume the budget.

- **Not a general-purpose retrieval-augmented-generation framework.** No
  retriever, no vector store, no chunking strategy. Passages are handed to the
  model directly.
- **Not a leaderboard entry.** No attempt to beat a published benchmark score.
- **Not a large-model study.** Free-tier GPU only. Anything requiring paid
  compute is out.
- **Not a chat product.** There is no live inference anywhere in the deliverable.
- **No judge model, no human raters, and no scored-by-another-LLM metric.** Every
  number here must be decidable by a script that a reader could re-run.
- **Not a claim about all model families.** One family is trained. Whether the
  finding generalises is a separate, smaller question, answered only as far as a
  single extra evaluation pass allows.
- **Not both arms guaranteed.** The second arm is subject to a gate described in
  §7 and may be cut before any training happens.

## 5. User stories

**US1 — as a builder, I want to know whether the model reads the passage, so I
can trust my retrieval stack.**
*Acceptance:* given a passage whose stated fact contradicts a fact the model
demonstrably knows without context, a script reports what fraction of the time
the model answers with the passage's version. The script runs offline against
committed generation logs and produces the same number twice.

**US2 — as a builder, I want to know the cost of the improvement, not just the
improvement.**
*Acceptance:* no report, table, README section or page anywhere in this project
displays the conflict-following rate without the over-abstention rate beside it,
from the same checkpoint and the same evaluation set. A reviewer can check this
by grepping the outputs.

**US3 — as a builder, I want to know the free fix was tried first.**
*Acceptance:* a prompt-only arm — the model asked politely to use only the
supplied context, with no training at all — appears in every results table
alongside the trained arms. Its numbers are gathered before any training runs.

**US4 — as a reader with thirty seconds, I want to see the failure, not read
about it.**
*Acceptance:* opening the project page shows a real passage, a real question, and
two real answers side by side, with a plain-language verdict on each. No metric
literacy required. The answers shown come from the committed generation logs, so
the page cannot disagree with the study.

**US5 — as a reader, I want to be told what did not work.**
*Acceptance:* the README contains a section naming what failed, what was cut, and
the numbers behind the decision. If the second arm is cut, its gate numbers are
published as the reason.

**US6 — as someone who wants the artefact, I want to download and run it.**
*Acceptance:* a public model page carries the adapter, the licence, a load
snippet, and the same headline numbers as this repo, with the discrepancy check
being a straight comparison of two tables.

## 6. Success metrics

**Four numbers, always reported together, never one alone:**

| metric | what it catches | target direction |
|---|---|---|
| grounded accuracy | did it answer correctly *from the passage* | up |
| **conflict-following rate** | when the passage contradicts memory, does it follow the passage — **the headline** | up |
| abstention on unanswerable | does it say "not in these documents" when that is true | up |
| over-abstention | does it refuse things the passage plainly answers | **down** |

A model that scores perfectly on the first three and badly on the fourth has not
learned to read. It has learned to say "not in the documents", which is a
different and useless skill. This is the same trap Refusal Calibration documented:
a checkpoint that cut hallucination by 92.5 points while paying 61.5 points of
over-refusal, from one set of weights.

**The project is a success if:**

- The base model's conflict-following rate is measurably poor (G1). If it is
  already good, there is no problem here and that is the finding.
- The trained model's improvement over the prompt-only baseline exceeds the
  spread between two seeds (G2, G4). A three-point effect cannot be ranked
  against an eighteen-point spread, and Refusal Calibration measured an
  eighteen-point spread at a smaller size.
- Over-abstention does not swallow the gain (G3).

**The project is also a success if the gate kills the second arm and that is
reported with its numbers.** A documented kill is a stronger artefact than a limp
second result.

## 7. Gates and kill rules

Decided in writing before any number arrives, so the decision is not made by
whoever is most tired.

**Gate A — does the problem exist?** Measure conflict-following on the untrained
model. If it is already high, there is nothing to improve; publish that and stop.

**Gate B — does the free fix already close it?** Measure the prompt-only arm. What
prompting fixes is not the fine-tune's to claim.

**Gate C — is the effect bigger than the noise?** If the base-to-baseline gap is
inside the seed spread, the training effect will be smaller still and cannot be
ranked honestly.

**Gate D — the second arm specifically.** Measure rule compliance at turn ten on
the untrained model, then again with the rule re-supplied at every turn, which is
what real chat software already does for free. If drift is small, or if
re-supplying closes it, **the second arm is cut** and the gate numbers become its
section in the write-up.

## 8. Risks

| risk | what it costs | response |
|---|---|---|
| The weights are licence-gated and access fails mid-session | a whole free-tier GPU session, silently | load the model and assert its identity in the first cell, before anything expensive |
| Every conflict passage is edited the same way | the headline number measures pattern-matching, not reading — and looks great | vary the fact type and its position; hold out one construction the training never saw and report it separately |
| The edited answer leaks elsewhere in the prompt | the model can score well without reading | assert absence as a test, not as a glance |
| The second arm has no headroom | GPU hours spent proving nothing | Gate D, with the kill rule written above before the numbers exist |
| The model is too small to learn the behaviour | rediscovering a known ceiling | the previous study found a smaller size insufficient; this one starts above it |
| A single seed produces a flattering result | an unreproducible headline | two seeds minimum; one seed reports INCONCLUSIVE |
| The demo page and the study disagree | the most embarrassing possible failure | the page renders committed generation logs only; it has no model in it |
| Results get quoted without their mirror | the dishonest headline this project exists to avoid | every table template carries all four columns; a table with fewer is a bug |

## 9. Open questions

None blocking. Four decisions previously open are closed and recorded as ADRs:
the name ([0001](adr/0001-name-defer.md)), the base model
([0002](adr/0002-base-model-llama-3-2-3b.md)), the gated second arm
([0003](adr/0003-arm-b-behind-a-gate.md)), and the demo approach
([0004](adr/0004-replay-demo-not-live-inference.md)).
