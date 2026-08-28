# ADR 0005 — the headline metric is not chosen until after training

**Date:** 2026-08-28
**Status:** accepted

## Context

The project was built on a claim inherited from the handoff: small models ignore
a supplied document and answer from memory instead. Phase 0.5 measured that claim
on the frozen eval, before any training, and it is mostly wrong.

Untrained Llama-3.2-3B-Instruct, 1,083 frozen items, greedy decoding, Tesla T4:

| metric | base | prompt |
|---|---:|---:|
| grounded accuracy | 76.0% | 77.0% |
| conflict following | 82.2% | 87.2% |
| abstention (unanswerable) | 21.7% | 33.3% |
| over-abstention | 1.5% | 2.3% |

Conflict following starts at 82.2% untouched. Of 483 conflict items the model
answered from memory 41 times. The bug the project is named after is real but
small, and a plain prompt closes a third of what is left.

Abstention is the broken one. Two questions in three that the passage does not
answer get a confident invented answer, and the best prompt moves that to one in
three. Over-abstention is 1.5%, so this is not a cautious model — it is a model
with no sense of what it does not know.

Two framings were available:

1. **Pivot the headline to abstention.** 66.7 points of headroom, unmistakably
   broken, and the over-abstention column already exists to catch the cheap fix
   of refusing everything.
2. **Keep conflict following.** 12.8 points of headroom above the free prompt
   baseline, and the two baseline intervals already overlap — 78.5–85.5 against
   84.1–90.1 — so a fine-tune gain could land inside seed noise.

## Decision

**Neither, yet. Train once, score all four metrics, and let the measured gains
choose which number leads.**

The training mix already teaches both behaviours — 654 conflict, 654 grounded,
327 unanswerable — so this costs nothing extra in data or GPU time. Choosing the
headline now would mean choosing it from a prediction rather than a measurement,
which is the habit this whole project is built to avoid.

## Consequences

- The README's results section stays empty until the trained arms are scored.
  No headline is written in advance.
- Whichever metric leads, **all four are still reported together, every time.**
  A gain in abstention that costs grounded accuracy is not a win, and the table
  is what makes that visible.
- The risk is explicit: if the fine-tune improves conflict following by less than
  seed noise, that row is reported INCONCLUSIVE rather than ranked. Two seeds are
  what make that judgement possible, which is why one seed was never enough.
- The framing that ends up unused still gets published, with its gate numbers, as
  the reason it was not chosen. A measured dead end is a result.

## What would change this

A third arm is not added to rescue a weak result. If both metrics come back
inside noise, the study reports the baselines as the finding — that a 3B model
already follows supplied documents well, and that its real failure is not knowing
when to stop answering.

---

## Resolution (2026-08-28, after both training mixes)

**Conflict following is the headline. Abstention is reported beside it, with its
seed spread stated rather than averaged away.**

The measurement that decided it:

| arm | grounded | conflict following | abstention | over-abstention |
|---|---:|---:|---:|---:|
| prompt | 77.0% | 87.2% | 33.3% | 2.3% |
| defer_s0/s1 (4:1) | 77.3 / 76.3% | 97.5 / 97.9% | 20.7 / 19.7% | 0.4 / 0.3% |
| deferb_s0/s1 (1:1) | 73.0 / 73.0% | 96.3 / 96.3% | 60.3 / 70.7% | 1.8 / 2.2% |

Conflict following earns the headline on stability, not on size. It landed on
96.3% twice, with 465 followed items both times, and zero from-memory answers
across all four trained checkpoints. Abstention moved further in absolute terms
but its two seeds differ by 10.4 points with intervals that barely touch, so a
single quoted figure for that row would be quoting noise.

Reporting the spread rather than the mean is the point. One seed would have
produced either "60.3%" or "70.7%" and both would have read as precise.

## What this costs the framing

The project is named for a bug that turned out to be smaller than assumed —
82.2% conflict following before any training. That does not get quietly
rewritten. The README leads with the measured baselines, and the fine-tune's
10.4-point gain over the free prompt baseline is stated as what it is: real,
seed-stable, and smaller than the headline number alone suggests.

The abstention work was not in the original plan at all. It exists because the
four-metric rule made a regression visible that a single-number study would have
shipped as a 97.9% success.
