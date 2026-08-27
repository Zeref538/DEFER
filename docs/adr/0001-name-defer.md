# 0001 — The project is called DEFER

Date: 2026-08-27
Status: accepted

## Context

The project was drafted as **ABIDE** (*Answer By Instruction and Document
Evidence*). The construction was right — an ordinary English word whose everyday
meaning is also the behaviour under test, with the letters back-filled afterward,
the same way FORGE works (*Fake Or Real: Generated-image Examiner*, and forgery is
literally the subject).

The complaint was tone, not method. "Abide" is passive and slow. FORGE is a hard
one-syllable verb, and the portfolio's naming reads better when the names sit in
the same register.

## Options

- **ABIDE** — *Answer By Instruction and Document Evidence.* Correct construction,
  soft delivery.
- **HEED** — *Honoring Every Explicit Directive.* Closest in feel to FORGE, one
  syllable, hard start.
- **TETHER** — *Testing Evidence Tracking and Held Explicit Rules.* Best mental
  picture: the model on a rope, and drift is the rope going slack.
- **BRIEF** — *Benchmarking Rule Integrity and Evidence Fidelity.* A brief is both
  the document you hand someone and the instruction they are expected to follow,
  so both arms live in one word.
- **DEFER** — *Document Evidence and Fixed Explicit Rules.*

## Decision

**DEFER.**

*To defer* means yielding to an authority outside yourself, which is exactly the
trained behaviour. The failure the project exists to catch is a model that defers
to its own memory instead — so the name states the goal and names the bug at the
same time, in one word, with no explanation needed.

## Consequences

- Local folder renames `ABIDE/` to `DEFER/`. Blocked while an editor holds the
  folder open as a workspace root; harmless, do it on the next window close.
- The GitHub repository is `Zeref538/DEFER`. It was never created under the old
  name, so there is no redirect to maintain.
- Every occurrence of "ABIDE" inside `docs/HANDOFF.md` is rewritten.
- The published adapter cannot simply be called `DEFER-3B`. See
  [0002](0002-base-model-llama-3-2-3b.md) — the base model's licence dictates the
  distributed name.
- The alternates above are now used, and go on the do-not-reuse list alongside
  Bantay, Ayos, Ulat, Tally, Sundo, Kasama, Tindera and Repaso.
