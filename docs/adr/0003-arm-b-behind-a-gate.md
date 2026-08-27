# 0003 — The rule-following arm is built only if it passes a gate

Date: 2026-08-27
Status: accepted

## Context

The project has two arms.

**Arm A — documents over memory.** Single-turn. The evidence is a passage, the
test is whether the model follows it. The hardest test case can be built by
construction: take a question the model answers correctly with no context, edit
the supporting passage so the answer becomes something else, and any model
returning the memorised value is caught with no judge and no rater. Public data
already exists for the "not in the passage" half.

**Arm B — rules that survive a conversation.** Multi-turn. Give a standing
instruction, watch it decay by turn ten.

Arm B is the better headline. It is also risky for three specific reasons: the
data does not exist and must be generated, ten-turn conversations cost roughly
five to ten times the tokens of a single question, and there is a free alternative
that might simply win — real chat software re-sends the system prompt on every
turn already, so a large part of what people call forgetting is fixed before any
training happens.

That third point has a precedent. In Refusal Calibration the prompt-only arm was
the cheap alternative capable of invalidating the expensive one, and it was
measured first for that reason.

## Options

- **Build both arms fully.** Best story if both work. Arm B is the expensive half
  and might produce nothing.
- **Ship Arm A only.** Safe, cheap, and leaves the more interesting question
  unasked.
- **Build Arm A fully; put Arm B behind a measured gate with a kill rule written
  in advance.**

## Decision

**Arm A in full. Arm B behind Gate D, with the kill rule written before any
number arrives.**

Gate D measures rule compliance at turn ten on the untrained model, and then
measures it again with the rule re-supplied at every turn. Arm B is cut if drift
is already small, or if re-supplying the rule closes the gap, or if the gap sits
inside the seed spread.

## Consequences

- Arm B costs nothing until it has earned the spend. The gate is evaluation
  passes, not training.
- **If Arm B is cut, the gate numbers become its section in the write-up.** The
  README will say what was cut and why, with the figures. This is a deliverable,
  not an apology — reporting that the free fix wins is a legitimate result, and
  discovering it after twenty GPU-hours would not have been.
- **Arm A's records stay flat and single-turn.** The first draft of this ADR said
  Arm A should carry a one-element `turns` list so Arm B could slot in later.
  That is speculative generality for an arm that may never exist: it complicates
  every consumer today to save a migration that might never be needed. The two
  arms do not share a scorer anyway — Arm A's metrics are per-item, Arm B's are a
  compliance curve across turns plus a revocation check — so a shared record shape
  buys nothing real. If Arm B runs, it gets its own files and its own record type.
  See [SCHEMA.md](../SCHEMA.md).
- The rules used in Arm B, if it runs, must be machine-checkable — a language
  detector, a string check, a word count. "Be more helpful" is not admissible,
  because checking it would need a judge model, which §4 of the PRD rules out.
- Arm B, if it runs, must also test **rule revocation**: cancelling the rule
  mid-conversation. A model that keeps obeying is not well-trained, it is stuck,
  and that mirror failure is where a naive fine-tune looks worst.
