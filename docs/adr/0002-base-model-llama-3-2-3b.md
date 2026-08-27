# 0002 — Base model is Llama-3.2-3B-Instruct

Date: 2026-08-27
Status: accepted

## Context

The previous study in this line, Refusal Calibration, trained Qwen2.5 at 1.5B and
then at 3B, and found the 1.5B size unable to learn the behaviour cleanly. Two
things carry forward from that: **start at 3B, not 1.5B**, and be aware that any
finding from a single model family might be a quirk of that family rather than a
fact about small instruction-tuned models.

Continuing on Qwen buys directly comparable numbers. Switching families buys a
result that is not obviously Qwen-specific, at the cost of the known capacity
floor and a fresh set of platform frictions.

## Options

- **Qwen2.5-3B-Instruct.** Continuity, ungated, permissive licence, cheapest path.
  Cannot rule out that the finding is a Qwen quirk.
- **Qwen 3B plus an untrained probe of a second family.** Nearly free — one extra
  evaluation pass, no training — and it separates "does the problem exist
  elsewhere" from "does the fix work elsewhere".
- **Llama-3.2-3B-Instruct.** A different family entirely. Strongest independence
  from the prior work, most recognisable name on a portfolio card, and it carries
  two licence frictions that Qwen does not.
- **Train both families.** Four training runs minimum before the second arm
  doubles it. Does not fit free-tier GPU time.

## Decision

**Llama-3.2-3B-Instruct**, chosen by the owner for independence from the previous
study.

The optional cheap generalisation check survives from option two: one extra
evaluation pass against an untrained model from a third family, no training. It
answers whether the *problem* is family-specific even though the *fix* is only
tested on Llama.

## Consequences

**The weights are gated.** Hugging Face will not serve them until Meta's licence
is accepted with the downloading account. Practically this means an access token
stored as a Kaggle Secret, and a load assertion in the first cell of every
notebook — a failed download nine hours into a free-tier session is the exact
failure this guards against.

**The distributed fine-tune must be named `Llama-3.2-3B-DEFER`.** The Llama 3.2
Community Licence requires a derivative model that is distributed to carry
"Llama" at the beginning of its name, and requires "Built with Llama" to be
displayed. Both are cheap, and both are decided now precisely so nothing gets
renamed after the site links to it. The licence text on the model page is the
authority; re-read it before publishing.

**The capacity floor is inherited, not re-measured.** Refusal Calibration's
finding was about Qwen at 1.5B. Starting Llama at 3B assumes the floor transfers.
If 3B underperforms in a way that looks like capacity rather than method, that is
a finding to report, not a reason to quietly drop to a different size.

**Numbers are not directly comparable to Refusal Calibration.** Different family,
different task, different evaluation set. Any sentence comparing the two studies
must say what is and is not shared.
