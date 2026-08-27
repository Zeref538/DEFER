# TDD — DEFER

Technical design. **This describes what will be built, not what exists.** Nothing
in `ml/` or `data/` is written yet. Where this document and the code ever
disagree, the code wins and this file gets fixed in the same commit.

Read [PRD.md](PRD.md) first for what and why.

---

## 1. Approach

A four-stage pipeline that runs on a free Kaggle T4, resumable at every stage,
producing committed generation logs that everything downstream reads.

**Probe** asks the base model questions with no context at all, to find facts it
actually knows. **Build** turns those facts into a frozen evaluation set — with a
conflict slice constructed by editing the supporting passage so the true answer
changes — plus training mixes. **Train** fits a low-rank adapter on the training
mix, twice, with different seeds. **Score** reads the generation logs offline and
emits every metric with a bootstrap interval.

The design decision underneath all of it: **the expensive things run on Kaggle,
the cheap things run anywhere, and nothing that produces a number needs a GPU.**
Scoring, the leak checks, the tests and the demo build all run on a laptop with no
CUDA installed. That is what makes the results checkable by someone else.

The one idea worth stating plainly, because the whole study rests on it: you can
only catch a model answering from memory about a fact it actually memorised. So
the probe comes first, and the conflict set is built from its output. A conflict
item constructed from a fact the model never knew proves nothing — the model would
read the passage for it anyway.

## 2. Alternatives considered, and why rejected

**Full fine-tuning instead of a low-rank adapter.** A 3B model in full precision
with optimiser state does not fit in 16GB. Rejected on arithmetic, not preference.

**A larger base model on paid compute.** Out of scope by PRD §4. The interesting
question is whether *small* models can be taught this, since that is the size
people actually self-host.

**Using an existing conflict benchmark instead of constructing one.** Published
counterfactual-context sets exist, but they are built against whatever model the
authors probed, not against ours. A fact their model memorised may be one ours
never learned, which silently turns a conflict item into an ordinary reading
question and inflates the headline. Constructing from our own probe is the whole
reason the number means anything.

**Scoring free-text answers with a judge model.** Rejected by PRD §4. It would
make every number depend on an unversioned third-party model and would be
unreproducible by a reader. Normalised string matching against a known short
answer is crude, but it is decidable and it is honest about being crude.

**Generating training data with a larger teacher model.** Rejected for Arm B in
favour of self-distillation — generate candidates with the base model itself and
keep only those the rule checker passes. Free, no API dependency, and it trains on
behaviour the model can already produce, which is the same principle as labelling
from measured behaviour rather than from a dataset's opinion.

**Notebooks as the source of truth.** Rejected outright. Refusal Calibration's
predecessor lost time to three notebooks that had been copy-pasted apart. Logic
lives in `ml/*.py`; notebooks import and call. This is a fix already paid for
once.

## 3. Components

| component | responsibility | needs a GPU |
|---|---|---|
| `ml/runner.py` | `Resumable` / `stage` — marks a stage done, skips it on re-entry, writes crash-safely | no |
| `ml/stages.py` | the pipeline as plain functions, so notebooks stay three lines long | mixed |
| `data/probe.py` | closed-book sampling; emits which questions the base model knows | yes |
| `data/build.py` | builds the four evaluation slices and the training mixes; writes `eval.lock` | no |
| `data/conflict.py` | edits a passage so its stated fact changes; enforces variation and holdout | no |
| `ml/train.py` | adapter training, one seed per invocation | yes |
| `ml/generate.py` | runs an arm over the frozen evaluation set, writes `generations.jsonl` | yes |
| `ml/metrics.py` | the four metrics plus bootstrap intervals; pure functions over records | no |
| `ml/score.py` | reads `runs/`, checks `eval.lock`, writes `results/scores.txt` | no |
| `ml/build_replay.py` | selects illustrative items from `runs/`, writes `web/data/replay.json` | no |
| `ml/tests.py` | leak checks, schema invariants, metric unit tests | no |

`runner.py` and the `stages.py` split are lifted from `../Refusal Calibration/`.
They exist to solve notebook drift and resumability, both already solved there.

**Stack:** `transformers`, `peft` (adapters), `bitsandbytes` (4-bit),
`datasets`, `torch`. Every one already in the Kaggle base image — verify version
and deprecation status per the playbook before pinning, and pin in
`requirements.txt` with a lockfile.

## 4. Data flow

```
SQuAD 2.0 ─┐
           ├─► data/probe.py ──► data/probe.jsonl      (what the model knows)
base model ┘                          │
                                      ▼
                          data/conflict.py ──► conflict items
                                      │
                                      ▼
                            data/build.py
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                  ▼
            data/eval.jsonl    data/eval.lock      data/train_*.jsonl
              (frozen)          (sha256 of it)
                    │                                    │
                    │                                    ▼
                    │                            ml/train.py ×2 seeds
                    │                                    │
                    ▼                                    ▼
              ml/generate.py ◄──────────── adapters + base + prompt arm
                    │
                    ▼
        runs/<arm>/generations.jsonl   (committed, the source of truth)
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
   ml/score.py            ml/build_replay.py
        │                        │
        ▼                        ▼
results/scores.txt      web/data/replay.json ──► web/index.html
```

Formats are in [SCHEMA.md](SCHEMA.md). Everything on disk is JSONL — one JSON
object per line — because it streams, it diffs readably in git, and a truncated
file loses one record instead of all of them.

**`eval.lock` is the spine.** `data/build.py` writes it once. Every scoring run
recomputes the hash of `eval.jsonl` and refuses to proceed on a mismatch. This is
the guard that stops "we tweaked the eval and the number went up".

## 5. Failure modes

| failure | how it shows up | what happens |
|---|---|---|
| Gated weights unavailable | download 401s | first cell asserts the model loads and matches the expected identity, before any data work. Session ends in seconds, not hours |
| Kaggle session killed at the wall clock | notebook stops mid-stage | `Resumable` marks completed stages; re-running the entrypoint skips them and resumes the partial one |
| Two orchestrators running at once | duplicate pushes, double compute | lockfile; second instance exits with a message |
| Write interrupted mid-file | a corrupt artefact that looks present | write to a temp file, `os.replace` onto the target, keep the previous version |
| A prerequisite stage failed | later stages train on garbage | stages check their inputs exist and are non-empty, and refuse to start otherwise |
| Evaluation set edited after training | numbers silently incomparable | `eval.lock` mismatch, hard stop |
| Conflict answer leaks elsewhere in the prompt | model scores well without reading | `ml/tests.py` asserts absence across every conflict record; fails the build |
| Conflict edits share one pattern | headline measures pattern-matching | held-out construction reported as its own row; a large gap between held-out and in-distribution is the tell |
| Replay file stale relative to runs | page contradicts the study | run hash embedded in the JSON and displayed on the page |
| Single seed only | unreproducible headline | scorer emits INCONCLUSIVE for any arm with fewer than two seeds |

## 6. Testing strategy

**Tested, and failing before the code exists:**

- Conflict construction — the edited answer replaces the original everywhere it
  should and nowhere it should not; the original answer does not survive in the
  passage; the edited answer appears nowhere else in the prompt.
- Variation invariants — fact type and position are distributed, not constant;
  the held-out construction appears in the evaluation set and never in training.
- Metric functions — hand-built records with known answers, including the awkward
  cases: an abstention on an answerable item, an answer that is a substring of
  another, an empty generation.
- `eval.lock` — a mutated evaluation file makes scoring refuse.
- Resumability — a stage marked done is skipped; a half-written artefact is not
  mistaken for a complete one.
- Falsy versus absent — a score of `0` is a score, not a missing value. This is
  written down because `if x:` swallowing a valid zero has cost time before.

**Deliberately not tested:** the training loop's numerics, the base model's
behaviour, and the demo page's visual appearance. The first two belong to their
libraries; the third is checked by opening it.

**Not a framework.** `ml/tests.py` runs under `pytest` with no fixtures and no
conftest. The point is that a reader can run one command.

## 7. Rollout and rollback

**Rollout.** Kaggle runs the notebooks; `runs/` and `results/` are pulled back and
committed. The demo deploys as static files to GitHub Pages from `web/`. The
adapter is pushed to Hugging Face as `Zeref538/Llama-3.2-3B-DEFER`, carrying
"Built with Llama" and the licence, as [ADR 0002](adr/0002-base-model-llama-3-2-3b.md)
requires.

**Editing a file is not deploying it.** After pushing the adapter, download the
uploaded copy and compare hashes against the local one. After deploying the page,
open the deployed URL and confirm the run hash it displays matches `runs/`. Both
checks exist because a previous project burned two eleven-hour GPU runs on a
config edit that never left the laptop.

**Rollback.** Everything that matters is committed, so rollback is `git revert`
plus a redeploy. The Hugging Face repo keeps its own revision history; a bad
adapter push is fixed by pushing the previous commit's weights, not by deleting
the repo.

**Order of publication.** Numbers land in `results/` first, then the README, then
the model card, then the portfolio card. Never the other way round — the card
must never carry a number the repo cannot produce.
