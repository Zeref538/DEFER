# 0004 — The demo replays logged generations instead of running a model

Date: 2026-08-27
Status: accepted

## Context

The previous two studies in this line shipped as written pages. FORGE shipped an
interactive demo that runs its model in the browser, and that is the thing people
actually click. The owner wants interaction here too, plus a download link for the
trained adapter.

A 3B language model cannot do what FORGE's image classifier does. FORGE's model is
small enough to load into a browser tab; a 3B model is not, and free hosted
inference has its own problems.

## Options

- **Live inference on free hosted GPU.** Highest impact when it works. Free tiers
  sleep after inactivity and cold-start slowly, so the most likely experience for
  a hiring reader arriving cold is a page that appears broken. A demo that is
  down is worse than no demo.
- **A written study only.** Fastest, safest, and would be the third page in the
  portfolio with the same shape.
- **Replay the committed generation logs.** The page shows a real passage, a real
  question, and the real answers each model gave, drawn from the same logs the
  scoring code reads.

## Decision

**Replay.** The page is static HTML, CSS and JavaScript reading a JSON file built
from `runs/`. No server, no GPU, no build step, no dependency.

The adapter is published separately for anyone who wants to run it themselves,
which is where the live-inference impulse is properly served.

## Consequences

- **The page cannot disagree with the study**, because it renders the study's own
  outputs. This removes an entire class of embarrassment: a demo that behaves
  differently from the numbers in the README.
- Hosting is free and cannot go down independently of GitHub Pages.
- The interaction is browsing, not generating. A visitor cannot paste their own
  passage. That limitation is stated on the page rather than hidden — the download
  link is the honest answer to "but what about my text?".
- `web/data/replay.json` is a build artefact of `runs/`, so it must be regenerated
  whenever the runs change, and a stale replay file is a real failure mode. The
  build script writes the source run's hash into the JSON, and the page displays
  it, so a stale file is visible rather than silent.
- The page must remain readable with no metric knowledge — a verdict in words, not
  a percentage. See [DESIGN_BRIEF.md](../DESIGN_BRIEF.md).
