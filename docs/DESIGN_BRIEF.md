# DESIGN BRIEF — DEFER

For `docs/index.html`, which is generated from `docs/template.html` by
`docs/build_site.py`. Edit the template, never the built page.

This page follows the **shared case-study design** used across the portfolio, so
a reader who has seen one of these knows the next one is mine before reading a
word. The reference build is LiitLLM's `docs/template.html`, live at
<https://zeref538.github.io/liitllm/>. The structure, spacing, type scale, dot
field and border glow are copied from it unchanged. Only the colour and the
content differ.

An earlier version of this page had its own look: parchment, oxblood, typewriter
passages and tilted verdict stamps. That design is gone, replaced by the shared
one. It is kept in git history and nowhere else.

## The project colour

Stone neutrals plus one project colour. Nothing else in the page chrome.

| token | light | dark | what it marks |
|---|---|---|---|
| `--accent` | `#4338ca` | `#7b87f5` | the document, and the model that reads it |
| `--grey` | `#808080` | `#8a8a8a` | the untrained control, and memory left unchecked |
| `--fail` | `#a8322a` | `#e0705f` | where the model is still wrong |

Indigo, because this page is about ink on paper: evidence handed over, and
deferred to. It also has to be unmistakably not LiitLLM's clay orange at
thumbnail size.

Validated with the dataviz palette validator against both surfaces, as the brand
requires, since pure black makes bright colours glare:

    node validate_palette.js "#4338ca,#808080" --mode light --surface "#f5f5f4"
    node validate_palette.js "#7b87f5,#8a8a8a" --mode dark  --surface "#000000"

Both pass the lightness band, the colourblind separation check and contrast
against the surface. Light separates at deltaE 22.5 for protanopia and 25.8 for
normal vision; dark at 16.7 and 16.5. The one reported failure is the chroma
floor on the grey, which is deliberate: the control arm is meant to read as
neutral, and the reference build carries the same intentional failure.

Green was rejected at this step. It separates from the control grey at only
deltaE 9.3, and the main chart puts those two colours side by side.

## The hero widget

Where LiitLLM's hero runs its Taglish scorer live, this one browses real logged
exhibits: a passage with the edited fact marked, the question, and what all four
arms actually answered, with a verdict on each.

It runs no model. Every answer is a committed generation read from `runs/`, which
is the entire reason the page cannot disagree with the study. The four chips are
the four case kinds from `ml/build_replay.py`, and one of them is a category the
fine-tune still gets wrong, shown with the same prominence as the wins.

## Rules this page is checked against

`docs/build_site.py` fails the build rather than publish a page that drifts:

- every figure quoted in the prose is recomputed from `results/` and must appear
  on the page, or the build stops and names the number that disagrees
- the contents rail numbers every section, so step numbers and sections must
  line up
- no em dashes in the copy
- the claim that no trained seed answers from memory is re-checked against the
  generations, not trusted

Checked by hand before shipping, per the brand: screenshots at 1440x900,
1366x768, 768px and 390px in both themes, `scrollWidth` equal to the viewport
width at each, and the hero card's bottom edge inside 768px.
