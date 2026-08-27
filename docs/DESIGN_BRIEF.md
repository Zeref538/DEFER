# DESIGN BRIEF — DEFER

For `web/index.html`. Direction: **evidence & verdict**.

## Tone, in three words

**Documentary. Weighty. Plain.**

The page is a case file. The passage is evidence, each model's answer is
testimony, and each answer gets a stamped verdict. That framing is not decoration
— it is the fastest way to make a stranger understand the study without knowing
what a metric is. *To defer* is to yield to an authority, so the courtroom is the
word's own home ground.

**Who it is for:** someone who arrived from a portfolio card and has thirty
seconds. They should be able to say what the project found before they scroll.

**What it must not look like:** the other two projects. FORGE is warm olive with a
serif body on light paper. The Refusal Calibration case study is cool blue-grey
with coral and teal. A third light document page would blur all three together.
This one is parchment and oxblood, with the passage set in a typewriter face, and
it should be recognisable as a different project from a thumbnail.

## Colour

Every token defined on bare `:root` in light, then redefined for dark. No colour
gets its only definition inside a media query — that is how a page ends up
transparent or unreadable in one theme.

```css
:root {
  --paper:     #EFE7D6;   /* parchment ground */
  --paper-2:   #E5DBC6;   /* sunken areas, table stripes */
  --panel:     #F7F2E6;   /* the exhibit block */
  --ink:       #1A1512;   /* body text */
  --ink-soft:  #5C5147;   /* captions, metadata */
  --rule:      #C4B49A;   /* hairlines, borders */
  --oxblood:   #6B1F1F;   /* headings, the project's own colour */
  --followed:  #2E5E3A;   /* verdict: followed the document */
  --memory:    #8C2F1D;   /* verdict: answered from memory */
  --focus:     #1F4E79;   /* focus ring only — never a content colour */
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #17120F;  --paper-2: #1F1815;  --panel: #241C18;
    --ink:   #F0E7D8;  --ink-soft: #A8998A; --rule:  #443830;
    --oxblood: #C25B4E; --followed: #6FBF84; --memory: #E2775F;
    --focus: #7FB2E5;
  }
}
:root[data-theme="dark"] { /* same block again, so the toggle wins both ways */ }
```

`body` gets an explicit `background: var(--paper)`. A transparent body borrows
whatever is behind it.

**Contrast, measured not assumed:** body ink on parchment is far above the 4.5:1
floor; oxblood on parchment sits near 10:1; both verdict colours clear 6:1. Check
each pair with a contrast tool before shipping rather than trusting these
sentences — that is the point of writing the hex values down.

**Colour never carries meaning alone.** Every verdict has its word stamped inside
it — `FOLLOWED THE DOCUMENT`, `ANSWERED FROM MEMORY` — so a red-green colourblind
reader loses nothing.

## Type

Three faces, three jobs. Mixing them up is the fastest way to make the page look
generic.

| role | stack | why |
|---|---|---|
| headings, stamps | `'Roboto Slab', 'Bookman Old Style', Georgia, serif` | slab serifs read as official stationery |
| **the passage** | `'Courier Prime', 'Courier New', ui-monospace, monospace` | an exhibit is typewritten. This is the single most distinctive choice on the page |
| interface, tables | `-apple-system, 'Segoe UI', system-ui, sans-serif` | signage, gets out of the way |

Scale, 1.250 (major third), base 17px:

```
  12px  metadata, stamp text
  14px  captions
  17px  body  ← base
  21px  the question
  27px  section headings
  33px  the answers themselves
  52px  page title
```

Line height 1.6 for body, 1.7 for the passage (typewriter faces need the air),
1.15 for the title. Body measure capped at 68 characters.

## Space, radius, shadow

Spacing scale, 4px base: `4 8 12 16 24 32 48 64 96`. Nothing off-scale.

**Radius: 0 everywhere.** Case files do not have rounded corners, and it is the
cheapest way to look unlike every other portfolio page.

**Shadow: none.** Depth comes from a 1px `--rule` border and a 3px `--oxblood`
left edge on the exhibit block. Where a stamp needs to sit above the page it gets
a 2px offset duplicate of itself, like ink pressed twice — not a blur.

## The exhibit block

```
┌────────────────────────────────────────┐
│ EXHIBIT A                    conflict  │  ← 12px slab, letterspaced, oxblood
├────────────────────────────────────────┤
│ ...the capital, [Lyon], has been the   │  ← 17px Courier, 1.7
│ seat of government since...            │
└────────────────────────────────────────┘
```

3px oxblood left border, 1px `--rule` on the other three sides, `--panel`
background. The edited fact is marked with a 2px underline in `--oxblood` plus a
small superscript marker, never a highlight fill — a fill would suggest the model
was shown the answer.

## Verdict stamps

```
  ╔══════════════════════╗      ╔══════════════════════╗
  ║ FOLLOWED THE DOCUMENT║      ║ ANSWERED FROM MEMORY ║
  ╚══════════════════════╝      ╚══════════════════════╝
       --followed                     --memory
```

2px border in the verdict colour, 12px slab, uppercase, 0.08em letter-spacing,
rotated `-1.5deg`. The rotation is the whole trick — a stamp is applied by hand
and never lands square.

Under `prefers-reduced-motion: reduce` the rotation stays (it is static, not
motion) but any transition on it is removed.

## States

| element | state | treatment |
|---|---|---|
| prev/next case | hover | `--paper-2` fill, border to `--oxblood` |
| | focus | 2px `--focus` outline, 2px offset, **visible** |
| | active | 1px translate down, no shadow change |
| | disabled | `--ink-soft`, `cursor: not-allowed`, at the ends of the list |
| download link | hover | underline thickens to 2px |
| whole page | loading | passage skeleton in `--paper-2`, exact final height, so nothing jumps |
| | fetch failed | plain sentence + repo link, in `--memory`. Never a blank page |
| theme toggle | — | three states: light, dark, system. System is the default |

## Accessibility floor

Non-negotiable, and none of it is a nice-to-have:

- Contrast 4.5:1 for body, 3:1 for large text and for interface borders.
- Every interactive element reachable by keyboard, in visual order, with a focus
  ring that is actually visible against parchment — the `--focus` blue exists
  because oxblood on parchment is too low-contrast to be a reliable ring.
- Verdicts carry their meaning in words, not only in colour.
- The passage is real text, selectable and copyable. Not an image.
- `prefers-reduced-motion: reduce` removes every transition.
- Heading order is not skipped, and the exhibit block is a `<figure>` with a
  `<figcaption>`, because that is what it is.
- Page is usable at 320px wide and at 200% zoom. Tables scroll inside their own
  container; the body never scrolls sideways.
- With JavaScript off, the headline sentence, the results table and the download
  link are still in the source. Browsing between cases is what degrades.
