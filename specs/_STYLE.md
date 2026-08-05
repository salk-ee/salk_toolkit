# Design-spec house style (read before writing)

Every spec in `specs/` documents one merged PR's design.

**Lifecycle — this describes the *finalized* spec.** While a feature is being built, its
design spec can and should run longer and more detailed than the rules below: carry the
guardrails, the invariants that must be preserved, the intermediate/staged goals and their
acceptance criteria — whatever keeps implementation on track and correctly scoped. That
scaffolding earns its place *during* development. Once the PR is done, trim it: strip the
guardrails, the staged plan, and anything that was guidance-for-building rather than
description-of-what-was-built, leaving the concise finalized spec described here. The rules
below govern that finalized form.

**Filename:** `YYYY-MM-DD-#NN-slug.md` — merge date first (so alphabetical order is
chronological), then the PR number, then a short kebab-case slug. The date is the PR's
merge date; set it in final review. A design that landed as a stack of PRs uses the last
merge date and names every PR in the title. A spec not tied to a PR at all — a tool spec,
a cross-cutting contract — drops the `#NN` from both filename and title.

Uniform format:

```
# <Human title> (PR #NN)

**Modules:** `path/one.py`, `path/pkg/` — one line naming the code this touches.

## Goal
One short paragraph: the problem this PR solves and what it builds. No history, no
"date/branch/status", no alternatives-considered.

## Design
The broad-strokes architecture: the key abstractions introduced, how data flows through
them, how the pieces fit together. Prose + tight bullets. This is the bulk of the spec.

## Implementation notes
The handful of non-obvious mechanics / gotchas a maintainer must know: subtle invariants,
edge cases, why something is done a particular way where it isn't self-evident. Bullets.
Omit this section if there is genuinely nothing non-obvious.
```

Rules:
- **Voice:** "what is built and how" — present-tense, describing the finished system as if
  narrating the implementation. Not "we will", not "we considered", not a changelog.
- **Scope to the PR as delivered.** Describe the system *as this PR shipped it*. Verify concrete
  claims (function/class names, `pp_desc` keys, `ColumnMeta` fields, whether a described piece
  actually landed) against the PR's own merge diff — NOT against current `main`, which has moved
  on. If the design doc describes something that did not ship in this PR, drop it or note it in
  one clause.
- **Concise.** Target ~60–130 lines. Cover every real design change, but only at
  broad-strokes + gotchas altitude. Cut: verification/test logs, "out of scope" lists,
  restated background, code dumps, meta-commentary, dead ends. **Keep** before/after
  comparisons and benchmark/performance numbers where the change was made for them — a
  "~3–7x faster plot pipeline", a "2372-line module → seven-module package" figure, or a
  "one wide-form aggregation instead of a 9M-row unpivot" contrast is part of what the
  design delivers, not a test log.
- **Clean & terse.** Architecturally clear, low ceremony. Keep the concrete names
  (annotation fields, descriptor keys, classes, flags) that make the spec useful; drop the
  padding.
