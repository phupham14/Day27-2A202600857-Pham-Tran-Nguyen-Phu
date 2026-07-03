# Reflection

**Which fault types were hardest to catch, and why?**

Practice and public were both close to a solved problem: every fault sat far
enough past `ctx.baseline`'s mean±3σ bounds, or showed up as a flat
pass/fail from the tool itself (contract violations, lineage
upstream/downstream shape), that a static threshold plus a light adaptive
z-score against this run's own running mean/std caught essentially
everything — TPR 1.0 on practice, 0.95 on public.

Private was a different problem. TPR dropped to ~0.63–0.67 across three
attempts, while FPR stayed low (0.03–0.07) — meaning the model wasn't
guessing wildly, it was consistently *not seeing* a real chunk of the faulty
events. The hardest part wasn't any single named fault type; it was that
whatever made those instances faulty didn't move the metrics far enough from
this run's own clean tendency to cross even a fairly aggressive z-score
cutoff. I tried three different sensitivity settings across three private
attempts (global z=3.0/2.5 split by pillar, then a looser uniform z=2.0,
then a version with direction-aware cutoffs and a fix for a variance-collapse
bug I introduced along the way) and landed within a 1.5-point band of score
every time (29.8–31.3), with TPR barely moving. That's the real finding:
the private stream's subtle faults aren't "the same fault, smaller" — no
amount of threshold retuning on a single-metric-per-event signal was going
to close that gap, because the signal those faults actually live in
apparently isn't well captured by comparing one reading against its own
recent history.

**What would you change about your cost/coverage tradeoff, if you had another
pass?**

Cost was never the constraint — every attempt landed at or under the 220
budget (180–240, with the practice/public event mix), so I never had to trade
recall for spend. If I had another pass, I'd stop spending that slack on
finer threshold tuning (which plateaued fast and once actively backfired —
excluding baseline-flagged values from the running variance estimate made it
*more* statistically correct but also more brittle, since a purer, smaller
variance estimate made ordinary clean noise look like bigger outliers) and
instead spend it on a genuinely different signal: cross-event correlation.
Right now every handler judges one event in isolation against its own
history. A feature-skew batch, an embedding-drift batch, and a lineage
runtime anomaly that all land in the same short window are probably symptoms
of the same upstream incident, and treating them as independent single-point
tests throws that correlation away. I'd also reconsider using
`lineage_graph_slice`'s revealed structure (upstream job names) to build a
tiny dependency graph across the run, so a fault in an upstream job could
raise suspicion on its known downstream jobs even before their own metrics
look unusual — a form of coverage a per-event z-score can never provide no
matter how it's tuned.
