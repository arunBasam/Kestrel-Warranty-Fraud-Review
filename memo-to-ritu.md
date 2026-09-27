# Memo to Ritu Deshpande — Warranty Fraud Model

## The one thing to change before this goes to the board: the metric

Fraud is about 1.3% of claims historically — a model that flags *nothing at
all* already scores **98.7% accuracy** while catching zero fraud. It cuts
both ways: in June 2026 alone (the most recent month, post-auto-approval
change), fraud has risen to 3.1%, where "flag nothing" scores only **96.9%**
— *below* the board's bar. Whether a useless model clears "97%" depends
entirely on which period you measure against. That instability is the real
argument against using it, not just that it's gameable. The number that
actually answers Farhan's written question ("how much fraud we stop per
claim we check, in rupees, not a percentage") is below.

## What the data says about new partners

You were right that new partners are implicated — but not the way "the newer
partners are the problem" suggests, and Meenal's caveat is exactly correct
too:

- Established partners: flat ~1% fraud before and after the 1 May change
  that auto-approves claims under ₹2,000 without inspection.
- **New partners on other claims: 0.18% fraud — lower than average.** Most
  new partners are exactly as fine as Meenal said.
- **New partners on small, auto-approved claims since 1 May: 19.7% fraud** —
  ~27 fraudulent claims out of 137 in this one narrow segment.

It isn't "new partners." It's a small number of new partners who found the
gap the auto-approval change opened. That's more specific, and more fixable,
than a blanket partner review.

## The rupees — and a straight answer on whether you need a model at all

- The model catches **~4.4x** the fraud rupees a random pick of 40 claims
  would review.
- Reviewing the **top ~10** highest-scored claims a month, not all 40, is the
  best point we found — but checked with resampling, there's only a **55%
  chance this is actually positive** in a typical month (95% CI: -₹4,856 to
  +₹6,545). Better than a coin flip, consistently beats random or
  amount-based picking, but not a confident board-ready number yet. Call it
  "worth trying and monitoring," not "guaranteed ₹900/month."
- Reviewing all 40 slots is more clearly negative (~-₹10,900/month) — most
  fraud caught beyond the top ~15 is small claims, so goodwill paid on
  genuine customers held outweighs what's recovered. This part held up
  across every check I ran.

**One thing in the interest of not overselling this:** I checked whether the
model finds anything the simple rule (new partner + small claim + after 1
May) can't find alone. It doesn't — in the one month I could test, model and
rule catch the exact same 9 fraud cases in their top 40. **My actual
recommendation is to run the queue off the rule directly** — simpler, no
retraining, same result today. Keep the model as infrastructure (it adapts
if the pattern shifts again, the way it did in May), but the rule is doing
the work right now.

All of this comes from one month of data since the policy changed (June
2026, 22 confirmed fraud cases) — real, but small. It'll sharpen as more
months come in.

## What I'd do next week

1. **Fix the gap, don't just detect it.** Tighten auto-approval specifically
   for partners onboarded in the last 6 months — restore inspection for
   their sub-₹2,000 claims, or lower their threshold. Closes the 19.7%-fraud
   segment at the source, for free. The one recommendation here that doesn't
   depend on any of the uncertain numbers above.
2. **Run the review queue off the rule** (new partner + under ₹2,000 + after
   1 May), prioritized by claim amount, reviewing roughly the top 10 a
   month — not the model. Same result, easier to audit.
3. **Don't publicize 97%+ accuracy either way** — clears the bar against 15
   months of history (98.7%), misses it against June alone (96.9%). If the
   board wants a number: "the model finds ~4.4x more fraud per claim
   reviewed than random."
4. **Re-check in 2-3 months.** Both the 55%-confidence economics figure and
   "the rule ties the model" rest on one month of 22 fraud cases — more data
   will sharpen both without new code.

## What this can't tell you (yet)

Claims under investigation in the old system before October 2025 that were
never resolved got recorded as "not fraud" by a data-entry limitation, not a
real verdict — the true historical fraud rate is likely a bit higher than
what's in the export. Doesn't change the new-partner finding above, which is
entirely post-migration data, but worth knowing before anyone quotes 1.3% as
precise.

— Kabir's team
