# Decision log

## Deduplication: resubmitted claims
`train.csv` has 681 duplicate `claim_id`s; `test_unlabelled.csv` has none. Checked:
every duplicate pair is byte-identical except `submitted_at`, which differs by a
few days — matches Tanmay's email ("partners resubmit claims when they're
bounced"). Kept the first (earliest) submission per `claim_id`.

## Legacy label contamination — disclosed, not corrected
Tanmay's email: "Cases still under investigation are blank in the CRM - but
Zoho couldn't store blanks, so the old ones came across as 0." Confirmed:
`legacy_zoho` rows have **zero** blank `is_fraud` values (vs 202 genuine blanks
in the current CRM period). Some unknown fraction of legacy "0" labels are
therefore "never investigated," not "confirmed genuine." There's no way to
tell which ones from the export alone. Did not attempt a heuristic fix (e.g.
guessing based on submission date near the migration cutoff) — a wrong guess
would be worse than an honest limitation. Disclosed in README and the memo.

## Time-based validation, not random split — and a bug this caught
`test_unlabelled.csv` is entirely Jul-Sep 2026, fully after the 1 May 2026
auto-approval change (README: claims under Rs 2,000 auto-approved without
inspection from that date). A random split of `train.csv` would validate on a
mix of pre- and post-change claims and overstate real-world performance.

First attempt: train on everything before 1 May 2026, validate on May-Jun 2026.
Result: **AUC 0.35 — worse than random.** Investigated before trusting it:
the two engineered features carrying the single strongest EDA signal
(`post_may2026`, `risky_segment`) have **zero variance in any pre-May-2026
row by construction** — a model trained on that window literally cannot learn
the pattern it most needs to. Confirmed directly:
```
training period risky_segment variance: {0.0: 9724}
training period post_may2026 variance:  {0.0: 9724}
```
Fixed split: train through 31 May 2026 (includes one month of the new regime),
validate on June 2026 only. This is the split used everywhere in this
submission.

## Model choice: logistic regression over gradient boosting
Compared `HistGradientBoostingClassifier` against a `LogisticRegression`
pipeline two ways:
- **Random 5-fold CV on all labeled data:** HGB wins (mean AUC 0.853 vs 0.828).
- **Walk-forward, train-on-past/validate-on-one-month, across 4 months:**

| Validate month | n_fraud | HGB AUC | LogReg AUC |
|---|---|---|---|
| Mar 2026 | 9  | 0.918 | 0.803 |
| Apr 2026 | 9  | 0.693 | 0.670 |
| May 2026 | 14 | 0.264 | 0.273 (both below random — cold start, first month of the new regime, no training exposure to it; expected, not a bug) |
| **Jun 2026** | 22 | **0.636** | **0.767** |

(This table is computed by `walk_forward_eval()` in `train_model.py` each
run — an earlier draft had it hand-typed, which silently drifted out of
sync with the data at least twice; see "Reviewer round 3" below.)

Random CV mixes future information into training (e.g. a partner's later
behavior leaks into folds containing that partner's earlier claims), which is
exactly why it disagrees with the walk-forward result. June is the only
labeled month sharing test's post-change regime with any training exposure —
the closest available analog to the real deployment distribution. Logistic
regression wins there by a wide margin; HGB's edge on stable historical
months doesn't transfer to a shifted distribution with only a few dozen
fraud examples to learn from. **Logistic regression was chosen as the final
model on this basis, not because it's simpler** (simplicity was a secondary
benefit — it's also more interpretable for the reasons the service returns).

## Probability calibration — a real bug caught before shipping
`LogisticRegression(class_weight='balanced')` is necessary for ranking (fraud
is ~1.2% of claims; without it the model barely learns the minority class),
but this makes its raw predicted probabilities badly inflated. Checked on
5-fold out-of-fold predictions across all labeled data: **sum of raw scores =
3,430 vs 141 actual fraud cases** — a ~24x overcount. Uncalibrated,
`predictions.csv` would have averaged ~36% predicted fraud likelihood, which
is absurd against a true rate near 1-3%.

Fixed with Platt scaling: fit a 1-D logistic regression mapping out-of-fold
raw scores to true labels. This is a monotonic transform (verified: AUC
unchanged, 0.8306 before and after), so it corrects magnitude without
touching ranking. After calibration, out-of-fold sum = 140.3 vs 141 actual.
`predictions.csv` now averages 2.2% (sum ≈ 50.5 expected fraud claims across
2,252 test rows) — plausible against June 2026's observed 3.1% rate.

The same calibrator is applied in the live `/score` endpoint, so a score from
the API means the same thing as a score in `predictions.csv`.

## Review-recommendation threshold: capacity-based, not 0.5
Once scores are properly calibrated, a fixed `0.5` cutoff for "recommend
review" is meaningless — the maximum calibrated score anywhere in the test
set is ~0.18. A single live claim has no other claims to rank against, so the
threshold is instead derived from the actual constraint (40 reviews/month,
avg ~743 claims/month): the score value at the percentile that would select
~40 claims out of a typical month (~94.6th percentile, threshold ≈ 0.0555 as
computed at training time). Stored in `model.joblib`, used identically by
`predictions.csv` consumers and the live service.

## The "97% accuracy" KPI — pushed back on, not delivered as asked
Computed the base rate directly: fraud is 141/11,146 = 1.26% of labeled
claims (post-dedup; 145/11,814 pre-dedup — the duplicate rows removed during
canonicalization don't change this meaningfully). A model that predicts "not
fraud" on every single claim already clears **98.7% accuracy** on the full
15-month history while catching precisely zero fraud.

**A further nuance, checked rather than assumed:** in June 2026 specifically
(the most recent month with confirmed outcomes, post-auto-approval-change),
fraud has already risen to 3.1% — at that rate, "flag nothing" scores only
**96.9%**, just *under* the board's 97% bar. So the "accuracy trap" isn't
even reliably a trap going forward: whether a do-nothing model clears 97%
depends entirely on which period you measure it against, which is itself
more evidence that accuracy is the wrong metric here, not just a
coincidentally-easy one to game. Accuracy
cannot distinguish that model from a useful one here. Recommended metric
instead: rupees of fraud stopped per claim reviewed, directly answering
Farhan's email ("how much fraud we stop per claim we check, in rupees - not
a percentage"). See `memo-to-ritu.md` for the number.

## Rule vs. model: the honest "why do you need ML" answer
Reviewer feedback on this engagement asked directly: if the 19.7%-fraud
segment was already found by hand, why build a model at all? Tested this
rather than asserting an answer. Built a rule-only baseline (new partner AND
post-May-2026 AND claim < ₹2,000, tie-broken by claim amount descending when
more than 40 are flagged) and compared it to the model at k=40 on June 2026:

| Approach | Fraud caught | Precision@40 | Fraud ₹ caught |
|---|---|---|---|
| Rule-only | 9 | 22.5% | ₹12,048 |
| Model (logistic regression) | 9 | 22.5% | ₹11,246 |

**They tie — the rule edges the model out slightly on raw rupees.** Checked
further: of the model's top-40 claims, zero are outside the rule's segment.
The model has rediscovered the same pattern via `amount_to_price_ratio` and
`customer_prior_claims`, not found something the rule can't see. Recommended
running the actual review queue off the rule directly (see memo-to-ritu.md)
— simpler, no retraining, same result today. Kept the model as the service's
engine regardless, as infrastructure that can adapt if the fraud pattern
shifts again (a hardcoded rule won't), not because it currently outperforms
the simpler alternative. Full baseline table (random, highest-amount,
new-partner-alone, etc.) in FINAL_VALIDATION_REPORT.md Section 5.

## A finding tested and retracted: the ₹1,800-2,000 "structuring" pattern
Initially reported (and briefly live in `explain.py`'s explanation text) that
claims priced ₹1,800-2,000 — just under the auto-approval cutoff — showed
roughly double the fraud rate of smaller claims in the same risky segment,
framed as possible deliberate structuring. Checked this by month before
leaving it in the shipped product, per the same discipline applied to the
main risky-segment finding (see "Ritu's new-partner hypothesis" below):

| Month | ₹1,800-2,000 band | Under ₹1,800 |
|---|---|---|
| May 2026 | 53.8% fraud (n=13) | 11.7% fraud (n=60) |
| June 2026 | 10.0% fraud (n=10) | 22.2% fraud (n=54) |

**It reverses sign between the two months** — not stable, almost certainly
noise from small cells. Pooled across both months it looked significant-ish
(Fisher's exact p=0.080, not even significant at 0.05 pooled), but the
month-by-month check is what actually matters and it fails. Removed
`near_threshold` from `FEATURE_COLUMNS` and from the live explanation
engine — it had been telling Kestrel staff "possible structuring," which
this check does not support. Documented here rather than quietly deleted,
because a retracted finding is itself worth showing the work on.

## Uncertainty was checked, not just point-estimated
Bootstrapped (3,000 resamples) the June 2026 validation set, via
`bootstrap_validation.py` — writes `bootstrap_metrics.json`, so these are
reproducible by running that script, not just numbers in this document:
- AUC 95% CI: [0.635, 0.886] (point estimate 0.767) — comfortably above random.
- Precision@40 95% CI: [0.100, 0.350] (point estimate 0.225).
- Net economic value at k=40, 95% CI: [-₹19,841, -₹2,250] (point estimate
  -₹10,934) — never positive in the bootstrap.
- **Net economic value at k=10, 95% CI: [-₹4,856, +₹6,545]** (point estimate
  +₹913). **P(net > 0) = 0.55.** This is the most important uncertainty
  finding in the whole submission: the headline "+₹900/month" economics
  figure has barely better than coin-flip odds of actually being positive,
  given only 22 fraud cases to estimate it from. The memo states this
  directly rather than presenting +₹913 as a confident number.

Also stress-tested the k=10 recommendation against cost-parameter changes
(review cost +10%, goodwill ±20%): stays net-positive under every single
one of those individually (worst case ₹197 under combined adverse
assumptions). The parameter sensitivity is much less concerning than the
sampling uncertainty above — the model of the world (₹380/₹260 costs) is
fairly robust; the small fraud count is what actually threatens the
conclusion.


## Ritu's new-partner hypothesis: confirmed, with a load-bearing caveat
Checked directly:

| Segment | Fraud rate |
|---|---|
| Established partners, any claim | 1.11% |
| Established partners, small claim, post-May (auto-approved) | 0.84% |
| New partners, any *other* claim | 0.18% |
| **New partners, small claim, post-May (auto-approved)** | **19.7%** (n=137) |

Ritu is right that new partners are implicated — but only in this specific
segment. New partners *outside* that segment are actually lower-risk than the
overall baseline (0.18% vs 1.3%), which is exactly Meenal's point ("most of
the new ones are fine"). Reported both halves; did not round this off to "new
partners are the problem" because that's not what the data says.

**Checked whether the 180-day "new partner" cutoff was cherry-picked** (a
natural worry given it directly drives the headline 19.7% number): swept it
from 90 to 365 days. Fraud rate in the segment rises smoothly from 9.3% (90
days) through 19.7% (180 days) to ~21% (270-365 days) — 180 days is not even
the peak, so this wasn't tuned to maximize the finding.

**Checked whether the segment itself is stable across the two months it can
be measured in:** 19.2% fraud in May 2026, 20.3% in June 2026 — nearly
identical. This is the finding that held up under scrutiny, unlike the
₹1,800-2,000 structuring pattern above, which didn't.

## What was tried and discarded
- **NLP features from `claim_description` / `inspector_note`.** Spot-checked
  fraud cases: inspector notes for confirmed-fraud claims read as
  reassuring ("Photos match fault, approved," "Customer has bill, serial
  verified") — the investigation verdict isn't leaked into the note text, and
  no obvious lexical fraud signal was found. Not pursued further given the
  time budget; a proper NLP feature pass is a plausible v2, not a shortcut
  skipped for convenience.
- **Partner-level historical fraud-rate target encoding.** With 141 total
  fraud cases spread across ~380+ partners, most partners have 0-1 known
  fraud events — a raw per-partner rate would be extremely noisy and prone to
  overfitting small counts. Left out rather than built carelessly.
- **SHAP-based explanations.** Not installed, no network access to install
  it in this environment, and unnecessary here: logistic regression's own
  coefficients give an *exact* linear decomposition of the score
  (contribution = coefficient × scaled feature value), not an approximation.
  A simpler, more honest choice than reaching for a heavier tool.
- **A bug in the explanation wording itself, caught by testing on real
  claims before shipping:** the first version of `explain.py` hard-coded
  "new partner" as a risk-raising phrase. But `partner_is_new`'s coefficient
  is *negative* (new partners are protective on their own — see the table
  above). Testing on a real high-scoring test claim produced a
  self-contradictory explanation ("new partner... this lowers the risk
  score" mislabeled as risk-raising). Rewrote so wording direction is always
  read off the model's actual computed contribution sign for that row, never
  assumed from the feature name.

## Reviewer round: reproducibility discrepancies fixed
A review of the submission found three real discrepancies between what the
documents claimed and what the code actually produced:
1. **AUC mismatch** — docs said 0.762, `validation_metrics.json` actually
   said 0.7669. The docs were wrong; fixed by updating every document to
   the real number (0.767), not by adjusting the code to hit 0.762.
2. **Walk-forward table was hand-typed**, and had drifted from the data —
   May showed 13 fraud in the docs vs. 14 in the actual (deduped) data, and
   March showed AUCs that didn't match a live rerun. Fixed by adding
   `walk_forward_eval()` to `train_model.py`, which computes all four
   monthly rows from the actual data every run. The June row now matches
   the main validation AUC by construction (same train/validate split),
   which is itself a useful internal-consistency check.
3. **Bootstrap CIs existed only in prose**, with no code anywhere in the
   repo that produced them. Added `bootstrap_validation.py`, which
   resamples the June holdout 3,000 times and writes `bootstrap_metrics.json`.
   The regenerated numbers landed close to (not identical to) the original
   prose claims — e.g. net-value CI at k=10 came out [-₹4,856, +₹6,545]
   vs. the original [-₹4,856, +₹6,487] — confirming the original analysis
   was done correctly by hand at some point, it just was never committed as
   runnable code. All three documents-vs-code mismatches are the exact
   failure mode this submission's own `validate.py` is meant to catch;
   none of the three were caught by it because the checks that would have
   caught them (recompute walk-forward, recompute bootstrap, diff prose
   against JSON) didn't exist yet. Added `bootstrap_validation.py` as a
   permanent, runnable artifact rather than a one-off fix specifically so
   this class of error can't silently recur.

## Reviewer round 2: a stale comment, an orphaned fragment, and a mismatched baseline
A second pass, after round 1's fixes, found three more issues - none of them
in the actual computed numbers this time, all in prose that had fallen out
of sync with the numbers:
1. **`fit_logreg()`'s docstring still said "Jun: 0.762 vs 0.615"** even
   after every document had been corrected to 0.767/0.636 in round 1 - the
   docstring itself was missed. Fixed by removing the hard-coded numbers
   from the docstring entirely (not just correcting them again), since a
   third drift is exactly as likely as the first two if a number is typed
   into a comment anywhere. The docstring now points to
   `validation_metrics.json` instead of repeating a value that can go stale.
2. **`FINAL_VALIDATION_REPORT.md`'s baseline table quoted two different
   "random selection" numbers in the same document** - the table said 1
   fraud caught / ₹17,866 (actually a leftover from an early single
   `permutation()` draw that happened to coincidentally match the
   deterministic "highest claim amount" row), while Section 10's "~4.4x
   lift over random" claim correctly used the analytical expected value
   (₹2,562). Fixed by replacing the table row with the expected-value
   figures (1.23 fraud claims, ₹2,562, -₹22,569), matching what
   `validation_metrics.json`'s `random_selection_baseline_at_40` already
   computed correctly - the bug was in the prose table, not in the
   underlying calculation.
3. **A sentence fragment was stranded in `submission-form.md`** - an edit
   that inserted a new bullet point had cut off the beginning of an
   adjacent bullet about the capacity-based `review_recommended` threshold,
   leaving its back half floating with no subject. Restored the full point
   rather than just deleting the fragment, since it was a real thing worth
   keeping, not filler.

The pattern across both rounds: every actual number in every JSON file was
correct throughout. Every mistake was in hand-written prose (a docstring, a
table, a sentence) that referenced a number or a claim without being
regenerated alongside the code that produces it. `validate.py` now includes
the rule-vs-model re-derivation specifically so at least one of these
comparisons is checked by running code, not by re-reading prose.

## Reviewer round 3: the baseline table itself was still unreproducible
Rounds 1 and 2 fixed numbers that were wrong or drifted. Round 3 found a
different problem: the fixes in round 2 corrected the "Random selection"
row's *value*, but nothing in the repo actually computed the other five
rows of the baseline table (highest-claim-amount, new-partner, rule-only,
model) or the 29/40 rule-vs-model overlap claim in Section 6 — they were
prose, asserted correctly, but not reproducible. Added
`baseline_comparison.py`, ran it, and confirmed every number it produces
matches what was already in the document exactly (no numbers changed) —
so the fix here is entirely about closing the reproducibility gap, not
correcting an error. This is the same failure mode as round 1's bootstrap
issue, in a different table; `validate.py` still doesn't check for
doc-vs-code numeric drift automatically, which is the one honest
limitation left in this review process — every round so far has depended
on a human (or reviewer) manually cross-checking prose against JSON.
