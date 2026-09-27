# Submission Form — Kestrel Home Warranty Claim Review (Variant C)

## Candidate
Arun Kumar Basam

## Deliverable 1 — Predictions: what I expect `predictions.csv` to score, and why
`predictions.csv` has 2,252 rows (one per `claim_id` in `test_unlabelled.csv`,
matching `sample_submission.csv`'s shape). Before sending it:

- **Expected implied fraud count: ~50.5 claims (≈2.2% of test).** This comes
  from the sum of calibrated scores (50.5), not a guess — after fixing a
  calibration bug (see decision-log.md) where the raw model overstated fraud
  probability by ~24x, the calibrated out-of-fold sum on training data
  matched the true fraud count almost exactly (140.3 predicted vs 141
  actual). If that calibration holds up out-of-sample, the sum of test scores
  is a reasonable estimate of the true test fraud count.
- **Why ~2.2%, not train's overall 1.3%:** `test_unlabelled.csv` is entirely
  Jul-Sep 2026, fully after the 1 May 2026 auto-approval change that drove
  fraud up in a specific segment. The one labeled month sharing that regime
  (June 2026) already shows 3.1% observed fraud. ~2.2% sits between train's
  historical average and June's post-change rate, which is what I'd expect
  given the auto-approval effect is real but concentrated in a ~9%-of-volume
  subsegment (204 of 2,252 test claims are new-partner + small + auto-approved).
- **Where I expect to be wrong:** precision at the top of the ranking has a
  **bootstrapped 95% CI of [10.0%, 35.0%]** around the 22.5% point estimate
  (3,000 resamples of the 22-fraud validation month — see decision-log.md). I
  would not be surprised if the actual test fraud count is anywhere from
  ~25 to ~80, not precisely 50.
- **Ranking, not just magnitude:** the score is calibrated but the more
  load-bearing property for Kestrel's actual use case (prioritizing 40
  reviews/month) is the ranking, which was validated separately via
  precision@40 and is unaffected by the calibration step (calibration is a
  monotonic transform — see decision-log.md).

## Deliverable 2 — Working service
`app.py` (Flask): `POST /score` takes a single claim as JSON, returns a
calibrated fraud score, a review recommendation (threshold derived from the
40/month capacity constraint, not an arbitrary 0.5), and 2-4 plain-English
reasons read directly off the model's own coefficients. One screen
(`GET /`) with a form that calls the same endpoint. No paid API key anywhere.
Verified: if `model.joblib` is missing, the service still starts; `/score`
returns a clear 503 with the fix instead of crashing (tested directly by
removing the model file and re-running).

## Deliverable 3 — Evidence it works, and how often it doesn't
Full detail in `FINAL_VALIDATION_REPORT.md` (13 sections: dataset integrity,
label limitations, leakage tests, walk-forward validation, baseline
comparison, model comparison, calibration, precision@K, bootstrap
uncertainty, business economics, threshold sensitivity, known failure cases,
final recommendation). Headlines:

- **Time-based validation** (train through May 2026, validate on June 2026
  only) rather than random split — see decision-log.md for why a random
  split would have hidden a real distribution shift, and for the AUC-0.35
  bug this caught in an earlier attempt.
- **ROC-AUC 0.767**, precision@40 = **22.5%** (~7.3x lift over the 3.1%
  base rate). Bootstrapped 95% CIs: AUC [0.635, 0.886], precision@40
  [0.100, 0.350].
- **Walk-forward comparison** across 4 monthly holdouts, used to choose
  logistic regression over a gradient-boosted alternative (full table in
  decision-log.md).
- **Baseline comparison at k=40**: random, highest-claim-amount, new-partner
  alone, and a rule-only baseline, alongside the model — see "rule vs model"
  finding below.
- **`validate.py` leakage tests**, including an adversarial check: built May
  2026 features, corrupted every June outcome and claim amount, rebuilt May's
  features, confirmed byte-identical — concrete evidence the pipeline is
  point-in-time.
- **Calibration check**: out-of-fold predicted fraud count (140.3) vs actual
  (141) on all labeled data, after fixing a ~24x overcount bug.
- **Economics sweep** (k=5 to 40) using the ops-policy's own cost figures,
  showing net value peaks around k=10 (+₹913) and turns negative by k=40
  (-₹10,934) — an honest answer to "how often is it wrong" in the terms that
  matter to Kestrel: rupees, not just a confusion matrix. **Bootstrapped this
  too: the k=10 net value has only a 55% probability of being positive at
  all** (95% CI [-₹4,856, +₹6,545]) — disclosed as the single most important
  uncertainty finding in the whole submission, not smoothed over.
- **How it's wrong, specifically:** at the review threshold used, roughly 3
  in 4 flagged claims will be genuine. This is disclosed as the headline
  precision number, not buried.

## Deliverable 4 — Memo to Ritu
`memo-to-ritu.md`. Non-technical, ~600 words (~2-3 minutes to read). States
the accuracy-KPI problem plainly, the new-partner finding with its caveat,
the rupee economics, and three concrete next-week actions.

## Deliverable 5 — Screen recording
`demo-script.md` — script for a ≤3-minute recording; timing sums to exactly
3:00. Recording itself: **[FILL IN AFTER UPLOAD]**.

## Deliverable 6 — This form
Filled in below.

## Did you change, narrow, or push back on the client's ask?
- **Pushed back on "accuracy above 97%" as the KPI**, in writing, with the
  naive-baseline calculation that makes it visibly the wrong metric for this
  problem. Did not simply report an accuracy number to satisfy the literal
  ask.
- **Did not treat "new partners are the problem" as confirmed** just because
  Ritu suspected it — reported the segment-specific finding instead, which
  both confirms her instinct and validates Meenal's pushback simultaneously.
- **Chose the simpler of two models** (logistic regression over gradient
  boosting) based on which one actually generalizes to the deployment
  distribution, not the more sophisticated option.
- **Recommended running the review queue off a simple rule, not the model** —
  tested whether the model beats the obvious rule (new partner + small claim
  + post-May), found it ties rather than beats it (identical precision@40,
  22.5%; rule edges the model out slightly on raw rupees). Rather than
  quietly keeping the more sophisticated-looking option, recommended the
  rule for the actual operational decision and kept the model as
  infrastructure for when the fraud pattern eventually shifts again.
- **Recommended against using all 40 review slots**, even though the ops
  policy states that as the desk's capacity — the economics say ~10 is
  closer to optimal given current fraud rarity and claim sizes. Then
  bootstrapped that recommendation and reported honestly that it has only a
  55% chance of being net-positive in a typical month, not presenting it as
  a confident number.

## What is wrong with what you are handing us?
- **Legacy label contamination is real and unfixed.** Some fraction of
  pre-Oct-2025 "not fraud" labels are actually "never investigated," coded as
  0 because the old system couldn't store blanks (4,560 legacy claims are
  affected; legacy fraud rate 1.00% vs 1.45% in the clean CRM period, a gap
  consistent with this dilution). This likely means the true historical
  fraud rate is somewhat higher than 1.3%; not correctable from this export.
- **The economics recommendation is not statistically solid.** Bootstrapped
  the net-value-at-k=10 figure: 95% CI is [-₹4,856, +₹6,545], and the
  probability it's actually positive in a given month is only 55% — barely
  better than a coin flip, given just 22 fraud cases to estimate from. The
  memo states this directly rather than presenting +₹913/month as reliable.
- **The model doesn't currently outperform a simple hand-written rule.**
  Tested this explicitly rather than assuming ML adds value: model and rule
  catch the identical 9 fraud cases out of the same top-40 in June 2026. The
  model's sophistication isn't earning its keep yet on the available data.
- **A finding I initially shipped turned out not to survive scrutiny, and I
  had to retract it.** Claims priced ₹1,800-2,000 (just under the
  auto-approval cutoff) looked like a "structuring" pattern in pooled data
  (34.8% vs 16.7% fraud), and this was briefly live in the explanation
  engine's output. Checking it month-by-month showed it reverses sign (May:
  53.8% vs 11.7%; June: 10.0% vs 22.2%) — not stable, almost certainly noise
  from small cells. Removed from the feature set and the live explanations.
  Flagging this prominently because shipping-then-retracting a finding is
  exactly the kind of mistake worth being visible about, not quietly fixing.
- **Validation sample for the post-change regime is small** (22 fraud cases
  in the one usable holdout month). Every precision and economics figure
  above should be treated as directional until more months of post-May-2026
  labels accumulate.
- **The model has a cold-start blind spot for sudden policy changes** — both
  models tested scored *below random* on May 2026 specifically, the first
  month of the auto-approval change, because neither had seen a single
  training example of the new regime yet. Any future policy change will
  likely reproduce this same blind spot for its first month.
- **Partner-level history isn't used** (see "left out" below) — a partner
  with, say, 3 of their last 5 claims already flagged as fraud isn't
  currently weighted any differently than a first-time offender, beyond what
  `partner_is_new` and `partner_type` capture.
- **`app.py` runs Flask's development server** (`debug=False`, but still the
  built-in dev server) — fine for this exercise, not what you'd put in front
  of real traffic without a production WSGI server.

## What did you deliberately leave out, and why that rather than something else?
- **NLP features from claim/inspector text** — spot-checked and found no
  obvious fraud-lexical signal (inspector notes for confirmed fraud read as
  reassuring, not suspicious); a real NLP pass is plausible future work, not
  a shortcut taken for convenience.
- **Partner-level fraud-rate target encoding** — with 141 total fraud cases
  across 380+ partners, most partners have 0-1 known fraud events; a raw
  per-partner rate would be noise dressed up as signal.
- **SHAP or LLM-based explanations** — logistic regression's own coefficients
  give an exact (not approximate) decomposition of the score; reaching for a
  heavier tool wouldn't have made the explanation more honest, just harder to
  audit.
- **A production deployment story** (WSGI server, auth, logging, monitoring)
  — out of scope for a 48-hour take-home; the brief asked for something that
  starts from the README, not something ready for real traffic.
- **The ₹1,800-2,000 "structuring" feature** — built it, shipped it briefly,
  then pulled it after a month-by-month check showed it doesn't survive (see
  above). Left out on evidence, not by initial choice.

## Anything you built or found that nobody asked for?
- **A rule-vs-model comparison nobody asked for, with an answer that argues
  against my own model.** Built a rule-only baseline (new partner + small
  claim + post-May) and found it ties the logistic regression at k=40 -
  identical fraud count and precision, marginally higher raw rupees caught.
  Recommended the rule for the actual operational decision rather than
  quietly presenting the model as the better option because it looks more
  sophisticated.
- **A leakage/integrity test suite (`validate.py`) with an adversarial
  check** — built features for May 2026, then corrupted every June outcome
  and claim amount, rebuilt May's features, and confirmed they were
  byte-identical. Concrete evidence the pipeline is point-in-time, not just
  an assertion that it is.
- **Bootstrap confidence intervals on every headline number** — nobody asked
  for uncertainty quantification specifically, but with only 22 fraud cases
  in the usable validation month, a point estimate alone would have been
  misleading. The 55%-probability-of-being-positive finding for the
  headline economics number came directly out of this and changed the
  memo's recommendation.
- **A capacity-based threshold for the live service's `review_recommended`
  flag.** Nothing in the brief asked for this specifically — a fixed 0.5
  cutoff becomes meaningless the moment scores are properly calibrated (the
  max calibrated score anywhere in the test set is ~0.18), and shipping a
  broken-looking default felt worse than deriving a real one from the
  desk's actual 40/month capacity.

## What did you use AI for?
- Data-quality inspection, feature engineering, and model code were built
  and debugged interactively with an AI coding assistant (Claude), which
  caught two real bugs before they shipped: the AUC-0.35 zero-variance
  training bug, and a sign error in the explanation engine (`partner_is_new`
  hard-coded as risk-raising when its actual coefficient is negative).
- Used for drafting and tightening the memo and README wording.
- **No paid API calls in the delivered product.** `app.py` and
  `train_model.py` use only local, offline libraries (pandas, scikit-learn,
  Flask) — no LLM call, no external API, at run time. This was a deliberate
  choice, not just a cost-saving one: it also means the explanation engine is
  exactly reproducible and auditable, not a black-box call to a model that
  could change its answer on a re-run.
- **Discarded:** an early idea to have an LLM generate the claim explanations
  in natural language. Rejected because it would (a) require a paid API key,
  violating "must start without one," and (b) be strictly less faithful than
  reading the explanation directly off the model's own coefficients, which
  costs nothing and can't hallucinate a reason that isn't actually in the
  model.

## Honest hours spent
_To be filled in by the candidate — the real number, not an estimate from
whoever helped prepare this submission._
