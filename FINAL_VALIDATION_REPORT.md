# Final Validation Report — Kestrel Warranty Fraud Model

Every number below is reproducible from the committed validation artifacts.
Run `python3 train_model.py`, `python3 baseline_comparison.py`,
`python3 bootstrap_validation.py`, and `python3 validate.py`, in that order. This document exists because a
reviewer of a related submission from this engagement specifically praised
a `validate.py` that "lets a reviewer re-derive all of it in one command" —
doing that again here, deliberately (across three commands now, since the
bootstrap CIs need their own script — see decision-log.md's "Reviewer round:
reproducibility discrepancies fixed" for why that split exists).

## 1. Dataset integrity
- `train.csv`: 12,029 raw rows → 681 exact duplicate `claim_id`s (identical
  except `submitted_at`, a few days apart — matches Tanmay's email about
  bounced resubmissions) → 11,348 canonical rows after keeping the first
  submission.
- `test_unlabelled.csv`: 2,252 rows, no duplicates, exact match to
  `sample_submission.csv`'s 2,252 `claim_id`s.
- Verified in `validate.py`: claim ID uniqueness (train + test), prediction
  row count, no missing/duplicate prediction scores, all scores in [0,1].

## 2. Label limitations
- Of 12,029 raw rows, 11,348 are canonical (post-dedup). Of those, 11,146
  have a non-null `is_fraud` (202 undecided at export, correctly blank —
  current-CRM period). **141 fraud, 1.26% base rate.**
- **Legacy contamination, quantified:** `legacy_zoho` (4,606 claims) has
  **zero** blank `is_fraud` values — the old system couldn't store undecided
  cases as blank, so they were coded `0`. All **4,560** legacy "not fraud"
  labels are therefore of unknown reliability — some unknown subset are
  "never investigated," not "confirmed genuine." Legacy fraud rate (1.00%) is
  measurably lower than the clean CRM-period rate (1.45%), consistent with
  this contamination diluting the legacy rate downward. **Not relabeled** —
  there is no investigation outcome available to support doing so; any
  correction would be a guess, and a wrong guess is worse than a disclosed
  limitation.

## 3. Leakage tests (`validate.py`)
```
PASS | Claim IDs unique in train (post-dedup)
PASS | Claim IDs unique in test
PASS | Prediction IDs match sample_submission IDs exactly
PASS | is_fraud excluded from FEATURE_COLUMNS
PASS | inspector_note / claim_description excluded from FEATURE_COLUMNS
PASS | near_threshold excluded (tested, unstable across months, not used)
PASS | predictions.csv row count matches test_unlabelled.csv
PASS | predictions.csv has no missing scores
PASS | predictions.csv has no duplicate claim_id
PASS | predictions.csv scores are in [0, 1]
PASS | Point-in-time: submitted_at itself never used as a raw numeric feature
PASS | Adversarial leakage test: corrupting June outcomes/amounts does not change May features
PASS | Explanation direction matches model contribution sign (n=100 sampled claims)
```
**Adversarial leakage test, specifically:** built May 2026 features once,
then corrupted every June row's `is_fraud` (flipped) and `claim_amount_inr`
(×7 + 999) and `customer_prior_claims` (set to 999), then rebuilt May's
features again. May's features were byte-identical before and after —
concrete evidence the feature pipeline is point-in-time and doesn't leak
future rows' data into past ones.

## 4. Walk-forward validation
Train-on-past, validate-on-one-future-month, across 4 months:

| Validate month | n_fraud | HGB AUC | LogReg AUC |
|---|---|---|---|
| Mar 2026 | 9  | 0.918 | 0.803 |
| Apr 2026 | 9  | 0.693 | 0.670 |
| May 2026 | 14 | 0.264 | 0.273 (cold start — first month of new regime, zero training exposure) |
| **Jun 2026** | 22 | **0.636** | **0.767** — decisive month, closest analog to test |

These four rows are computed directly by `walk_forward_eval()` in
`train_model.py` (not hand-typed) each time it runs, specifically so this
table cannot drift out of sync with the code the way an earlier draft of
this report did. The June row matches `validation_metrics.json`'s main
`roc_auc` (0.767) and `roc_auc_hgb_comparison_not_chosen` (0.636) exactly,
by construction — both come from the identical train/validate split.

June is the relevant holdout because it's the only labeled month sharing
`test_unlabelled.csv`'s post-1-May-2026 regime with any training exposure to
that regime. All headline numbers below are from June unless stated otherwise.

## 5. Baseline comparison (June 2026, k=40)
| Approach | Fraud caught | Precision | Fraud ₹ caught | Net ₹ |
|---|---|---|---|---|
| Random selection (expected value, not one draw) | 1.23 | 3.1% | 2,562 | -22,569 |
| Highest claim amount | 1 | 2.5% | 17,866 | -7,354 |
| New partner (any) | 4 | 10.0% | 6,573 | -17,507 |
| New partner + post-May (no amount filter) | 4 | 10.0% | 6,573 | -17,507 |
| **Rule-only** (new + small + post-May, tie-broken by amount) | **9** | **22.5%** | **12,048** | -10,132 |
| **Model (logistic regression)** | **9** | **22.5%** | 11,246 | -10,934 |

All six rows are computed by `baseline_comparison.py` → `baseline_metrics.json`,
not hand-typed. Re-run it to re-derive this table.
"Random selection" is the analytical expectation (k × base rate, and base
rate × mean fraud claim amount), not the outcome of one arbitrary
`permutation()` call — an earlier draft used a single random draw here,
which happened to coincidentally match "highest claim amount" and was
inconsistent with the properly-computed random baseline used later in this
same report (Section 10's "~4.4x lift over random"). Fixed so there's one
random baseline number throughout, not two.

## 6. Model comparison — and the honest "why ML" answer
**The rule ties the model on fraud count and precision, and edges it out
slightly on raw rupees caught.** Checked directly: of the model's top-40
scored claims in June, **zero** are outside the rule's `risky_segment`
(new + small + post-May) — the model has essentially rediscovered the same
segment via `amount_to_price_ratio` and `customer_prior_claims`, not found
signal the rule can't see. The overlap between the model's top-40 and the
rule's top-40 is 29/40 (`rule_vs_model_overlap_at_k` in
`baseline_metrics.json`).

**Recommendation:** deploy the rule as the primary, auditable filter — it's
simpler, requires no retraining, and performs at least as well as the model
on the one period that matters. Keep the model as the live service's engine
anyway, for three reasons that don't show up in this month's numbers: (1) it
produces a continuous score rather than a binary flag, useful if the review
capacity ever changes without needing to hand-tune a new rule; (2) a
hardcoded rule won't adapt if fraud patterns shift again the way they did in
May 2026 — the model will re-learn a new pattern once enough labeled data
exists, the rule won't; (3) the explanation engine (Section 10) already
exists on top of the model and gives the same auditability the rule offers,
without giving up the option value of #1 and #2. This is not a case for
"ML because ML" — it's a case for keeping ML as infrastructure while
trusting the rule for the actual decision today.

## 7. Calibration
`LogisticRegression(class_weight='balanced')` is necessary for ranking (fraud
is 1.2% of claims) but badly inflates raw probabilities. Out-of-fold (5-fold
CV) check on all labeled data: **3,430 predicted vs 141 actual** before
calibration (~24x over). Fixed with Platt scaling (monotonic — AUC unchanged,
0.8306 before and after): **140.3 predicted vs 141 actual** after. Same
calibrator used in `predictions.csv` and the live `/score` endpoint.

## 8. Precision@K
Precision@40 = 22.5% (9/40), a 7.3x lift over the 3.1% June base rate.
Full k-sweep in Section 11.

## 9. Bootstrap uncertainty
3,000-resample bootstrap on the June validation set (713 rows, 22 fraud),
computed by `bootstrap_validation.py` and written to
`bootstrap_metrics.json` — reproducible by running that script directly,
not a number that exists only in this document:
- **AUC 95% CI: [0.635, 0.886]** (point estimate 0.767) — lower bound stays
  meaningfully above random.
- **Precision@40 95% CI: [0.100, 0.350]** (point estimate 0.225) — wide, but
  the lower bound (10.0%) still clears the 3.1% base rate.
- **Net value at k=40, 95% CI: [-₹19,841, -₹2,250]** (point estimate
  -₹10,934). Never positive in this bootstrap — reviewing all 40 slots is
  reliably a net cost, not just a weak point estimate.
- **Net value at k=10, 95% CI: [-₹4,856, +₹6,545]** (point estimate +₹913).
  **P(net > 0) = 0.55** — barely better than a coin flip. **This is the
  single most important uncertainty finding in this report:** the "positive
  economics at k=10" headline number is not statistically robust on 22 fraud
  cases. See Section 13.

## 10. Business economics
Using ops-policy.pdf's own figures (₹380 goodwill/genuine claim held, ₹260
per claim reviewed): net value peaks around k=10 (+₹913) and turns clearly
negative by k=20 (-₹2,786) and beyond (Section 11). Model-prioritized review
at k=40 catches ~4.4x the fraud rupees a random pick of 40 would (₹11,246 vs
an expected ₹2,562) — the value is in concentrating limited capacity, not in
making review free or reliably profitable in isolation (Section 13).

## 11. Threshold sensitivity
| k | Fraud caught | Precision | Fraud ₹ caught | Net ₹ |
|---|---|---|---|---|
| 5  | 2 | 40.0% | 3,076  | +636 |
| 10 | 4 | 40.0% | 5,793  | **+913 (peak)** |
| 15 | 6 | 40.0% | 7,734  | +414 |
| 20 | 6 | 30.0% | 7,734  | -2,786 |
| 25 | 6 | 24.0% | 7,734  | -5,986 |
| 30 | 6 | 20.0% | 7,734  | -9,186 |
| 35 | 8 | 22.9% | 10,379 | -8,981 |
| 40 | 9 | 22.5% | 11,246 | -10,934 |

The "use ~10, not 40" recommendation is evidence-driven from this table, not
a guess — but see Section 9's bootstrap CI before treating +₹913 as reliable.

## 12. Known failure cases
- **Cold start on sudden policy changes:** both models score *below random*
  on May 2026 specifically (the first month of the auto-approval change),
  because neither had a single training example of the new regime yet. Any
  future policy shift will likely reproduce this same one-month blind spot.
- **A tested-and-rejected finding, disclosed rather than hidden:** claims
  priced ₹1,800-2,000 (just under the auto-approval cutoff) showed a
  suggestive "structuring" pattern in the pooled data (34.8% vs 16.7% fraud
  rate, Fisher's exact p=0.080 — not significant at 0.05). Checked by month:
  it **reverses sign** between May (53.8% vs 11.7%) and June (10.0% vs
  22.2%). Not stable, almost certainly noise from small cells (n=13-23 per
  month). Removed from the feature set and the live explanation engine —
  it was previously telling Kestrel staff "possible structuring," which the
  month-by-month check does not support.
- **New-partner cutoff robustness, checked**: the 180-day "new partner"
  definition was a reasonable a priori choice (not tuned to maximize the
  finding) — fraud rate in the risky segment rises smoothly from 9.3% (90-day
  cutoff) through 19.7% (180 days) to ~21% (270-365 days), i.e. 180 days is
  not even the cherry-picked peak.
- **The risky-segment finding itself is stable**, unlike the structuring
  claim: 19.2% fraud in May, 20.3% in June — nearly identical across the only
  two months it can be measured in.

## 13. Final recommendation
1. **Fix the gap at the source.** Tighten auto-approval specifically for
   partners onboarded in the last 6 months — this closes the ~20%-fraud
   segment for free, instead of paying to detect it after the fact. This is
   the one recommendation in this report that doesn't depend on any
   uncertain estimate above.
2. **Deploy the simple rule operationally**, not the model, for the actual
   review-queue decision today — it ties the model on this validation month
   and is far more auditable (Section 6).
3. **Do not present the ₹913/month economics figure as a confident business
   case.** Its own bootstrap CI spans -₹4,856 to +₹6,545 with only a 55%
   chance of being positive at all. The honest framing for Ritu: "worth
   trying and monitoring, not a guaranteed return" — see the revised memo.
4. **Do not chase 97%+ accuracy.** A model that flags nothing already clears
   that bar on the full historical base rate (98.7%) — but would actually
   fall *below* it (96.9%) if measured against June 2026 alone, the most
   recent month. That instability is itself proof accuracy is the wrong
   metric here, not just a coincidentally-easy trap — this was the trap in
   the original brief and remains the one number this report actively
   recommends against using.
5. **Re-run this validation in 2-3 months** once more post-May-2026 labeled
   data exists. Every uncertainty caveat in this report (Sections 9, 12)
   stems from having only 22 fraud cases in the one usable holdout month;
   that will improve mechanically as more months accumulate, without
   changing any code.
