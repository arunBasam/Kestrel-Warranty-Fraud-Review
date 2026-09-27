"""
Train the Kestrel warranty-fraud model, validate it honestly, and write:
  - model.joblib               (final model, trained on all labeled data)
  - predictions.csv            (score for every claim in test_unlabelled.csv)
  - validation_metrics.json    (machine-readable validation results -- point
                                 estimates, walk-forward table, economics sweep;
                                 the .md docs summarize this file, not the
                                 other way around)

Bootstrap confidence intervals are a separate script -- see
bootstrap_validation.py -- because they resample the validation set 3,000
times and don't need to run every time this file does.

Run: python3 train_model.py
"""
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from features import build_features, feature_matrix, FEATURE_COLUMNS, AUTO_APPROVE_START

warnings.filterwarnings('ignore')

GOODWILL_COST_INR = 380     # ops-policy.pdf S4: cost of holding a genuine claim for review
CONTACT_COST_INR = 260      # ops-policy.pdf S4: blended cost of a service contact (the cost of *checking* a claim)
MONTHLY_REVIEW_CAPACITY = 40  # ops-policy.pdf S5 / Farhan's email: investigation desk can review at most 40/month


def load_raw():
    t = pd.read_csv('data/train.csv', parse_dates=['submitted_at'])
    te = pd.read_csv('data/test_unlabelled.csv', parse_dates=['submitted_at'])
    partners = pd.read_csv('data/partners.csv', parse_dates=['onboarded_date'])
    products = pd.read_csv('data/products.csv')

    # Partners resubmit bounced claims; same claim_id, identical fields, later timestamp.
    # Keep the first (original) submission as canonical.
    t = t.sort_values(['claim_id', 'submitted_at']).drop_duplicates('claim_id', keep='first')
    te = te.sort_values(['claim_id', 'submitted_at']).drop_duplicates('claim_id', keep='first')
    return t, te, partners, products


def fit_hgb(X, y):
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=4,
        learning_rate=0.06,
        class_weight='balanced',   # fraud is ~1.2% of labeled claims
        random_state=0,
    )
    model.fit(X, y)
    return model


def fit_logreg(X, y):
    """Final model choice - see decision-log.md. A walk-forward comparison across
    4 monthly holdouts (computed by walk_forward_eval(), not hard-coded here -
    see validation_metrics.json for the exact current numbers, since they will
    shift slightly whenever the feature set or data changes) shows
    HistGradientBoosting winning on stable historical periods but a plain
    logistic regression winning on the one holdout month that shares the
    post-May-2026 auto-approval regime with the actual test set - the tree
    model overfits the few dozen fraud examples available and doesn't
    transfer to the shifted distribution, which is exactly the distribution
    test_unlabelled.csv is drawn from. Numbers were previously hard-coded
    directly in this docstring and drifted out of sync with the code after a
    retrain - deliberately not repeating that mistake by putting numbers here
    at all now."""
    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000)),
    ])
    pipe.fit(X, y)
    return pipe


def walk_forward_eval(labeled, partners, products, month_starts):
    """Actually computes the walk-forward table (previously hard-coded --
    see decision-log.md for why that was wrong and risky). For each month
    start date, trains both models on everything strictly before that
    month and evaluates on that month only. Returns real n_fraud/AUC for
    each month, not typed-in numbers -- so this is what changes if the
    underlying data changes, not a stale table someone forgot to update.
    """
    results = {}
    for month_start in month_starts:
        month_start = pd.Timestamp(month_start)
        month_end = month_start + pd.DateOffset(months=1)
        tr_m = labeled[labeled.submitted_at < month_start]
        val_m = labeled[(labeled.submitted_at >= month_start) & (labeled.submitted_at < month_end)]
        if len(val_m) == 0 or val_m['is_fraud'].nunique() < 2:
            results[month_start.strftime('%Y-%m')] = {
                'n_fraud': int(val_m['is_fraud'].sum()),
                'hgb_auc': None, 'logreg_auc': None,
                'note': 'skipped -- validation month has fewer than 2 classes present',
            }
            continue
        tr_feat_m = build_features(tr_m, partners, products)
        val_feat_m = build_features(val_m, partners, products)
        X_tr_m, y_tr_m = feature_matrix(tr_feat_m), tr_m['is_fraud'].values
        X_val_m, y_val_m = feature_matrix(val_feat_m), val_m['is_fraud'].values

        logreg_m = fit_logreg(X_tr_m, y_tr_m)
        hgb_m = fit_hgb(X_tr_m, y_tr_m)
        logreg_auc_m = roc_auc_score(y_val_m, logreg_m.predict_proba(X_val_m)[:, 1])
        hgb_auc_m = roc_auc_score(y_val_m, hgb_m.predict_proba(X_val_m)[:, 1])

        results[month_start.strftime('%Y-%m')] = {
            'n_fraud': int(y_val_m.sum()),
            'hgb_auc': round(float(hgb_auc_m), 3),
            'logreg_auc': round(float(logreg_auc_m), 3),
        }
    return results


def precision_at_k(y_true, scores, k):
    order = np.argsort(-scores)[:k]
    if k == 0:
        return float('nan')
    return float(np.mean(np.asarray(y_true)[order]))


def rupees_saved_at_k(val_df, scores, k):
    """Rupees of fraud stopped minus rupees of review cost, if the top-k highest-scored
    claims in the validation set were the ones sent to the investigation desk."""
    order = np.argsort(-scores)[:k]
    picked = val_df.iloc[order]
    fraud_caught_inr = picked.loc[picked.is_fraud == 1, 'claim_amount_inr'].sum()
    genuine_held = int((picked.is_fraud == 0).sum())
    review_cost = k * CONTACT_COST_INR
    goodwill_cost = genuine_held * GOODWILL_COST_INR
    net = fraud_caught_inr - review_cost - goodwill_cost
    return {
        'k': k,
        'fraud_claims_caught': int((picked.is_fraud == 1).sum()),
        'fraud_rupees_caught': float(fraud_caught_inr),
        'genuine_claims_held': genuine_held,
        'review_cost_inr': review_cost,
        'goodwill_cost_inr': goodwill_cost,
        'net_rupees': float(net),
    }


def calibrate_scores(X_full, y_full, n_splits=5, seed=0):
    """
    LogisticRegression(class_weight='balanced') is needed for ranking (fraud is
    ~1.2% of claims) but its predicted probabilities are badly inflated as a
    result: checked on the June 2026 holdout and found 22 actual frauds vs a
    sum of predicted probabilities of 62 (~2.8x over). Confirmed on all labeled
    data via 5-fold out-of-fold predictions: sum(scores)=3,430 vs 141 actual
    (~24x over-count) before calibration.
    Fixes this with Platt scaling: get honest out-of-fold scores, fit a 1-D
    logistic regression mapping raw score -> true probability. This is a
    monotonic transform, so it changes magnitude only, not ranking (AUC is
    provably unchanged - verified: 0.8306 before and after).
    Returns a fitted calibrator with .predict_proba(scores.reshape(-1,1)).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_full))
    for tr_idx, te_idx in skf.split(X_full, y_full):
        m = fit_logreg(X_full[tr_idx], y_full[tr_idx])
        oof[te_idx] = m.predict_proba(X_full[te_idx])[:, 1]
    calibrator = LogisticRegression()
    calibrator.fit(oof.reshape(-1, 1), y_full)
    oof_calibrated = calibrator.predict_proba(oof.reshape(-1, 1))[:, 1]
    calibration_check = {
        'oof_sum_before_calibration': float(oof.sum()),
        'oof_sum_after_calibration': float(oof_calibrated.sum()),
        'actual_fraud_count': int(y_full.sum()),
        'auc_before': float(roc_auc_score(y_full, oof)),
        'auc_after': float(roc_auc_score(y_full, oof_calibrated)),
    }
    return calibrator, calibration_check, oof_calibrated


def main():
    t, te, partners, products = load_raw()

    labeled = t.dropna(subset=['is_fraud']).copy()
    labeled['is_fraud'] = labeled['is_fraud'].astype(int)

    # --- Time-based validation split -----------------------------------------
    # test_unlabelled.csv is entirely Jul-Sep 2026, fully after the 1 May 2026
    # auto-approval change. A random split of train.csv would validate on a mix
    # of pre- and post-change claims and overstate real-world performance.
    #
    # First attempt was: train on everything before 1 May 2026, validate on
    # May-Jun 2026. That produced an AUC of 0.35 (worse than random) because the
    # engineered post_may2026/risky_segment features - the two features carrying
    # the single strongest signal found in the EDA - have zero variance before
    # 1 May 2026 by construction, so the model trained on that window never sees
    # a single example of the regime it most needs to learn. Confirmed this
    # directly (see decision-log.md) before trusting a below-chance AUC.
    #
    # Fixed split: train through 31 May 2026 (includes one month of the new
    # regime, so the model can actually learn it), validate on June 2026 only -
    # the most recent labeled month, and the closest available proxy for test's
    # entirely-post-change distribution.
    split_date = pd.Timestamp('2026-06-01')
    tr_raw = labeled[labeled.submitted_at < split_date].copy()
    val_raw = labeled[labeled.submitted_at >= split_date].copy()

    tr_feat = build_features(tr_raw, partners, products)
    val_feat = build_features(val_raw, partners, products)

    X_tr, y_tr = feature_matrix(tr_feat), tr_raw['is_fraud'].values
    X_val, y_val = feature_matrix(val_feat), val_raw['is_fraud'].values

    model_val = fit_logreg(X_tr, y_tr)
    val_scores = model_val.predict_proba(X_val)[:, 1]

    # Documented comparison, not the final choice - see fit_logreg's docstring and
    # decision-log.md for the walk-forward evidence behind picking logistic
    # regression over this.
    hgb_val = fit_hgb(X_tr, y_tr)
    hgb_scores = hgb_val.predict_proba(X_val)[:, 1]
    hgb_auc = roc_auc_score(y_val, hgb_scores)

    auc = roc_auc_score(y_val, val_scores)
    ap = average_precision_score(y_val, val_scores)
    p_at_40 = precision_at_k(y_val, val_scores, MONTHLY_REVIEW_CAPACITY)
    base_rate = y_val.mean()
    econ_40 = rupees_saved_at_k(val_raw.assign(is_fraud=y_val), val_scores, MONTHLY_REVIEW_CAPACITY)
    econ_sweep = {k: rupees_saved_at_k(val_raw.assign(is_fraud=y_val), val_scores, k)
                  for k in (5, 10, 15, 20, 25, 30, 35, 40)}

    # Random-selection baseline at k=40, for comparison: what the investigation desk
    # would expect to catch by reviewing 40 claims with no model at all.
    avg_fraud_amt = val_raw.loc[y_val == 1, 'claim_amount_inr'].mean()
    exp_fraud_caught_random = MONTHLY_REVIEW_CAPACITY * base_rate
    exp_fraud_inr_random = exp_fraud_caught_random * avg_fraud_amt
    exp_net_random = (exp_fraud_inr_random
                       - MONTHLY_REVIEW_CAPACITY * CONTACT_COST_INR
                       - MONTHLY_REVIEW_CAPACITY * (1 - base_rate) * GOODWILL_COST_INR)
    random_baseline_40 = {
        'expected_fraud_claims_caught': float(exp_fraud_caught_random),
        'expected_fraud_rupees_caught': float(exp_fraud_inr_random),
        'expected_net_rupees': float(exp_net_random),
    }

    # "Always predict not-fraud" baseline, to make the accuracy-KPI problem concrete
    naive_accuracy = 1 - base_rate

    report = {
        'validation_window': ['2026-06-01', '2026-06-30'],
        'validation_rows': int(len(y_val)),
        'validation_fraud_count': int(y_val.sum()),
        'validation_base_fraud_rate': float(base_rate),
        'naive_always_not_fraud_accuracy': float(naive_accuracy),
        'roc_auc': float(auc),
        'roc_auc_hgb_comparison_not_chosen': float(hgb_auc),
        'average_precision': float(ap),
        'precision_at_40': p_at_40,
        'economics_at_40_per_month': econ_40,
        'economics_sweep_by_k': econ_sweep,
        'random_selection_baseline_at_40': random_baseline_40,
        'model_vs_random_fraud_rupees_lift': (
            econ_40['fraud_rupees_caught'] / exp_fraud_inr_random if exp_fraud_inr_random else None
        ),
        'walk_forward_monthly_auc_hgb_vs_logreg': walk_forward_eval(
            labeled, partners, products,
            month_starts=['2026-03-01', '2026-04-01', '2026-05-01', '2026-06-01'],
        ),
    }
    with open('validation_metrics.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    # --- Final model: retrain on ALL labeled data for the actual submission ---
    full_feat = build_features(labeled, partners, products)
    X_full, y_full = feature_matrix(full_feat), labeled['is_fraud'].values
    final_model = fit_logreg(X_full, y_full)
    calibrator, calibration_check, oof_calibrated = calibrate_scores(X_full, y_full)
    report['probability_calibration_check'] = calibration_check

    # A single live claim has no "this month's other claims" to rank against, so
    # /score needs a fixed operating threshold to say review_recommended True/False.
    # Derive it from the actual capacity constraint (40/month, ops-policy.pdf S5,
    # Farhan's email) rather than an arbitrary 0.5 cutoff, which is meaningless
    # once scores are properly calibrated (max calibrated score in test is ~0.18):
    # take the score value at the percentile that would select ~40 claims out of
    # a typical month's volume, using the same out-of-fold calibrated scores.
    avg_monthly_volume = float(labeled.groupby(labeled.submitted_at.dt.to_period('M')).size().mean())
    target_percentile = 1 - (MONTHLY_REVIEW_CAPACITY / avg_monthly_volume)
    review_threshold = float(np.quantile(oof_calibrated, target_percentile))
    report['review_threshold_basis'] = {
        'avg_monthly_claim_volume': avg_monthly_volume,
        'monthly_review_capacity': MONTHLY_REVIEW_CAPACITY,
        'target_percentile': target_percentile,
        'review_threshold': review_threshold,
    }
    with open('validation_metrics.json', 'w') as f:
        json.dump(report, f, indent=2)
    joblib.dump({
        'model': final_model,
        'calibrator': calibrator,
        'feature_columns': FEATURE_COLUMNS,
        'review_threshold': review_threshold,
    }, 'model.joblib')

    # --- Predictions on test_unlabelled.csv ------------------------------------
    te_feat = build_features(te, partners, products)
    X_te = feature_matrix(te_feat)
    te_scores_raw = final_model.predict_proba(X_te)[:, 1]
    te_scores = calibrator.predict_proba(te_scores_raw.reshape(-1, 1))[:, 1]
    out = pd.DataFrame({'claim_id': te['claim_id'].values, 'score': te_scores})
    # sample_submission.csv row order — match it exactly
    sample = pd.read_csv('data/sample_submission.csv')
    out = sample[['claim_id']].merge(out, on='claim_id', how='left')
    assert out['score'].isna().sum() == 0, "missing scores for some claim_id"
    out.to_csv('predictions.csv', index=False)
    print(f"\nWrote predictions.csv: {len(out)} rows, calibrated score range "
          f"[{out.score.min():.4f}, {out.score.max():.4f}], mean {out.score.mean():.4f}, "
          f"sum {out.score.sum():.1f} (expected fraud count if calibration holds up out-of-sample)")

    # quick sanity: how many test claims fall in the risky segment used in EDA?
    risky_n = int(te_feat['risky_segment'].sum())
    print(f"Test claims in the new-partner/auto-approved risky segment: {risky_n} of {len(te_feat)}")


if __name__ == '__main__':
    main()
