"""
baseline_comparison.py
------------------------
FINAL_VALIDATION_REPORT.md's baseline-comparison table (section 5) and the
rule-vs-model analysis (section 6) were written as prose with no script in
this repo actually producing them -- a reviewer found the "Random
selection" and "Highest claim amount" rows were identical (a copy-paste
bug), and asked for internal consistency with validation_metrics.json's
own `random_selection_baseline_at_40`, which uses an expected-value
calculation, not a single arbitrary draw.

This script computes every row for real, on the same June 2026 holdout and
the same model-fitting procedure train_model.py uses for its headline
numbers, and writes baseline_metrics.json. The report's table should be
copied from that file's output, not typed by hand.

Run after train_model.py (uses the same train/validation split logic).
"""
import json
import numpy as np
import pandas as pd

from features import build_features, feature_matrix
from train_model import (
    load_raw, fit_logreg, rupees_saved_at_k,
    GOODWILL_COST_INR, CONTACT_COST_INR, MONTHLY_REVIEW_CAPACITY,
)

K = MONTHLY_REVIEW_CAPACITY


def main():
    t, te, partners, products = load_raw()
    labeled = t.dropna(subset=['is_fraud']).copy()
    labeled['is_fraud'] = labeled['is_fraud'].astype(int)

    split_date = pd.Timestamp('2026-06-01')
    tr_raw = labeled[labeled.submitted_at < split_date].copy()
    val_raw = labeled[labeled.submitted_at >= split_date].copy().reset_index(drop=True)

    tr_feat = build_features(tr_raw, partners, products)
    val_feat = build_features(val_raw, partners, products)
    X_tr, y_tr = feature_matrix(tr_feat), tr_raw['is_fraud'].values
    X_val, y_val = feature_matrix(val_feat), val_raw['is_fraud'].values

    val_raw = val_raw.assign(is_fraud=y_val)
    amt = val_raw['claim_amount_inr'].values
    base_rate = float(y_val.mean())  # June's own observed rate -- matches train_model.py's
                                      # random_selection_baseline_at_40 exactly, NOT the
                                      # training-period rate (an earlier draft of this script
                                      # used y_tr.mean() and got a different, wrong answer)

    rows = {}

    exp_fraud_claims = K * base_rate
    exp_fraud_rupees = K * base_rate * val_raw.loc[val_raw.is_fraud == 1, 'claim_amount_inr'].mean()
    exp_net = (exp_fraud_rupees - K * CONTACT_COST_INR
               - K * (1 - base_rate) * GOODWILL_COST_INR)
    rows['random_selection_expected_value'] = {
        'fraud_claims_caught': round(exp_fraud_claims, 2),
        'precision': round(exp_fraud_claims / K, 4),
        'fraud_rupees_caught': round(exp_fraud_rupees, 0),
        'net_rupees': round(exp_net, 0),
        'note': 'expected value under random selection at base_rate, not a single draw -- matches validation_metrics.json random_selection_baseline_at_40',
    }

    econ = rupees_saved_at_k(val_raw, amt, K)
    rows['highest_claim_amount'] = {
        'fraud_claims_caught': econ['fraud_claims_caught'],
        'precision': round(econ['fraud_claims_caught'] / K, 4),
        'fraud_rupees_caught': econ['fraud_rupees_caught'],
        'net_rupees': econ['net_rupees'],
    }

    scores = val_feat['partner_is_new'].values * 1e6 + amt
    econ = rupees_saved_at_k(val_raw, scores, K)
    n_segment = int(val_feat['partner_is_new'].sum())
    rows['new_partner_any'] = {
        'fraud_claims_caught': econ['fraud_claims_caught'],
        'precision': round(econ['fraud_claims_caught'] / K, 4),
        'fraud_rupees_caught': econ['fraud_rupees_caught'],
        'net_rupees': econ['net_rupees'],
        'n_matching_in_val_month': n_segment,
    }

    seg2 = (val_feat['partner_is_new'].values.astype(bool) & val_feat['post_may2026'].values.astype(bool))
    scores = seg2.astype(float) * 1e6 + amt
    econ = rupees_saved_at_k(val_raw, scores, K)
    rows['new_partner_and_post_may_no_amount_filter'] = {
        'fraud_claims_caught': econ['fraud_claims_caught'],
        'precision': round(econ['fraud_claims_caught'] / K, 4),
        'fraud_rupees_caught': econ['fraud_rupees_caught'],
        'net_rupees': econ['net_rupees'],
        'n_matching_in_val_month': int(seg2.sum()),
    }

    risky = val_feat['risky_segment'].values.astype(bool)
    scores = risky.astype(float) * 1e6 + amt
    econ = rupees_saved_at_k(val_raw, scores, K)
    rule_top_k_idx = set(np.argsort(-scores)[:K].tolist())
    rows['rule_only_risky_segment'] = {
        'fraud_claims_caught': econ['fraud_claims_caught'],
        'precision': round(econ['fraud_claims_caught'] / K, 4),
        'fraud_rupees_caught': econ['fraud_rupees_caught'],
        'net_rupees': econ['net_rupees'],
        'n_matching_in_val_month': int(risky.sum()),
        'note': ('fewer claims match the rule than K -- the remaining slots in the '
                 'top-K are filled by amount ordering outside the segment, since the '
                 'rule itself does not rank non-matching claims'
                 if risky.sum() < K else 'rule alone produced >= K matching claims'),
    }

    model_val = fit_logreg(X_tr, y_tr)
    model_scores = model_val.predict_proba(X_val)[:, 1]
    econ = rupees_saved_at_k(val_raw, model_scores, K)
    model_top_k_idx = set(np.argsort(-model_scores)[:K].tolist())
    rows['model_logistic_regression'] = {
        'fraud_claims_caught': econ['fraud_claims_caught'],
        'precision': round(econ['fraud_claims_caught'] / K, 4),
        'fraud_rupees_caught': econ['fraud_rupees_caught'],
        'net_rupees': econ['net_rupees'],
    }

    overlap = len(rule_top_k_idx & model_top_k_idx)
    model_outside_rule = len(model_top_k_idx - rule_top_k_idx)
    model_outside_rule_and_risky = sum(
        1 for i in (model_top_k_idx - rule_top_k_idx) if not risky[i]
    )
    rows['rule_vs_model_overlap_at_k'] = {
        'k': K,
        'overlap_count': overlap,
        'model_claims_outside_rule_top_k': model_outside_rule,
        'of_those_also_outside_risky_segment_entirely': model_outside_rule_and_risky,
    }

    with open('baseline_metrics.json', 'w') as f:
        json.dump(rows, f, indent=2, default=str)

    print(json.dumps(rows, indent=2, default=str))


if __name__ == '__main__':
    main()
