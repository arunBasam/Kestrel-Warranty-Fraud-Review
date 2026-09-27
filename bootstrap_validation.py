"""
bootstrap_validation.py
-------------------------
The FINAL_VALIDATION_REPORT previously quoted bootstrap confidence
intervals (AUC, precision@40, net economic value) that existed only in
prose -- no code in this repo actually produced them, so a reviewer
running `python train_model.py` could not reproduce them. This script
fixes that: it resamples the June 2026 validation set with replacement
(the standard non-parametric bootstrap for a fixed classifier's
performance on a fixed population) and reports the empirical 95% interval
for each metric.

Run: python3 bootstrap_validation.py   (run after train_model.py)

Writes bootstrap_metrics.json. Whatever numbers this actually produces are
what should appear in the docs -- do not hand-edit either side to match
the other.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import build_features, feature_matrix
from train_model import load_raw, fit_logreg, rupees_saved_at_k, MONTHLY_REVIEW_CAPACITY

N_RESAMPLES = 3000
SEED = 0
RECOMMENDED_K = 10  # the actual operating point recommended to Ritu -- see memo-to-ritu.md
CAPACITY_K = MONTHLY_REVIEW_CAPACITY  # 40 -- the desk's hard capacity ceiling, reported for comparison


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

    model_val = fit_logreg(X_tr, y_tr)
    val_scores = model_val.predict_proba(X_val)[:, 1]

    n = len(y_val)
    rng = np.random.RandomState(SEED)

    aucs = []
    precisions_at_k = {CAPACITY_K: [], RECOMMENDED_K: []}
    nets_at_k = {CAPACITY_K: [], RECOMMENDED_K: []}
    skipped_single_class = 0

    for _ in range(N_RESAMPLES):
        idx = rng.randint(0, n, size=n)
        y_b = y_val[idx]
        s_b = val_scores[idx]
        val_raw_b = val_raw.iloc[idx].reset_index(drop=True)

        if len(np.unique(y_b)) < 2:
            skipped_single_class += 1
            continue

        aucs.append(roc_auc_score(y_b, s_b))

        for k in (CAPACITY_K, RECOMMENDED_K):
            order = np.argsort(-s_b)[:k]
            precisions_at_k[k].append(float(np.mean(y_b[order])))
            econ = rupees_saved_at_k(val_raw_b.assign(is_fraud=y_b), s_b, k)
            nets_at_k[k].append(econ['net_rupees'])

    aucs = np.array(aucs)
    precisions_at_k = {k: np.array(v) for k, v in precisions_at_k.items()}
    nets_at_k = {k: np.array(v) for k, v in nets_at_k.items()}

    def ci(arr):
        return {
            'point_estimate_on_full_val_set': None,
            'bootstrap_mean': float(arr.mean()),
            'ci_95_low': float(np.percentile(arr, 2.5)),
            'ci_95_high': float(np.percentile(arr, 97.5)),
            'n_resamples_used': int(len(arr)),
        }

    full_auc = float(roc_auc_score(y_val, val_scores))
    auc_ci = ci(aucs); auc_ci['point_estimate_on_full_val_set'] = full_auc

    per_k_report = {}
    p_net_positive = {}
    for k in (CAPACITY_K, RECOMMENDED_K):
        order_full = np.argsort(-val_scores)[:k]
        full_p_k = float(np.mean(y_val[order_full]))
        full_econ_k = rupees_saved_at_k(val_raw.assign(is_fraud=y_val), val_scores, k)
        full_net_k = full_econ_k['net_rupees']

        p_ci = ci(precisions_at_k[k]); p_ci['point_estimate_on_full_val_set'] = full_p_k
        n_ci = ci(nets_at_k[k]); n_ci['point_estimate_on_full_val_set'] = full_net_k

        per_k_report[k] = {'precision_at_k': p_ci, 'net_rupees_at_k': n_ci}
        p_net_positive[k] = float(np.mean(nets_at_k[k] > 0))

    report = {
        'method': 'non-parametric bootstrap, resampling the June 2026 validation set '
                  'with replacement, scored by the model fit on pre-June training data',
        'n_resamples_requested': N_RESAMPLES,
        'n_resamples_skipped_single_class_in_resample': skipped_single_class,
        'roc_auc': auc_ci,
        f'at_k={CAPACITY_K}_desk_capacity_ceiling': per_k_report[CAPACITY_K],
        f'at_k={RECOMMENDED_K}_recommended_operating_point': per_k_report[RECOMMENDED_K],
        'p_net_rupees_positive': {
            f'k={CAPACITY_K}': p_net_positive[CAPACITY_K],
            f'k={RECOMMENDED_K}': p_net_positive[RECOMMENDED_K],
        },
    }

    with open('bootstrap_metrics.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nAUC:                     {full_auc:.3f}  95% CI [{auc_ci['ci_95_low']:.3f}, {auc_ci['ci_95_high']:.3f}]")
    for k in (CAPACITY_K, RECOMMENDED_K):
        p_ci = per_k_report[k]['precision_at_k']
        n_ci = per_k_report[k]['net_rupees_at_k']
        print(f"\n-- k={k} --")
        print(f"Precision@{k}:            {p_ci['point_estimate_on_full_val_set']:.3f}  "
              f"95% CI [{p_ci['ci_95_low']:.3f}, {p_ci['ci_95_high']:.3f}]")
        print(f"Net Rs @ k={k}:           {n_ci['point_estimate_on_full_val_set']:,.0f}  "
              f"95% CI [Rs {n_ci['ci_95_low']:,.0f}, Rs {n_ci['ci_95_high']:,.0f}]")
        print(f"P(net > 0) at k={k}:      {p_net_positive[k]:.2f}")


if __name__ == '__main__':
    main()
