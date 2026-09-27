"""
Feature engineering for the Kestrel warranty-fraud model.

Shared by train_model.py (offline training) and service/app.py (live scoring),
so a claim is featurized identically whether it comes from a CSV or a live
JSON request. Keep this file free of anything that reads is_fraud.
"""
import numpy as np
import pandas as pd

NEW_PARTNER_DAYS = 180          # "new partner" cutoff used in the EDA (email: partners onboarded "over the past year")
AUTO_APPROVE_CUTOFF_INR = 2000  # ops-policy.pdf S5: claims under this were auto-approved without inspection from 1 May 2026
AUTO_APPROVE_START = pd.Timestamp('2026-05-01')

FEATURE_COLUMNS = [
    'claim_amount_inr',
    'log_claim_amount',
    'amount_to_price_ratio',
    'days_since_purchase',
    'customer_prior_claims',
    'photo_attached_bin',
    'partner_inspected_bin',
    'partner_age_days',
    'partner_is_new',
    'post_may2026',
    'risky_segment',
    'partner_type_code',
    'family_code',
    # near_threshold (claims priced just under the auto-approval cutoff) was
    # tried and REMOVED from the feature set: the "structuring" pattern
    # (18.8% vs 34.8% vs 16.7% aggregate) reverses sign between May (53.8% vs
    # 11.7%) and June (10.0% vs 22.2%) - not stable across the only 2 months
    # available, almost certainly noise from a small cell (n=13-23/month).
    # Tested, found unstable, not used - see decision-log.md. The raw feature
    # is still computed below (for anyone who wants to re-examine it) but is
    # not part of the model or the live explanation.
]

# Fixed category orderings so train-time and serve-time encodings always match,
# even if a live request never happens to see every category.
PARTNER_TYPES = ['authorised_service_centre', 'franchise', 'freelance_technician']
FAMILIES = [
    'Air Fryer', 'Mixer Grinder', 'Water Purifier', 'Robot Vacuum',
    'Induction Cooktop', 'Ceiling Fan', 'Room Heater',
]


def _code(series, categories):
    return pd.Categorical(series, categories=categories).codes.astype(float)


def build_features(df, partners, products):
    """
    df: one row per claim, columns as in README (submitted_at as string or datetime OK).
    partners: partner_id, city, onboarded_date, partner_type
    products: sku, family, list_price_inr, warranty_months
    Returns a new dataframe with FEATURE_COLUMNS added. Does not mutate inputs.
    """
    d = df.copy()
    d['submitted_at'] = pd.to_datetime(d['submitted_at'])

    d = d.merge(partners[['partner_id', 'onboarded_date', 'partner_type']], on='partner_id', how='left')
    d['onboarded_date'] = pd.to_datetime(d['onboarded_date'])
    d = d.merge(products[['sku', 'family', 'list_price_inr']], on='sku', how='left')

    d['log_claim_amount'] = np.log1p(d['claim_amount_inr'].clip(lower=0))
    d['amount_to_price_ratio'] = d['claim_amount_inr'] / d['list_price_inr'].replace(0, np.nan)

    d['photo_attached_bin'] = (d['photo_attached'].astype(str).str.upper() == 'Y').astype(float)
    d['partner_inspected_bin'] = (d['partner_inspected'].astype(str).str.upper() == 'Y').astype(float)

    d['partner_age_days'] = (d['submitted_at'] - d['onboarded_date']).dt.days
    d['partner_is_new'] = (d['partner_age_days'] <= NEW_PARTNER_DAYS).astype(float)

    d['post_may2026'] = (d['submitted_at'] >= AUTO_APPROVE_START).astype(float)
    under_threshold = d['claim_amount_inr'] < AUTO_APPROVE_CUTOFF_INR
    d['risky_segment'] = (d['partner_is_new'].astype(bool) & under_threshold & d['post_may2026'].astype(bool)).astype(float)
    # claims priced just under the auto-approval cutoff (possible structuring)
    d['near_threshold'] = (
        (d['claim_amount_inr'] >= AUTO_APPROVE_CUTOFF_INR * 0.9) & under_threshold
    ).astype(float)

    d['partner_type_code'] = _code(d['partner_type'], PARTNER_TYPES)
    d['family_code'] = _code(d['family'], FAMILIES)

    return d


def feature_matrix(d):
    """Return the numeric feature matrix (numpy-friendly) for a featurized dataframe."""
    return d[FEATURE_COLUMNS].astype(float).values
