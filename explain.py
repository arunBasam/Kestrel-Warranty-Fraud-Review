"""
Turns a single claim's score into plain-English reasons a Kestrel employee can
read, using the logistic regression's own coefficients - not an approximation,
not an LLM call. contribution_i = coefficient_i * scaled_feature_i is exactly
what the model adds to the log-odds for that feature, so "why did this score
this way" has one honest answer straight from the model itself.

Design rule that fixed a real bug during testing: each feature's DESCRIPTION is
a fact about the claim (independent of the model), and whether that fact is
framed as raising or lowering the score comes ONLY from the sign of its actual
computed contribution for this row - never assumed in advance. For example
partner_is_new has a *negative* coefficient (new partners are, on their own,
slightly lower risk per the training data - see decision-log.md), so "new
partner" must not be hard-coded as a risk-raising phrase; the very first
version of this file did exactly that and produced a self-contradictory
explanation, caught by testing on real claims before this shipped.

No paid API, no network call - this only needs the trained pipeline.
"""
from features import FEATURE_COLUMNS


def _describe_amount_to_price_ratio(v):
    return f"claimed amount is {v * 100:.0f}% of the product's retail price"


def _describe_risky_segment(v):
    return ("new partner, under ₹2,000, filed after the 1 May 2026 auto-approval change - "
            "this exact combination has a historical fraud rate of ~20%, vs ~1.2% overall")


def _describe_customer_prior_claims(v):
    n = int(round(v))
    if n == 0:
        return "customer has no prior warranty claims on file"
    return f"customer has {n} prior warranty claim{'s' if n != 1 else ''} on file"


def _describe_partner_inspected(v):
    return "claim was signed off by a partner inspection" if v >= 0.5 else "claim was not signed off by a partner inspection"


def _describe_partner_is_new(v):
    return "partner was onboarded within the last 6 months" if v >= 0.5 else "partner has been onboarded for more than 6 months"


def _describe_partner_type(v):
    # codes follow features.PARTNER_TYPES order
    names = ['an authorised service centre', 'a franchise', 'a freelance technician']
    idx = int(round(v))
    name = names[idx] if 0 <= idx < len(names) else 'an unrecorded partner type'
    return f"filed through {name}"


def _describe_days_since_purchase(v):
    return f"filed {int(round(v))} days after purchase"


def _describe_photo_attached(v):
    return "a photo was attached" if v >= 0.5 else "no photo was attached"


def _describe_near_threshold(v):
    # Deliberately unused (removed from DESCRIBERS below) - the "structuring"
    # pattern this would describe did not survive a month-by-month check
    # (reversed sign between May and June). Kept only so the function exists
    # if someone re-examines this with more data later.
    return "claim amount is priced just under the ₹2,000 auto-approval cutoff"


DESCRIBERS = {
    'amount_to_price_ratio': _describe_amount_to_price_ratio,
    'risky_segment': _describe_risky_segment,
    'customer_prior_claims': _describe_customer_prior_claims,
    'partner_inspected_bin': _describe_partner_inspected,
    'partner_is_new': _describe_partner_is_new,
    'partner_type_code': _describe_partner_type,
    'days_since_purchase': _describe_days_since_purchase,
    'photo_attached_bin': _describe_photo_attached,
    # near_threshold deliberately excluded - see decision-log.md (didn't survive
    # a month-by-month check, reversed sign between May and June)
    # Deliberately no describer for log_claim_amount / claim_amount_inr (redundant
    # with amount_to_price_ratio), post_may2026 (redundant with risky_segment),
    # partner_age_days (redundant with partner_is_new), family_code (weak, low
    # signal per model coefficients) - keeping the reason list short and non-repetitive.
}

# Only show risky_segment / near_threshold when they're actually true (raw value 0
# for these engineered flags isn't a meaningful "reason", it's just the default).
SHOW_ONLY_WHEN_TRUE = {'risky_segment'}


def explain_claim(model_artifact, features_row, max_reasons=4, min_contribution=0.05):
    """
    model_artifact: the dict loaded from model.joblib ({'model': pipeline, 'feature_columns': [...]})
    features_row: a single-row dataframe already run through features.build_features()
    Returns: {'score': float, 'reasons': [str, ...], 'contributions': {feature: float}}
    """
    pipe = model_artifact['model']
    calibrator = model_artifact.get('calibrator')
    imp = pipe.named_steps['impute']
    scaler = pipe.named_steps['scale']
    clf = pipe.named_steps['clf']

    x_raw = features_row[FEATURE_COLUMNS].astype(float).values
    x_imputed = imp.transform(x_raw)
    x_scaled = scaler.transform(x_imputed)

    raw_score = float(clf.predict_proba(x_scaled)[0, 1])
    # Raw score from the class_weight='balanced' model is badly miscalibrated
    # (overstates fraud likelihood by ~3x on validation - see train_model.py's
    # calibrate_scores docstring). Apply the same Platt-scaling calibrator used
    # for predictions.csv so a live score means the same thing as a submitted one.
    if calibrator is not None:
        score = float(calibrator.predict_proba([[raw_score]])[0, 1])
    else:
        score = raw_score
    contributions = {
        col: float(coef * x_scaled[0, i])
        for i, (col, coef) in enumerate(zip(FEATURE_COLUMNS, clf.coef_[0]))
    }

    ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))
    reasons = []
    for col, contrib in ranked:
        if len(reasons) >= max_reasons:
            break
        if abs(contrib) < min_contribution:
            continue
        describer = DESCRIBERS.get(col)
        if describer is None:
            continue
        raw_val = features_row[col].iloc[0]
        if col in SHOW_ONLY_WHEN_TRUE and raw_val < 0.5:
            continue
        fact = describer(raw_val)
        direction = "raises the risk score" if contrib > 0 else "lowers the risk score"
        fact_sentence = fact[0].upper() + fact[1:]  # capitalize first letter only - fact.capitalize() would lowercase "May 2026" etc.
        reasons.append(f"{fact_sentence} - this {direction}.")

    if not reasons:
        reasons = ["No single factor stands out; this claim looks broadly typical."]

    return {'score': score, 'reasons': reasons, 'contributions': contributions}


def risk_band(score, review_threshold):
    """High/Medium/Low label for the UI, derived from the same capacity-based
    threshold used for review_recommended (see decision-log.md) - not an
    arbitrary second cutoff. High = at/above the review threshold. Medium =
    at least half the threshold (still worth a human's attention if capacity
    allows). Low = everything else."""
    if score >= review_threshold:
        return 'High'
    if score >= review_threshold / 2:
        return 'Medium'
    return 'Low'