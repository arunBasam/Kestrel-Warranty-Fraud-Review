"""
Kestrel warranty-fraud scoring service.

Run:  python3 app.py
Then open http://127.0.0.1:5000/

- POST /score   { claim JSON }  -> { score, reasons, review_recommended, ... }
- GET  /        a one-page form that calls /score and shows the result
- GET  /health  liveness check, also reports whether the model loaded

No paid API key anywhere. If model.joblib is missing (e.g. train_model.py has
not been run yet), the service still starts; /score returns a clear 503 with
instructions instead of crashing, and the UI shows the same message.
"""
import os
import traceback
from flask import Flask, request, jsonify, render_template_string

import pandas as pd
import joblib

from features import build_features, FEATURE_COLUMNS
from explain import explain_claim, risk_band

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'model.joblib')
DATA_DIR = os.path.join(APP_DIR, 'data')

app = Flask(__name__)

_state = {'model_artifact': None, 'partners': None, 'products': None, 'load_error': None}


def _load():
    try:
        _state['model_artifact'] = joblib.load(MODEL_PATH)
        _state['partners'] = pd.read_csv(os.path.join(DATA_DIR, 'partners.csv'), parse_dates=['onboarded_date'])
        _state['products'] = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
        _state['load_error'] = None
    except Exception as e:
        # Deliberately caught broadly: the service must still start and serve
        # a clear error rather than crash on import if the model or reference
        # data hasn't been generated yet.
        _state['load_error'] = f"{type(e).__name__}: {e}"


_load()

REQUIRED_FIELDS = [
    'claim_id', 'submitted_at', 'partner_id', 'sku', 'product_serial',
    'days_since_purchase', 'claim_amount_inr', 'photo_attached',
    'partner_inspected', 'customer_prior_claims',
]


@app.route('/health')
def health():
    ok = _state['load_error'] is None
    return jsonify({
        'status': 'ok' if ok else 'model_unavailable',
        'error': _state['load_error'],
    }), (200 if ok else 503)


@app.route('/score', methods=['POST'])
def score():
    if _state['load_error']:
        return jsonify({
            'error': 'Model not available on this server.',
            'detail': _state['load_error'],
            'fix': 'Run `python3 train_model.py` from the project root to generate model.joblib, then restart app.py.',
        }), 503

    claim = request.get_json(force=True, silent=True)
    if not claim:
        return jsonify({'error': 'Request body must be JSON.'}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in claim]
    if missing:
        return jsonify({'error': f'Missing required field(s): {missing}'}), 400

    try:
        row = pd.DataFrame([claim])
        # optional text/reference fields the model doesn't use but featurization tolerates missing
        for opt, default in [('inspector_note', None), ('claim_description', ''), ('source', 'crm')]:
            if opt not in row.columns:
                row[opt] = default
        feat = build_features(row, _state['partners'], _state['products'])
        result = explain_claim(_state['model_artifact'], feat)
    except Exception as e:
        return jsonify({
            'error': 'Could not score this claim.',
            'detail': f"{type(e).__name__}: {e}",
        }), 400

    return jsonify({
        'claim_id': claim.get('claim_id'),
        'score': round(result['score'], 4),
        'risk_band': risk_band(result['score'], _state['model_artifact'].get('review_threshold', 0.5)),
        'review_recommended': result['score'] >= _state['model_artifact'].get('review_threshold', 0.5),
        'review_threshold': _state['model_artifact'].get('review_threshold'),
        'reasons': result['reasons'],
    })


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Kestrel Warranty Claim Review</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 760px; margin: 40px auto; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  label { display:block; margin-top: 10px; font-size: 0.85rem; color:#444; }
  input, select { width: 100%; padding: 6px 8px; font-size: 0.95rem; box-sizing: border-box; }
  button { margin-top: 18px; padding: 10px 18px; font-size: 1rem; cursor: pointer; }
  #result { margin-top: 24px; padding: 16px; border-radius: 8px; }
  .fraud { background: #fdecea; border: 1px solid #f5c2c0; }
  .clean { background: #eaf7ea; border: 1px solid #c2e6c2; }
  .err { background: #fff3cd; border: 1px solid #ffe08a; }
  .score { font-size: 2rem; font-weight: 700; }
  ul { padding-left: 18px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 16px; }
  .note { color:#666; font-size:0.8rem; margin-top:4px; }
</style>
</head>
<body>
<h1>Kestrel warranty claim review</h1>
<p class="note">Enter a claim below and score it. This calls the same /score endpoint the real integration would use.</p>
<div class="grid">
  <div>
    <label>Claim ID<input id="claim_id" value="WC900001"></label>
    <label>Submitted at (YYYY-MM-DD HH:MM)<input id="submitted_at" value="2026-08-15 10:00"></label>
    <label>Partner ID<input id="partner_id" value="SP3219"></label>
    <label>SKU<input id="sku" value="KH-MG-03"></label>
    <label>Product serial<input id="product_serial" value="KH395147108"></label>
  </div>
  <div>
    <label>Days since purchase<input id="days_since_purchase" type="number" value="120"></label>
    <label>Claim amount (INR)<input id="claim_amount_inr" type="number" value="1850"></label>
    <label>Photo attached
      <select id="photo_attached"><option>Y</option><option>N</option></select>
    </label>
    <label>Partner inspected
      <select id="partner_inspected"><option>N</option><option>Y</option></select>
    </label>
    <label>Customer prior claims<input id="customer_prior_claims" type="number" value="2"></label>
  </div>
</div>
<button onclick="submitClaim()">Score this claim</button>
<div id="result" style="display:none"></div>

<script>
async function submitClaim() {
  const ids = ['claim_id','submitted_at','partner_id','sku','product_serial',
               'days_since_purchase','claim_amount_inr','photo_attached',
               'partner_inspected','customer_prior_claims'];
  const claim = {};
  ids.forEach(id => claim[id] = document.getElementById(id).value);
  claim.days_since_purchase = Number(claim.days_since_purchase);
  claim.claim_amount_inr = Number(claim.claim_amount_inr);
  claim.customer_prior_claims = Number(claim.customer_prior_claims);

  const box = document.getElementById('result');
  box.style.display = 'block';
  box.className = '';
  box.innerHTML = 'Scoring...';

  try {
    const res = await fetch('/score', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(claim)
    });
    const data = await res.json();
    if (!res.ok) {
      box.className = 'err';
      box.innerHTML = '<b>' + (data.error || 'Error') + '</b><br>' + (data.detail || data.fix || '');
      return;
    }
    box.className = data.review_recommended ? 'fraud' : 'clean';
    box.innerHTML = '<div class="score">' + (data.score*100).toFixed(1) + '% likely fraudulent</div>' +
      '<div>' + (data.review_recommended ? 'Recommended: send to investigation desk.' : 'Recommended: no review needed.') + '</div>' +
      '<ul>' + data.reasons.map(r => '<li>' + r + '</li>').join('') + '</ul>';
  } catch (e) {
    box.className = 'err';
    box.innerHTML = 'Request failed: ' + e;
  }
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(PAGE)


if __name__ == '__main__':
    if _state['load_error']:
        print(f"WARNING: model did not load ({_state['load_error']}). "
              f"Service will still start; /score will return 503 until you run train_model.py.")
    app.run(host='127.0.0.1', port=5000, debug=False)
