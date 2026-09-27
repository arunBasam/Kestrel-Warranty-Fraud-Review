# 3-minute recording script

## 0:00–0:20 — The trap in the brief

Say: "The board asked for over 97% accuracy. Fraud is 1.3% of claims
historically, so a do-nothing model already clears that — 98.7%. It even
cuts both ways: in June alone, the most recent month, fraud rose to 3.1%,
where a do-nothing model would score only 96.9% — below the bar. Accuracy
just isn't a usable metric here, in either direction."

## 0:20–0:45 — The real driver, and a bug in my first attempt

Say: "Ritu suspected new partners. The data says new partners alone are
actually lower risk, 0.18%. But new partners on small claims after the May
auto-approval change hit 19.7% fraud. My first validation attempt gave an
AUC of 0.35 — worse than random — because my training window ended before
the policy even changed. The model had zero examples of the thing it needed
to learn."

## 0:45–1:15 — Two more bugs I caught before shipping

Say: "The model needs class-weighting to see fraud at all, but that made its
raw scores 24 times too high — I calibrated before writing predictions.csv,
not after. Separately, my first explanation engine hard-coded 'new partner'
as risk-raising, but the model's own coefficient says the opposite in
isolation. Testing on a real claim produced a self-contradiction, so I
rewrote it to always read direction off the model's actual math."

## 1:15–1:45 — The finding that argues against my own model

Say: "Here's the one I didn't expect. I built a simple rule — new partner,
small claim, after May 1 — and compared it to the model head to head. They
tie. Same 9 fraud cases, same precision, the rule even edges out slightly on
rupees. My actual recommendation to Ritu is to run the review queue off the
rule, not the model. I'd rather tell her that than oversell what I built."

## 1:45–2:15 — Show the live service and validate.py

Open the app, score a claim; then show `validate.py` running.

Say: "One endpoint, one screen, no paid API key. Reasons come straight from
the model's coefficients. And this — validate.py — includes an adversarial
leakage test: I corrupted June's outcomes and rebuilt May's features, and
they came back byte-identical. That's evidence the pipeline can't leak the
future into the past, not just a claim that it can't."

## 2:15–2:45 — What I threw away, and the uncertainty I didn't hide

Say: "I found what looked like claims structured just under the ₹2,000
cutoff — until I checked it by month and the pattern reversed sign between
May and June. Pulled it from the live product. And the headline economics
number, reviewing the top 10 claims a month — I bootstrapped it. It's only
55% likely to actually be positive in a given month. I'd rather Ritu hear
that from me than find out later."

## 2:45–3:00 — Close

Say: "The one recommendation that doesn't depend on any of that uncertainty:
fix the auto-approval gap for new partners directly. Everything else here is
worth trying and monitoring, not a guarantee."
