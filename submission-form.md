1. What did you build, and what business outcome does it move?

A calibrated fraud-scoring service and review queue. At 40 reviews/month, the model catches about 4.4× more fraud rupees than random selection — ₹11,246 vs ₹2,562 in June. A simpler rule catches the same 9 fraud cases, so I recommend using the rule operationally for now.

2. What score do you expect predictions.csv to get, on which metric, and why?

I expect about 0.767 ROC-AUC. The 95% bootstrap CI is 0.635–0.886, so I would not treat 0.767 as guaranteed. ROC-AUC is appropriate because the submission is a continuous fraud score and ranking matters.

3. How do you know it works?

I used 11,146 labeled claims and held out June 2026 completely: 713 claims with 22 fraud cases. June AUC was 0.767 and precision@40 was 22.5%. I also ran leakage, calibration, bootstrap and explanation checks. The main weakness is the small number of post-policy fraud cases.

4. Did you change, narrow, or push back on the client's ask?

Yes. I pushed back on 97% accuracy because a model predicting no fraud already achieves about 98.7% historical accuracy while catching nothing. I used fraud rupees per review instead. I also found that a simple rule currently performs as well as the model.

5. What is wrong with what you are handing us, or with the data?

The main limitation is only 22 June fraud cases, so the economics are uncertain. At k=10, estimated value is +₹913, but the 95% CI is -₹4,856 to +₹6,545, with about a 55% chance of being positive. Older Zoho records also contain unresolved cases incorrectly recorded as non-fraud.

6. What did you deliberately leave out, and why?

I left out LLM explanations, SHAP, partner target encoding and a ₹1,800–2,000 "structuring" feature. The structuring pattern changed direction between months, so I treated it as noise. Explanations instead use the model's actual coefficients.

7. Anything you built or found that nobody asked for?

I added bootstrap confidence intervals, adversarial leakage tests, calibration checks, threshold/economics analysis, baseline comparisons and explanation validation. The key finding was that the simple rule currently matches the model's fraud catches.

8. What did you use AI for?

I used ChatGPT and Claude for coding, debugging, review and challenging assumptions. I spent about 7 hours on the project. AI helped identify bugs and reproducibility issues, but I verified the final numbers against the actual data and code. No paid API or LLM is used inside the final service.

Loom recording: https://www.loom.com/share/208720c9bd5a4378b7d48dc43d196a17

9. What does one prediction cost, and what would a month cost at Kestrel's volume?

₹0 per prediction and ₹0 for 750 claims/month in paid API costs. The model runs locally with no paid API calls. This excludes any normal infrastructure/hosting cost if Kestrel later deploys the service.

10. Someone picks this up on Monday — the three things they need to know.
Run train_model.py → baseline_comparison.py → bootstrap_validation.py → validate.py.
Use the simple rule for today's review queue; keep the model for future ranking.
Do not overclaim the economics — only 22 fraud cases support the current estimate.