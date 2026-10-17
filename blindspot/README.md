# Blindspot

An independent auditor for a bank's anti-money-laundering transaction monitoring.
It takes the bank's **rulebook** and its transactions, and measures what the rules
miss. UK viewpoint throughout: all amounts are GBP.

**Code computes every number, the AI reads regulation and writes the report, and a
person makes every decision.**

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate          # in every new terminal window
pip install -r requirements.txt
```

For the LLM steps, create a `.env` file (git ignores it):

```
GROQ_API_KEY=your-key-here
```

## Quick test without SAML-D (a 2,000-customer fake bank)

```bash
python -m blindspot.make_sample
python -m blindspot.run_all --data data/sample
```

## Real data

1. Download SAML-D from
   https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
2. Convert a realistic sample (about 1 launderer per 100 customers):

```bash
python -m blindspot.normalise path/to/SAML-D.csv --customers 50000 --laundering 500   # -> data/base/
python -m blindspot.run_all                                                         # everything else
streamlit run app.py                                                                # dashboard
```

To audit a different rulebook, rerun from the engine:

```bash
python -m blindspot.run_all --from engine --rules rulebooks/northbridge_bank_remediated.yaml
```

The dashboard opens empty. On **Audit setup**, upload or choose a rulebook, adjust it
if you like, and press Run audit.

## The bank's rulebook

The input Blindspot audits: a YAML file listing the bank's rules, what payments each
can see, its trigger and its points, plus the case threshold and any suppression
policy. The format is documented at the top of `blindspot/rulebook.py`.

| File | What it is |
|---|---|
| `rulebooks/northbridge_bank.yaml` | Fictional bank modelled on Bank of America's failures. The default. |
| `rulebooks/northbridge_bank_remediated.yaml` | The same bank after acting on Blindspot's findings. |
| `rulebooks/harbourside_building_society.yaml` | Untested test bank: stale rules. Predicted flaws in `rulebooks/PREDICTIONS.md`. |
| `rulebooks/meridian_pay.yaml` | Untested test bank: watches only money going out. Predicted flaws in `rulebooks/PREDICTIONS.md`. |

## Pipeline

| Step | Module | Reads | Writes |
|---|---|---|---|
| Normalise | `normalise.py` | SAML-D CSV | `data/base/` transactions, labels |
| Plant | `plant.py` | `data/base/` | `data/` transactions, labels, planted_customers |
| Engine | `engine.py` | transactions **only** + the rulebook | alerts, cases, rules, rulebook_used.yaml |
| Evaluate | `evaluate.py` | answers | prints what the bank's rules caught |
| Investigate | `investigate.py` | answers, for reviewed customers only | investigations |
| 1 Coverage | `checks/coverage.py` | transactions, rules, alerts | coverage, findings |
| 2 Rule weights | `checks/calibration.py` | alerts, rules, investigations | calibration, findings |
| 3 Regulator alerts | `checks/advisory.py` | an advisory (LLM), transactions, rules, cases | advisory, findings |
| 4 Below the line | `checks/below_line.py` | transactions, cases (ML) | below_line, findings |
| Report | `report.py` | all findings (LLM writes text) | findings, review_queue, summary.json |
| Evidence | `evidence.py` | review queue, transactions, alerts | evidence |
| Dashboard | `app.py` | the files above | the screen |

`run_all` runs Plant → Evidence in order and stops at the first failure.
All table shapes live in `blindspot/schema.py`.

## Rules everyone follows

- **The engine and the four checks never read the answers** (`labels.parquet`,
  `planted_customers.parquet`). Only `evaluate`, `investigate` and the sections marked
  PROOF do. Test it: move `data/labels.parquet` away and run the checks with `--no-proof`.
- **Monitor by `customer_id`, not `sender_id`.** Inbound payments belong to the receiver.
- **"Caught" is customer-level:** a launderer is caught if the bank opened a case.
- **Only public text and totals go to the LLM**, never transactions or customer IDs.
  Code rejects any LLM sentence containing a number it wasn't given.
- **Seed 42 everywhere**, so every run gives the same results.

## Planted patterns (GBP)

| Typology | What it looks like |
|---|---|
| `structured_money_orders` | Sets of money orders on one day totalling £2,400–£2,499, just under the £2,500 ID limit |
| `high_risk_wires` | 10–25 inbound wires of £15k–£60k from Belarus or Georgia within 60 days |
| `credit_card_funnel` | One shop repaying 15–40 unrelated people's credit cards |
| `structured_cash` | 6–15 cash deposits of £9,000–£9,900 |
| `slow_burn` | Small deposits and high-risk payments every 25–45 days, too thin to score |

Innocent look-alikes are planted too (occasional money orders, people repaying their
own card, shops banking takings, normal overseas wires), so no payment type gives
the launderers away.

## Known limits (say these before a judge does)

- **The planted results measure patterns we designed.** On SAML-D's own, independent
  laundering, Blindspot adds little: it's mostly networks of accounts, and every check
  looks at one customer at a time. A network check is the next module.
- **Slow burn is rarely found.** It's designed to look normal.
- **Investigators are simulated as always right.**
- **The £2,500 and £10,000 limits are the fictional bank's own policies**, not UK law.
- **The advisory in `advisories/` is illustrative**, written by the team in the style
  of a UK alert. In production Blindspot reads the real text.

## Back pocket

IBM AMLworld could be added with its own `normalise_ibm.py` mapping into the same schema.
