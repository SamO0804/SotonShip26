# Blindspot

Independent auditor for bank AML transaction monitoring. Hackathon build.

## Setup

```bash
pip install -r requirements.txt
python -m blindspot.make_sample          # fake data in data/sample/ (seconds)
```

Real data: download SAML-D from
https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
then:

```bash
python -m blindspot.normalise path/to/SAML-D.csv --customers 20000
```

This writes `data/transactions.parquet` and `data/labels.parquet`. It reads the
CSV in chunks, so it is safe on a laptop, but the first pass takes a few minutes.

## Pipeline and owners

| Step | Module | Reads | Writes |
|---|---|---|---|
| Normalise | `normalise.py` (done) | SAML-D CSV | transactions, labels |
| Plant patterns | `plant.py` (todo) | transactions, labels | transactions, labels (with `P*` rows) |
| Replica engine | `engine.py` (todo) | transactions **only** | alerts, cases |
| Blindspot checks | `checks/` (todo) | all of the above | findings |
| Dashboard | `app.py` (todo) | all tables | screen |

All table shapes live in `blindspot/schema.py`. Do not change a schema without
telling the team.

## Rules everyone follows

- **The engine never reads `labels`.** Ground truth goes only to Blindspot
  scoring and the dashboard.
- **"Caught" means customer-level:** a suspicious customer is caught if the
  engine opened a case for them. Use `schema.headline()` for the number;
  report it separately for `source="samld"` and `source="planted"`.
- **Seed is `schema.SEED` (42) everywhere.**
- **Sampling is by customer, not transaction**, so laundering patterns stay whole.

## Open decision: currency

SAML-D amounts are mostly UK pounds; the Bank of America thresholds are in
dollars ($3,000 money order ID limit, $10,000 cash). We treat `amount` as one
base currency and plant the Bank of America patterns at their dollar values.
Say so if a judge asks.

## Back pocket

IBM AMLworld can be added later with its own `normalise_ibm.py` mapping into the
same schema (`wire`, `ach`, `cheque`, `cash_*` already exist in the enum).
