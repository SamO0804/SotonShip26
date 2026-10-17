# Blindspot

Independent auditor for bank AML transaction monitoring. Hackathon build.
UK viewpoint throughout: all amounts are GBP.

## First-time setup (macOS)

```bash
cd ~/Downloads/blindspot          # wherever you unzipped it
python3 -m venv .venv             # private Python for this project
source .venv/bin/activate         # run this in every new terminal window
pip install -r requirements.txt
python -m blindspot.make_sample   # fake data in data/sample/ - takes seconds
```

If `make_sample` prints two "caught" lines and "wrote ... to data/sample/",
everything works.

## Real data

1. Download SAML-D from
   https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
   and unzip it into the project's `data/` folder.
2. Run (replace the file name with the CSV's real name):

```bash
python -m blindspot.normalise data/SAML-D.csv --customers 20000   # -> data/base/
python -m blindspot.plant --per-pattern 25                        # -> data/
```

`normalise` reads the CSV in chunks, so it is safe on a laptop; it takes a few
minutes. `plant` takes seconds and prints a table of what it planted.

## Pipeline and owners

| Step | Module | Reads | Writes |
|---|---|---|---|
| Normalise | `normalise.py` (done) | SAML-D CSV | data/base/ transactions, labels |
| Plant patterns | `plant.py` (done) | data/base/ | data/ transactions, labels, planted_customers |
| Replica engine | `engine.py` (todo) | transactions **only** | alerts, cases |
| Blindspot checks | `checks/` (todo) | all of the above | findings |
| Dashboard | `app.py` (todo) | all tables | screen |

All table shapes live in `blindspot/schema.py`. Do not change a schema without
telling the team.

## Planted patterns (GBP)

| Typology | What it looks like | Customer |
|---|---|---|
| `structured_money_orders` | Sets of 3 money orders, one set per day, totalling £2,400-£2,499 (just under the £2,500 ID limit) | small business |
| `high_risk_wires` | 10-25 inbound wires of £15k-£60k from Belarus/Georgia within 60 days | small business |
| `credit_card_funnel` | One shop paying off 15-40 unrelated people's credit cards | small business |
| `structured_cash` | 6-15 cash deposits of £9,000-£9,900 (just under the £10,000 review limit) | any |
| `slow_burn` | Small cash deposits and high-risk-country payments every 25-45 days, too thin to reach 20 points | any |

Patterns are planted on top of existing clean customers, so they have a normal
history too. 20% of all customers are tagged `small_business`, so the segment
alone doesn't give the planted ones away.

## Rules everyone follows

- **The engine never reads `labels` or `planted_customers`.** Ground truth goes
  only to Blindspot scoring and the dashboard.
- **Monitor by `customer_id`, not `sender_id`.** Inbound wires and money-order
  deposits belong to the receiving customer.
- **"Caught" means customer-level:** a suspicious customer is caught if the
  engine opened a case for them. Use `schema.headline()` and report it
  separately for `source="samld"` and `source="planted"`.
- **Seed is `schema.SEED` (42) everywhere.**

## Assumptions to state if asked

- SAML-D is a UK dataset; amounts are treated as GBP.
- The £2,500 money order ID limit and £10,000 cash review limit are the
  fictional bank's internal policy limits, chosen to mirror the Bank of America
  case. They are not statutory UK thresholds.

## Back pocket

IBM AMLworld can be added later with its own `normalise_ibm.py` mapping into the
same schema (`wire`, `ach`, `cheque`, `cash_*` already exist in the enum).
