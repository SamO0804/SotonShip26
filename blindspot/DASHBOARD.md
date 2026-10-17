# Blindspot dashboard

The screen a bank's compliance team uses: load the bank's rules, see what they miss.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

A browser tab opens. It reads only the pipeline's real output in `data/`. If the results
aren't there, it shows an error telling you which command to run, rather than showing
anything made up.

## Plug in the real engine

`app.py` lives in the outer `blindspot/` folder, next to `data/`. Produce the files with:

```bash
python -m blindspot.run_all          # whole pipeline, ends with report + evidence
# or, if only the report or evidence changed:
python -m blindspot.report && python -m blindspot.evidence
```

Once `data/summary.json` exists, the app uses those files. Files it reads from `data/`:

| File | Used on |
|---|---|
| summary.json | Overview, Method & limits |
| findings.parquet | Findings |
| review_queue.parquet | Review queue |
| evidence.parquet | Review queue: per-customer evidence payments and timeline |
| rules.parquet, rulebook_used.yaml | Audit setup: the rulebook behind the results |
| coverage.parquet | Coverage |
| calibration.parquet | Rule weights, default rulebook |
| advisory.parquet, advisory_patterns.json | Regulator alerts (advisory text read from `advisories/`) |
| below_line.parquet | Below the line |

It never loads transactions.parquet.

## How it works now

The app **opens empty** on Audit setup, with "No audit yet" on every results page,
until a rulebook has been audited in this browser session. Refreshing the browser
starts empty again.

1. **Upload** the bank's rulebook (YAML), or choose an example from `rulebooks/`.
   Mistakes in the file are reported in plain words.
2. **Check the rules:** switch rules on or off and change points, amounts, windows,
   counts, the case threshold and how many valued clients are exempt. Which payments
   a rule can see is changed in the YAML file itself.
3. **Run audit:** saves the rulebook to `rulebooks/active.yaml` and runs the pipeline,
   showing each step. It takes a minute or two. Then every page shows the results and
   the top line says which rulebook was audited.

**No time to run live?** Audit setup has a "Show the last saved audit" button that loads
the results already in `data/`.

Running replaces the results in `data/`. Run audit needs the test bank
(`data/transactions.parquet`, or `data/base/` on a fresh setup). Set
`BLINDSPOT_READONLY=1` to switch running off.

## Example rulebooks (`rulebooks/`)

| File | Purpose |
|---|---|
| `northbridge_bank.yaml` | Bank of America-style flaws. The main demo. |
| `northbridge_bank_remediated.yaml` | Northbridge after acting on Blindspot's findings. |
| `harbourside_building_society.yaml` | Untested. Predicted flaws in `PREDICTIONS.md`. |
| `meridian_pay.yaml` | Untested. Predicted flaws in `PREDICTIONS.md`. |

## Before the demo

- Do a full run with `northbridge_bank.yaml` the day before, so "Show the last saved audit" has a fallback.
- `SAMLD_REASON` near the top of `app.py` explains the low SAML-D result; keep it honest.
- Turn wifi off and click through every page.
