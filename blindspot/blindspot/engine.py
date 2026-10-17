"""Replica of a flawed bank monitoring system, modelled on Bank of America's
"Event Processor". This is the system Blindspot audits.

    python -m blindspot.engine                 # reads data/, writes data/
    python -m blindspot.engine --data data/sample

RULE: this module reads transactions ONLY. It must never read labels or
planted_customers - a real bank's system doesn't know the answers.

How it works (deliberately flawed, like the real one):
  - Each rule gives points to a customer when it fires.
  - A case opens when a customer's points inside any 13-month window reach 20.
  - Older alerts expire after 13 months.

The flaws Blindspot should find:
  1. COVERAGE GAP   - no rule looks at money order deposits at all.
  2. BAD WEIGHTS    - "cash just under the review limit" is a strong red flag
                      but is worth only 1 point.
  3. THRESHOLD      - slow, thin activity never reaches 20 points.
  4. SUPPRESSION    - the bank's biggest small-business clients are on a
                      "valued client" list and never get a case.
  5. NO ADVISORY    - nothing watches for one business paying many people's
                      credit cards (the FinCEN-style advisory pattern).

Tune with the CONFIG dict below; keep the flaws, change the numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import schema

CONFIG = {
    # rule_id: points
    "R01_large_cash": 10,           # cash deposit at or over the review limit
    "R02_cash_near_limit": 1,       # cash deposit 90-99.9% of the limit  (flaw 2)
    "R03_high_risk_country": 1,     # wire/cross-border to or from a high-risk country
    "R04_large_wire": 5,            # wire/cross-border of LARGE_WIRE or more
    "R05_large_payment": 2,         # cheque/ACH/cash withdrawal of LARGE_PAYMENT or more
    "R06_velocity": 3,              # VELOCITY_COUNT+ outbound payments in a 30-day block

    "LARGE_WIRE": 50_000,
    "LARGE_PAYMENT": 25_000,
    "VELOCITY_COUNT": 15,
    "SUPPRESS_TOP_PCT": 0.5,        # top % of small businesses by inbound £ are "valued clients"
    "UNMONITORED_TYPES": ["money_order"],   # flaw 1
}


def _alerts(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": df["customer_id"].values,
        "rule_id": rule,
        "points": CONFIG[rule],
        "timestamp": df["timestamp"].values,
        "txn_id": df["txn_id"].values,
    })


def run_rules(tx: pd.DataFrame) -> pd.DataFrame:
    t = tx[~tx["payment_type"].astype(str).isin(CONFIG["UNMONITORED_TYPES"])]
    pt = t["payment_type"].astype(str)
    lim = schema.CASH_REVIEW_LIMIT
    hr = schema.HIGH_RISK_COUNTRIES

    cash = t[pt == "cash_deposit"]
    intl = t[pt.isin(["wire", "cross_border"])]
    other = t[pt.isin(["cheque", "ach", "cash_withdrawal"])]

    parts = [
        _alerts(cash[cash["amount"] >= lim], "R01_large_cash"),
        _alerts(cash[(cash["amount"] >= 0.9 * lim) & (cash["amount"] < lim)], "R02_cash_near_limit"),
        _alerts(intl[intl["sender_country"].isin(hr) | intl["receiver_country"].isin(hr)],
                "R03_high_risk_country"),
        _alerts(intl[intl["amount"] >= CONFIG["LARGE_WIRE"]], "R04_large_wire"),
        _alerts(other[other["amount"] >= CONFIG["LARGE_PAYMENT"]], "R05_large_payment"),
    ]

    # Velocity: count outbound payments per customer in fixed 30-day blocks.
    out = t[t["direction"].astype(str) == "out"].copy()
    if len(out):
        out["block"] = (out["timestamp"] - t["timestamp"].min()).dt.days // 30
        g = out.sort_values("timestamp").groupby(["customer_id", "block"], observed=True)
        busy = g.tail(1).merge(g.size().rename("n").reset_index(), on=["customer_id", "block"])
        parts.append(_alerts(busy[busy["n"] >= CONFIG["VELOCITY_COUNT"]], "R06_velocity"))

    alerts = pd.concat(parts, ignore_index=True)
    return schema.conform(alerts, schema.ALERTS, "alerts")


def suppression_list(tx: pd.DataFrame) -> set[str]:
    """Biggest small-business clients by inbound money: kept for revenue (flaw 4)."""
    sb = tx[(tx["customer_segment"].astype(str) == "small_business")
            & (tx["direction"].astype(str) == "in")]
    vol = sb.groupby("customer_id")["amount"].sum()
    if vol.empty:
        return set()
    cut = vol.quantile(1 - CONFIG["SUPPRESS_TOP_PCT"] / 100)
    return set(vol[vol >= cut].index)


def score_cases(tx: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    """Score = most points inside any 13-month window (older alerts expire)."""
    if len(alerts):
        a = alerts.sort_values("timestamp").set_index("timestamp")
        rolling = (a.groupby("customer_id")["points"]
                   .rolling(f"{schema.WINDOW_DAYS}D").sum())
        score = rolling.groupby(level=0).max()
    else:
        score = pd.Series(dtype="float64")

    suppressed = suppression_list(tx)
    cases = pd.DataFrame({"customer_id": pd.unique(tx["customer_id"])})
    cases["score"] = cases["customer_id"].map(score).fillna(0).astype(int)
    cases["suppressed"] = cases["customer_id"].isin(suppressed)
    cases["case_opened"] = (cases["score"] >= schema.CASE_THRESHOLD) & ~cases["suppressed"]
    return schema.conform(cases, schema.CASES, "cases")


def run(tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    alerts = run_rules(tx)
    return alerts, score_cases(tx, alerts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()

    tx = pd.read_parquet(args.data / "transactions.parquet")   # transactions ONLY
    alerts, cases = run(tx)
    alerts.to_parquet(args.data / "alerts.parquet", index=False)
    cases.to_parquet(args.data / "cases.parquet", index=False)

    print(alerts.groupby("rule_id").agg(alerts=("points", "size"),
                                        customers=("customer_id", "nunique")).to_string())
    print(f"\n{len(cases):,} customers scored, {cases['case_opened'].sum():,} cases opened, "
          f"{cases['suppressed'].sum():,} suppressed")
    print(f"wrote alerts.parquet and cases.parquet to {args.data}/")
    print("next: python -m blindspot.evaluate" + (f" --data {args.data}" if str(args.data) != "data" else ""))


if __name__ == "__main__":
    main()