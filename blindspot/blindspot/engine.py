"""Replays a bank's transaction monitoring: runs the bank's rulebook over the
transactions and decides who gets a case. This is the system Blindspot audits.

    python -m blindspot.engine                                   # default rulebook
    python -m blindspot.engine --rules rulebooks/my_bank.yaml
    python -m blindspot.engine --data data/sample

The rules come from a rulebook file (see blindspot/rulebook.py for the format).
The default, rulebooks/northbridge_bank.yaml, is modelled on Bank of America's
flawed "Event Processor".

RULE: this module reads transactions ONLY. It must never read labels or
planted_customers - a real bank's system doesn't know the answers.

How a case opens:
  - Each rule gives the customer its points when it fires.
  - A customer's score is the most points inside any rolling window_days.
  - A case opens when the score reaches case_threshold, unless the customer is
    on the valued-client suppression list.

Writes alerts.parquet, cases.parquet, rules.parquet (what each rule watches,
for Blindspot's checks) and rulebook_used.yaml (a copy of the input).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from . import rulebook as rbk
from . import schema


def _alerts(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": df["customer_id"].values,
        "rule_id": rule["id"],
        "points": rule["points"],
        "timestamp": df["timestamp"].values,
        "txn_id": df["txn_id"].values,
    })


def _matching(tx: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Payments a rule looks at: its payment types, direction, amount and countries."""
    t = tx[tx["payment_type"].astype(str).isin(rule["payment_types"])]
    if rule["direction"] != "any":
        t = t[t["direction"].astype(str) == rule["direction"]]
    trig = rule["trigger"]
    if trig["amount_min"] is not None:
        t = t[t["amount"] >= trig["amount_min"]]
    if trig["amount_max"] is not None:
        t = t[t["amount"] <= trig["amount_max"]]
    if trig["countries"]:
        t = t[t["sender_country"].isin(trig["countries"]) | t["receiver_country"].isin(trig["countries"])]
    return t


def _repeated(t: pd.DataFrame, trig: dict) -> pd.DataFrame:
    """Each payment that is at least the min_count-th inside a rolling window."""
    if t.empty:
        return t
    # Sorted by customer then time, so grouped rolling counts come back in row order.
    t = t.sort_values(["customer_id", "timestamp"], kind="stable")
    n = (t.set_index("timestamp").groupby("customer_id", sort=True)["amount"]
         .rolling(f"{int(trig['window_days'])}D").count().values)
    return t[n >= trig["min_count"]]


def _blocks(t: pd.DataFrame, trig: dict, start: pd.Timestamp) -> pd.DataFrame:
    """Velocity and aggregate: fixed blocks of window_days, one alert per qualifying
    block, on the block's last payment."""
    if t.empty:
        return t
    t = t.assign(block=(t["timestamp"] - start).dt.days // int(trig["window_days"]))
    is_out = t["direction"].astype(str) == "out"
    t = t.assign(cp=t["receiver_id"].where(is_out, t["sender_id"]))
    g = t.sort_values("timestamp").groupby(["customer_id", "block"], observed=True)
    stats = g.agg(n=("amount", "size"), cps=("cp", "nunique"), total=("amount", "sum")).reset_index()
    ok = pd.Series(True, index=stats.index)
    if trig["min_count"] is not None:
        ok &= stats["n"] >= trig["min_count"]
    if trig["type"] == "aggregate":
        if trig["min_distinct_counterparties"] is not None:
            ok &= stats["cps"] >= trig["min_distinct_counterparties"]
        if trig["min_total"] is not None:
            ok &= stats["total"] >= trig["min_total"]
        if trig["max_total"] is not None:
            ok &= stats["total"] <= trig["max_total"]
    last = g.tail(1)
    return last.merge(stats[ok][["customer_id", "block"]], on=["customer_id", "block"])


def run_rules(tx: pd.DataFrame, rb: dict) -> pd.DataFrame:
    start = tx["timestamp"].min()
    parts = []
    on = [r for r in rb["rules"] if r["enabled"]]
    print(f"→ checking {len(tx):,} transactions against {len(on)} of the bank's rules")
    for rule in rb["rules"]:
        if not rule["enabled"]:
            print(f"→ {rule['id']} is switched off")
            continue
        t = _matching(tx, rule)
        trig = rule["trigger"]
        if trig["type"] == "single":
            hit = t
        elif trig["type"] == "repeated":
            hit = _repeated(t, trig)
        else:
            hit = _blocks(t, trig, start)
        parts.append(_alerts(hit, rule))
        print(f"→ {rule['id']} ({rule['description']}): fired {len(hit):,} times")
    alerts = pd.concat(parts, ignore_index=True) if parts else schema.empty(schema.ALERTS)
    return schema.conform(alerts, schema.ALERTS, "alerts")


def suppression_list(tx: pd.DataFrame, rb: dict) -> set[str]:
    """Biggest small-business clients by money in: kept for revenue."""
    n = rb["suppression"]["valued_clients_top_n"]
    if n <= 0:
        return set()
    sb = tx[(tx["customer_segment"].astype(str) == "small_business")
            & (tx["direction"].astype(str) == "in")]
    vol = sb.groupby("customer_id")["amount"].sum()
    return set(vol.nlargest(n).index) if len(vol) else set()


def score_cases(tx: pd.DataFrame, alerts: pd.DataFrame, rb: dict) -> pd.DataFrame:
    """Score = most points inside any rolling window (older alerts expire)."""
    if len(alerts):
        a = alerts.sort_values("timestamp").set_index("timestamp")
        rolling = a.groupby("customer_id")["points"].rolling(f"{rb['window_days']}D").sum()
        score = rolling.groupby(level=0).max()
    else:
        score = pd.Series(dtype="float64")

    suppressed = suppression_list(tx, rb)
    cases = pd.DataFrame({"customer_id": pd.unique(tx["customer_id"])})
    cases["score"] = cases["customer_id"].map(score).fillna(0).astype(int)
    cases["suppressed"] = cases["customer_id"].isin(suppressed)
    cases["case_opened"] = (cases["score"] >= rb["case_threshold"]) & ~cases["suppressed"]
    return schema.conform(cases, schema.CASES, "cases")


def run(tx: pd.DataFrame, rb: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    alerts = run_rules(tx, rb)
    print(f"→ adding up each customer's points over rolling {rb['window_days']}-day windows; "
          f"a case opens at {rb['case_threshold']}")
    return alerts, score_cases(tx, alerts, rb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--rules", type=Path, default=rbk.DEFAULT_RULEBOOK)
    args = ap.parse_args()

    try:
        rb = rbk.load(args.rules)
    except rbk.RulebookError as e:
        raise SystemExit(f"rulebook problem: {e}")

    tx = pd.read_parquet(args.data / "transactions.parquet")   # transactions ONLY
    alerts, cases = run(tx, rb)
    alerts.to_parquet(args.data / "alerts.parquet", index=False)
    cases.to_parquet(args.data / "cases.parquet", index=False)
    rbk.catalogue(rb).to_parquet(args.data / "rules.parquet", index=False)
    if args.rules.resolve() != (args.data / "rulebook_used.yaml").resolve():
        shutil.copyfile(args.rules, args.data / "rulebook_used.yaml")

    enabled = sum(r["enabled"] for r in rb["rules"])
    print(f"rulebook: {rb['bank']} {rb['version']} ({enabled} of {len(rb['rules'])} rules on) "
          f"from {args.rules}")
    if len(alerts):
        print(alerts.groupby("rule_id").agg(alerts=("points", "size"),
                                            customers=("customer_id", "nunique")).to_string())
    print(f"\n{len(cases):,} customers scored, {cases['case_opened'].sum():,} cases opened, "
          f"{cases['suppressed'].sum():,} suppressed")
    print(f"wrote alerts, cases, rules and rulebook_used.yaml to {args.data}/")


if __name__ == "__main__":
    main()
