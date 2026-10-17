"""Blindspot check 1: the coverage map.

    python -m blindspot.checks.coverage
    python -m blindspot.checks.coverage --data data/sample

Question it answers: is any rule watching this kind of money at all?

For every payment type and direction, it counts the money and transactions,
then checks the bank's rule documentation (rules.parquet) for rules that can
fire on it. Three levels:
  - unmonitored     no rule can ever fire          (Bank of America: money orders)
  - count-only      only a velocity rule, which never looks at amount,
                    country or counterparty
  - monitored       at least one amount, country or pattern rule

Reads transactions, rules and alerts. Never reads labels.
Writes data/coverage.parquet and data/findings_coverage.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import schema

NICE = {
    "cash_deposit": "Cash deposits", "cash_withdrawal": "Cash withdrawals",
    "cheque": "Cheques", "money_order": "Money order deposits", "ach": "ACH payments",
    "credit_card": "Credit card payments", "card_repayment": "Credit card repayments", "debit_card": "Debit card payments",
    "cross_border": "Cross-border payments", "wire": "Wires",
}


def _in_scope(rule: pd.Series, ptype: str, direction: str) -> bool:
    return ptype in list(rule["payment_types"]) and rule["direction"] in ("any", direction)


def coverage_table(tx: pd.DataFrame, rules: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    t = tx[["txn_id", "payment_type", "direction", "amount", "customer_id"]].copy()
    t["payment_type"] = t["payment_type"].astype(str)
    t["direction"] = t["direction"].astype(str)
    t["alerted"] = t["txn_id"].isin(alerts["txn_id"])

    g = t.groupby(["payment_type", "direction"]).agg(
        transactions=("txn_id", "size"), money_gbp=("amount", "sum"),
        customers=("customer_id", "nunique"), alerted_txns=("alerted", "sum"),
    ).reset_index()
    g = g[g["transactions"] > 0]

    rule_ids, kinds = [], []
    for _, row in g.iterrows():
        hit = rules[rules.apply(_in_scope, axis=1, args=(row["payment_type"], row["direction"]))]
        rule_ids.append(list(hit["rule_id"]))
        kinds.append(set(hit["kind"]))
    g["rules"] = rule_ids
    g["status"] = ["unmonitored" if not k else "count-only" if k == {"count"} else "monitored"
                   for k in kinds]
    g["share_of_money"] = (100 * g["money_gbp"] / g["money_gbp"].sum()).round(2)
    g["label"] = g["payment_type"].map(NICE) + " (" + g["direction"] + ")"
    order = {"unmonitored": 0, "count-only": 1, "monitored": 2}
    return g.sort_values(["status", "money_gbp"], key=lambda s: s.map(order) if s.name == "status"
                         else -s).reset_index(drop=True)


def findings(cov: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t = tx.assign(payment_type=tx["payment_type"].astype(str), direction=tx["direction"].astype(str))
    for i, r in cov[cov["status"] != "monitored"].iterrows():
        unmon = r["status"] == "unmonitored"
        sub = t[(t["payment_type"] == r["payment_type"]) & (t["direction"] == r["direction"])]
        top = sub.groupby("customer_id")["amount"].sum().sort_values(ascending=False)
        rows.append({
            "finding_id": f"CV{len(rows) + 1}",
            "check": "coverage",
            "severity": "high" if unmon else "medium",
            "title": (f"{r['label']} are not monitored at all" if unmon else
                      f"{r['label']} are only checked by a transaction count"),
            "metric": (f"0% monitored: £{r['money_gbp']:,.0f} across {r['transactions']:,} "
                       f"transactions" if unmon else
                       f"only {', '.join(r['rules'])}: £{r['money_gbp']:,.0f}"),
            "evidence_ids": list(top.index[:100]),
            "suggested_fix": (f"Add scenarios for {r['label'].lower()}, including amount, "
                              "frequency and split-deposit patterns." if unmon else
                              f"Add amount and counterparty rules for {r['label'].lower()}."),
            "explanation": (f"No rule can fire on {r['label'].lower()}, so £{r['money_gbp']:,.0f} "
                            f"from {r['customers']:,} customers was never checked."
                            if unmon else
                            "The only rule that can fire counts payments; it never looks at "
                            "amounts, countries or who is being paid."),
        })
    cols = list(schema.FINDINGS)
    return schema.conform(pd.DataFrame(rows, columns=cols), schema.FINDINGS, "findings")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()
    d = args.data
    tx = pd.read_parquet(d / "transactions.parquet")
    rules = pd.read_parquet(d / "rules.parquet")
    alerts = pd.read_parquet(d / "alerts.parquet")

    print(f"→ comparing {len(tx):,} transactions with the payment types each rule can see")
    cov = coverage_table(tx, rules, alerts)
    fnd = findings(cov, tx)
    cov.to_parquet(d / "coverage.parquet", index=False)
    fnd.to_parquet(d / "findings_coverage.parquet", index=False)

    show = cov[["label", "status", "transactions", "money_gbp", "share_of_money", "alerted_txns"]].copy()
    show["money_gbp"] = show["money_gbp"].map("£{:,.0f}".format)
    print(show.to_string(index=False))
    mon = cov.loc[cov["status"] == "monitored", "money_gbp"].sum() / cov["money_gbp"].sum()
    print(f"\n{100 * mon:.1f}% of money is watched by a rule that checks amounts, countries or patterns")
    print(f"{len(fnd)} findings: " + "; ".join(fnd["title"]))


if __name__ == "__main__":
    main()
