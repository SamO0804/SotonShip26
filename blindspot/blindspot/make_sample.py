"""Build a tiny fake dataset in data/sample/ so everyone can build before
the real pipeline exists.

    python -m blindspot.make_sample

It writes a SAML-D-shaped CSV, pushes it through the real normaliser (so the
normaliser is tested), then adds dummy engine output and findings. The
numbers are made up; the shapes are real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schema
from .normalise import convert

OUT = Path("data/sample")
rng = np.random.default_rng(schema.SEED)


def fake_samld(n: int = 60) -> pd.DataFrame:
    types = ["Cash Deposit", "Cash Withdrawal", "Cheque", "ACH",
             "Credit card", "Debit card", "Cross-border"]
    senders = [f"{1000 + i}" for i in range(12)]
    ts = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24, n), "h")
    laund = rng.random(n) < 0.15
    return pd.DataFrame({
        "Time": ts.strftime("%H:%M:%S"),
        "Date": ts.strftime("%Y-%m-%d"),
        "Sender_account": rng.choice(senders, n),
        "Receiver_account": rng.choice([f"{9000 + i}" for i in range(20)], n),
        "Amount": rng.lognormal(7, 1.2, n).round(2),
        "Sender_bank_location": "UK",
        "Receiver_bank_location": rng.choice(["UK", "UK", "UK", "UAE", "Turkey"], n),
        "Payment_type": rng.choice(types, n),
        "Is_laundering": laund.astype(int),
        "Laundering_type": np.where(laund, "Structuring", "Normal_Small_Fan_Out"),
    })


def fake_planted(start: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A few rows of the Bank of America money-order pattern."""
    rows = []
    for i in range(6):
        rows.append({
            "txn_id": f"P{start + i}",
            "timestamp": pd.Timestamp("2023-06-01") + pd.Timedelta(hours=i),
            "sender_id": "AUCTION_01",
            "receiver_id": "AUCTION_01_DEP",
            "amount": float(rng.integers(960, 999)),
            "payment_type": "money_order",
            "sender_country": "USA",
            "receiver_country": "USA",
            "customer_segment": "small_business",
        })
    tx = schema.conform(pd.DataFrame(rows), schema.TRANSACTIONS, "planted")
    lab = schema.conform(pd.DataFrame({
        "txn_id": tx["txn_id"],
        "is_suspicious": True,
        "source": "planted",
        "typology": "structured_money_orders",
    }), schema.LABELS, "planted labels")
    return tx, lab


def fake_engine(tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Money orders are unmonitored in the replica, so they never alert.
    watched = tx[tx["payment_type"] != "money_order"].sample(frac=0.3, random_state=schema.SEED)
    alerts = schema.conform(pd.DataFrame({
        "customer_id": watched["sender_id"],
        "rule_id": rng.choice(["R01_large_cash", "R02_high_risk_country", "R03_velocity"],
                              len(watched)),
        "points": rng.choice([1, 5, 10], len(watched)),
        "timestamp": watched["timestamp"],
        "txn_id": watched["txn_id"],
    }), schema.ALERTS, "alerts")
    score = alerts.groupby("customer_id")["points"].sum()
    cases = pd.DataFrame({"customer_id": tx["sender_id"].unique()})
    cases["score"] = cases["customer_id"].map(score).fillna(0).astype(int)
    cases["case_opened"] = cases["score"] >= schema.CASE_THRESHOLD
    cases["suppressed"] = False
    return alerts, schema.conform(cases, schema.CASES, "cases")


def fake_findings(planted_ids: list[str]) -> pd.DataFrame:
    rows = [
        ("F1", "coverage", "Money order deposits are not monitored", "high", "0% monitored",
         planted_ids, "Add a scenario for money order deposits, including sets just under $3,000.",
         "No rule can fire on money order deposits, so this money is never checked."),
        ("F2", "below_line", "Customers just under the threshold look riskier", "high",
         "Scores 16-19: 2.1x anomaly rate", ["1003", "1007"],
         "Lower the case threshold or test a sample of scores 16-19 every month.",
         "Customers scoring 16 to 19 are more unusual than those scoring 20 to 24."),
        ("F3", "calibration", "Strong red flag worth only 1 point", "medium",
         "R01: 1 pt, 56.8% precision", ["R01_large_cash"],
         "Raise R01's weight to match how often it is right.",
         "This rule is right more than half the time but barely moves the score."),
        ("F4", "advisory", "Credit-card funnel pattern not covered", "medium",
         "0 of 4 red flags covered", [],
         "Add a rule for one business paying many unrelated people's cards.",
         "The advisory's red flags do not match any current rule."),
    ]
    df = pd.DataFrame(rows, columns=list(schema.FINDINGS))
    return schema.conform(df, schema.FINDINGS, "findings")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = fake_samld()
    raw.to_csv(OUT / "samld_tiny.csv", index=False)
    tx, lab = convert(raw)
    ptx, plab = fake_planted(0)
    tx = pd.concat([tx, ptx], ignore_index=True)
    lab = pd.concat([lab, plab], ignore_index=True)
    schema.validate_transactions(tx)
    alerts, cases = fake_engine(tx)
    findings = fake_findings(list(ptx["txn_id"]))

    tx.to_parquet(OUT / "transactions.parquet", index=False)
    lab.to_parquet(OUT / "labels.parquet", index=False)
    alerts.to_parquet(OUT / "alerts.parquet", index=False)
    cases.to_parquet(OUT / "cases.parquet", index=False)
    findings.to_parquet(OUT / "findings.parquet", index=False)

    for src in ("samld", "planted"):
        print(src, schema.headline(lab, tx, cases, source=src))
    print(f"wrote {len(tx)} transactions, {len(alerts)} alerts, "
          f"{len(cases)} customers, {len(findings)} findings to {OUT}/")


if __name__ == "__main__":
    main()
