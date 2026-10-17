"""Build a small fake bank in data/sample/ for testing without the SAML-D download.

    python -m blindspot.make_sample
    python -m blindspot.run_all --data data/sample

Makes a SAML-D-shaped CSV with 2,000 customers and runs it through the real
normaliser into data/sample/base/. run_all then plants the patterns, runs the
bank's rulebook and every check, exactly as on the real data.

The fake "SAML-D" laundering here is random noise, so only the planted results
mean anything. Never present sample results as real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schema
from .normalise import convert

OUT = Path("data/sample")


def fake_samld(n_customers: int = 2_000, n: int = 100_000, seed: int = schema.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = ["Cash Deposit", "Cash Withdrawal", "Cheque", "ACH",
             "Credit card", "Debit card", "Cross-border"]
    senders = np.array([f"{100000 + i}" for i in range(n_customers)])
    sender = rng.choice(senders, n)
    ts = pd.Timestamp("2022-10-07") + pd.to_timedelta(rng.integers(0, 330 * 24, n), "h")
    laund = np.isin(sender, senders[:4]) & (rng.random(n) < 0.3)   # 4 noise "launderers"
    return pd.DataFrame({
        "Time": ts.strftime("%H:%M:%S"),
        "Date": ts.strftime("%Y-%m-%d"),
        "Sender_account": sender,
        "Receiver_account": rng.choice([f"{900000 + i}" for i in range(2_000)], n),
        "Amount": rng.lognormal(6.5, 1.1, n).round(2),
        "Sender_bank_location": "UK",
        "Receiver_bank_location": rng.choice(["UK"] * 8 + ["UAE", "Turkey"], n),
        "Payment_type": rng.choice(types, n),
        "Is_laundering": laund.astype(int),
        "Laundering_type": np.where(laund, "Structuring", "Normal_Small_Fan_Out"),
    })


def main() -> None:
    base = OUT / "base"
    base.mkdir(parents=True, exist_ok=True)
    raw = fake_samld()
    raw.to_csv(OUT / "samld_tiny.csv", index=False)
    tx, lab = convert(raw)
    tx.to_parquet(base / "transactions.parquet", index=False)
    lab.to_parquet(base / "labels.parquet", index=False)
    print(f"wrote {len(tx):,} transactions for {tx['customer_id'].nunique()} customers to {base}/")
    print("next: python -m blindspot.run_all --data data/sample")


if __name__ == "__main__":
    main()
