"""Load SAML-D, sample it, and convert it to the Blindspot schema.

Usage:
    python -m blindspot.normalise path/to/SAML-D.csv --customers 20000

Writes data/base/transactions.parquet and data/base/labels.parquet
(clean SAML-D, before planting). Amounts are treated as GBP.

Sampling is by CUSTOMER, not by transaction: we keep every transaction for
the chosen sender accounts, so laundering patterns stay intact. All senders
with any laundering are kept, plus a random set of clean senders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

# SAML-D payment type -> canonical enum. Matched case-insensitively.
PAYMENT_MAP = {
    "cash deposit": "cash_deposit",
    "cash withdrawal": "cash_withdrawal",
    "cheque": "cheque",
    "ach": "ach",
    "credit card": "credit_card",
    "debit card": "debit_card",
    "cross-border": "cross_border",
}

SAMLD_COLS = [
    "Time", "Date", "Sender_account", "Receiver_account", "Amount",
    "Sender_bank_location", "Receiver_bank_location", "Payment_type",
    "Is_laundering", "Laundering_type",
]


def _slug(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))


def pick_customers(csv: Path, n_clean: int, seed: int, chunksize: int) -> set[str]:
    """First pass: find laundering senders, then sample clean ones."""
    dirty, everyone = set(), set()
    for chunk in pd.read_csv(csv, usecols=["Sender_account", "Is_laundering"],
                             dtype={"Sender_account": "string"}, chunksize=chunksize):
        everyone.update(chunk["Sender_account"].unique())
        dirty.update(chunk.loc[chunk["Is_laundering"] == 1, "Sender_account"].unique())
    clean = sorted(everyone - dirty)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(clean, size=min(n_clean, len(clean)), replace=False)
    print(f"senders: {len(everyone):,} total, {len(dirty):,} laundering kept, "
          f"{len(chosen):,} clean sampled")
    return dirty | set(chosen)


def convert(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a SAML-D frame (original columns, original row index) to our schema."""
    pt = raw["Payment_type"].astype(str).str.strip().str.lower()
    unknown = sorted(set(pt) - set(PAYMENT_MAP))
    if unknown:
        raise ValueError(f"unmapped SAML-D payment types: {unknown} - add to PAYMENT_MAP")

    txn_id = "S" + raw.index.astype(str)
    mapped = pt.map(PAYMENT_MAP)
    tx = pd.DataFrame({
        "txn_id": txn_id,
        "timestamp": pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str)),
        # SAML-D is sender-centric: the sender account is the bank's customer.
        "customer_id": raw["Sender_account"].astype(str),
        "direction": np.where(mapped == "cash_deposit", "in", "out"),
        "sender_id": raw["Sender_account"].astype(str),
        "receiver_id": raw["Receiver_account"].astype(str),
        "amount": raw["Amount"].astype(float),
        "payment_type": mapped,
        "sender_country": raw["Sender_bank_location"].astype(str),
        "receiver_country": raw["Receiver_bank_location"].astype(str),
        "customer_segment": "unknown",
    })
    is_sus = raw["Is_laundering"].astype(int) == 1
    lab = pd.DataFrame({
        "txn_id": txn_id,
        "is_suspicious": is_sus.values,
        "source": "samld",
        "typology": np.where(is_sus, _slug(raw["Laundering_type"]), "normal"),
    })
    tx = schema.conform(tx, schema.TRANSACTIONS, "transactions")
    lab = schema.conform(lab, schema.LABELS, "labels")
    schema.validate_transactions(tx)
    return tx, lab


def load_samld(csv: Path, n_clean: int = 20000, seed: int = schema.SEED,
               chunksize: int = 1_000_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = pick_customers(csv, n_clean, seed, chunksize)
    parts, offset = [], 0
    for chunk in pd.read_csv(csv, usecols=SAMLD_COLS,
                             dtype={"Sender_account": "string", "Receiver_account": "string"},
                             chunksize=chunksize):
        chunk.index = range(offset, offset + len(chunk))  # stable row ids across chunks
        offset += len(chunk)
        parts.append(chunk[chunk["Sender_account"].isin(keep)])
    raw = pd.concat(parts)
    tx, lab = convert(raw)
    print(f"kept {len(tx):,} transactions; {lab['is_suspicious'].sum():,} suspicious "
          f"({100 * lab['is_suspicious'].mean():.2f}%)")
    return tx, lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--customers", type=int, default=20000,
                    help="number of clean sender accounts to sample")
    ap.add_argument("--out", type=Path, default=Path("data/base"))
    args = ap.parse_args()
    tx, lab = load_samld(args.csv, args.customers)
    args.out.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(args.out / "transactions.parquet", index=False)
    lab.to_parquet(args.out / "labels.parquet", index=False)
    print(f"wrote {args.out}/transactions.parquet and labels.parquet")


if __name__ == "__main__":
    main()
