"""How much did the bank's rules catch? Uses the ground truth, so it lives
outside the engine.

    python -m blindspot.evaluate
    python -m blindspot.evaluate --data data/sample

Prints the headline number for planted and SAML-D laundering, then a
per-typology table. Use it when trying a different rulebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import schema


def by_typology(tx: pd.DataFrame, lab: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    sus = lab[lab["is_suspicious"]].merge(tx[["txn_id", "customer_id"]], on="txn_id")
    pairs = sus[["source", "typology", "customer_id"]].drop_duplicates()
    pairs = pairs.merge(cases, on="customer_id", how="left")
    out = pairs.groupby(["source", "typology"], observed=True).agg(
        customers=("customer_id", "nunique"),
        caught=("case_opened", "sum"),
        suppressed=("suppressed", "sum"),
        avg_score=("score", "mean"),
    )
    out["caught_pct"] = (100 * out["caught"] / out["customers"]).round(1)
    out["avg_score"] = out["avg_score"].round(1)
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()

    tx = pd.read_parquet(args.data / "transactions.parquet")
    lab = pd.read_parquet(args.data / "labels.parquet")
    cases = pd.read_parquet(args.data / "cases.parquet")

    for src in ("planted", "samld"):
        h = schema.headline(lab, tx, cases, source=src)
        print(f"{src:8} caught {h['caught']:>5} of {h['suspicious_customers']:>5} "
              f"suspicious customers = {h['caught_pct']}%")

    clean = set(tx["customer_id"]) - set(
        tx.loc[tx["txn_id"].isin(lab.loc[lab["is_suspicious"], "txn_id"]), "customer_id"])
    fp = cases[cases["customer_id"].isin(clean)]["case_opened"].sum()
    opened = cases["case_opened"].sum()
    print(f"false positives: {fp} of {opened} cases opened "
          f"({100 * fp / opened if opened else 0:.0f}%)\n")
    print(by_typology(tx, lab, cases).to_string(index=False))


if __name__ == "__main__":
    main()
