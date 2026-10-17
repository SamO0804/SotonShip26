"""The bank's human investigators. Produces the investigation outcome log a
real bank keeps: which customers were reviewed and whether a suspicious
activity report (SAR) was filed.

    python -m blindspot.investigate
    python -m blindspot.investigate --data data/sample

Who gets investigated (as in a real bank):
  - every customer with an open case
  - below-the-line testing: for each rule, a random sample of customers that
    rule alerted on who did NOT reach a case (BTL_SAMPLE_PER_RULE)

Investigators are assumed accurate: they file a SAR exactly when the customer
really is laundering. That is the only place the ground truth is used, and
only for customers a real bank would actually have reviewed.

Writes data/investigations.parquet: customer_id, reason, sar_filed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

BTL_SAMPLE_PER_RULE = 40


def investigate(tx, lab, alerts, cases, per_rule: int = BTL_SAMPLE_PER_RULE,
                seed: int = schema.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    opened = set(cases.loc[cases["case_opened"], "customer_id"])

    btl = set()
    no_case = alerts[~alerts["customer_id"].isin(opened)]
    for _, g in no_case.groupby("rule_id"):
        pool = sorted(g["customer_id"].unique())
        btl.update(rng.choice(pool, size=min(per_rule, len(pool)), replace=False))

    bad = set(tx.loc[tx["txn_id"].isin(lab.loc[lab["is_suspicious"], "txn_id"]), "customer_id"])
    rows = [{"customer_id": c, "reason": "case", "sar_filed": c in bad} for c in sorted(opened)]
    rows += [{"customer_id": c, "reason": "btl_sample", "sar_filed": c in bad}
             for c in sorted(btl - opened)]
    return pd.DataFrame(rows, columns=["customer_id", "reason", "sar_filed"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()
    d = args.data
    print("→ simulating the bank's investigators: every case, plus a sample below the threshold")
    inv = investigate(pd.read_parquet(d / "transactions.parquet"), pd.read_parquet(d / "labels.parquet"),
                      pd.read_parquet(d / "alerts.parquet"), pd.read_parquet(d / "cases.parquet"))
    inv.to_parquet(d / "investigations.parquet", index=False)
    s = inv.groupby("reason")["sar_filed"].agg(["size", "sum"])
    for reason, r in s.iterrows():
        print(f"{reason:11} {r['size']:>6,} investigated, {r['sum']:>5,} SARs filed")
    print(f"wrote {d}/investigations.parquet")


if __name__ == "__main__":
    main()
