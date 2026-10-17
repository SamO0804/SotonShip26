"""Blindspot check 2: the below-the-line audit (machine learning).

    python -m blindspot.checks.below_line
    python -m blindspot.checks.below_line --data data/sample

Question it answers: how much suspicious-looking activity sits UNDER the
bank's alarm level, where nobody ever looks?

How:
  1. Build a behaviour profile per customer from transactions (cash habits,
     high-risk countries, how many people they pay, and so on).
  2. Fit an IsolationForest - an unsupervised anomaly detector. It needs no
     labels, which is the point: a real bank doesn't know who the criminals are.
  3. Rank the customers the bank's system NEVER opened a case on, most
     unusual first, with plain reasons for each.
  4. Compare engine score bands: are customers just under 20 points more
     unusual than those just over? (The Bank of America finding.)

Blindspot itself never reads labels. The PROOF section at the bottom does,
only to show judges how well the ranking worked.

Writes data/below_line.parquet and data/findings_below_line.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .. import schema

# Feature -> plain-English name used in reasons.
FEATURES = {
    "n_txn": "transactions",
    "total_in": "money in (£)",
    "total_out": "money out (£)",
    "max_amount": "largest transaction (£)",
    "cash_in_share": "share of money in that is cash",
    "near_limit_cash": "cash deposits just under £10,000",
    "money_order_in": "money order deposits (£)",
    "max_deposits_one_day": "most deposits in one day",
    "high_risk_share": "share of money via high-risk countries",
    "high_risk_txns": "high-risk-country transactions",
    "intl_in": "money in from abroad (£)",
    "unique_payees": "different people paid",
    "card_payees": "different credit cards paid off",
    "payees_per_payment": "new payee rate",
    "max_out_30d": "most payments in 30 days",
    "small_business": "small business",
}
# Heavy-tailed features get log-scaled before modelling.
LOG = ["n_txn", "total_in", "total_out", "max_amount", "near_limit_cash", "money_order_in",
       "max_deposits_one_day", "high_risk_txns", "intl_in", "unique_payees", "card_payees",
       "max_out_30d"]

BANDS = [(-1, 0, "0"), (1, 9, "1-9"), (10, 15, "10-15"), (16, 19, "16-19"),
         (20, 24, "20-24"), (25, 10**9, "25+")]


def customer_features(tx: pd.DataFrame) -> pd.DataFrame:
    t = tx[["customer_id", "timestamp", "amount", "receiver_id"]].copy()
    pt = tx["payment_type"].astype(str)
    d = tx["direction"].astype(str)
    hr = (tx["sender_country"].isin(schema.HIGH_RISK_COUNTRIES)
          | tx["receiver_country"].isin(schema.HIGH_RISK_COUNTRIES))
    lim = schema.CASH_REVIEW_LIMIT
    amt = tx["amount"]

    t["in_amt"] = amt.where(d == "in", 0.0)
    t["out_amt"] = amt.where(d == "out", 0.0)
    t["cash_in"] = amt.where(pt == "cash_deposit", 0.0)
    t["near"] = ((pt == "cash_deposit") & (amt >= 0.9 * lim) & (amt < lim)).astype(int)
    t["mo_in"] = amt.where(pt == "money_order", 0.0)
    t["hr_amt"] = amt.where(hr, 0.0)
    t["hr"] = hr.astype(int)
    t["intl_in"] = amt.where((d == "in") & pt.isin(["wire", "cross_border"]), 0.0)

    f = t.groupby("customer_id").agg(
        n_txn=("amount", "size"), total_in=("in_amt", "sum"), total_out=("out_amt", "sum"),
        max_amount=("amount", "max"), cash_in=("cash_in", "sum"), near_limit_cash=("near", "sum"),
        money_order_in=("mo_in", "sum"), hr_amt=("hr_amt", "sum"), high_risk_txns=("hr", "sum"),
        intl_in=("intl_in", "sum"),
    )
    total = f["total_in"] + f["total_out"]
    f["cash_in_share"] = (f["cash_in"] / f["total_in"].where(f["total_in"] > 0)).fillna(0)
    f["high_risk_share"] = (f["hr_amt"] / total.where(total > 0)).fillna(0)

    out = t[d == "out"]
    f["unique_payees"] = out.groupby("customer_id")["receiver_id"].nunique()
    f["payees_per_payment"] = f["unique_payees"] / out.groupby("customer_id").size()
    f["card_payees"] = t[(d == "out") & (pt == "card_repayment")].groupby("customer_id")["receiver_id"].nunique()

    ins = t[d == "in"]
    f["max_deposits_one_day"] = (ins.groupby(["customer_id", ins["timestamp"].dt.floor("D")])
                                 .size().groupby(level=0).max())
    block = (out["timestamp"] - tx["timestamp"].min()).dt.days // 30
    f["max_out_30d"] = out.groupby(["customer_id", block]).size().groupby(level=0).max()

    seg = tx.groupby("customer_id")["customer_segment"].first().astype(str)
    f["small_business"] = (seg == "small_business").astype(int)
    return f[list(FEATURES)].fillna(0)


def fit_scores(f: pd.DataFrame) -> pd.Series:
    x = f.copy()
    x[LOG] = np.log1p(x[LOG])
    model = IsolationForest(n_estimators=300, random_state=schema.SEED, n_jobs=-1)
    model.fit(x.values)
    # score_samples: lower = more anomalous. Flip so higher = more unusual, 0-100 scale.
    raw = -model.score_samples(x.values)
    return pd.Series(100 * (raw - raw.min()) / (raw.max() - raw.min() + 1e-12), index=f.index)


def reasons(f: pd.DataFrame, ids, top: int = 3) -> dict[str, str]:
    """For each customer, the features where they are most extreme, as
    'higher than X% of customers'. Only features above the 95th percentile."""
    feats = [c for c in f.columns if c != "small_business"]
    out = {}
    for cid in ids:
        row = f.loc[cid, feats]
        # share of customers strictly below this customer's value
        pct = pd.Series({c: (f[c] < row[c]).mean() for c in feats})
        parts = []
        for feat in pct.sort_values(ascending=False).index[:top]:
            if pct[feat] < 0.95:
                break
            v = row[feat]
            val = f"{v:.0%}" if "share" in feat or feat == "payees_per_payment" else f"{v:,.0f}"
            top_pct = max(100 * (1 - pct[feat]), 0.1)   # never claim "top 0%"
            band = f"{top_pct:.1f}" if top_pct < 1 else f"{top_pct:.0f}"
            parts.append(f"{FEATURES[feat]} {val} (top {band}% of customers)")
        out[cid] = "; ".join(parts) or "unusual combination of ordinary behaviours"
    return out


def run(tx: pd.DataFrame, cases: pd.DataFrame, top_n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = customer_features(tx)
    score = fit_scores(f)

    res = cases.set_index("customer_id").join(score.rename("anomaly_score"), how="inner")
    res = res.rename(columns={"score": "engine_score"}).reset_index()
    below = res[~res["case_opened"]].sort_values("anomaly_score", ascending=False)
    below["rank"] = np.arange(1, len(below) + 1)
    top = below.head(top_n)
    why = reasons(f, top["customer_id"])
    below["reasons"] = below["customer_id"].map(why).fillna("")

    # Band comparison: share of each engine-score band in the top 2% most unusual.
    cut = res["anomaly_score"].quantile(0.98)
    band_rows = []
    for lo, hi, name in BANDS:
        b = res[(res["engine_score"] >= lo) & (res["engine_score"] <= hi)]
        band_rows.append({"band": name, "customers": len(b),
                          "pct_highly_unusual": round(100 * (b["anomaly_score"] >= cut).mean(), 1)
                          if len(b) else 0.0})
    bands = pd.DataFrame(band_rows)

    findings = []
    findings.append({
        "finding_id": "BL1", "check": "below_line", "severity": "high",
        "title": f"{top_n} customers with no case look highly unusual",
        "metric": f"top {top_n} of {len(below):,} unflagged customers",
        "evidence_ids": list(top["customer_id"]),
        "suggested_fix": "Send these customers for manual review, starting from rank 1.",
        "explanation": "These customers never reached the alarm level, but their behaviour "
                       "is among the most unusual in the bank. Top reasons: "
                       + " | ".join(f"{c}: {why[c]}" for c in list(top["customer_id"])[:3]),
    })
    b = bands.set_index("band")["pct_highly_unusual"]
    if b.get("16-19", 0) > b.get("20-24", 0):
        findings.append({
            "finding_id": "BL2", "check": "below_line", "severity": "high",
            "title": "Customers just under the threshold look riskier than those just over it",
            "metric": f"scores 16-19: {b['16-19']}% highly unusual vs 20-24: {b['20-24']}%",
            "evidence_ids": list(below.loc[below["engine_score"].between(16, 19), "customer_id"]
                                 .head(100)),
            "suggested_fix": "Lower the case threshold or sample scores 16-19 for review every month.",
            "explanation": "The threshold of 20 points cuts off customers who look more suspicious "
                           "than many it lets through.",
        })
    sup = below[below["suppressed"]].sort_values("anomaly_score", ascending=False)
    hot = sup[sup["anomaly_score"] >= cut]
    if len(hot):
        findings.append({
            "finding_id": "BL3", "check": "below_line", "severity": "high",
            "title": "The valued-client list hides highly unusual customers",
            "metric": f"{len(hot)} of {len(sup)} suppressed customers in the top 2% most unusual",
            "evidence_ids": list(hot["customer_id"]),
            "suggested_fix": "Remove automatic suppression; review valued clients like anyone else.",
            "explanation": "These customers scored enough for a case but were exempted as valued "
                           "clients, and their behaviour is among the most unusual in the bank.",
        })
    fdf = schema.conform(pd.DataFrame(findings), schema.FINDINGS, "findings")
    cols = ["rank", "customer_id", "anomaly_score", "engine_score", "suppressed", "reasons"]
    return below[cols], fdf, bands


# ---------------------------------------------------------------------------
# PROOF ONLY - uses the ground truth to show judges the ranking works.
# ---------------------------------------------------------------------------

def proof(below: pd.DataFrame, tx: pd.DataFrame, lab: pd.DataFrame) -> None:
    sus = lab[lab["is_suspicious"]].merge(tx[["txn_id", "customer_id"]], on="txn_id")
    cust = sus.groupby("customer_id").agg(source=("source", "first"), typology=("typology", "first"))
    below = below.merge(cust, left_on="customer_id", right_index=True, how="left")
    below["bad"] = below["source"].notna()
    base = below["bad"].mean()
    print(f"\nPROOF: {below['bad'].sum():,} of {len(below):,} unflagged customers are really "
          f"laundering (base rate {100 * base:.2f}%)")
    for k in (25, 50, 100, 250):
        hit = below.head(k)["bad"].mean()
        print(f"  top {k:>3}: {100 * hit:5.1f}% laundering  ({hit / base if base else 0:4.1f}x random)")
    missed = below[below["source"] == "planted"]
    print("\nmissed planted customers found in the top 100 / 250:")
    for ty, g in missed.groupby("typology"):
        print(f"  {ty:25} {(g['rank'] <= 100).sum():>3} / {(g['rank'] <= 250).sum():>3}  of {len(g)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--no-proof", action="store_true", help="skip the ground-truth proof")
    args = ap.parse_args()

    tx = pd.read_parquet(args.data / "transactions.parquet")
    cases = pd.read_parquet(args.data / "cases.parquet")
    below, findings, bands = run(tx, cases, args.top)
    below.to_parquet(args.data / "below_line.parquet", index=False)
    findings.to_parquet(args.data / "findings_below_line.parquet", index=False)

    print("engine score band vs share in the top 2% most unusual customers:")
    print(bands.to_string(index=False))
    print(f"\ntop 5 unflagged customers:")
    for _, r in below.head(5).iterrows():
        print(f"  #{r['rank']} {r['customer_id']} (engine {r['engine_score']} pts): {r['reasons']}")
    print(f"\n{len(findings)} findings: " + "; ".join(findings["title"]))

    if not args.no_proof:
        proof(below, tx, pd.read_parquet(args.data / "labels.parquet"))


if __name__ == "__main__":
    main()