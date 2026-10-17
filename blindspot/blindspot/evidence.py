"""Evidence layer: link every customer in the review queue to the specific
transactions behind each reason, then score and rank those transactions.

    python -m blindspot.evidence
    python -m blindspot.evidence --data data/sample

Runs after report.py. For each queued customer, every check that flagged them
contributes the transactions that support its reason:

  check        evidence transactions                                    weight
  advisory     transactions inside the matching red-flag windows            3
  suppression  transactions that triggered alerts on a suppressed client    3
  calibration  transactions that triggered an under-weighted rule           2
  below_line   transactions behind the ML's top reasons for this customer   2
  coverage     transactions of a type no rule can see                       1

A transaction supporting several reasons collects all their weights, plus up
to 0.5 for its size relative to the customer's other evidence. Each
transaction also records whether the bank's rules fired on it, so a reviewer
sees at a glance what the bank's system never looked at.

Never reads labels (except PROOF).
Writes data/evidence.parquet and adds evidence columns to review_queue.parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import rulebook as rbk
from . import schema
from .checks import advisory as adv
from .checks import below_line as bl

MAX_PER_CUSTOMER = 25     # evidence rows kept per customer
MAX_PER_REASON = 10       # transactions one ML reason may contribute
MIN_PCT = 0.95            # ML features must be in the top 5% to count as a reason

COLS = ["txn_id", "customer_id", "timestamp", "direction", "payment_type", "amount",
        "sender_id", "receiver_id", "sender_country", "receiver_country"]


def _ev(df: pd.DataFrame, check: str, why: str, weight: float) -> pd.DataFrame:
    return pd.DataFrame({"txn_id": df["txn_id"].values, "customer_id": df["customer_id"].values,
                         "check": check, "why": why, "weight": weight})


# ---------------------------------------------------------------------------
# Which transactions sit behind each ML reason
# ---------------------------------------------------------------------------

def _largest(t: pd.DataFrame, n: int = MAX_PER_REASON) -> pd.DataFrame:
    return t.nlargest(n, "amount")


def _busiest_day(t: pd.DataFrame) -> pd.DataFrame:
    if t.empty:
        return t
    day = t["timestamp"].dt.floor("D")
    return t[day == day.value_counts().idxmax()]


def _busiest_block(t: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    if t.empty:
        return t
    block = (t["timestamp"] - start).dt.days // 30
    return t[block == block.value_counts().idxmax()]


def ml_selectors(start: pd.Timestamp) -> dict:
    """feature name -> function(customer's transactions) -> evidence transactions."""
    lim = schema.CASH_REVIEW_LIMIT
    hr = schema.HIGH_RISK_COUNTRIES

    def ptype(t, *types):
        return t[t["payment_type"].astype(str).isin(types)]

    def inbound(t):
        return t[t["direction"].astype(str) == "in"]

    def outbound(t):
        return t[t["direction"].astype(str) == "out"]

    def high_risk(t):
        return t[t["sender_country"].isin(hr) | t["receiver_country"].isin(hr)]

    def one_per_payee(t):
        return outbound(t).sort_values("amount", ascending=False).drop_duplicates("receiver_id")

    return {
        "n_txn": lambda t: _largest(_busiest_block(t, start)),
        "total_in": lambda t: _largest(inbound(t), 5),
        "total_out": lambda t: _largest(outbound(t), 5),
        "max_amount": lambda t: _largest(t, 1),
        "cash_in_share": lambda t: _largest(ptype(t, "cash_deposit")),
        "near_limit_cash": lambda t: ptype(t, "cash_deposit").pipe(
            lambda c: c[(c["amount"] >= 0.9 * lim) & (c["amount"] < lim)]),
        "money_order_in": lambda t: _largest(ptype(t, "money_order"), 15),
        "max_deposits_one_day": lambda t: _busiest_day(inbound(t)),
        "high_risk_share": lambda t: _largest(high_risk(t)),
        "high_risk_txns": lambda t: _largest(high_risk(t)),
        "intl_in": lambda t: _largest(inbound(ptype(t, "wire", "cross_border"))),
        "unique_payees": lambda t: _largest(one_per_payee(t)),
        "payees_per_payment": lambda t: _largest(one_per_payee(t)),
        "card_payees": lambda t: _largest(ptype(outbound(t), "card_repayment").drop_duplicates("receiver_id")),
        "max_out_30d": lambda t: _largest(_busiest_block(outbound(t), start)),
    }


def ml_evidence(tx: pd.DataFrame, txq: pd.DataFrame, customers: list[str]) -> pd.DataFrame:
    f = bl.customer_features(tx)
    feats = [c for c in f.columns if c != "small_business"]
    sorted_vals = {c: np.sort(f[c].values) for c in feats}
    n = len(f)
    sel = ml_selectors(tx["timestamp"].min())
    by_cust = dict(tuple(txq.groupby("customer_id")))
    parts = []
    for c in customers:
        if c not in f.index or c not in by_cust:
            continue
        row = f.loc[c]
        pct = {k: np.searchsorted(sorted_vals[k], row[k], side="left") / n for k in feats}
        top = [k for k in sorted(pct, key=pct.get, reverse=True)[:3] if pct[k] >= MIN_PCT]
        for k in top:
            picked = sel[k](by_cust[c])
            if len(picked):
                band = max(100 * (1 - pct[k]), 0.1)
                band = f"{band:.1f}" if band < 1 else f"{band:.0f}"
                parts.append(_ev(picked, "below_line",
                                 f"ML: {bl.FEATURES[k]} in the top {band}% of customers", 2))
    return pd.concat(parts) if parts else None


# ---------------------------------------------------------------------------
# Build the evidence table
# ---------------------------------------------------------------------------

EV_COLS = ["queue_rank", "customer_id", "evidence_rank", "txn_id", "timestamp", "direction",
           "payment_type", "amount", "counterparty", "counterparty_country", "checks", "why",
           "score", "bank_alerted", "bank_rules"]


def _empty_evidence() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in EV_COLS})


def build(d: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    queue = pd.read_parquet(d / "review_queue.parquet")
    customers = list(queue["customer_id"])
    tx = pd.read_parquet(d / "transactions.parquet", columns=COLS + ["customer_segment"])
    txq = tx[tx["customer_id"].isin(customers)]
    alerts = pd.read_parquet(d / "alerts.parquet")
    alerts_q = alerts[alerts["customer_id"].isin(customers)]
    rules = pd.read_parquet(d / "rules.parquet").set_index("rule_id")
    cases = pd.read_parquet(d / "cases.parquet").set_index("customer_id")

    parts = []

    # Advisory: transactions inside the matching red-flag windows.
    pat_path = d / "advisory_patterns.json"
    if pat_path.exists():
        for p in json.loads(pat_path.read_text()).get("patterns", []):
            _, hit = adv.match(tx, p)
            hit = hit[hit["customer_id"].isin(customers)] if len(hit) else hit
            if len(hit):
                parts.append(_ev(hit, "advisory", f"advisory red flag: {p['name']}", 3))

    # Suppression: alerted transactions of valued clients who never got a case.
    rb = rbk.used(d)
    case_at = rb["case_threshold"] if rb else schema.CASE_THRESHOLD
    sup = set(cases.index[cases["suppressed"] & (cases["score"] >= case_at)]) & set(customers)
    for rid, g in alerts_q[alerts_q["customer_id"].isin(sup)].groupby("rule_id"):
        parts.append(_ev(g, "suppression",
                         f"triggered {rid} ({rules.at[rid, 'points']} pt{'s' if rules.at[rid, 'points'] != 1 else ''}) but the case was "
                         f"blocked: valued client", 3))

    # Calibration: transactions that triggered an under-weighted rule.
    cal_path = d / "calibration.parquet"
    if cal_path.exists():
        cal = pd.read_parquet(cal_path)
        for _, r in cal[cal["verdict"] == "under-weighted"].iterrows():
            g = alerts_q[alerts_q["rule_id"] == r["rule_id"]]
            if len(g):
                parts.append(_ev(g, "calibration",
                                 f"triggered {r['rule_id']}: worth {r['points']} pt but right "
                                 f"{100 * r['hit_rate']:.0f}% of the time", 2))

    # Below the line: transactions behind the ML's top reasons.
    ml = ml_evidence(tx, txq, customers)
    if ml is not None:
        parts.append(ml)

    # Coverage: transactions of a type no rule can see.
    cov_path = d / "coverage.parquet"
    if cov_path.exists():
        cov = pd.read_parquet(cov_path)
        for _, r in cov[cov["status"] == "unmonitored"].iterrows():
            g = txq[(txq["payment_type"].astype(str) == r["payment_type"])
                    & (txq["direction"].astype(str) == r["direction"])]
            if len(g):
                parts.append(_ev(g, "coverage", f"no rule can see {r['label'].lower()}", 1))

    if not parts:
        return _empty_evidence(), queue.assign(evidence_txns=0, evidence_gbp=0.0,
                                               unalerted_pct=0.0, top_evidence="")
    ev = pd.concat(parts, ignore_index=True)
    # One row per transaction: collect every reason it supports.
    ev = ev.drop_duplicates(["txn_id", "why"])
    agg = ev.groupby(["customer_id", "txn_id"]).agg(
        weight=("weight", "sum"),
        checks=("check", lambda s: ", ".join(sorted(set(s)))),
        why=("why", lambda s: " | ".join(s)),
    ).reset_index()

    agg = agg.merge(txq[COLS], on=["customer_id", "txn_id"], how="left")
    is_out = agg["direction"].astype(str) == "out"
    agg["counterparty"] = agg["receiver_id"].where(is_out, agg["sender_id"])
    agg["counterparty_country"] = agg["receiver_country"].where(is_out, agg["sender_country"])
    agg = agg.drop(columns=["sender_id", "receiver_id", "sender_country", "receiver_country"])

    fired = alerts_q.groupby("txn_id")["rule_id"].agg(lambda s: ", ".join(sorted(set(s))))
    agg["bank_rules"] = agg["txn_id"].map(fired).fillna("")
    agg["bank_alerted"] = agg["bank_rules"] != ""

    # Score: weights, plus up to 0.5 for size within this customer's evidence.
    agg["size_pct"] = agg.groupby("customer_id")["amount"].rank(pct=True)
    agg["score"] = (agg["weight"] + 0.5 * agg["size_pct"]).round(2)
    agg = agg.sort_values(["customer_id", "score", "amount"], ascending=[True, False, False])
    agg["evidence_rank"] = agg.groupby("customer_id").cumcount() + 1

    # Per-customer totals (before trimming), added to the review queue.
    totals = agg.groupby("customer_id").agg(
        evidence_txns=("txn_id", "size"), evidence_gbp=("amount", "sum"),
        unalerted_pct=("bank_alerted", lambda s: round(100 * (1 - s.mean()), 1)))
    top1 = agg[agg["evidence_rank"] == 1].set_index("customer_id")
    totals["top_evidence"] = (top1["amount"].map("£{:,.0f}".format) + " "
                              + top1["payment_type"].astype(str) + ": " + top1["why"])

    agg = agg[agg["evidence_rank"] <= MAX_PER_CUSTOMER]
    agg = agg.merge(queue[["customer_id", "queue_rank"]], on="customer_id")
    agg = agg.sort_values(["queue_rank", "evidence_rank"])[EV_COLS].reset_index(drop=True)

    queue = queue.drop(columns=[c for c in totals.columns if c in queue.columns])
    queue = queue.merge(totals.reset_index(), on="customer_id", how="left")
    queue["evidence_txns"] = queue["evidence_txns"].fillna(0).astype(int)
    return agg, queue


# ---------------------------------------------------------------------------
# PROOF ONLY
# ---------------------------------------------------------------------------

def proof(ev: pd.DataFrame, d: Path) -> None:
    lab = pd.read_parquet(d / "labels.parquet")
    bad_tx = set(lab.loc[lab["is_suspicious"], "txn_id"])
    tx = pd.read_parquet(d / "transactions.parquet", columns=["txn_id", "customer_id"])
    bad_cust = set(tx.loc[tx["txn_id"].isin(bad_tx), "customer_id"])
    e = ev[ev["customer_id"].isin(bad_cust)].assign(bad=lambda x: x["txn_id"].isin(bad_tx))
    if e.empty:
        print("\nPROOF: no laundering customers in the queue")
        return
    print(f"\nPROOF: for queued customers who really are laundering "
          f"({e['customer_id'].nunique()} customers):")
    for k in (1, 3, 10):
        top = e[e["evidence_rank"] <= k]
        print(f"  top {k:>2} evidence transactions per customer: "
              f"{100 * top['bad'].mean():.0f}% are laundering transactions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--no-proof", action="store_true")
    args = ap.parse_args()
    d = args.data

    ev, queue = build(d)
    ev.to_parquet(d / "evidence.parquet", index=False)
    queue.to_parquet(d / "review_queue.parquet", index=False)

    if ev.empty:
        print("review queue is empty: no evidence to link")
        return
    print(f"{len(ev):,} evidence transactions for {ev['customer_id'].nunique():,} customers "
          f"({100 * (~ev['bank_alerted']).mean():.0f}% never alerted by the bank's rules)")
    print("\nby check:")
    for chk, n in ev["checks"].str.split(", ").explode().value_counts().items():
        print(f"  {chk:12} {n:>6,}")
    first = queue.iloc[0]
    print(f"\nqueue #1: customer {first['customer_id']} - {first['evidence_txns']} evidence "
          f"transactions, £{first['evidence_gbp']:,.0f}, "
          f"{first['unalerted_pct']}% never alerted")
    show = ev[ev["queue_rank"] == 1].head(8).copy()
    show["amount"] = show["amount"].map("£{:,.0f}".format)
    show["timestamp"] = show["timestamp"].dt.strftime("%Y-%m-%d")
    print(show[["evidence_rank", "timestamp", "payment_type", "amount", "counterparty_country",
                "bank_alerted", "why"]].to_string(index=False, max_colwidth=70))
    if not args.no_proof:
        proof(ev, d)


if __name__ == "__main__":
    main()
