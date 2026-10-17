"""Plant the five Bank of America laundering patterns into the data, in GBP.

    python -m blindspot.plant                    # data/base -> data
    python -m blindspot.plant --per-pattern 25

Patterns are planted onto EXISTING clean customers, so they sit on top of a
normal transaction history and are not trivially easy to spot. Every planted
transaction gets a label (source="planted") with its typology.

Also assigns a customer segment (individual / small_business) to every
customer, so segment alone does not give the planted customers away.

Writes:
    data/transactions.parquet
    data/labels.parquet
    data/planted_customers.parquet   # who got which pattern (for the demo)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

PATTERNS = [
    "structured_money_orders",
    "high_risk_wires",
    "credit_card_funnel",
    "structured_cash",
    "slow_burn",
]
BUSINESS_PATTERNS = {"structured_money_orders", "high_risk_wires", "credit_card_funnel"}
SMALL_BUSINESS_SHARE = 0.2


class Planter:
    def __init__(self, start: pd.Timestamp, end: pd.Timestamp, rng: np.random.Generator):
        self.start, self.end, self.rng = start, end, rng
        self._n = 0

    # -- helpers -----------------------------------------------------------
    def _uid(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n:06d}"

    def _time(self, a: pd.Timestamp | None = None, b: pd.Timestamp | None = None) -> pd.Timestamp:
        a, b = a or self.start, b or self.end
        span = max(int((b - a).total_seconds()), 1)
        return a + pd.Timedelta(seconds=int(self.rng.integers(0, span)))

    def _window(self, days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        """A random window of `days` inside the data range."""
        latest = max(self.end - pd.Timedelta(days=days), self.start)
        a = self._time(self.start, latest)
        return a, min(a + pd.Timedelta(days=days), self.end)

    @staticmethod
    def _row(ts, customer, direction, sender, receiver, amount, ptype,
             s_country="UK", r_country="UK") -> dict:
        return {
            "timestamp": ts, "customer_id": customer, "direction": direction,
            "sender_id": sender, "receiver_id": receiver, "amount": round(float(amount), 2),
            "payment_type": ptype, "sender_country": s_country, "receiver_country": r_country,
        }

    # -- patterns ----------------------------------------------------------
    def structured_money_orders(self, c: str) -> list[dict]:
        """Sets of three money orders deposited together, totalling just under
        the ID limit (£2,400-£2,499). Mirrors the car-auction customers."""
        rows = []
        lim = schema.MONEY_ORDER_ID_LIMIT
        n_days = max((self.end - self.start).days, 1)
        n_sets = min(int(self.rng.integers(6, 16)), n_days)
        days = self.rng.choice(n_days, size=n_sets, replace=False)   # one set per day
        for d in days:
            day = (self.start.normalize() + pd.Timedelta(days=int(d))
                   + pd.Timedelta(hours=int(self.rng.integers(9, 17))))
            total = self.rng.uniform(lim * 0.96, lim - 1)
            parts = np.round(self.rng.dirichlet([5, 5, 5]) * total, 2)
            for k, amt in enumerate(parts):
                ts = day + pd.Timedelta(minutes=3 * k + int(self.rng.integers(0, 3)))
                rows.append(self._row(ts, c, "in", self._uid("MO_PAYER_"), c, amt, "money_order"))
        return rows

    def high_risk_wires(self, c: str) -> list[dict]:
        """A small business suddenly receives large wires from Belarus/Georgia."""
        a, b = self._window(60)
        senders = [(self._uid("FX_"), self.rng.choice(["Belarus", "Georgia"]))
                   for _ in range(int(self.rng.integers(2, 5)))]
        rows = []
        for _ in range(int(self.rng.integers(10, 26))):
            sid, country = senders[int(self.rng.integers(len(senders)))]
            rows.append(self._row(self._time(a, b), c, "in", sid, c,
                                  self.rng.uniform(15_000, 60_000), "wire",
                                  s_country=country, r_country="UK"))
        return rows

    def credit_card_funnel(self, c: str) -> list[dict]:
        """One shop pays off many unrelated people's credit cards."""
        a, b = self._window(90)
        rows = []
        for _ in range(int(self.rng.integers(15, 41))):
            person = self._uid("CARD_")
            for _ in range(int(self.rng.integers(1, 4))):
                rows.append(self._row(self._time(a, b), c, "out", c, person,
                                      self.rng.uniform(300, 2_500), "credit_card"))
        return rows

    def structured_cash(self, c: str) -> list[dict]:
        """Repeated cash deposits just under the review limit (£9,000-£9,900)."""
        a, b = self._window(int(self.rng.integers(30, 91)))
        lim = schema.CASH_REVIEW_LIMIT
        return [self._row(self._time(a, b), c, "in", c, c,
                          self.rng.uniform(lim * 0.90, lim * 0.99), "cash_deposit")
                for _ in range(int(self.rng.integers(6, 16)))]

    def slow_burn(self, c: str) -> list[dict]:
        """Weak signals spread thinly across the whole period, so the points
        never reach the case threshold inside 13 months."""
        rows, ts = [], self.start + pd.Timedelta(days=int(self.rng.integers(0, 30)))
        while ts < self.end:
            if self.rng.random() < 0.5:
                rows.append(self._row(ts, c, "in", c, c,
                                      self.rng.uniform(1_500, 3_000), "cash_deposit"))
            else:
                rows.append(self._row(ts, c, "out", c, self._uid("HR_"),
                                      self.rng.uniform(500, 1_500), "cross_border",
                                      r_country=str(self.rng.choice(schema.HIGH_RISK_COUNTRIES))))
            ts += pd.Timedelta(days=int(self.rng.integers(25, 46)))
        return rows


def plant(tx: pd.DataFrame, lab: pd.DataFrame, per_pattern: int = 25,
          seed: int = schema.SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    p = Planter(tx["timestamp"].min(), tx["timestamp"].max(), rng)

    # Hosts: customers with no SAML-D laundering, so the two label sets stay separate.
    dirty = set(tx.loc[tx["txn_id"].isin(lab.loc[lab["is_suspicious"], "txn_id"]), "customer_id"])
    candidates = sorted(set(tx["customer_id"]) - dirty)
    if len(candidates) < per_pattern * len(PATTERNS):
        per_pattern = len(candidates) // len(PATTERNS)
        print(f"warning: only {len(candidates)} clean customers; planting {per_pattern} per pattern")
    hosts = rng.permutation(candidates)[: per_pattern * len(PATTERNS)]

    rows, manifest = [], []
    for i, pattern in enumerate(PATTERNS):
        for c in hosts[i * per_pattern:(i + 1) * per_pattern]:
            new = getattr(p, pattern)(str(c))
            for r in new:
                r["typology"] = pattern
            rows.extend(new)
            manifest.append({"customer_id": str(c), "typology": pattern, "n_txns": len(new),
                             "total_gbp": round(sum(r["amount"] for r in new), 2)})

    planted = pd.DataFrame(rows)
    planted["txn_id"] = [f"P{i}" for i in range(len(planted))]

    # Segments for every customer. Business patterns need small_business hosts.
    customers = pd.Index(sorted(set(tx["customer_id"]) | set(planted["customer_id"])))
    seg = pd.Series(np.where(rng.random(len(customers)) < SMALL_BUSINESS_SHARE,
                             "small_business", "individual"), index=customers)
    man = pd.DataFrame(manifest)
    seg[man.loc[man["typology"].isin(BUSINESS_PATTERNS), "customer_id"]] = "small_business"

    new_lab = pd.DataFrame({"txn_id": planted["txn_id"], "is_suspicious": True,
                            "source": "planted", "typology": planted["typology"]})
    planted = planted.drop(columns="typology")
    planted["customer_segment"] = "unknown"

    out_tx = pd.concat([tx, schema.conform(planted, schema.TRANSACTIONS, "planted")],
                       ignore_index=True)
    out_tx["customer_segment"] = out_tx["customer_id"].map(seg).astype("category")
    out_tx = schema.conform(out_tx.sort_values("timestamp", kind="stable").reset_index(drop=True),
                            schema.TRANSACTIONS, "transactions")
    out_lab = schema.conform(pd.concat([lab, new_lab], ignore_index=True), schema.LABELS, "labels")
    schema.validate_transactions(out_tx)
    return out_tx, out_lab, man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("data/base"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--per-pattern", type=int, default=25)
    args = ap.parse_args()

    tx = pd.read_parquet(args.base / "transactions.parquet")
    lab = pd.read_parquet(args.base / "labels.parquet")
    tx, lab, man = plant(tx, lab, args.per_pattern)

    args.out.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(args.out / "transactions.parquet", index=False)
    lab.to_parquet(args.out / "labels.parquet", index=False)
    man.to_parquet(args.out / "planted_customers.parquet", index=False)
    print(man.groupby("typology").agg(customers=("customer_id", "count"),
                                      txns=("n_txns", "sum"),
                                      total_gbp=("total_gbp", "sum")).round(0).to_string())
    print(f"wrote {len(tx):,} transactions ({(lab['source'] == 'planted').sum():,} planted) to {args.out}/")


if __name__ == "__main__":
    main()
