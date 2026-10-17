"""Blindspot check 3: the calibration check.

    python -m blindspot.checks.calibration
    python -m blindspot.checks.calibration --data data/sample

Question it answers: are the rules weighted sensibly?

For each rule, it takes the customers that rule alerted on who were later
investigated (from the bank's own investigation log) and measures how often
a suspicious activity report was filed: the rule's hit rate. A rule that is
usually right should be worth a lot of points; a rule that is usually wrong
should be worth few.

Uses a 95% Wilson interval so small samples don't produce confident claims:
a rule is only called under- or over-weighted if the WHOLE interval supports it.

Reads alerts, rules and investigations. Never reads labels.
Writes data/calibration.parquet and data/findings_calibration.parquet.
"""

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema

MIN_INVESTIGATED = 10
MIN_GAP = 3   # points


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def points_for(hit_rate: float) -> int:
    """Suggested weight: a rule right X% of the time earns X% of a case.
    So a rule right half the time gets 10 points, two hits open a case."""
    return int(np.clip(round(schema.CASE_THRESHOLD * hit_rate), 1, schema.CASE_THRESHOLD))


def calibration_table(alerts, rules, inv) -> pd.DataFrame:
    base = inv["sar_filed"].mean() if len(inv) else 0.0
    rows = []
    for _, r in rules.iterrows():
        alerted = set(alerts.loc[alerts["rule_id"] == r["rule_id"], "customer_id"])
        seen = inv[inv["customer_id"].isin(alerted)]
        n, k = len(seen), int(seen["sar_filed"].sum())
        lo, hi = wilson(k, n)
        rate = k / n if n else np.nan
        verdict = "too few investigated"
        pts = int(r["points"])
        if n >= MIN_INVESTIGATED:
            # Report only gaps the whole confidence interval supports AND that matter:
            # at least double (or half) the weight, and at least MIN_GAP points apart.
            if points_for(lo) >= max(2 * pts, pts + MIN_GAP):
                verdict = "under-weighted"
            elif points_for(hi) <= min(pts / 2, pts - MIN_GAP):
                verdict = "over-weighted"
            else:
                verdict = "reasonable"
        rows.append({
            "rule_id": r["rule_id"], "description": r["description"], "points": int(r["points"]),
            "customers_alerted": len(alerted), "investigated": n, "sars": k,
            "hit_rate": rate, "ci_low": lo, "ci_high": hi,
            "lift": rate / base if n and base else np.nan,
            "suggested_points": points_for(rate) if n else np.nan,
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def findings(cal: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in cal[cal["verdict"].isin(["under-weighted", "over-weighted"])].iterrows():
        under = r["verdict"] == "under-weighted"
        ev = alerts.loc[alerts["rule_id"] == r["rule_id"], "customer_id"].unique()[:100]
        rows.append({
            "finding_id": f"CA{len(rows) + 1}",
            "check": "calibration",
            "severity": "high" if under else "medium",
            "title": (f"Strong red flag worth only {r['points']} point"
                      f"{'s' if r['points'] != 1 else ''}: {r['rule_id']}" if under else
                      f"Weak rule over-weighted: {r['rule_id']}"),
            "metric": (f"{r['points']} pts, right {100 * r['hit_rate']:.1f}% of the time "
                       f"({100 * r['ci_low']:.0f}-{100 * r['ci_high']:.0f}%, n={r['investigated']})"),
            "evidence_ids": list(ev),
            "suggested_fix": (f"Raise {r['rule_id']} from {r['points']} to about "
                              f"{int(r['suggested_points'])} points."
                              if under else
                              f"Lower {r['rule_id']} from {r['points']} to about "
                              f"{int(r['suggested_points'])} points to cut false alarms."),
            "explanation": (f"{r['description']}. When investigators reviewed customers this rule "
                            f"flagged, {100 * r['hit_rate']:.0f}% were laundering, "
                            f"{r['lift']:.1f}x the average, yet it adds only {r['points']} of the "
                            f"{schema.CASE_THRESHOLD} points needed for a case."
                            if under else
                            f"{r['description']}. Only {100 * r['hit_rate']:.0f}% of reviewed "
                            f"customers it flagged were laundering, but it is worth "
                            f"{r['points']} points, so it mostly generates false alarms."),
        })
    return schema.conform(pd.DataFrame(rows, columns=list(schema.FINDINGS)),
                          schema.FINDINGS, "findings")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()
    d = args.data
    alerts = pd.read_parquet(d / "alerts.parquet")
    rules = pd.read_parquet(d / "rules.parquet")
    inv = pd.read_parquet(d / "investigations.parquet")

    cal = calibration_table(alerts, rules, inv)
    fnd = findings(cal, alerts)
    cal.to_parquet(d / "calibration.parquet", index=False)
    fnd.to_parquet(d / "findings_calibration.parquet", index=False)

    show = cal.copy()
    show["hit_rate"] = show.apply(
        lambda r: f"{100 * r['hit_rate']:.1f}% ({100 * r['ci_low']:.0f}-{100 * r['ci_high']:.0f})"
        if r["investigated"] else "-", axis=1)
    show["lift"] = show["lift"].map(lambda v: f"{v:.1f}x" if pd.notna(v) else "-")
    print(f"average hit rate across all investigations: {100 * inv['sar_filed'].mean():.1f}%\n")
    print(show[["rule_id", "points", "investigated", "hit_rate", "lift", "suggested_points",
                "verdict"]].to_string(index=False))
    print(f"\n{len(fnd)} findings: " + "; ".join(fnd["title"]))


if __name__ == "__main__":
    main()