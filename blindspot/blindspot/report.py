"""Combine the four checks into one report.

    python -m blindspot.report
    python -m blindspot.report --data data/sample

1. Merges every finding into data/findings.parquet, most severe first.
2. Builds ONE review queue: customers the bank never opened a case on that
   Blindspot says a human should look at, with the reasons from each check.
3. The LLM rewrites each finding in plain English for an auditor and writes a
   short executive summary. It sees only finding titles and aggregate
   numbers - never transactions or customer IDs - and code rejects any
   sentence containing a number that wasn't in the facts it was given.
4. PROOF (uses the planted answers): of the laundering the bank missed, how
   much is in Blindspot's review queue? This is the headline for the demo.

Writes data/findings.parquet, data/review_queue.parquet, data/summary.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from . import llm, schema

CHECK_ORDER = {"coverage": 0, "below_line": 1, "calibration": 2, "advisory": 3}
SEV_ORDER = {"high": 0, "medium": 1, "low": 2}
BELOW_LINE_TOP = 100      # how far down the ML ranking the queue reaches


# ---------------------------------------------------------------------------
# 1-2. Findings and review queue (no labels)
# ---------------------------------------------------------------------------

def load_findings(d: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(d.glob("findings_*.parquet"))]
    if not parts:
        return schema.empty(schema.FINDINGS)
    f = pd.concat(parts, ignore_index=True)
    f = f.assign(_s=f["severity"].astype(str).map(SEV_ORDER),
                 _c=f["check"].astype(str).map(CHECK_ORDER))
    return f.sort_values(["_s", "_c"]).drop(columns=["_s", "_c"]).reset_index(drop=True)


def review_queue(d: Path) -> pd.DataFrame:
    cases = pd.read_parquet(d / "cases.parquet").set_index("customer_id")
    no_case = set(cases.index[~cases["case_opened"]])
    reasons: dict[str, list[str]] = {}
    priority: dict[str, int] = {}

    def add(c, why, pts):
        if c in no_case:
            reasons.setdefault(c, []).append(why)
            priority[c] = priority.get(c, 0) + pts

    # Advisory: matches a regulator's red-flag pattern.
    adv_path = d / "advisory.parquet"
    if adv_path.exists():
        for _, r in pd.read_parquet(adv_path).iterrows():
            for c in r["customer_ids"]:
                add(c, f"matches advisory red flag: {r['name']}", 3)

    # Below-the-line ML ranking.
    below = pd.read_parquet(d / "below_line.parquet")
    for _, r in below[below["rank"] <= BELOW_LINE_TOP].iterrows():
        add(r["customer_id"], f"ML: #{r['rank']} most unusual unflagged customer "
                              f"({r['reasons']})", 3 if r["rank"] <= 25 else 2)

    # Suppressed valued clients: scored enough for a case, never got one.
    for c, r in cases[cases["suppressed"]].iterrows():
        add(c, f"scored {r['score']} points but exempted as a valued client", 3)

    # Calibration: customers hit by an under-weighted rule.
    cal_path = d / "calibration.parquet"
    if cal_path.exists():
        cal = pd.read_parquet(cal_path)
        under = cal[cal["verdict"] == "under-weighted"]
        if len(under):
            alerts = pd.read_parquet(d / "alerts.parquet")
            for _, r in under.iterrows():
                hit = alerts.loc[alerts["rule_id"] == r["rule_id"], "customer_id"].unique()
                for c in hit:
                    add(c, f"triggered {r['rule_id']}: worth {r['points']} pt but right "
                           f"{100 * r['hit_rate']:.0f}% of the time", 2)

    q = pd.DataFrame({"customer_id": list(reasons),
                      "priority": [priority[c] for c in reasons],
                      "checks": [len(reasons[c]) for c in reasons],
                      "reasons": [" | ".join(reasons[c]) for c in reasons]})
    if q.empty:
        return q
    q = q.merge(below[["customer_id", "anomaly_score"]], on="customer_id", how="left")
    q["engine_score"] = q["customer_id"].map(cases["score"])
    q["suppressed"] = q["customer_id"].map(cases["suppressed"])
    q = q.sort_values(["priority", "anomaly_score"], ascending=False).reset_index(drop=True)
    q.insert(0, "queue_rank", range(1, len(q) + 1))
    return q


def facts(d: Path, findings: pd.DataFrame, queue: pd.DataFrame) -> dict:
    """Aggregate numbers only. This is everything the LLM is allowed to see."""
    cases = pd.read_parquet(d / "cases.parquet")
    cov = pd.read_parquet(d / "coverage.parquet")
    total = cov["money_gbp"].sum()
    unmon = cov[cov["status"] == "unmonitored"]
    return {
        "customers_monitored": int(len(cases)),
        "cases_opened": int(cases["case_opened"].sum()),
        "customers_suppressed": int(cases["suppressed"].sum()),
        "pct_money_watched_by_amount_or_country_rule":
            round(100 * cov.loc[cov["status"] == "monitored", "money_gbp"].sum() / total, 1),
        "unmonitored_payment_types": list(unmon["label"]),
        "unmonitored_money_gbp": round(float(unmon["money_gbp"].sum())),
        "findings_total": int(len(findings)),
        "findings_high": int((findings["severity"].astype(str) == "high").sum()),
        "review_queue_customers": int(len(queue)),
        "findings": [{"id": r["finding_id"], "check": str(r["check"]), "severity": str(r["severity"]),
                      "title": r["title"], "metric": r["metric"]} for _, r in findings.iterrows()],
    }


# ---------------------------------------------------------------------------
# 3. LLM writing, with a number check
# ---------------------------------------------------------------------------

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUM.findall(text):
        try:
            out.add(round(float(m.replace(",", "")), 2))
        except ValueError:
            pass
    return out


def numbers_ok(written: str, source: str) -> bool:
    """Every number in `written` must appear in `source` (small counts 0-10 allowed)."""
    allowed = _numbers(source)
    return all(n in allowed or (n <= 10 and n == int(n)) for n in _numbers(written))


REWRITE_SYSTEM = """You are writing for a bank's internal auditor, in UK English.
For each finding, write 2-3 plain sentences: what is wrong, why it matters, what to do.
Use ONLY the numbers given. Do not add, round, combine or calculate numbers.
Do not name customers. Return JSON: {"findings": {"<id>": "<text>", ...}}"""

SUMMARY_SYSTEM = """You are writing the executive summary of an independent audit of a
bank's anti-money-laundering transaction monitoring, for its board, in UK English.
Write one paragraph of at most 120 words: the overall conclusion, the most serious
gaps, and the next step. Use ONLY the numbers in the facts. Do not calculate new
numbers. Do not name customers. Return JSON: {"summary": "<text>"}"""


def rewrite_findings(findings: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    src = {r["finding_id"]: f"{r['title']}. {r['metric']}. {r['explanation']} "
                            f"Suggested fix: {r['suggested_fix']}"
           for _, r in findings.iterrows()}
    try:
        out = llm.chat_json([{"role": "system", "content": REWRITE_SYSTEM},
                             {"role": "user", "content": json.dumps(src, ensure_ascii=False)}])
        texts = out.get("findings", {})
    except Exception as e:
        print(f"  LLM rewrite skipped ({e}); keeping code-written explanations")
        return findings.assign(plain_english=findings["explanation"]), 0
    plain, rejected = [], 0
    for _, r in findings.iterrows():
        t = texts.get(r["finding_id"])
        if isinstance(t, str) and t.strip() and numbers_ok(t, src[r["finding_id"]]):
            plain.append(t.strip())
        else:
            rejected += t is not None
            plain.append(r["explanation"])
    return findings.assign(plain_english=plain), rejected


def executive_summary(f: dict) -> tuple[str, str]:
    source = json.dumps(f, ensure_ascii=False)
    fallback = (f"Blindspot reviewed monitoring for {f['customers_monitored']:,} customers and "
                f"raised {f['findings_total']} findings, {f['findings_high']} of them high "
                f"severity. Only {f['pct_money_watched_by_amount_or_country_rule']}% of money is "
                f"watched by an amount or country rule. {f['review_queue_customers']:,} customers "
                f"with no case are recommended for review.")
    try:
        text = llm.chat_json([{"role": "system", "content": SUMMARY_SYSTEM},
                              {"role": "user", "content": source}]).get("summary", "")
    except Exception as e:
        print(f"  LLM summary skipped ({e}); using code-written summary")
        return fallback, "code"
    if text and numbers_ok(text, source):
        return text.strip(), "llm"
    print("  LLM summary contained a number not in the facts; using code-written summary")
    return fallback, "code"


# ---------------------------------------------------------------------------
# 4. PROOF ONLY - uses the planted answers
# ---------------------------------------------------------------------------

def validation(d: Path, queue: pd.DataFrame) -> dict:
    tx = pd.read_parquet(d / "transactions.parquet", columns=["txn_id", "customer_id"])
    lab = pd.read_parquet(d / "labels.parquet")
    cases = pd.read_parquet(d / "cases.parquet").set_index("customer_id")
    sus = lab[lab["is_suspicious"]].merge(tx, on="txn_id")
    cust = sus.groupby("customer_id").agg(source=("source", "first"), typology=("typology", "first"))
    cust["caught_by_bank"] = cust.index.map(cases["case_opened"]).fillna(False).astype(bool)
    cust["in_queue"] = cust.index.isin(set(queue["customer_id"]))

    out = {}
    for src, g in cust.groupby("source"):
        missed = g[~g["caught_by_bank"]]
        found = int(missed["in_queue"].sum())
        out[src] = {
            "suspicious_customers": int(len(g)),
            "caught_by_bank": int(g["caught_by_bank"].sum()),
            "bank_pct": round(100 * g["caught_by_bank"].mean(), 1),
            "missed_by_bank": int(len(missed)),
            "found_by_blindspot": found,
            "blindspot_pct_of_missed": round(100 * found / len(missed), 1) if len(missed) else 0.0,
            "combined_pct": round(100 * (g["caught_by_bank"] | g["in_queue"]).mean(), 1),
            "by_typology": {
                t: {"customers": int(len(x)), "bank": int(x["caught_by_bank"].sum()),
                    "blindspot_found": int((~x["caught_by_bank"] & x["in_queue"]).sum())}
                for t, x in g.groupby("typology")},
        }
    bad = set(cust.index)
    out["queue_precision_pct"] = round(100 * queue["customer_id"].isin(bad).mean(), 1) if len(queue) else 0.0
    no_case = (~cases["case_opened"])
    out["base_rate_pct"] = round(100 * cases.index[no_case].isin(bad).mean(), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--no-llm", action="store_true", help="skip LLM writing")
    args = ap.parse_args()
    d = args.data

    findings = load_findings(d)
    queue = review_queue(d)
    f = facts(d, findings, queue)

    if args.no_llm:
        findings = findings.assign(plain_english=findings["explanation"])
        summary, by, rejected = (f"Blindspot raised {f['findings_total']} findings; "
                                 f"{f['review_queue_customers']:,} customers recommended "
                                 f"for review."), "code", 0
    else:
        findings, rejected = rewrite_findings(findings)
        summary, by = executive_summary(f)

    findings.to_parquet(d / "findings.parquet", index=False)
    queue.to_parquet(d / "review_queue.parquet", index=False)
    val = validation(d, queue)
    (d / "summary.json").write_text(json.dumps(
        {"facts": f, "executive_summary": summary, "summary_written_by": by,
         "validation_uses_planted_answers": val}, indent=2, default=str))

    print("EXECUTIVE SUMMARY" + (" (LLM)" if by == "llm" else " (code)"))
    print(summary)
    if rejected:
        print(f"\n  {rejected} LLM finding rewrites rejected for unsupported numbers")
    print(f"\n{len(findings)} findings ({f['findings_high']} high):")
    for _, r in findings.iterrows():
        print(f"  [{r['severity']:6}] {r['check']:11} {r['title']}")
    print(f"\nreview queue: {len(queue):,} customers with no case")
    print(queue.head(5)[["queue_rank", "customer_id", "priority", "reasons"]].to_string(index=False))

    print("\nPROOF (planted answers): of the laundering the bank missed, how much does Blindspot find?")
    for src in ("planted", "samld"):
        v = val.get(src)
        if v:
            print(f"  {src:8} bank caught {v['caught_by_bank']}/{v['suspicious_customers']} "
                  f"({v['bank_pct']}%); Blindspot found {v['found_by_blindspot']} of the "
                  f"{v['missed_by_bank']} missed ({v['blindspot_pct_of_missed']}%); "
                  f"together {v['combined_pct']}%")
    if "planted" in val:
        for t, x in val["planted"]["by_typology"].items():
            print(f"    {t:25} bank {x['bank']:>2}/{x['customers']}, Blindspot found "
                  f"{x['blindspot_found']} more")
    print(f"  review queue precision: {val['queue_precision_pct']}% laundering "
          f"(vs {val['base_rate_pct']}% if you picked at random)")


if __name__ == "__main__":
    main()