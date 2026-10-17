"""Blindspot check 3: the advisory check (LLM + code).

    python -m blindspot.checks.advisory
    python -m blindspot.checks.advisory advisories/other_alert.md --data data/sample

Question it answers: would our rules catch the scheme a regulator just warned about?

How:
  1. The LLM reads the advisory (public text only) and turns each red flag
     into a structured, testable pattern (JSON). It must quote the red flag it
     implements and state any threshold it had to assume.
  2. Code VALIDATES every pattern (allowed fields, payment types, sane numbers)
     and drops anything malformed. The LLM never touches the data.
  3. Code searches the transactions for customers matching each pattern, then
     checks whether the bank's rules could see them and whether a case opened.
  4. Red flags the pattern language can't express are listed for a human,
     not silently dropped.

Writes data/advisory_patterns.json (for human review), data/advisory.parquet
and data/findings_advisory.parquet. Never reads labels (except PROOF).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .. import llm, schema

FIELDS = {
    "id": str, "name": str, "red_flag_quote": str, "assumptions": str,
    "direction": str, "payment_types": list, "segment": str, "countries": list,
    "amount_min": (int, float), "amount_max": (int, float),
    "window_days": int, "min_count": int, "min_distinct_counterparties": int,
    "min_total": (int, float), "max_total": (int, float), "min_windows": int,
}
DEFAULTS = {"assumptions": "", "direction": "any", "payment_types": [], "segment": "any",
            "countries": [], "amount_min": 0, "amount_max": 1e12, "window_days": 30,
            "min_count": 1, "min_distinct_counterparties": 1, "min_total": 0,
            "max_total": 1e15, "min_windows": 1}

SYSTEM = f"""You turn financial-crime advisories into testable transaction-monitoring patterns.
Return ONLY a JSON object: {{"patterns": [...], "untestable": [...]}}.

Each pattern describes, for ONE customer, a group of transactions inside a time window:
- id: short snake_case id
- name: plain-English name, under 10 words
- red_flag_quote: the advisory words this implements, under 15 words, copied exactly
- assumptions: any threshold you had to choose because the advisory was vague
- direction: "in" (money arriving), "out" (money leaving) or "any"
- payment_types: subset of {schema.PAYMENT_TYPES} (empty list = all)
- segment: "individual", "small_business" or "any"
- countries: counterparty countries to require (empty list = any)
- amount_min, amount_max: bounds on EACH transaction, in GBP
- window_days: length of the window in days (1 = same day)
- min_count: minimum number of matching transactions in a window
- min_distinct_counterparties: minimum number of different counterparties in a window
- min_total, max_total: bounds on the window's total, in GBP
- min_windows: in how many separate windows it must happen (e.g. "repeated on different days")

Rules:
- Only use what the advisory supports. Do not invent red flags.
- "credit_card" means SPENDING with a card. Paying off a credit card account
  (your own or someone else's) is "card_repayment".
- One pattern per suspicious BEHAVIOUR. If several red flags describe the same
  behaviour (e.g. who is paid AND how much each payment is), combine them into
  ONE pattern. Never make a pattern from an amount range alone.
- If the advisory names the type of customer (e.g. "a business account"), set segment.
- red_flag_quote must be UNDER 15 words: quote the key phrase, not the whole red flag.
- If the advisory is vague ("many", "several"), choose a concrete, conservative
  number and say so in "assumptions".
- If a red flag CANNOT be expressed (e.g. it depends on the ORDER of different
  kinds of transactions), put it in "untestable" as {{"red_flag": "...", "reason": "..."}}.
"""


# ---------------------------------------------------------------------------
# 1-2. Extract and validate
# ---------------------------------------------------------------------------

def extract(advisory_text: str) -> dict:
    return llm.chat_json([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": advisory_text},
    ])


def validate(raw: dict) -> tuple[list[dict], list[dict], list[str]]:
    """Keep only well-formed patterns. Returns (patterns, untestable, warnings)."""
    patterns, warnings = [], []
    for i, p in enumerate(raw.get("patterns", [])):
        if not isinstance(p, dict):
            warnings.append(f"pattern {i}: not an object, dropped")
            continue
        clean = {**DEFAULTS, **{k: v for k, v in p.items() if k in FIELDS and v is not None}}
        bad = [k for k, t in FIELDS.items() if k in clean and not isinstance(clean[k], t)]
        # accept whole numbers given as floats, e.g. 30.0 days
        for k in list(bad):
            if FIELDS[k] is int and isinstance(clean[k], float) and clean[k].is_integer():
                clean[k] = int(clean[k])
                bad.remove(k)
        if bad or not clean.get("id") or not clean.get("name"):
            warnings.append(f"pattern {i} ({p.get('id', '?')}): bad fields {bad or ['id/name']}, dropped")
            continue
        unknown = set(clean["payment_types"]) - set(schema.PAYMENT_TYPES)
        if unknown:
            warnings.append(f"{clean['id']}: unknown payment types {sorted(unknown)} removed")
            clean["payment_types"] = [t for t in clean["payment_types"] if t in schema.PAYMENT_TYPES]
        if clean["direction"] not in ("in", "out", "any"):
            clean["direction"] = "any"
        if clean["segment"] not in ("individual", "small_business", "any"):
            clean["segment"] = "any"
        clean["window_days"] = max(1, min(clean["window_days"], schema.WINDOW_DAYS))
        words = str(clean.get("red_flag_quote", "")).split()
        if len(words) > 15:   # keep quotes short even if the model doesn't
            clean["red_flag_quote"] = " ".join(words[:14]) + " ..."
        for k in ("min_count", "min_distinct_counterparties", "min_windows"):
            clean[k] = max(1, clean[k])
        if clean["amount_min"] > clean["amount_max"] or clean["min_total"] > clean["max_total"]:
            warnings.append(f"{clean['id']}: impossible bounds, dropped")
            continue
        patterns.append(clean)
    untestable = [u for u in raw.get("untestable", []) if isinstance(u, dict)]
    return patterns, untestable, warnings


# ---------------------------------------------------------------------------
# 3. Search the data and test the bank's rules
# ---------------------------------------------------------------------------

def match(tx: pd.DataFrame, p: dict) -> tuple[set, pd.DataFrame]:
    """Customers matching pattern p, and the transactions in their matching windows."""
    t = tx
    d = t["direction"].astype(str)
    if p["direction"] != "any":
        t = t[d == p["direction"]]
    if p["payment_types"]:
        t = t[t["payment_type"].astype(str).isin(p["payment_types"])]
    if p["segment"] != "any":
        t = t[t["customer_segment"].astype(str) == p["segment"]]
    t = t[(t["amount"] >= p["amount_min"]) & (t["amount"] <= p["amount_max"])]
    if t.empty:
        return set(), t
    is_out = t["direction"].astype(str) == "out"
    t = t.assign(cp=t["receiver_id"].where(is_out, t["sender_id"]),
                 cp_country=t["receiver_country"].where(is_out, t["sender_country"]))
    if p["countries"]:
        t = t[t["cp_country"].isin(p["countries"])]
        if t.empty:
            return set(), t

    w = p["window_days"]
    start = tx["timestamp"].min().normalize()
    days = (t["timestamp"].dt.normalize() - start).dt.days
    # Fixed windows at two offsets approximate a rolling window.
    offsets = [0] if w == 1 else [0, w // 2]
    hits = []
    for off in offsets:
        tt = t.assign(win=(days + off) // w)
        g = tt.groupby(["customer_id", "win"]).agg(
            n=("amount", "size"), cps=("cp", "nunique"), total=("amount", "sum"))
        ok = g[(g["n"] >= p["min_count"]) & (g["cps"] >= p["min_distinct_counterparties"])
               & (g["total"] >= p["min_total"]) & (g["total"] <= p["max_total"])]
        hits.append(tt.merge(ok.reset_index()[["customer_id", "win"]], on=["customer_id", "win"]))
    hit_tx = pd.concat(hits).drop_duplicates("txn_id")
    if hit_tx.empty:
        return set(), hit_tx
    # Require the behaviour in enough separate windows (counted at offset 0).
    wins = hit_tx.assign(win=(hit_tx["timestamp"].dt.normalize() - start).dt.days // w) \
                 .groupby("customer_id")["win"].nunique()
    customers = set(wins[wins >= p["min_windows"]].index)
    return customers, hit_tx[hit_tx["customer_id"].isin(customers)]


def rules_in_scope(rules: pd.DataFrame, p: dict) -> list[str]:
    types = set(p["payment_types"] or schema.PAYMENT_TYPES)
    out = []
    for _, r in rules.iterrows():
        if types & set(r["payment_types"]) and (p["direction"] == "any" or
                                                r["direction"] in ("any", p["direction"])):
            out.append(f"{r['rule_id']} ({r['kind']})")
    return out


def run(tx, rules, alerts, cases, patterns) -> tuple[pd.DataFrame, pd.DataFrame]:
    opened = set(cases.loc[cases["case_opened"], "customer_id"])
    rows, findings = [], []
    for p in patterns:
        custs, hit_tx = match(tx, p)
        scope = rules_in_scope(rules, p)
        with_case = custs & opened
        alerted = hit_tx["txn_id"].isin(alerts["txn_id"]).mean() if len(hit_tx) else 0.0
        case_rate = len(with_case) / len(custs) if custs else 0.0
        money = hit_tx["amount"].sum() if len(hit_tx) else 0.0
        rows.append({"pattern_id": p["id"], "name": p["name"], "customers": len(custs),
                     "with_case": len(with_case), "case_rate": round(100 * case_rate, 1),
                     "txns": len(hit_tx), "money_gbp": round(money, 2),
                     "alerted_txn_pct": round(100 * alerted, 1),
                     "rules_in_scope": scope, "customer_ids": sorted(custs)})
        if custs and case_rate < 0.5:
            only_count = scope and all("(count)" in s for s in scope)
            gap = ("no rule can see these transactions" if not scope else
                   "only a transaction-count rule can see them" if only_count else
                   "the rules that can see them rarely add up to a case")
            findings.append({
                "finding_id": f"AD{len(findings) + 1}", "check": "advisory",
                "severity": "high" if case_rate < 0.2 else "medium",
                "title": f"Advisory pattern not caught: {p['name']}",
                "metric": (f"{len(custs)} customers match; {100 * case_rate:.0f}% have a case; "
                           f"£{money:,.0f}"),
                "evidence_ids": sorted(custs)[:100],
                "suggested_fix": f"Add a scenario for: {p['name'].lower()} "
                                 f"(red flag: \"{p['red_flag_quote']}\").",
                "explanation": f"The advisory warns: \"{p['red_flag_quote']}\". "
                               f"{len(custs)} customers match this pattern, but {gap}, "
                               f"so only {len(with_case)} have a case."
                               + (f" Assumption: {p['assumptions']}" if p["assumptions"] else ""),
            })
    res = pd.DataFrame(rows)
    fnd = schema.conform(pd.DataFrame(findings, columns=list(schema.FINDINGS)),
                         schema.FINDINGS, "findings")
    return res, fnd


# ---------------------------------------------------------------------------
# PROOF ONLY
# ---------------------------------------------------------------------------

def proof(res: pd.DataFrame, tx: pd.DataFrame, lab: pd.DataFrame) -> None:
    sus = lab[lab["is_suspicious"]].merge(tx[["txn_id", "customer_id"]], on="txn_id")
    typ = sus.groupby("customer_id")["typology"].first()
    print("\nPROOF: who matched each pattern")
    for _, r in res.iterrows():
        ids = r["customer_ids"]
        if not ids:
            print(f"  {r['pattern_id']}: no matches")
            continue
        found = typ.reindex(ids).fillna("innocent").value_counts()
        print(f"  {r['pattern_id']}: " + ", ".join(f"{k} {v}" for k, v in found.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("advisory", nargs="?", type=Path,
                    default=Path("advisories/third_party_card_repayments.md"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--no-proof", action="store_true")
    args = ap.parse_args()
    d = args.data

    saved = d / "advisory_patterns.json"
    try:
        raw = extract(args.advisory.read_text())
        patterns, untestable, warnings = validate(raw)
        print(f"LLM extracted {len(raw.get('patterns', []))} patterns; "
              f"{len(patterns)} passed validation")
    except Exception as e:  # noqa: BLE001  LLM offline, no key, rate limit...
        # Patterns depend only on the advisory text, so the last extraction of the
        # SAME advisory is still valid; only the search below needs re-running.
        prev = json.loads(saved.read_text()) if saved.exists() else {}
        if prev.get("advisory") != str(args.advisory):
            raise
        print(f"LLM unavailable ({e}); reusing the patterns last extracted from this advisory")
        patterns, untestable, warnings = prev["patterns"], prev["untestable"], prev["warnings"]
    saved.write_text(json.dumps(
        {"advisory": str(args.advisory), "patterns": patterns, "untestable": untestable,
         "warnings": warnings}, indent=2))
    for w in warnings:
        print("  warning:", w)
    for p in patterns:
        print(f"\n  [{p['id']}] {p['name']}\n    red flag: \"{p['red_flag_quote']}\"")
        if p["assumptions"]:
            print(f"    assumed: {p['assumptions']}")
    for u in untestable:
        print(f"\n  NOT TESTABLE (needs a human): {u.get('red_flag')} - {u.get('reason')}")

    tx = pd.read_parquet(d / "transactions.parquet")
    res, fnd = run(tx, pd.read_parquet(d / "rules.parquet"), pd.read_parquet(d / "alerts.parquet"),
                   pd.read_parquet(d / "cases.parquet"), patterns)
    res.to_parquet(d / "advisory.parquet", index=False)
    fnd.to_parquet(d / "findings_advisory.parquet", index=False)

    print("\n" + res[["pattern_id", "customers", "with_case", "case_rate", "money_gbp",
                      "alerted_txn_pct"]].to_string(index=False))
    print(f"\n{len(fnd)} findings: " + "; ".join(fnd["title"]))
    if not args.no_proof:
        proof(res, tx, pd.read_parquet(d / "labels.parquet"))


if __name__ == "__main__":
    main()
