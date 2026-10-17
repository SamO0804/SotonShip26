"""The bank's rulebook: its transaction-monitoring rules as a plain YAML file.

This is the input a bank hands Blindspot. The engine executes it (to replay
the bank's monitoring on the test data) and Blindspot's checks read it (to
see what each rule can and can't see). One file, one source of truth.

Format (see rulebooks/northbridge_bank.yaml for a full example):

    bank: Northbridge Bank
    version: "2026-10"
    case_threshold: 20          # points needed to open a case
    window_days: 396            # points older than this expire (13 months)
    high_risk_countries: [Belarus, Georgia, ...]
    suppression:
      valued_clients_top_n: 5   # biggest small businesses by money in never get a case (0 = off)
    rules:
      - id: R01_large_cash
        description: Cash deposit of £10,000 or more
        enabled: true           # optional, default true
        points: 10
        applies_to:
          payment_types: [cash_deposit]     # or "all"
          direction: in                     # in, out or any
        trigger:
          type: single                      # single | repeated | velocity | aggregate
          amount_min: 10000                 # optional bounds on each payment (£)
          amount_max: null
          countries: high_risk              # optional: "high_risk" or a list; either end of the payment

Trigger types:
    single      every matching payment fires the rule
    repeated    fires on each matching payment once there are min_count of them
                within a rolling window_days
    velocity    counts matching payments in fixed blocks of window_days; fires
                once per block that reaches min_count
    aggregate   fixed blocks of window_days; fires once per block where the
                matching payments reach min_count, min_distinct_counterparties,
                and a total between min_total and max_total
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema

DEFAULT_RULEBOOK = Path("rulebooks/northbridge_bank.yaml")
TRIGGERS = {"single", "repeated", "velocity", "aggregate"}
DIRECTIONS = {"in", "out", "any"}


class RulebookError(ValueError):
    """A problem with the rulebook, worded for the person who wrote it."""


def _num(v, name, rid, minimum=None):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise RulebookError(f"{rid}: {name} must be a number, got {v!r}")
    if minimum is not None and v < minimum:
        raise RulebookError(f"{rid}: {name} must be at least {minimum}, got {v}")
    return v


def validate(raw: dict) -> dict:
    """Check a parsed rulebook and fill in defaults. Raises RulebookError."""
    if not isinstance(raw, dict) or not isinstance(raw.get("rules"), list) or not raw["rules"]:
        raise RulebookError("The rulebook needs a 'rules:' list with at least one rule.")
    rb = {
        "bank": str(raw.get("bank", "Unnamed bank")),
        "version": str(raw.get("version", "")),
        "case_threshold": _num(raw.get("case_threshold", schema.CASE_THRESHOLD), "case_threshold", "rulebook", 1),
        "window_days": int(_num(raw.get("window_days", schema.WINDOW_DAYS), "window_days", "rulebook", 1)),
        "high_risk_countries": list(raw.get("high_risk_countries", schema.HIGH_RISK_COUNTRIES)),
        "suppression": {"valued_clients_top_n": int(_num(
            (raw.get("suppression") or {}).get("valued_clients_top_n", 0),
            "suppression.valued_clients_top_n", "rulebook", 0))},
        "rules": [],
    }
    seen = set()
    for i, r in enumerate(raw["rules"], 1):
        if not isinstance(r, dict):
            raise RulebookError(f"Rule {i} is not a set of fields.")
        rid = str(r.get("id", "")).strip()
        if not rid:
            raise RulebookError(f"Rule {i} has no id.")
        if rid in seen:
            raise RulebookError(f"Two rules are called {rid}.")
        seen.add(rid)

        scope = r.get("applies_to") or {}
        types = scope.get("payment_types", "all")
        if types == "all":
            types = list(schema.PAYMENT_TYPES)
        if isinstance(types, str):
            types = [types]
        unknown = sorted(set(types) - set(schema.PAYMENT_TYPES))
        if unknown:
            raise RulebookError(f"{rid}: unknown payment types {unknown}. "
                                f"Allowed: {schema.PAYMENT_TYPES} or 'all'.")
        direction = scope.get("direction", "any")
        if direction not in DIRECTIONS:
            raise RulebookError(f"{rid}: direction must be in, out or any, got {direction!r}.")

        t = dict(r.get("trigger") or {})
        ttype = t.get("type", "single")
        if ttype not in TRIGGERS:
            raise RulebookError(f"{rid}: trigger type must be one of {sorted(TRIGGERS)}, got {ttype!r}.")
        countries = t.get("countries")
        if countries == "high_risk":
            countries = rb["high_risk_countries"]
        elif isinstance(countries, str):
            countries = [countries]
        trig = {
            "type": ttype,
            "amount_min": _num(t.get("amount_min"), "amount_min", rid, 0),
            "amount_max": _num(t.get("amount_max"), "amount_max", rid, 0),
            "countries": countries or None,
            "window_days": _num(t.get("window_days"), "window_days", rid, 1),
            "min_count": _num(t.get("min_count"), "min_count", rid, 1),
            "min_distinct_counterparties": _num(t.get("min_distinct_counterparties"),
                                                "min_distinct_counterparties", rid, 1),
            "min_total": _num(t.get("min_total"), "min_total", rid, 0),
            "max_total": _num(t.get("max_total"), "max_total", rid, 0),
        }
        if ttype in ("repeated", "velocity", "aggregate") and trig["window_days"] is None:
            raise RulebookError(f"{rid}: a {ttype} trigger needs window_days.")
        if ttype in ("repeated", "velocity") and trig["min_count"] is None:
            raise RulebookError(f"{rid}: a {ttype} trigger needs min_count.")
        if ttype == "aggregate" and not any(trig[k] is not None for k in
                                            ("min_count", "min_distinct_counterparties", "min_total")):
            raise RulebookError(f"{rid}: an aggregate trigger needs min_count, "
                                f"min_distinct_counterparties or min_total.")
        if (trig["amount_min"] is not None and trig["amount_max"] is not None
                and trig["amount_min"] > trig["amount_max"]):
            raise RulebookError(f"{rid}: amount_min is bigger than amount_max.")

        rb["rules"].append({
            "id": rid,
            "description": str(r.get("description", rid)),
            "enabled": bool(r.get("enabled", True)),
            "points": int(_num(r.get("points", 1), "points", rid, 0)),
            "payment_types": types,
            "direction": direction,
            "trigger": trig,
        })
    return rb


def load(path: Path | str = DEFAULT_RULEBOOK) -> dict:
    import yaml
    path = Path(path)
    if not path.exists():
        raise RulebookError(f"Rulebook not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise RulebookError(f"{path.name} isn't valid YAML: {e}") from None
    return validate(raw)


def kind(rule: dict) -> str:
    """What sort of thing a rule looks at; the coverage map uses this."""
    t = rule["trigger"]
    if t["type"] == "velocity":
        return "count"
    if t["type"] == "aggregate":
        return "pattern"
    if t["countries"]:
        return "country"
    return "amount"


def catalogue(rb: dict) -> pd.DataFrame:
    """The enabled rules as the table Blindspot's checks read (rules.parquet)."""
    return pd.DataFrame([
        {"rule_id": r["id"], "points": r["points"], "payment_types": r["payment_types"],
         "direction": r["direction"], "kind": kind(r), "description": r["description"]}
        for r in rb["rules"] if r["enabled"]
    ], columns=["rule_id", "points", "payment_types", "direction", "kind", "description"])


def used(data_dir: Path) -> dict | None:
    """The rulebook the engine last ran with (copied into the data folder)."""
    p = Path(data_dir) / "rulebook_used.yaml"
    return load(p) if p.exists() else None
