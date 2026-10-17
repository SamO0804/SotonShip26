"""Blindspot data contract.

Every module reads and writes these four tables. Agree changes as a team
before editing this file: the data, engine and dashboard all depend on it.

    transactions   -> canonical transaction table (SAML-D + planted rows)
    labels         -> ground truth. ONLY Blindspot scoring and the dashboard
                      may read this. The replica rule engine must never see it.
    engine_output  -> what the flawed bank system did (alerts + cases)
    findings       -> what Blindspot reports; the only thing Streamlit renders
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

# Canonical payment types. SAML-D has no money orders; planted rows add them.
# The coverage-map check counts money by these values, so keep them fixed.
PAYMENT_TYPES = [
    "cash_deposit",
    "cash_withdrawal",
    "cheque",
    "money_order",   # not in SAML-D; created by the planting step
    "ach",
    "credit_card",
    "debit_card",
    "cross_border",
    "wire",          # planted Belarus/Georgia wires
]

SEGMENTS = ["individual", "small_business", "unknown"]

LABEL_SOURCES = ["samld", "planted"]

CHECKS = ["coverage", "below_line", "calibration", "advisory"]

SEVERITIES = ["high", "medium", "low"]

# ---------------------------------------------------------------------------
# Table schemas: column -> pandas dtype
# ---------------------------------------------------------------------------

TRANSACTIONS = {
    "txn_id": "string",            # unique; SAML-D rows "S<row>", planted "P<n>"
    "timestamp": "datetime64[ns]",
    "sender_id": "string",         # sender account = customer for monitoring
    "receiver_id": "string",
    "amount": "float64",           # one base currency (see README)
    "payment_type": "category",    # one of PAYMENT_TYPES
    "sender_country": "string",
    "receiver_country": "string",
    "customer_segment": "category",  # one of SEGMENTS; SAML-D rows are "unknown"
}

LABELS = {
    "txn_id": "string",
    "is_suspicious": "bool",
    "source": "category",          # "samld" or "planted"
    "typology": "string",          # e.g. "structured_money_orders"; "normal" if clean
}

# One row per (customer, rule firing). Written by the replica engine.
ALERTS = {
    "customer_id": "string",
    "rule_id": "string",
    "points": "int64",
    "timestamp": "datetime64[ns]",
    "txn_id": "string",            # the transaction that triggered the rule
}

# One row per customer the engine scored.
CASES = {
    "customer_id": "string",
    "score": "int64",              # points inside the rolling 13-month window
    "case_opened": "bool",         # True when score reached CASE_THRESHOLD
    "suppressed": "bool",          # True if on the suppression list
}

FINDINGS = {
    "finding_id": "string",
    "check": "category",           # one of CHECKS
    "title": "string",
    "severity": "category",        # one of SEVERITIES
    "metric": "string",            # the headline number, formatted, e.g. "0% monitored"
    "evidence_ids": "object",      # list[str] of txn_ids or customer_ids
    "suggested_fix": "string",
    "explanation": "string",       # plain English; LLM-written, code-computed numbers only
}

# ---------------------------------------------------------------------------
# Shared constants for the replica and the headline number
# ---------------------------------------------------------------------------

CASE_THRESHOLD = 20       # points needed to open a case
WINDOW_DAYS = 396         # 13 months
SEED = 42                 # fix everywhere for a reproducible demo

# Headline definition (agree as a team; do not change mid-build):
# A suspicious customer counts as CAUGHT if the engine opened a case for them
# (case_opened is True). Measured at customer level, not transaction level.
CAUGHT_DEFINITION = "customer_level_case_opened"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def empty(schema: dict) -> pd.DataFrame:
    """Return an empty DataFrame with the schema's columns and dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in schema.items()})


def conform(df: pd.DataFrame, schema: dict, name: str = "table") -> pd.DataFrame:
    """Check columns, then cast to the schema's dtypes. Raises on mismatch."""
    missing = set(schema) - set(df.columns)
    extra = set(df.columns) - set(schema)
    if missing or extra:
        raise ValueError(f"{name}: missing {sorted(missing)}, unexpected {sorted(extra)}")
    out = df[list(schema)].copy()
    for col, dtype in schema.items():
        if dtype == "object":
            continue
        out[col] = out[col].astype(dtype)
    return out


def validate_transactions(df: pd.DataFrame) -> None:
    bad = set(df["payment_type"].astype(str)) - set(PAYMENT_TYPES)
    if bad:
        raise ValueError(f"unknown payment types: {sorted(bad)}")
    if df["txn_id"].duplicated().any():
        raise ValueError("duplicate txn_id values")


def headline(labels: pd.DataFrame, transactions: pd.DataFrame, cases: pd.DataFrame,
             source: str | None = None) -> dict:
    """The headline number: share of suspicious customers the engine caught.

    Pass source="planted" or "samld" to report on each label set separately.
    """
    lab = labels[labels["is_suspicious"]]
    if source:
        lab = lab[lab["source"] == source]
    sus_customers = set(
        transactions.loc[transactions["txn_id"].isin(lab["txn_id"]), "sender_id"]
    )
    caught = set(cases.loc[cases["case_opened"], "customer_id"]) & sus_customers
    total = len(sus_customers)
    return {
        "suspicious_customers": total,
        "caught": len(caught),
        "caught_pct": round(100 * len(caught) / total, 1) if total else 0.0,
    }
