"""
Blindspot: an independent audit of a bank's AML transaction monitoring.

The dashboard reads the engine's output files. It does not run the pipeline
unless RUN_COMMAND is set below, so the demo can't break live.

    pip install streamlit plotly pandas pyarrow pyyaml
    streamlit run app.py

Data folder: ./data, written by the pipeline (python -m blindspot.run_all).
Override with:  BLINDSPOT_DATA=path/to/folder streamlit run app.py
"""
import html
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from blindspot import rulebook as rbk

# ----------------------------------------------------------------- settings
ROOT = Path(__file__).parent
DATA_DIR = Path(os.environ.get("BLINDSPOT_DATA", ROOT / "data"))
RULEBOOK_DIR = ROOT / "rulebooks"
ACTIVE_RULEBOOK = RULEBOOK_DIR / "active.yaml"   # what Run audit uses; written by Save

# Run audit re-runs the pipeline from the engine onwards with the saved rulebook.
# Set BLINDSPOT_READONLY=1 to disable it (e.g. on a laptop without the transactions).
READ_ONLY = os.environ.get("BLINDSPOT_READONLY") == "1"
RUN_COMMAND = [sys.executable, "-m", "blindspot.run_all", "--from", "engine",
               "--data", str(DATA_DIR), "--rules", str(ACTIVE_RULEBOOK)]

# One sentence explaining the low SAML-D result, shown on Method & limits.
# Replace with the engine team's actual explanation once they've checked.
SAMLD_REASON = ("SAML-D's laundering is mostly networks of accounts (fan-ins, cycles, layering), and every "
                "check today looks at one customer at a time. A network check is the next module.")

# ----------------------------------------------------------------- look
INK, MUTED, LINE, BG, CARD = "#15171C", "#5F6673", "#E3E1DC", "#F6F5F2", "#FFFFFF"
ACCENT, ACCENT_SOFT = "#0B6E5F", "#D7EBE6"          # Blindspot
BANK = "#8A93A3"                                      # the bank's own system
MISSED = "#D9D6CF"
HIGH, MEDIUM = "#B42318", "#B5650B"                   # severity only

st.set_page_config(page_title="Blindspot", page_icon="◐", layout="wide")
st.markdown(f"""
<style>
:root {{ --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
.stApp {{ background: {BG}; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
.bs-eyebrow {{ font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; margin-bottom: 4px; }}
.bs-stat {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 8px; padding: 20px 22px; height: 100%; }}
.bs-stat .v {{ font-family: var(--mono); font-size: 54px; line-height: 1; font-weight: 600; color: {INK}; }}
.bs-stat.accent {{ border-color: {ACCENT}; box-shadow: inset 0 3px 0 {ACCENT}; }}
.bs-stat.accent .v {{ color: {ACCENT}; }}
.bs-stat .l {{ font-size: 14px; color: {MUTED}; margin-top: 10px; }}
.bs-card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 8px; padding: 16px 18px; margin-bottom: 12px; }}
.bs-card.high {{ border-left: 4px solid {HIGH}; }}
.bs-card.medium {{ border-left: 4px solid {MEDIUM}; }}
.bs-card .t {{ font-weight: 600; font-size: 16px; color: {INK}; }}
.bs-card .m {{ font-family: var(--mono); font-size: 13px; color: {MUTED}; margin: 2px 0 8px; }}
.bs-card .fix {{ background: {ACCENT_SOFT}; border-radius: 6px; padding: 8px 10px; margin-top: 8px; font-size: 14px; }}
.bs-pill {{ display: inline-block; font-size: 11px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase;
           padding: 2px 8px; border-radius: 99px; margin-right: 6px; }}
.bs-pill.high {{ background: #FBE9E7; color: {HIGH}; }}
.bs-pill.medium {{ background: #FDF0E1; color: {MEDIUM}; }}
.bs-pill.check {{ background: #ECEAE5; color: {MUTED}; }}
.bs-tag {{ display: inline-block; font-size: 12px; background: #ECEAE5; color: {INK}; border-radius: 4px;
          padding: 2px 7px; margin: 2px 4px 2px 0; }}
.bs-sample {{ border: 1px dashed {MUTED}; border-radius: 6px; padding: 6px 12px; font-size: 13px; color: {MUTED};
             background: {CARD}; margin-bottom: 12px; }}
.bs-foot {{ font-size: 12px; color: {MUTED}; border-top: 1px solid {LINE}; margin-top: 32px; padding-top: 10px; }}
.bs-quote {{ background: {ACCENT_SOFT}; padding: 1px 3px; border-radius: 3px; }}
.bs-num {{ font-family: var(--mono); }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_data
def load_json(name):
    p = DATA_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_parquet(name):
    p = DATA_DIR / name
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_text(name):
    p = DATA_DIR / name
    return p.read_text() if p.exists() else None


# The app opens empty. Results appear only after an audit has been run in this
# browser session (or the user deliberately loads the last saved one).
summary = load_json("summary.json") or {}
HAS_SAVED_RESULTS = bool(summary)
SHOW_RESULTS = HAS_SAVED_RESULTS and st.session_state.get("audit_ran", False)

facts = summary.get("facts", {})
val = summary.get("validation_uses_planted_answers") or {}   # absent at a real bank
planted = val.get("planted") or {}
samld = val.get("samld") or summary.get("validation", {}).get("samld")
IS_SAMPLE = summary.get("is_sample", False)


def _rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def gbp(x):
    return f"£{x:,.0f}"


def stat(value, label, accent=False, eyebrow=""):
    eb = f'<div class="bs-eyebrow">{eyebrow}</div>' if eyebrow else ""
    return f'<div class="bs-stat {"accent" if accent else ""}">{eb}<div class="v">{value}</div><div class="l">{label}</div></div>'


def header(eyebrow, title, point=None):
    if IS_SAMPLE:
        st.markdown('<div class="bs-sample">SAMPLE DATA: illustrative files, '
                    'not a pipeline run. Run python -m blindspot.run_all before the demo.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bs-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if point:
        st.markdown(f"<p style='font-size:17px;color:{MUTED};margin-top:-6px'>{point}</p>", unsafe_allow_html=True)


def footer():
    st.markdown(
        f'<div class="bs-foot">Source: {html.escape(summary.get("dataset", "synthetic dataset"))} · '
        f'files in <span class="bs-num">{html.escape(_rel(DATA_DIR))}</span> · '
        f'run {html.escape(str(summary.get("generated_at", "date unknown")))}. '
        "All transactions are synthetic. Blindspot points to possible gaps; a person reviews every finding.</div>",
        unsafe_allow_html=True)


def show(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=30, b=8), paper_bgcolor=CARD, plot_bgcolor=CARD,
                      font=dict(color=INK, size=13), legend=dict(orientation="h", y=1.12, x=0, traceorder="normal"))
    fig.update_xaxes(gridcolor="#EEECE8", zeroline=False)
    fig.update_yaxes(gridcolor="#EEECE8", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def as_pct(s):
    """Series as 0-100. Engine files store percentages; the old sample stored fractions."""
    s = pd.to_numeric(s, errors="coerce")
    return s * 100 if s.max(skipna=True) <= 1 else s


def pretty(s):
    return str(s).replace("_", " ").capitalize()


# ----------------------------------------------------------------- rulebook
# Rulebooks are YAML files in the format described in blindspot/rulebook.py.
# The table edits the numbers; which payments a rule covers is edited in the file.
EDIT_TRIGGER = ["amount_min", "amount_max", "window_days", "min_count"]


def load_raw(text):
    raw = yaml.safe_load(text)
    rbk.validate(raw)            # raises RulebookError with a readable message
    return raw


def rules_table(raw):
    rows = []
    for r in raw["rules"]:
        a, t = r.get("applies_to") or {}, r.get("trigger") or {}
        types = a.get("payment_types", "all")
        types = "all" if types == "all" else ", ".join(types if isinstance(types, list) else [types])
        rows.append({"enabled": bool(r.get("enabled", True)), "id": r["id"],
                     "description": r.get("description", ""), "points": r.get("points", 1),
                     "type": t.get("type", "single"),
                     **{k: t.get(k) for k in EDIT_TRIGGER},
                     "watches": f"{types} ({a.get('direction', 'any')})"})
    return pd.DataFrame(rows)


def apply_table(raw, table):
    """Copy the table's edits back into the rulebook, keeping everything else."""
    out = yaml.safe_load(yaml.safe_dump(raw))          # deep copy
    by_id = {r["id"]: r for r in table.to_dict(orient="records")}
    for r in out["rules"]:
        row = by_id.get(r["id"])
        if not row:
            continue
        r["enabled"] = bool(row["enabled"])
        r["description"] = row["description"]
        r["points"] = int(row["points"]) if pd.notna(row["points"]) else 0
        trig = r.setdefault("trigger", {})
        for k in EDIT_TRIGGER:
            v = row.get(k)
            if v is None or pd.isna(v):
                trig.pop(k, None)
            else:
                trig[k] = int(v) if k in ("window_days", "min_count") else float(v)
    return out


STEP_LABELS = {
    "plant": "Planting known laundering into the test bank",
    "engine": "Replaying the bank's rules",
    "evaluate": "Scoring the bank against the planted answers",
    "investigate": "Simulating the bank's investigators",
    "coverage": "Check 1: coverage",
    "calibration": "Check 2: rule weights",
    "advisory": "Check 3: regulator alerts (LLM)",
    "below_line": "Check 4: below the line (machine learning)",
    "report": "Writing the report (LLM)",
    "evidence": "Linking the evidence payments",
}


def run_audit(rulebook_path):
    """Run the pipeline from the engine (or from planting, on a fresh test bank),
    showing each step as it happens."""
    start = "engine" if (DATA_DIR / "transactions.parquet").exists() else "plant"
    cmd = [sys.executable, "-m", "blindspot.run_all", "--from", start,
           "--data", str(DATA_DIR), "--rules", str(rulebook_path)]
    with st.status("Starting the audit…", expanded=True) as status:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        log = []
        for line in proc.stdout:
            log.append(line)
            step = line.strip().lower()
            if step in STEP_LABELS:
                st.write(STEP_LABELS[step])
                status.update(label=STEP_LABELS[step] + "…")
        proc.wait()
        if proc.returncode == 0:
            status.update(label="Audit complete", state="complete", expanded=False)
            return True, "".join(log)
        status.update(label="The audit stopped with an error", state="error")
        return False, "".join(log)


def page_setup():
    header("Step 1", "Audit setup",
           "Give Blindspot the bank's monitoring rules. It replays them on a test bank with "
           "known laundering planted in it, then audits what they miss.")
    can_run = not READ_ONLY and ((DATA_DIR / "transactions.parquet").exists()
                                 or (DATA_DIR / "base" / "transactions.parquet").exists())

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("#### 1. The bank's rulebook")
        up = st.file_uploader("Upload the bank's rulebook (YAML)", type=["yaml", "yml"])
        files = {f.name: f for f in sorted(RULEBOOK_DIR.glob("*.yaml")) if f.name != ACTIVE_RULEBOOK.name}
        pick = st.selectbox("…or use one of the example banks", list(files), index=None,
                            placeholder="Choose an example rulebook")
        source = (up.name, up.getvalue().decode("utf-8")) if up else \
                 (pick, files[pick].read_text()) if pick else None
        if source and st.session_state.get("rb_source") != source[0]:
            try:
                st.session_state.rb_raw = load_raw(source[1])
                st.session_state.rb_source = source[0]
                st.session_state.pop("rule_editor", None)
            except (rbk.RulebookError, yaml.YAMLError) as e:
                st.session_state.pop("rb_raw", None)
                st.session_state.rb_source = None
                st.error(f"Couldn't use {source[0]}: {e}")
        raw = st.session_state.get("rb_raw") if source else None

        new_raw = None
        if raw is None:
            st.info("Upload the bank's rulebook, or choose an example, to begin.")
        else:
            st.markdown(f"#### 2. Check the rules · {html.escape(str(raw.get('bank', 'Bank')))}")
            k1, k2 = st.columns(2)
            threshold = k1.number_input("Points to open a case", min_value=1,
                                        value=int(raw.get("case_threshold", 20)))
            top_n = k2.number_input("Valued clients exempt from cases", min_value=0,
                                    value=int((raw.get("suppression") or {}).get("valued_clients_top_n", 0)))
            edited = st.data_editor(
                rules_table(raw), num_rows="fixed", width="stretch", key="rule_editor", hide_index=True,
                disabled=["id", "type", "watches"],
                column_config={
                    "enabled": st.column_config.CheckboxColumn("On"),
                    "id": "Rule", "description": st.column_config.TextColumn("What it looks for", width="medium"),
                    "points": st.column_config.NumberColumn("Points", min_value=0, step=1),
                    "type": "Trigger",
                    "amount_min": st.column_config.NumberColumn("Min £", min_value=0, format="£%,.0f"),
                    "amount_max": st.column_config.NumberColumn("Max £", min_value=0, format="£%,.0f"),
                    "window_days": st.column_config.NumberColumn("Window (days)", min_value=1, step=1),
                    "min_count": st.column_config.NumberColumn("Min count", min_value=1, step=1),
                    "watches": st.column_config.TextColumn("Payments it can see", width="medium"),
                })
            st.caption("You can switch rules off and change their numbers here. To change which "
                       "payments a rule can see, or add a rule, edit the YAML file and upload it again.")
            new_raw = apply_table(raw, edited)
            new_raw["case_threshold"] = int(threshold)
            new_raw.setdefault("suppression", {})["valued_clients_top_n"] = int(top_n)
            st.download_button("Download this rulebook (YAML)",
                               yaml.safe_dump(new_raw, sort_keys=False, allow_unicode=True),
                               file_name="bank_rulebook.yaml", mime="application/x-yaml")

    with c2:
        st.markdown("#### 3. Run the audit")
        st.markdown(
            '<div class="bs-card"><div class="t">Test bank</div>'
            '<div class="m">SAML-D (Bournemouth University) with known laundering planted in it</div>'
            'Every catch and miss can be checked against the answer key. No real customer data is used.'
            '</div>', unsafe_allow_html=True)
        if not can_run:
            st.button("Run audit", type="primary", width="stretch", disabled=True)
            st.caption("Running is switched off (BLINDSPOT_READONLY is set)." if READ_ONLY else
                       "There's no test bank in the data folder yet. Run `python -m blindspot.normalise` "
                       "first (see README).")
        elif st.button("Run audit", type="primary", width="stretch", disabled=new_raw is None):
            try:
                rbk.validate(new_raw)
            except rbk.RulebookError as e:
                st.error(f"Can't run: {e}")
            else:
                RULEBOOK_DIR.mkdir(exist_ok=True)
                ACTIVE_RULEBOOK.write_text(yaml.safe_dump(new_raw, sort_keys=False, allow_unicode=True))
                ok, log = run_audit(ACTIVE_RULEBOOK)
                if ok:
                    st.cache_data.clear()
                    st.session_state.audit_ran = True
                    st.session_state.just_ran = True
                    st.rerun()
                else:
                    st.code(log[-3000:])
        if new_raw is None and can_run:
            st.caption("Choose a rulebook first. The audit takes a minute or two.")

        if st.session_state.pop("just_ran", False) or SHOW_RESULTS:
            st.success(f"Audit complete: {facts.get('rulebook', 'rulebook')}. "
                       f"{facts.get('findings_total', 0)} findings.")
            st.page_link(OVERVIEW, label="See the results →")
        elif HAS_SAVED_RESULTS:
            st.divider()
            st.caption(f"No time to run live? The last audit ({facts.get('rulebook', 'unknown rulebook')}, "
                       f"{summary.get('generated_at', 'date unknown')}) is saved.")
            if st.button("Show the last saved audit", width="stretch"):
                st.session_state.audit_ran = True
                st.rerun()
    footer()


def empty_state():
    header("Results", "No audit yet")
    st.markdown('<div class="bs-card" style="padding:28px">Blindspot hasn\'t audited a rulebook yet. '
                'Give it the bank\'s rules on <b>Audit setup</b>, press <b>Run audit</b>, and the results '
                'appear here.</div>', unsafe_allow_html=True)
    st.page_link(SETUP, label="Go to Audit setup →")


def needs_audit(page):
    """Wrap a results page so it shows the empty state until an audit has run."""
    def wrapped():
        if SHOW_RESULTS:
            page()
        else:
            empty_state()
    wrapped.__name__ = page.__name__
    return wrapped


# ----------------------------------------------------------------- pages
def page_overview():
    header("Results", "What the bank's rules missed",
           f"Rulebook audited: {facts.get('rulebook', 'unknown')}. Validated on laundering planted "
           "in synthetic data, so we know every right answer.")
    bank_pct = planted.get("bank_pct", 0)
    comb_pct = planted.get("combined_pct", 0)
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.markdown(stat(f"{bank_pct:.0f}%", f"of planted laundering caught by the bank's rules "
                     f"({planted.get('caught_by_bank', 0)} of {planted.get('caught_by_bank', 0) + planted.get('missed_by_bank', 0)})",
                     eyebrow="Bank alone"), unsafe_allow_html=True)
    c2.markdown(stat(f"{comb_pct:.0f}%", f"caught with Blindspot's review list added "
                     f"(+{planted.get('found_by_blindspot', 0)} customers)", accent=True, eyebrow="With Blindspot"),
                unsafe_allow_html=True)
    qp, br = val.get("queue_precision_pct"), val.get("base_rate_pct")
    if qp and br:
        c3.markdown(stat(f"{qp:.0f}%", f"of Blindspot's {facts.get('review_queue_customers', 0)}-customer review list is laundering, "
                         f"vs {br:.2f}% if you checked customers at random ({qp / br:.0f}× better)",
                         eyebrow="Review list"), unsafe_allow_html=True)

    by = planted.get("by_typology", {})
    if by:
        st.markdown("#### By laundering pattern")
        names = [pretty(k) for k in by]
        bank = [v["bank"] for v in by.values()]
        found = [v["blindspot_found"] for v in by.values()]
        missed = [v["customers"] - v["bank"] - v["blindspot_found"] for v in by.values()]
        fig = go.Figure()
        fig.add_bar(y=names, x=bank, name="Caught by bank", orientation="h", marker_color=BANK)
        fig.add_bar(y=names, x=found, name="Found by Blindspot", orientation="h", marker_color=ACCENT)
        fig.add_bar(y=names, x=missed, name="Still missed", orientation="h", marker_color=MISSED)
        fig.update_layout(barmode="stack", xaxis_title="Customers")
        fig.update_yaxes(autorange="reversed")
        show(fig, 320)
        worst = min(by.items(), key=lambda kv: kv[1]["bank"] + kv[1]["blindspot_found"])
        st.caption(f"Hardest pattern: {pretty(worst[0])}. Blindspot and the bank together found "
                   f"{worst[1]['bank'] + worst[1]['blindspot_found']} of {worst[1]['customers']}. "
                   "See Method & limits.")

    st.markdown("#### Summary for the board")
    st.markdown(f'<div class="bs-card">{html.escape(summary.get("executive_summary", ""))}</div>', unsafe_allow_html=True)
    st.caption("Written by the language model from numbers the code computed. It doesn't calculate anything itself.")
    footer()


def page_findings():
    header("Results", "Findings", "Each gap, the evidence for it, and a suggested fix. Most severe first.")
    df = load_parquet("findings.parquet")
    if df is None:
        st.info("findings.parquet not found.")
        return footer()
    checks = ["All"] + sorted(df["check"].unique())
    c1, c2 = st.columns([3, 1])
    pick = c1.radio("Check", checks, horizontal=True, format_func=pretty, label_visibility="collapsed")
    c2.markdown(f"<div style='text-align:right;color:{MUTED}'>{(df.severity == 'high').sum()} high · "
                f"{(df.severity == 'medium').sum()} medium</div>", unsafe_allow_html=True)
    if pick != "All":
        df = df[df["check"] == pick]
    for _, r in df.iterrows():
        sev = r.severity if r.severity in ("high", "medium") else "medium"
        st.markdown(
            f'<div class="bs-card {sev}"><span class="bs-pill {sev}">{sev}</span>'
            f'<span class="bs-pill check">{pretty(r["check"])}</span>'
            f'<div class="t" style="margin-top:8px">{html.escape(r.title)}</div>'
            f'<div class="m">{html.escape(str(r.metric))}</div>{html.escape(r.plain_english)}'
            f'<div class="fix"><b>Suggested fix:</b> {html.escape(r.suggested_fix)}</div></div>',
            unsafe_allow_html=True)
        ids = list(r.evidence_ids) if r.evidence_ids is not None else []
        if ids:
            with st.expander(f"Evidence ({len(ids)} IDs)"):
                st.markdown(" ".join(f'<span class="bs-tag bs-num">{html.escape(str(i))}</span>' for i in ids),
                            unsafe_allow_html=True)
    footer()


def page_queue():
    header("Results", "Review queue", "Customers the bank never flagged that a person should look at, highest priority first.")
    df = load_parquet("review_queue.parquet")
    if df is None:
        st.info("review_queue.parquet not found.")
        return footer()
    c1, c2, c3 = st.columns(3)
    c1.markdown(stat(f"{len(df)}", "customers to review"), unsafe_allow_html=True)
    c2.markdown(stat(f"{int(df.suppressed.sum())}", "hidden by the valued-client list"), unsafe_allow_html=True)
    if val.get("queue_precision_pct"):
        c3.markdown(stat(f"{val['queue_precision_pct']:.0f}%", "of this list is laundering (planted answers)", accent=True),
                    unsafe_allow_html=True)

    st.markdown("#### Top 5")
    for _, r in df.head(5).iterrows():
        tags = "".join(f'<span class="bs-tag">{html.escape(t.strip())}</span>' for t in str(r.reasons).split("|") if t.strip())
        sup = '<span class="bs-pill high">suppressed</span>' if r.suppressed else ""
        st.markdown(f'<div class="bs-card"><span class="bs-num" style="color:{MUTED}">#{r.queue_rank}</span> '
                    f'<span class="t bs-num">{html.escape(str(r.customer_id))}</span> {sup}'
                    f'<div class="m">priority {r.priority} · flagged by {r.checks} check(s) · '
                    f'unusualness {r.anomaly_score:.0f}/100 · bank score {r.engine_score}'
                    + (f' · {int(r.evidence_txns)} evidence payments, '
                       f'{r.unalerted_pct:.0f}% never alerted' if "evidence_txns" in df else "")
                    + f'</div>{tags}</div>',
                    unsafe_allow_html=True)

    st.markdown("#### Full list")
    only_sup = st.toggle("Only show suppressed customers")
    view = df[df.suppressed] if only_sup else df
    view = view.assign(reasons=view.reasons.str.replace(" | ", " · ", regex=False))
    st.dataframe(
        view.style.apply(lambda row: ["background-color:#FBE9E7" if row.suppressed else "" for _ in row], axis=1),
        width="stretch", hide_index=True, height=420,
        column_config={"queue_rank": "Rank", "customer_id": "Customer", "priority": "Priority", "checks": "Checks",
                       "anomaly_score": st.column_config.ProgressColumn("Unusualness", min_value=0, max_value=100, format="%d"),
                       "engine_score": "Bank score", "suppressed": "Suppressed", "reasons": st.column_config.TextColumn("Why", width="large"),
                       "evidence_txns": "Evidence payments",
                       "evidence_gbp": st.column_config.NumberColumn("Evidence £", format="£%,.0f"),
                       "unalerted_pct": st.column_config.NumberColumn("Never alerted", format="%.0f%%"),
                       "top_evidence": st.column_config.TextColumn("Strongest evidence", width="large")})
    st.download_button("Download CSV", df.to_csv(index=False), file_name="blindspot_review_queue.csv", mime="text/csv")
    evidence_drilldown(df)
    footer()


def evidence_drilldown(queue):
    """One customer's ranked evidence payments, from evidence.parquet."""
    ev = load_parquet("evidence.parquet")
    st.markdown("#### Evidence: the payments behind each customer")
    if ev is None:
        st.info("evidence.parquet not found. Run `python -m blindspot.evidence` after the report.")
        return
    labels = {r.customer_id: f"#{r.queue_rank} · {r.customer_id}" + (" · suppressed" if r.suppressed else "")
              for r in queue.itertuples()}
    cust = st.selectbox("Customer", list(labels), format_func=labels.get)
    e = ev[ev.customer_id == cust].sort_values("evidence_rank")
    if e.empty:
        st.info("No linked payments for this customer.")
        return
    q = queue[queue.customer_id == cust].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.markdown(stat(f"{int(q.get('evidence_txns', len(e)))}", "evidence payments"), unsafe_allow_html=True)
    c2.markdown(stat(gbp(q.get("evidence_gbp", e.amount.sum())), "in those payments"), unsafe_allow_html=True)
    c3.markdown(stat(f"{(~e.bank_alerted).mean() * 100:.0f}%", "never alerted by the bank's rules", accent=True),
                unsafe_allow_html=True)

    fig = go.Figure()
    for alerted, name, colour in [(False, "Never alerted", ACCENT), (True, "Alerted by a bank rule", BANK)]:
        d = e[e.bank_alerted == alerted]
        if len(d):
            fig.add_scatter(x=d.timestamp, y=d.amount, mode="markers", name=name,
                            marker=dict(color=colour, size=10, line=dict(color=CARD, width=1)),
                            customdata=d[["payment_type", "why"]].values,
                            hovertemplate="%{x|%d %b %Y} · £%{y:,.0f} · %{customdata[0]}<br>"
                                          "%{customdata[1]}<extra></extra>")
    fig.update_layout(yaxis_title="Amount (£)")
    fig.update_yaxes(tickprefix="£", tickformat="~s")
    show(fig, 300)

    view = e.assign(timestamp=e.timestamp.dt.strftime("%d %b %Y"),
                    payment_type=e.payment_type.astype(str).map(pretty),
                    why=e.why.str.replace(" | ", " · ", regex=False))
    cols = ["evidence_rank", "timestamp", "direction", "payment_type", "amount", "counterparty",
            "counterparty_country", "why", "bank_alerted", "bank_rules"]
    st.dataframe(
        view[cols].style.apply(lambda row: ["" if row.bank_alerted else f"background-color:{ACCENT_SOFT}"
                                            for _ in row], axis=1),
        width="stretch", hide_index=True, height=420,
        column_config={"evidence_rank": "Rank", "timestamp": "Date", "direction": "In/out",
                       "payment_type": "Type", "amount": st.column_config.NumberColumn("Amount", format="£%,.0f"),
                       "counterparty": "Counterparty", "counterparty_country": "Country",
                       "why": st.column_config.TextColumn("Why it matters", width="large"),
                       "bank_alerted": "Bank alerted", "bank_rules": "Bank rules fired"})
    st.caption("Highlighted rows: payments the bank's rules never alerted on. "
               "Ranked by how many checks point at each payment, then by size.")


def page_coverage():
    pct = facts.get("pct_money_watched_by_amount_or_country_rule")
    header("Check 1", "Coverage", f"Only {pct}% of money is watched by a rule that checks amounts, countries or patterns." if pct else None)
    df = load_parquet("coverage.parquet")
    if df is None:
        st.info("coverage.parquet not found.")
        return footer()
    colour = {"unmonitored": HIGH, "count-only": MEDIUM, "monitored": BANK}
    order = {"unmonitored": 0, "count-only": 1, "monitored": 2}
    df = df.assign(_o=df.status.map(order)).sort_values(["_o", "money_gbp"], ascending=[True, False]).drop(columns="_o")
    fig = go.Figure()
    for status in ["unmonitored", "count-only", "monitored"]:
        d = df[df.status == status]
        if len(d):
            fig.add_bar(y=d.label, x=d.money_gbp, orientation="h", name=pretty(status), marker_color=colour[status],
                        customdata=d.transactions, hovertemplate="%{y}<br>£%{x:,.0f} · %{customdata:,} transactions<extra></extra>")
    fig.update_layout(barmode="stack", xaxis_title="Money (£)")
    fig.update_yaxes(categoryorder="array", categoryarray=list(df.label), autorange="reversed")
    fig.update_xaxes(tickprefix="£", tickformat="~s")
    show(fig, 360)
    un = df[df.status == "unmonitored"]
    if len(un):
        st.caption("No rule can fire on: " + ", ".join(f"{r.label} ({gbp(r.money_gbp)}, {r.transactions:,} transactions)"
                                                     for r in un.itertuples()))
    with st.expander("Table"):
        st.dataframe(df.assign(money_gbp=df.money_gbp.apply(gbp), share_of_money=as_pct(df.share_of_money).round(1),
                               rules=df.rules.apply(lambda x: ", ".join(x) if len(x) else "none")),
                     width="stretch", hide_index=True,
                     column_config={"label": "Payment type", "status": "Status", "money_gbp": "Money",
                                    "transactions": "Transactions", "rules": "Rules that can fire",
                                    "share_of_money": st.column_config.NumberColumn("Share", format="%.1f%%")})
    footer()


def page_calibration():
    header("Check 2", "Rule weights", "Rules that are worth a lot but rarely right fill the queue with false alarms; "
           "strong rules worth little let crime through.")
    df = load_parquet("calibration.parquet")
    if df is None:
        st.info("calibration.parquet not found.")
        return footer()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Points now vs suggested**")
        fig = go.Figure()
        for r in df.itertuples():
            fig.add_shape(type="line", x0=r.points, x1=r.suggested_points, y0=r.rule_id, y1=r.rule_id, line=dict(color=LINE, width=4))
        fig.add_scatter(x=df.points, y=df.rule_id, mode="markers", name="Now", marker=dict(color=BANK, size=13))
        fig.add_scatter(x=df.suggested_points, y=df.rule_id, mode="markers", name="Suggested", marker=dict(color=ACCENT, size=13))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="Points")
        show(fig, 340)
    with c2:
        st.markdown("**How often each rule is right** (with range)")
        fig = go.Figure(go.Scatter(
            x=df.hit_rate * 100, y=df.rule_id, mode="markers", marker=dict(color=INK, size=11),
            error_x=dict(type="data", symmetric=False, array=(df.ci_high - df.hit_rate) * 100,
                         arrayminus=(df.hit_rate - df.ci_low) * 100, color=MUTED), showlegend=False))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="Right when it fires (%)")
        show(fig, 340)
    st.dataframe(df.assign(hit_rate=(df.hit_rate * 100).round(1), ci_low=(df.ci_low * 100).round(1),
                           ci_high=(df.ci_high * 100).round(1)),
                 width="stretch", hide_index=True)
    footer()


_DEFAULTS = {"direction": "any", "segment": "any", "payment_types": [], "countries": [], "amount_min": 0,
             "amount_max": 1e12, "min_count": 1, "min_distinct_counterparties": 1, "min_total": 0,
             "max_total": 1e15, "min_windows": 1}
_TERMS = [("direction", "Direction"), ("payment_types", "Payment types"), ("segment", "Customer"),
          ("countries", "Countries"), ("amount_min", "Each at least"), ("amount_max", "Each at most"),
          ("window_days", "Window (days)"), ("min_count", "At least (payments)"),
          ("min_distinct_counterparties", "Different counterparties"), ("min_total", "Window total at least"),
          ("max_total", "Window total at most"), ("min_windows", "Repeated in windows")]


def pattern_terms(p):
    """Readable (label, value) pairs for an extracted pattern, skipping defaults."""
    if p.get("thresholds"):
        return [(pretty(k), v) for k, v in p["thresholds"].items()]
    out = []
    for key, label in _TERMS:
        v = p.get(key)
        if v is None or v == _DEFAULTS.get(key) or v == []:
            continue
        if isinstance(v, list):
            v = ", ".join(map(str, v))
        elif key in ("amount_min", "amount_max", "min_total", "max_total"):
            v = gbp(v)
        out.append((label, v))
    return out


def page_advisory():
    header("Check 3 · AI", "Regulator alerts",
           "The language model reads an alert, turns each red flag into a testable pattern, and code checks if any rule catches it.")
    pats = load_json("advisory_patterns.json") or {}
    src = pats.get("source") or pats.get("advisory") or ""
    text = None
    for base in (ROOT, DATA_DIR):            # engine stores a project-relative path
        p = base / src
        if src and p.is_file():
            text = p.read_text()
            break
    if text is None:
        mds = sorted((DATA_DIR / "advisories").glob("*.md")) if (DATA_DIR / "advisories").exists() else []
        text = mds[0].read_text() if mds else None

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**The alert, as published**")
        if text:
            body = html.escape(text)
            for p in pats.get("patterns", []):
                q = html.escape(p.get("red_flag_quote", ""))
                if q:
                    body = body.replace(q, f'<span class="bs-quote">{q}</span>')
            st.markdown(f'<div class="bs-card" style="white-space:pre-wrap">{body}</div>', unsafe_allow_html=True)
        else:
            st.info("No advisory text found.")
    with c2:
        st.markdown("**What the AI extracted**")
        for p in pats.get("patterns", []):
            th = " · ".join(f"{html.escape(k)}: <span class='bs-num'>{html.escape(str(v))}</span>"
                            for k, v in pattern_terms(p))
            a = p.get("assumptions") or []
            a = [a] if isinstance(a, str) else a
            assumptions = "".join(f"<li>{html.escape(str(x))}</li>" for x in a) or "<li>none stated</li>"
            st.markdown(f'<div class="bs-card"><div class="t">{html.escape(p.get("name", ""))}</div>'
                        f'<div style="margin:6px 0"><span class="bs-quote">“{html.escape(p.get("red_flag_quote", ""))}”</span></div>'
                        f'<div class="m">{th}</div>'
                        f'<div style="font-size:13px;color:{MUTED}">Assumptions the AI made, for a person to check:<ul>{assumptions}</ul></div></div>',
                        unsafe_allow_html=True)

    adv = load_parquet("advisory.parquet")
    if adv is not None:
        st.markdown("#### Would the bank's rules catch it?")
        cols = ["name", "customers", "with_case", "case_rate", "money_gbp", "rules_in_scope"]
        st.dataframe(adv[cols].assign(case_rate=as_pct(adv.case_rate).round(1), money_gbp=adv.money_gbp.apply(gbp),
                                      rules_in_scope=adv.rules_in_scope.apply(lambda x: ", ".join(x) if len(x) else "none")),
                     width="stretch", hide_index=True,
                     column_config={"name": "Pattern", "customers": "Matching customers", "with_case": "With a case",
                                    "case_rate": st.column_config.NumberColumn("Caught", format="%.1f%%"),
                                    "money_gbp": "Money", "rules_in_scope": "Rules that could fire"})
    if pats.get("untestable"):
        st.markdown("#### Not testable from transactions: needs a person")
        for u in pats["untestable"]:
            if isinstance(u, dict):
                st.markdown(f"- **{u.get('red_flag', '')}**: {u.get('reason', '')}")
            else:
                st.markdown(f"- {u}")
    footer()


def page_below_line():
    header("Check 4 · AI", "Below the line",
           "A model ranks every customer the bank never investigated by how unusual they look. It never sees the answers.")
    df = load_parquet("below_line.parquet")
    if df is None:
        st.info("below_line.parquet not found.")
        return footer()
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        fig = go.Figure(go.Histogram(x=df.anomaly_score, nbinsx=30, marker_color=BANK))
        top = df.head(50).anomaly_score.min()
        fig.add_vline(x=top, line=dict(color=ACCENT, width=2, dash="dash"), annotation_text="top 50", annotation_position="top")
        fig.update_layout(xaxis_title="Unusualness (0–100)", yaxis_title="Customers", showlegend=False)
        show(fig, 360)
        st.caption(f"{len(df):,} customers with no case. The top 50 go to the review queue with reasons.")
    with c2:
        st.dataframe(df.head(50), width="stretch", hide_index=True, height=360,
                     column_config={"rank": "Rank", "customer_id": "Customer",
                                    "anomaly_score": st.column_config.ProgressColumn("Unusualness", min_value=0, max_value=100, format="%.0f"),
                                    "engine_score": "Bank score", "suppressed": "Suppressed",
                                    "reasons": st.column_config.TextColumn("Why", width="large")})
    footer()


def page_method():
    header("Trust", "Method & limits", "What Blindspot does, how we tested it, and where it falls short.")
    st.markdown("""
1. **Test bank.** A synthetic bank with known laundering planted in it, so every catch and miss can be checked.
2. **The bank's rules** run over it exactly as configured.
3. **Blindspot's four checks** look for gaps: coverage, rule weights, regulator alerts (AI) and below the line (AI).
4. **A person decides.** Blindspot never labels anyone a criminal. It produces findings and a review list for an analyst.
""")
    st.markdown("#### Test results")
    if not SHOW_RESULTS:
        st.caption("Run an audit on Audit setup to see how Blindspot scored on the test bank.")
    rows = []
    for name, v in [("Planted patterns (we designed these)", planted), ("SAML-D labels (independent, we did not design these)", samld)]:
        if v and SHOW_RESULTS:
            total = v.get("caught_by_bank", 0) + v.get("missed_by_bank", 0)
            rows.append({"Test": name, "Known laundering": total, "Bank caught": f"{v.get('bank_pct', 0):.1f}%",
                         "Blindspot found of the missed": f"{v.get('found_by_blindspot', 0)} of {v.get('missed_by_bank', 0)} "
                                                          f"({v.get('blindspot_pct_of_missed', 0):.1f}%)",
                         "Together": f"{v.get('combined_pct', 0):.1f}%"})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown("#### Known limits")
    samld_line = ""
    if samld and SHOW_RESULTS:
        samld_line = (f"- **On the independent SAML-D labels, Blindspot adds little:** it found "
                      f"{samld.get('found_by_blindspot', 0)} of the {samld.get('missed_by_bank', 0)} cases the bank missed. "
                      f"{SAMLD_REASON}\n")
    st.markdown("""
- **Synthetic data only contains patterns someone imagined.** Results show the method works, not how a real bank would score.
""" + samld_line + """- **Some patterns stay hidden.** Slow, spread-out activity is built to look normal, and the model rarely finds it.
- **False alarms.** Most of the review list is not laundering, so it costs analyst time. It is still far better than random checking.
- **The AI's reading of an alert involves assumptions** (shown on the Regulator alerts page). A person should check them before any rule is changed.

**Next step:** a pilot run inside a bank's own environment on anonymised data, with no data leaving the bank.
""")
    footer()


# ----------------------------------------------------------------- nav
# Navigation bar across the top, grouped. Opens on Audit setup, empty.
SETUP = st.Page(page_setup, title="Audit setup", url_path="setup", default=True)
OVERVIEW = st.Page(needs_audit(page_overview), title="Overview", url_path="overview")
SECTIONS = {
    "Setup": [SETUP],
    "Results": [OVERVIEW,
                st.Page(needs_audit(page_findings), title="Findings", url_path="findings"),
                st.Page(needs_audit(page_queue), title="Review queue", url_path="review-queue")],
    "The four checks": [
        st.Page(needs_audit(page_coverage), title="1 · Coverage", url_path="coverage"),
        st.Page(needs_audit(page_calibration), title="2 · Rule weights", url_path="rule-weights"),
        st.Page(needs_audit(page_advisory), title="3 · Regulator alerts", url_path="regulator-alerts"),
        st.Page(needs_audit(page_below_line), title="4 · Below the line", url_path="below-the-line")],
    "About": [st.Page(page_method, title="Method & limits", url_path="method")],
}

# Brand and status line, shown above every page.
if SHOW_RESULTS:
    status = " · ".join(x for x in [
        f"Auditing {facts['rulebook']}" if facts.get("rulebook") else "",
        f"{facts.get('findings_total', 0)} findings ({facts.get('findings_high', 0)} high)",
        "SAMPLE DATA" if IS_SAMPLE else "",
    ] if x)
else:
    status = "No audit yet"
st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
            f"border-bottom:1px solid {LINE};padding-bottom:6px;margin-bottom:18px'>"
            f"<span style='font-size:20px;font-weight:700;letter-spacing:-0.02em;color:{INK}'>◐ Blindspot</span>"
            f"<span style='font-size:13px;color:{MUTED}'>{html.escape(status)}</span></div>",
            unsafe_allow_html=True)

_test_page = os.environ.get("BLINDSPOT_TEST_PAGE")   # used by automated tests only
if _test_page:
    if os.environ.get("BLINDSPOT_TEST_SHOW") == "1":
        st.session_state.audit_ran = True
        SHOW_RESULTS = HAS_SAVED_RESULTS
    {"Audit setup": page_setup, "Overview": needs_audit(page_overview),
     "Findings": needs_audit(page_findings), "Review queue": needs_audit(page_queue),
     "1 · Coverage": needs_audit(page_coverage), "2 · Rule weights": needs_audit(page_calibration),
     "3 · Regulator alerts": needs_audit(page_advisory),
     "4 · Below the line": needs_audit(page_below_line), "Method & limits": page_method}[_test_page]()
else:
    st.navigation(SECTIONS, position="top").run()
