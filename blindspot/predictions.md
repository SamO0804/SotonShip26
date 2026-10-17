# Predictions, written before testing

These predictions were written **before** `harbourside_building_society.yaml` or
`meridian_pay.yaml` was ever run through Blindspot. The files were only checked
to load. Run each one, then fill in the **Result** lines. Don't edit a
prediction after seeing results: a miss is a finding about Blindspot.

"Bank caught" comes from Overview (the bar chart per pattern). "Blindspot found"
is the green part of the same bar. Findings are on the Findings page.

---

## Harbourside Building Society (`harbourside_building_society.yaml`)

A cautious lender whose rules haven't been reviewed since 2019. Its money-order
and card-funnel rules are good, so those two patterns are **negative controls**:
Blindspot should *not* report gaps there.

### H-1. No structuring rule for cash
**The flaw:** the only cash rule fires at £10,000 or more. Repeated deposits of
£9,000 to £9,900 never trigger anything.
**Bank should miss:** almost all 25 `structured_cash` customers.
**Prediction: Blindspot mostly MISSES this.** Coverage won't flag it, because
cash deposits *are* watched by an amount rule. Calibration can't weigh a rule
that doesn't exist. The advisory doesn't mention cash. Only the ML could
notice, and it was weak on this pattern before (1 of 23).
**Counts as a pass:** Blindspot recovers fewer than a third of the
structured-cash customers, confirming a known blind spot: *a missing scenario on
a payment type that is otherwise watched*.
**Result:**

### H-2. Out-of-date high-risk country list, and a wire threshold set too high
**The flaw:** the list omits Belarus and Georgia, and the large-wire rule starts
at £100,000. The planted wires are £15k–£60k from Belarus and Georgia.
**Bank should miss:** nearly all 25 `high_risk_wires` customers.
**Prediction: Blindspot FINDS these.** The ML uses Blindspot's own, independent
high-risk list and money-from-abroad measures, so these businesses should rank
near the top of Below the line and the review queue. Coverage will *not* flag
it, because wires are watched.
**Counts as a pass:** at least 20 of the 25 wire businesses appear in the
review queue.
**Note:** no check directly says "your country list is out of date". Comparing
the bank's list with a reference list would be an easy new check.
**Result:**

### H-3. A noisy rule worth too many points
**The flaw:** H04 gives 8 points for just 8 outgoing payments in 30 days,
ordinary behaviour for many customers.
**Prediction: Blindspot FINDS this, less confidently.** Rule weights should mark
H04 as over-weighted, with a low hit rate and a suggestion well below 8 points,
and most cases should be false alarms. Less confident because the card-funnel
launderers also make many payments, which would push H04's hit rate up.
**Counts as a pass:** a "Weak rule over-weighted: H04_busy_account" finding.
**Result:**

### Negative controls
- **Money orders and card funnels should be caught by the bank** (most of each 25).
- **Regulator alerts should show no "not caught" findings**, and Coverage should
  not list money orders or card repayments as a gap.
- **If Blindspot reports either as a gap, that's a false alarm by Blindspot.**
**Result:**

---

## Meridian Pay (`meridian_pay.yaml`)

A fintech whose monitoring was built only for money going out.

### M-1. Money coming in is never monitored
**The flaw:** every rule is `direction: out`. Cash deposits, money orders and
incoming wires are invisible.
**Bank should miss:** essentially all `structured_cash`,
`structured_money_orders` and `high_risk_wires` customers.
**Prediction: Blindspot FINDS the gap, but not every customer behind it.**
- Coverage should raise **three high findings**: cash deposits (in), money order
  deposits (in) and wires (in) "not monitored at all".
- The ML should put most wire businesses in the review queue (as in H-2).
- The advisory check should catch the money-order structurers.
- Individual structured-cash customers will mostly **not** be in the queue (as in H-1).
**Counts as a pass:** all three coverage findings appear; at least 20 of 25 wire
businesses and 20 of 25 money-order structurers are in the queue.
**Result:**

### M-2. Card repayments are only counted
**The flaw:** the only rule that can see card repayments is the 25-payment
velocity rule.
**Prediction: Blindspot FINDS this.** Coverage should raise a medium "Credit
card repayments (out) are only checked by a transaction count" finding.
Regulator alerts should report the card-funnel pattern as not caught.
**Counts as a pass:** both findings appear, and most of the 25 card-funnel
customers are in the review queue.
**Result:**

### M-3. Case threshold too high for the points on offer
**The flaw:** a case needs 40 points, but no rule is worth more than 10.
**Prediction: Blindspot partly FINDS this.** Very few cases should open.
Rule weights should mark at least one rule as under-weighted (suggestions scale
with the 40-point threshold). The "just under the threshold" comparison now uses
32–39 vs 40–48 points; with so few customers in those bands, it may not fire.
**Counts as a pass:** at least one "Strong red flag worth only N points" finding.
**Result:**

### M-4. A long valued-client list
**The flaw:** the 25 biggest small businesses by money received are exempt.
These include the planted wire businesses.
**Prediction: Blindspot FINDS the risk.** Below the line should raise "The
valued-client list hides highly unusual customers". Because incoming money isn't
monitored, those customers probably never reach 40 points, so the finding
should say few or none of them have *already* scored enough for a case. The
damage comes from M-1, not the exemption.
**Counts as a pass:** the finding appears, and its explanation gives a small number.
**Result:**

---

## Expected blind spots for both banks
- **Slow burn**: designed to look normal; expect fewer than 5 of 25 found.
- **SAML-D's own laundering**: mostly networks of accounts; expect little added.