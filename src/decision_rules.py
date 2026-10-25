"""Turn churn probability and predicted value into a business action, and backtest campaign strategies.

Two different "values" are used on purpose:
- conditional_amount: what the customer would spend in 6 months IF they stay and buy
  (part 2 of the spend model, after smearing and cap). Used in the PROFIT RULE.
  Using expected spend there would count churn twice: likely churners already get a low P(spend > 0).
- expected_spend: P(spend > 0) x conditional_amount. Shown to users and used for the
  high/low VALUE split in the action matrix.

Profit rule (the offer is a discount; all rates are ASSUMPTIONS in config.py):
  expected_net = p_churn * response_rate * amount * (margin - discount)    gain: churners we keep
               - (1 - p_churn) * stayer_redeem_rate * discount * amount     cost: stayers who use the discount
               - contact_cost                                               cost: contacting the customer
  Send the offer only if expected_net > 0.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config
from feature_selection import make_model_matrix
from hurdle_model import amount_gbp, predict_hurdle

ACTIONS = {
    (True, True): "Retention offer (priority)",      # high value, high risk
    (True, False): "Keep happy: loyalty perks",       # high value, low risk
    (False, True): "Low-cost email only",             # low value, high risk
    (False, False): "No action",                      # low value, low risk
}

SEGMENT_NOTES = {
    "Champions": "top customer: best handled with a personal contact",
    "Loyal": "regular buyer: a reminder or small incentive fits",
    "One-off buyers": "has ordered only once or twice: aim for a second order",
    "At-Risk": "has been quiet for months: needs a reason to come back",
}


def default_assumptions():
    """Return the business assumptions from config.py as a dict (all are ASSUMPTIONS, not data)."""
    return {"response_rate": config.OFFER_RESPONSE_RATE, "discount_rate": config.DISCOUNT_RATE,
            "gross_margin": config.GROSS_MARGIN, "contact_cost": config.CONTACT_COST,
            "stayer_redeem_rate": config.STAYER_REDEEM_RATE}


def expected_net(churn_prob, amount, response_rate=config.OFFER_RESPONSE_RATE, discount_rate=config.DISCOUNT_RATE,
                 gross_margin=config.GROSS_MARGIN, contact_cost=config.CONTACT_COST,
                 stayer_redeem_rate=config.STAYER_REDEEM_RATE):
    """Expected profit in GBP of sending one discount offer to a customer (works on numbers or arrays)."""
    gain_from_kept_churners = churn_prob * response_rate * amount * (gross_margin - discount_rate)
    discount_given_to_stayers = (1 - churn_prob) * stayer_redeem_rate * discount_rate * amount
    return gain_from_kept_churners - discount_given_to_stayers - contact_cost


def passes_profit_rule(net):
    """Send the offer only if its expected net profit is positive."""
    return net > 0


def score_customers(table, assumptions=None):
    """Return churn probability, conditional amount, expected spend, expected loss and expected_net per customer."""
    assumptions = assumptions or default_assumptions()
    churn = joblib.load(config.CHURN_MODEL)
    spend = joblib.load(config.SPEND_MODEL)
    encoder = joblib.load(config.SEGMENT_ENCODER)
    X = make_model_matrix(table, encoder)

    scores = pd.DataFrame(index=table.index)
    scores["segment_name"] = table[config.SEGMENT_COLUMN]
    scores["churn_probability"] = churn["model"].predict_proba(X[churn["features"]])[:, 1]
    X_spend = X[spend["features"]]
    scores["conditional_amount"] = np.minimum(amount_gbp(spend["regressor"], X_spend, spend["smearing_factor"]),
                                              spend["cap_gbp"])
    scores["expected_spend"] = predict_hurdle(spend, X_spend, spend["cap_gbp"])
    # Expected loss is shown for information: sales we might lose if this buyer leaves.
    scores["expected_loss"] = scores["churn_probability"] * scores["conditional_amount"]
    scores["expected_net"] = expected_net(scores["churn_probability"], scores["conditional_amount"], **assumptions)
    return scores


def fit_split_points(train_scores):
    """Choose the high/low split points from TRAIN scores (medians) and save them with the profit-rule settings."""
    thresholds = {
        "risk_threshold": float(train_scores["churn_probability"].median()),
        "value_threshold": float(train_scores["expected_spend"].median()),
        "split_rule": "medians of TRAIN churn probability and TRAIN expected spend",
        "profit_rule": "send retention offer only if expected_net > 0, where expected_net = "
                       "p*response_rate*amount*(gross_margin-discount_rate) "
                       "- (1-p)*stayer_redeem_rate*discount_rate*amount - contact_cost",
        "assumptions": default_assumptions(),
    }
    config.DECISION_THRESHOLDS_JSON.write_text(json.dumps(thresholds, indent=2))
    return thresholds


def load_split_points():
    """Load the saved split points and profit-rule settings."""
    return json.loads(config.DECISION_THRESHOLDS_JSON.read_text())


def recommend_action(churn_prob, expected_spend, conditional_amount, segment, thresholds=None, assumptions=None):
    """Return (action, reason) for one customer.

    The action matrix uses churn risk (high/low) and expected spend (high/low). A retention offer is
    only kept if the profit rule passes (expected_net > 0); otherwise it becomes a low-cost email.
    """
    thresholds = thresholds or load_split_points()
    assumptions = assumptions or default_assumptions()
    high_risk = churn_prob >= thresholds["risk_threshold"]
    high_value = expected_spend >= thresholds["value_threshold"]
    net = expected_net(churn_prob, conditional_amount, **assumptions)
    action = ACTIONS[(high_value, high_risk)]
    profit_ok = passes_profit_rule(net)
    if action.startswith("Retention offer") and not profit_ok:
        action = "Low-cost email only"

    reason = (f"{segment} ({SEGMENT_NOTES.get(segment, '')}). "
              f"Churn risk {churn_prob:.1%} is {'HIGH' if high_risk else 'low'} "
              f"(split {thresholds['risk_threshold']:.1%}); expected 6-month spend {expected_spend:,.0f} GBP is "
              f"{'HIGH' if high_value else 'low'} (split {thresholds['value_threshold']:,.0f} GBP). "
              f"Expected net profit of a {assumptions['discount_rate']:.0%} discount offer: {net:,.0f} GBP "
              f"-> profit rule {'passes' if profit_ok else 'fails'}.")
    return action, reason


def add_actions(scores, thresholds, assumptions=None):
    """Add action and reason columns to a scores table."""
    results = [recommend_action(r.churn_probability, r.expected_spend, r.conditional_amount, r.segment_name,
                                thresholds, assumptions) for r in scores.itertuples()]
    scores = scores.copy()
    scores["action"] = [a for a, _ in results]
    scores["reason"] = [r for _, r in results]
    return scores


def strategy_masks(scores, thresholds, assumptions):
    """Return which customers each strategy contacts (True = gets the discount offer)."""
    n_top = int(round(config.TOP_N_SHARE * len(scores)))
    top = scores["churn_probability"].rank(ascending=False, method="first") <= n_top
    net = expected_net(scores["churn_probability"], scores["conditional_amount"], **assumptions)
    priority = add_actions(scores, thresholds, assumptions)["action"] == "Retention offer (priority)"
    return {
        "(a) Mail everyone": pd.Series(True, index=scores.index),
        f"(b) Top {config.TOP_N_SHARE:.0%} highest churn risk": top,
        "(c) Profit rule (expected_net > 0)": passes_profit_rule(net),
        "(d) Priority offers only": priority,
        "(e) No campaign": pd.Series(False, index=scores.index),
    }


def backtest(scores, churned, actual_spend, thresholds, assumptions=None):
    """Backtest every strategy on customers with KNOWN outcomes (TEST labels).

    For each contacted customer:
      actual churner     -> + response_rate * predicted amount * (margin - discount)   (some are kept)
      actual non-churner -> - stayer_redeem_rate * discount * their ACTUAL future spend (wasted discount)
      everyone           -> - contact_cost
    """
    a = assumptions or default_assumptions()
    churned = np.asarray(churned).astype(bool)
    actual_spend = np.asarray(actual_spend)
    amount = scores["conditional_amount"].to_numpy()
    rows = []
    for name, mask in strategy_masks(scores, thresholds, a).items():
        mask = np.asarray(mask)
        reached = mask & churned
        stayers = mask & ~churned
        gain = float((a["response_rate"] * amount[reached] * (a["gross_margin"] - a["discount_rate"])).sum())
        wasted = float((a["stayer_redeem_rate"] * a["discount_rate"] * actual_spend[stayers]).sum())
        contact = float(mask.sum() * a["contact_cost"])
        rows.append({"strategy": name, "contacted": int(mask.sum()), "churners_reached": int(reached.sum()),
                     "share_of_all_churners": reached.sum() / churned.sum(), "stayers_contacted": int(stayers.sum()),
                     "gain_from_kept_churners": gain, "wasted_discount_cost": wasted, "contact_cost": contact,
                     "net_profit": gain - wasted - contact})
    return pd.DataFrame(rows)


def sensitivity(scores, churned, actual_spend, thresholds, response_rates=None, discount_rates=None):
    """Net profit of each strategy over several response rates and discount rates (both ASSUMPTIONS).

    The profit rule and the priority offers are re-decided for every combination.
    """
    rows = []
    for discount in discount_rates or config.SENSITIVITY_DISCOUNT_RATES:
        for rate in response_rates or config.SENSITIVITY_RESPONSE_RATES:
            assumptions = {**default_assumptions(), "response_rate": rate, "discount_rate": discount}
            for r in backtest(scores, churned, actual_spend, thresholds, assumptions).itertuples():
                rows.append({"discount_rate": discount, "response_rate": rate, "strategy": r.strategy,
                             "contacted": r.contacted, "net_profit": r.net_profit})
    return pd.DataFrame(rows)


def priority_list(scores, n=config.PRIORITY_LIST_SIZE):
    """Top-n customers to contact first: retention offers first, then by expected net profit (highest first)."""
    ranked = scores.assign(is_offer=scores["action"].str.startswith("Retention offer"))
    ranked = ranked.sort_values(["is_offer", "expected_net"], ascending=False).head(n)
    columns = ["segment_name", "churn_probability", "conditional_amount", "expected_spend", "expected_net",
               "action", "reason"]
    return ranked[columns].round({"churn_probability": 3, "conditional_amount": 0, "expected_spend": 0,
                                  "expected_net": 1})
