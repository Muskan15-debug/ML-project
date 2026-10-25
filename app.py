"""Streamlit app: check one customer, score a CSV of customers, and see the dashboard.

Run from the project folder:   streamlit run app.py

The app uses ONLY the saved models in models/ (written by src/train.py) through src/predict.py,
so it applies exactly the same feature rules and models as training.
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
import config
import decision_rules as dr
import predict

st.set_page_config(page_title="Churn & CLV Predictor", layout="wide")

PROBABILITY_NOTE = "Probabilities run about 5 points high on recent data (see known issues)."
# Example customers for the form. The first one shows a clear change when the recency slider moves.
PRESETS = {
    "Loyal customer, 8 orders (default)": {"recency": 60, "frequency": 8, "monetary": 3000.0, "tenure_days": 300,
                                           "unique_products": 80, "spend_last_3m": 900.0, "returned_orders": 0,
                                           "is_uk": 1},
    "Spec section 9 example": {"recency": 85, "frequency": 3, "monetary": 240.0, "tenure_days": 200,
                               "unique_products": 6, "spend_last_3m": 80.0, "returned_orders": 0, "is_uk": 1},
}
TREND_DAYS = config.TREND_WINDOW_MONTHS * config.DAYS_PER_MONTH


# ---------------------------------------------------------------- loading (once)
@st.cache_resource
def load_models():
    """Load every saved model file once for the whole app session."""
    return predict.load_artifacts()


@st.cache_data
def load_test_data():
    """TEST table and its scores (used by the dashboard and the campaign simulator)."""
    test = pd.read_parquet(config.TEST_TABLE_PARQUET)
    scores = dr.add_actions(dr.score_customers(test), dr.load_split_points())
    return test, scores


@st.cache_data
def sample_csv():
    """A small CSV of real TEST customers in the upload format."""
    test = pd.read_parquet(config.TEST_TABLE_PARQUET)
    sample = test.sample(8, random_state=config.RANDOM_STATE)
    return predict.table_to_inputs(sample).to_csv(index=False).encode("utf-8")


@st.cache_data
def business_finding():
    """The 'Main business finding' section of reports/RESULTS_NOTES.md."""
    text = (config.REPORTS_DIR / "RESULTS_NOTES.md").read_text(encoding="utf-8")
    match = re.search(r"## Main business finding\n(.*?)(?=\n## )", text, flags=re.S)
    return match.group(1).strip() if match else "Section not found in RESULTS_NOTES.md."


@st.cache_data
def run_backtest(discount, response, redeem):
    """Backtest every campaign strategy on TEST for the chosen ASSUMPTIONS."""
    test, scores = load_test_data()
    assumptions = {**dr.default_assumptions(), "discount_rate": discount, "response_rate": response,
                   "stayer_redeem_rate": redeem}
    return dr.backtest(scores, test["churn"], test["future_spend"], dr.load_split_points(), assumptions), assumptions


# ---------------------------------------------------------------- shared display
def show_result(result, base=None):
    """Show one prediction. With `base`, also show the change compared with the base prediction."""
    def delta(key, fmt):
        return None if base is None else fmt(result[key] - base[key])

    for warning in result["warnings"]:
        st.warning(warning)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", result["verdict"])
    c2.metric("Churn probability", f"{result['churn_probability']:.0%}",
              delta=delta("churn_probability", lambda d: f"{d:+.0%}"), delta_color="inverse")
    c2.caption(PROBABILITY_NOTE)
    c3.metric("Risk level", result["risk_level"])
    c4.metric("Segment", result["segment"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted 6-month spend", f"{result['expected_spend_6m']:,.0f} GBP",
              delta=delta("expected_spend_6m", lambda d: f"{d:+,.0f} GBP"))
    c1.caption("Expected value = chance of buying x amount if they buy.")
    c2.metric("If the customer stays and buys", f"{result['conditional_amount_6m']:,.0f} GBP")
    c3.metric("Expected profit of a discount offer", f"{result['expected_net']:,.1f} GBP",
              delta=delta("expected_net", lambda d: f"{d:+,.1f} GBP"))
    c3.caption("Based on ASSUMED margin, discount, response and redeem rates.")

    st.markdown(f"**Recommended action:** {result['action']}")
    st.caption(result["action_reason"])
    st.markdown("**Why (SHAP, strongest first):**")
    for reason in result["top_reasons"]:
        st.markdown(f"- {reason}")


def show_filled_values(result):
    """Table of calculated and assumed values, so the user sees what was not typed."""
    rows = [{"value": config.FEATURE_LABELS.get(k, k), "number": round(v["value"], 3),
             "type": "calculated", "how": v["how"]} for k, v in result["derived_values"].items()]
    rows += [{"value": config.FEATURE_LABELS.get(k, k), "number": round(v["value"], 3),
              "type": "ASSUMED", "how": v["how"]} for k, v in result["assumed_values"].items()]
    if result["assumed_values"]:
        st.info("Assumed value(s): " + ", ".join(config.FEATURE_LABELS.get(k, k) for k in result["assumed_values"])
                + " - filled with a training median because it cannot be typed.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ---------------------------------------------------------------- tab 1
def tab_check_customer():
    """Form for one customer, then What-If sliders that update live."""
    st.subheader("Check a customer")
    preset_name = st.selectbox("Start from an example customer", list(PRESETS))
    preset = PRESETS[preset_name]
    k = list(PRESETS).index(preset_name)        # widget keys change with the preset, so the form refills
    with st.form("customer_form"):
        c1, c2, c3, c4 = st.columns(4)
        recency = c1.number_input("Days since last order", min_value=0, value=preset["recency"], step=1, key=f"rec{k}")
        frequency = c2.number_input("Number of orders", min_value=1, value=preset["frequency"], step=1, key=f"frq{k}")
        monetary = c3.number_input("Total spent (GBP)", min_value=0.0, value=preset["monetary"], step=10.0,
                                   key=f"mon{k}")
        tenure = c4.number_input("Days since first order", min_value=0, value=preset["tenure_days"], step=1,
                                 key=f"ten{k}")
        c1, c2, c3, c4 = st.columns(4)
        products = c1.number_input("Different products bought", min_value=0, value=preset["unique_products"], step=1,
                                   key=f"prd{k}")
        last_3m = c2.number_input("Spend in last 3 months (GBP)", min_value=0.0, value=preset["spend_last_3m"],
                                  step=10.0, key=f"l3m{k}")
        returned = c3.number_input("Returned / cancelled orders", min_value=0, value=preset["returned_orders"], step=1,
                                   key=f"ret{k}")
        uk = c4.radio("Customer in UK?", ["Yes", "No"], index=0 if preset["is_uk"] else 1, horizontal=True,
                      key=f"uk{k}")
        submitted = st.form_submit_button("Check customer", type="primary")

    if submitted:
        inputs = {"recency": recency, "frequency": frequency, "monetary": monetary, "tenure_days": tenure,
                  "unique_products": products, "spend_last_3m": last_3m, "returned_orders": returned,
                  "is_uk": 1 if uk == "Yes" else 0}
        try:
            st.session_state["base_result"] = predict.predict_customer(inputs)
            st.session_state["base_inputs"] = inputs
            st.session_state.pop("form_error", None)
        except ValueError as error:
            st.session_state["form_error"] = str(error)

    if "form_error" in st.session_state:
        st.error(f"Please check the inputs: {st.session_state['form_error']}")
    if "base_result" not in st.session_state:
        st.info("Pick an example (or type your own values) and click **Check customer**.")
        return

    base_inputs, base_result = st.session_state["base_inputs"], st.session_state["base_result"]
    show_result(base_result)
    with st.expander("Calculated and assumed values"):
        show_filled_values(base_result)

    st.divider()
    st.subheader("What-if simulator")
    st.caption("Move a slider: the prediction updates immediately. Other values stay as in the form above.")
    key = str(hash(tuple(base_inputs.items())))     # new form values -> sliders restart from them
    c1, c2, c3, c4 = st.columns(4)
    w_recency = c1.slider("Days since last order", 0, 365, int(base_inputs["recency"]), key=f"r{key}")
    w_frequency = c2.slider("Number of orders", 1, max(50, int(base_inputs["frequency"]) * 2),
                            int(base_inputs["frequency"]), key=f"f{key}")
    w_monetary = c3.slider("Total spent (GBP)", 1, max(5000, int(base_inputs["monetary"]) * 2),
                           max(1, int(base_inputs["monetary"])), key=f"m{key}")
    w_tenure = c4.slider("Days since first order", 0, 730, min(730, int(base_inputs["tenure_days"])), key=f"t{key}")

    what_if = {**base_inputs, "recency": w_recency, "frequency": w_frequency, "monetary": w_monetary,
               "tenure_days": w_tenure}
    notes = []
    if w_tenure < w_recency:
        what_if["tenure_days"] = w_recency
        notes.append(f"days since first order raised to {w_recency} (it cannot be below days since last order)")
    if w_recency > TREND_DAYS + 1 and what_if["spend_last_3m"] > 0:
        what_if["spend_last_3m"] = 0.0
        notes.append("spend in last 3 months set to 0 (the last order is more than 3 months ago)")
    if what_if["spend_last_3m"] > w_monetary:
        what_if["spend_last_3m"] = float(w_monetary)
        notes.append("spend in last 3 months lowered to the total spent")
    for note in notes:
        st.info("Adjusted to keep the inputs consistent: " + note)
    try:
        show_result(predict.predict_customer(what_if), base=base_result)
    except ValueError as error:
        st.error(f"These slider values are not possible: {error}")


# ---------------------------------------------------------------- tab 2
def tab_batch():
    """Upload a CSV, get a ranked 'contact first' table, download it."""
    st.subheader("Batch prediction")
    st.markdown("Upload a CSV with one row per customer and these columns (an optional `customer_id` column is kept):")
    st.dataframe(pd.DataFrame({"column": list(predict.INPUT_FIELDS), "meaning": list(predict.INPUT_FIELDS.values())}),
                 hide_index=True, width="stretch")
    st.caption("`tenure_days` may be empty for customers with one order. `is_uk` accepts 1/0 or yes/no.")
    st.download_button("Download sample CSV (8 real test customers)", sample_csv(), "sample_customers.csv", "text/csv")

    uploaded = st.file_uploader("Upload customer CSV", type="csv")
    if uploaded is None:
        return
    try:
        data = pd.read_csv(uploaded)
    except Exception as error:      # any parsing problem: show it instead of crashing
        st.error(f"Could not read the file as CSV: {error}")
        return
    missing = [c for c in predict.REQUIRED_FIELDS if c not in data.columns and c != "tenure_days"]
    if missing:
        st.error("The file is missing these columns: " + ", ".join(missing))
        return
    if "tenure_days" not in data.columns:
        data["tenure_days"] = None
    if data.empty:
        st.error("The file has no rows.")
        return

    with st.spinner(f"Scoring {len(data):,} customers..."):
        result = predict.predict_batch(data)
    errors = result[result["error"].notna()] if "error" in result.columns else pd.DataFrame()
    scored = result.drop(errors.index).drop(columns=["error"], errors="ignore")

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", len(scored))
    c2.metric("Retention offers", int((scored["action"] == "Retention offer (priority)").sum()) if len(scored) else 0)
    c3.metric("Rows with errors", len(errors))
    if len(errors):
        st.error("Some rows could not be scored:")
        st.dataframe(errors[["customer_id", "error"]], hide_index=True, width="stretch")
    if len(scored):
        if (scored["warnings"] != "").any():
            st.warning(f"{(scored['warnings'] != '').sum()} customer(s) have warnings - see the 'warnings' column.")
        st.markdown("**Contact first** (ranked by expected profit of a discount offer):")
        st.dataframe(scored, hide_index=True, width="stretch")
        st.download_button("Download results CSV", result.to_csv(index=False).encode("utf-8"),
                           "churn_predictions.csv", "text/csv")


# ---------------------------------------------------------------- tab 3
def tab_dashboard():
    """Business finding, segments, revenue at risk, model comparison, SHAP and the campaign simulator."""
    st.subheader("Main business finding")
    st.markdown(business_finding())

    st.divider()
    st.subheader("Campaign simulator (TEST backtest)")
    st.caption("All three inputs are ASSUMPTIONS, not data. Margin 30% and contact cost 1 GBP stay fixed.")
    c1, c2, c3 = st.columns(3)
    discount = c1.slider("Discount rate (ASSUMPTION)", 5, 20, int(config.DISCOUNT_RATE * 100), format="%d%%")
    response = c2.slider("Response rate of churners (ASSUMPTION)", 5, 30, int(config.OFFER_RESPONSE_RATE * 100),
                         format="%d%%")
    redeem = c3.slider("Stayers who redeem the discount (ASSUMPTION)", 30, 100,
                       int(config.STAYER_REDEEM_RATE * 100), step=5, format="%d%%")
    impact, assumptions = run_backtest(discount / 100, response / 100, redeem / 100)
    best = impact.sort_values(["net_profit", "contacted"], ascending=[False, True]).iloc[0]
    winner = "(e) No campaign" if best["contacted"] == 0 else best["strategy"]
    st.success(f"**Best at these values:** {winner} (net profit {best['net_profit']:,.0f} GBP)")
    fig = dr.plot_headline(impact, assumptions, save=False)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(impact.round(0), hide_index=True, width="stretch")

    st.divider()
    test, scores = load_test_data()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Customer segments")
        st.image(str(config.FIGURES_DIR / "segments_pca_train.png"), width="stretch")
    with c2:
        st.subheader("Churn rate per segment (TEST)")
        summary = pd.read_csv(config.TABLES_DIR / "segment_summary_test.csv", index_col=0)
        st.bar_chart(summary["churn_rate"])
        st.dataframe(summary[["customers", "churn_rate", "avg_future_spend"]].round(3), width="stretch")

    st.subheader("Revenue at risk (TEST snapshot, estimate)")
    st.caption("Churn probability x amount the customer would spend in 6 months if they stay and buy.")
    at_risk = scores.groupby("segment_name")["expected_loss"].sum().sort_values(ascending=False)
    c1, c2 = st.columns([1, 2])
    c1.metric("Total revenue at risk", f"{at_risk.sum():,.0f} GBP")
    c2.bar_chart(at_risk)

    st.subheader("Model comparison")
    churn = pd.read_csv(config.TABLES_DIR / "churn_model_comparison.csv")
    st.markdown("**Churn models (TEST)**")
    st.dataframe(churn[["model", "imbalance", "cv_roc_auc_train", "precision", "recall", "f1", "roc_auc", "pr_auc",
                        "mean_pred_prob"]].round(3), hide_index=True, width="stretch")
    st.markdown("**Spend models (TEST, GBP)**")
    spend = pd.read_csv(config.TABLES_DIR / "spend_two_part_test.csv")
    st.dataframe(spend[["model", "mae", "rmse", "r2", "mae_spenders_only", "mean_predicted", "mean_actual"]].round(3),
                 hide_index=True, width="stretch")

    st.subheader("What drives churn risk (SHAP)")
    st.image(str(config.FIGURES_DIR / "shap_summary_churn.png"), width="stretch")


# ---------------------------------------------------------------- page
st.title("Customer Churn & Lifetime Value Predictor")
st.caption("Online Retail II - models trained on active customers (bought in the last 180 days).")
try:
    load_models()
except FileNotFoundError as error:
    st.error(f"Saved models not found ({error}). Run `python src/train.py` first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Check a Customer", "Batch Prediction", "Dashboard"])
with tab1:
    tab_check_customer()
with tab2:
    tab_batch()
with tab3:
    tab_dashboard()
