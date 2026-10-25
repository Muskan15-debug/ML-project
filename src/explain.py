"""SHAP explanations for the XGBoost churn model, plus a small readable CART tree.

SHAP values are in log-odds: a positive value pushes the churn risk up, a negative value pushes it down.
The model sees log1p-transformed features, but every plot and sentence shows the REAL values
(e.g. "12 orders", not "log(13)"), with plain-English feature names.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.append(str(Path(__file__).resolve().parent))
import config
from eda import save_matplotlib
from feature_selection import make_model_matrix


def load_churn_parts():
    """Load the saved churn model bundle and the segment encoder."""
    return joblib.load(config.CHURN_MODEL), joblib.load(config.SEGMENT_ENCODER)


def model_inputs(table, bundle, encoder):
    """Return the exact feature matrix the churn model was trained on (log1p + one-hot, selected columns)."""
    return make_model_matrix(table, encoder)[bundle["features"]]


def display_values(table, bundle, encoder):
    """Return the same columns with REAL (untransformed) values, for showing to people."""
    shown = make_model_matrix(table, encoder)[bundle["features"]].copy()
    for column in bundle["features"]:
        if column in table.columns:
            shown[column] = table[column]          # undo log1p by taking the raw column
    return shown


def plain_names(columns):
    """Map feature column names to plain-English labels."""
    return [config.FEATURE_LABELS.get(column, column) for column in columns]


def shap_explanation(table, bundle, encoder):
    """Return a SHAP Explanation for the given customers, with real values and plain-English names."""
    explainer = shap.TreeExplainer(bundle["model"].named_steps["model"])
    raw = explainer(model_inputs(table, bundle, encoder))
    return shap.Explanation(values=raw.values, base_values=raw.base_values,
                            data=display_values(table, bundle, encoder).to_numpy().astype(float).round(1),
                            feature_names=plain_names(bundle["features"]))


def plot_shap_summary(explanation):
    """Save the global SHAP summary (beeswarm) plot."""
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=len(explanation.feature_names), show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP summary: what drives churn risk (sample of test customers)", y=1.02)
    save_matplotlib(fig, "shap_summary_churn")
    return fig


def plot_shap_waterfall(explanation, index, file_name, title):
    """Save a SHAP waterfall plot for one customer (row `index` of the explanation)."""
    plt.figure()
    shap.plots.waterfall(explanation[index], max_display=len(explanation.feature_names), show=False)
    fig = plt.gcf()
    fig.suptitle(title, y=1.02)
    save_matplotlib(fig, file_name)
    return fig


def describe_value(column, value):
    """Turn one feature value into a short plain-English phrase, e.g. '85 days since last order'."""
    if column == "recency":
        return f"{value:.0f} days since last order"
    if column == "frequency":
        return f"{value:.0f} order{'s' if value != 1 else ''} in total"
    if column == "monetary":
        return f"{value:,.0f} GBP spent in total"
    if column == "avg_order_value":
        return f"{value:,.0f} GBP average order"
    if column == "avg_gap_days":
        return f"{value:.0f} days between orders on average"
    if column == "tenure_days":
        return f"first order {value:.0f} days ago"
    if column == "unique_products":
        return f"{value:.0f} different products bought"
    if column == "return_rate":
        return f"{value:.0%} of orders returned"
    if column == "spend_trend":
        return f"{value:.0%} of spend in the last 3 months"
    if column == "purchase_regularity":
        return f"gaps between orders vary by about {value:.0f} days"
    if column.startswith("seg_"):
        segment = column[4:]
        return f"is in the {segment} segment" if value == 1 else f"is not in the {segment} segment"
    return f"{config.FEATURE_LABELS.get(column, column)} = {value}"


def explain_table(table, bundle=None, encoder=None, top_k=4):
    """Explain many customers at once. Returns (churn probabilities, list of sentence lists), one per row.

    `table` has raw features + segment_name. One SHAP call for all rows keeps batch prediction fast.
    """
    if bundle is None or encoder is None:
        bundle, encoder = load_churn_parts()
    inputs = model_inputs(table, bundle, encoder)
    shap_values = shap.TreeExplainer(bundle["model"].named_steps["model"])(inputs).values
    probabilities = bundle["model"].predict_proba(inputs)[:, 1]
    shown = display_values(table, bundle, encoder)

    all_sentences = []
    for i in range(len(table)):
        contributions = pd.Series(shap_values[i], index=bundle["features"])
        sentences = []
        for column in contributions.abs().sort_values(ascending=False).index[:top_k]:
            direction = "raises risk" if contributions[column] > 0 else "lowers risk"
            sentences.append(f"{describe_value(column, shown.iloc[i][column])} -> {direction}")
        all_sentences.append(sentences)
    return probabilities, all_sentences


def explain_customer(row, bundle=None, encoder=None, top_k=4):
    """Explain one customer's churn risk in plain English.

    `row` is one row of a customer table (raw features + segment_name), e.g. table.loc[[customer_id]].
    Returns (churn probability, list of sentences), strongest reasons first.
    """
    table = row if isinstance(row, pd.DataFrame) else row.to_frame().T
    probabilities, sentences = explain_table(table, bundle, encoder, top_k)
    return float(probabilities[0]), sentences[0]


def cart_rules(train_df, test_df, features):
    """Fit a small CART tree on RAW feature values and return (rules text, its test ROC-AUC).

    Trees do not care about log transforms, so raw values give the same splits but readable thresholds.
    """
    def raw_matrix(table):
        X = pd.DataFrame(index=table.index)
        for column in features:
            X[column] = table[column] if column in table.columns else \
                (table[config.SEGMENT_COLUMN] == column[4:]).astype(int)
        return X

    tree = DecisionTreeClassifier(max_depth=config.CART_RULES_DEPTH, min_samples_leaf=20,
                                  random_state=config.RANDOM_STATE)
    tree.fit(raw_matrix(train_df), train_df["churn"])
    rules = export_text(tree, feature_names=plain_names(features), class_names=["stays", "churns"],
                        show_weights=True, decimals=1)
    test_auc = roc_auc_score(test_df["churn"], tree.predict_proba(raw_matrix(test_df))[:, 1])
    return rules, test_auc


def save_cart_rules(rules, test_auc):
    """Save the CART rules as a text file in reports/tables/."""
    path = config.TABLES_DIR / "cart_rules.txt"
    header = (f"CART decision tree (max depth {config.CART_RULES_DEPTH}) for churn, trained on TRAIN.\n"
              f"weights = [customers who stay, customers who churn] in each leaf. Test ROC-AUC: {test_auc:.3f}\n\n")
    path.write_text(header + rules)
    return path
