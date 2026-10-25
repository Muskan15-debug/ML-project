"""Feature selection for the churn and spend models: filter, wrapper and embedded methods.

Everything here is FITTED ON TRAIN ONLY. The test table is used only in compare_on_test().
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
import config
from eda import save_matplotlib
from feature_builder import FEATURE_COLUMNS


def fit_segment_encoder(train_df):
    """Fit a one-hot encoder for the segment name on TRAIN only and save it to models/."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[[config.SEGMENT_COLUMN]])
    joblib.dump(encoder, config.SEGMENT_ENCODER)
    return encoder


def make_model_matrix(table, encoder):
    """Return the feature matrix: log1p of skewed features + other features + one-hot segment columns."""
    X = table[FEATURE_COLUMNS].copy()
    for column in config.LOG1P_FEATURES:
        X[column] = np.log1p(X[column])
    one_hot = encoder.transform(table[[config.SEGMENT_COLUMN]])
    names = [f"seg_{name}" for name in encoder.categories_[0]]
    return X.join(pd.DataFrame(one_hot, columns=names, index=table.index).astype(int))


def churn_target(table):
    """Return the churn label (1 = churned)."""
    return table["churn"]


def spend_target(table):
    """Return log1p(future_spend); the log tames the very long tail of big spenders."""
    return np.log1p(table["future_spend"])


def plot_correlation(X, threshold=0.8):
    """Save a correlation heatmap and return the feature pairs with |r| above the threshold."""
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature correlation (TRAIN, after log1p of skewed features)")
    save_matplotlib(fig, "feature_correlation_heatmap")

    pairs = []
    columns = corr.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if abs(corr.iloc[i, j]) > threshold:
                pairs.append({"feature_1": columns[i], "feature_2": columns[j], "r": round(corr.iloc[i, j], 3)})
    return fig, pd.DataFrame(pairs, columns=["feature_1", "feature_2", "r"])


def filter_scores(X, y_churn, y_spend):
    """Return mutual information and F-test scores for churn and for log spend, one row per feature."""
    scores = pd.DataFrame(index=X.columns)
    scores["mi_churn"] = mutual_info_classif(X, y_churn, random_state=config.RANDOM_STATE)
    scores["anova_f_churn"] = f_classif(X, y_churn)[0]
    scores["mi_spend"] = mutual_info_regression(X, y_spend, random_state=config.RANDOM_STATE)
    scores["f_reg_spend"] = f_regression(X, y_spend)[0]
    return scores


def rfe_logistic(X, y):
    """Recursive Feature Elimination with Logistic Regression; return the rank of each feature (1 = kept)."""
    # Scale first so the logistic coefficients are comparable between features.
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=config.N_SELECTED_FEATURES)
    rfe.fit(X_scaled, y)
    return pd.Series(rfe.ranking_, index=X.columns, name="rfe_rank")


def embedded_scores(X, y_churn, y_spend):
    """Return Random Forest importances and L1 coefficients for churn and spend, one row per feature."""
    scores = pd.DataFrame(index=X.columns)
    rf_c = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=config.RANDOM_STATE, n_jobs=-1)
    scores["rf_importance_churn"] = rf_c.fit(X, y_churn).feature_importances_
    rf_r = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=config.RANDOM_STATE, n_jobs=-1)
    scores["rf_importance_spend"] = rf_r.fit(X, y_spend).feature_importances_

    X_scaled = StandardScaler().fit_transform(X)
    # L1 pushes useless coefficients to exactly 0; C is chosen by 5-fold CV on TRAIN.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    l1_logistic = LogisticRegressionCV(Cs=20, cv=cv, l1_ratios=[1.0], solver="saga", scoring="roc_auc",
                                       max_iter=5000, random_state=config.RANDOM_STATE,
                                       use_legacy_attributes=True)
    scores["l1_coef_churn"] = l1_logistic.fit(X_scaled, y_churn).coef_[0]
    lasso = LassoCV(cv=5, random_state=config.RANDOM_STATE, max_iter=10000)
    scores["lasso_coef_spend"] = lasso.fit(X_scaled, y_spend).coef_
    return scores


def top_n(series, n=config.N_SELECTED_FEATURES):
    """Return the set of the n features with the largest scores."""
    return set(series.sort_values(ascending=False).head(n).index)


def selection_table(filters, rfe_rank, embedded):
    """Return a table of ticks: which features each method keeps, plus a vote count per task."""
    churn_methods = {
        "MI (churn)": top_n(filters["mi_churn"]),
        "ANOVA F (churn)": top_n(filters["anova_f_churn"]),
        "RFE-LogReg (churn)": set(rfe_rank[rfe_rank == 1].index),
        "RF importance (churn)": top_n(embedded["rf_importance_churn"]),
        "L1-logistic (churn)": set(embedded.index[embedded["l1_coef_churn"].abs() > 1e-6]),
    }
    spend_methods = {
        "MI (spend)": top_n(filters["mi_spend"]),
        "f_regression (spend)": top_n(filters["f_reg_spend"]),
        "RF importance (spend)": top_n(embedded["rf_importance_spend"]),
        "Lasso (spend)": set(embedded.index[embedded["lasso_coef_spend"].abs() > 1e-6]),
    }
    table = pd.DataFrame(index=filters.index)
    for name, kept in {**churn_methods, **spend_methods}.items():
        table[name] = table.index.isin(kept).astype(int)
    table["votes_churn"] = table[list(churn_methods)].sum(axis=1)
    table["votes_spend"] = table[list(spend_methods)].sum(axis=1)
    return table


def choose_features(table, filters, votes_column, mi_column, n=config.N_SELECTED_FEATURES):
    """Pick the n features with the most votes; ties are broken by mutual information."""
    ranked = table[[votes_column]].join(filters[[mi_column]])
    ranked = ranked.sort_values([votes_column, mi_column], ascending=False)
    return list(ranked.head(n).index)


def plot_method_scores(filters, embedded):
    """Save one bar chart per churn method score so the rankings can be compared visually."""
    churn_scores = {
        "Mutual information": filters["mi_churn"],
        "ANOVA F": filters["anova_f_churn"],
        "Random Forest importance": embedded["rf_importance_churn"],
        "|L1-logistic coefficient|": embedded["l1_coef_churn"].abs(),
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    order = filters["mi_churn"].sort_values().index
    for ax, (title, score) in zip(axes, churn_scores.items()):
        ax.barh(order, score[order], color="steelblue")
        ax.set_title(title)
    fig.suptitle("Churn feature scores by method (TRAIN)")
    fig.tight_layout()
    save_matplotlib(fig, "feature_scores_churn")
    return fig


def evaluate_logistic(X_train, y_train, X_test, y_test, columns):
    """Fit scaled Logistic Regression on TRAIN with the given columns and return TEST metrics."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_train[columns], y_train)
    probability = model.predict_proba(X_test[columns])[:, 1]
    predicted = (probability >= 0.5).astype(int)
    return {
        "n_features": len(columns),
        "precision": precision_score(y_test, predicted),
        "recall": recall_score(y_test, predicted),
        "f1": f1_score(y_test, predicted),
        "roc_auc": roc_auc_score(y_test, probability),
        "pr_auc": average_precision_score(y_test, probability),
        "mean_predicted_prob": probability.mean(),
    }


def recency_baseline_row(train_df, test_df):
    """Recency-only reference: churn if recency > TRAIN median; recency itself is the ranking score."""
    threshold = train_df["recency"].median()      # threshold from TRAIN, so nothing is tuned on test
    predicted = (test_df["recency"] > threshold).astype(int)
    y = test_df["churn"]
    return {
        "n_features": 1,
        "precision": precision_score(y, predicted),
        "recall": recall_score(y, predicted),
        "f1": f1_score(y, predicted),
        "roc_auc": roc_auc_score(y, test_df["recency"]),
        "pr_auc": average_precision_score(y, test_df["recency"]),
        "mean_predicted_prob": np.nan,
    }


def compare_on_test(train_df, test_df, encoder, selected):
    """Compare Logistic Regression with all features vs the selected subset on TEST, plus the recency baseline."""
    X_train, X_test = make_model_matrix(train_df, encoder), make_model_matrix(test_df, encoder)
    y_train, y_test = churn_target(train_df), churn_target(test_df)
    rows = {
        "Recency only (baseline, threshold = train median)": recency_baseline_row(train_df, test_df),
        "LogReg - all features": evaluate_logistic(X_train, y_train, X_test, y_test, list(X_train.columns)),
        "LogReg - selected features": evaluate_logistic(X_train, y_train, X_test, y_test, selected),
    }
    result = pd.DataFrame(rows).T
    result["n_features"] = result["n_features"].astype(int)
    return result.round(3)


def segment_ablation(train_df, test_df, encoder):
    """Check what the segment adds: Logistic Regression on TEST with and without the segment columns."""
    X_train, X_test = make_model_matrix(train_df, encoder), make_model_matrix(test_df, encoder)
    y_train, y_test = churn_target(train_df), churn_target(test_df)
    segment_cols = [c for c in X_train.columns if c.startswith("seg_")]
    rfm = config.RFM_COLUMNS
    feature_sets = {
        "Segment only": segment_cols,
        "RFM only": rfm,
        "RFM + segment": rfm + segment_cols,
        "All features without segment": [c for c in X_train.columns if c not in segment_cols],
        "All features with segment": list(X_train.columns),
    }
    rows = {name: evaluate_logistic(X_train, y_train, X_test, y_test, cols) for name, cols in feature_sets.items()}
    result = pd.DataFrame(rows).T
    result["n_features"] = result["n_features"].astype(int)
    return result.round(3)


def save_selected_features(churn_features, spend_features):
    """Save the chosen feature lists to models/selected_features.json for training and the app."""
    data = {"churn": churn_features, "spend": spend_features,
            "log1p_features": config.LOG1P_FEATURES, "segment_column": config.SEGMENT_COLUMN}
    config.SELECTED_FEATURES_JSON.write_text(json.dumps(data, indent=2))
    return data


def run_feature_selection(train_df):
    """Run every selection method on TRAIN, choose the churn and spend feature lists and save them.

    Same steps as notebook 03, in one call (used by train.py).
    """
    encoder = fit_segment_encoder(train_df)
    X = make_model_matrix(train_df, encoder)
    y_churn, y_spend = churn_target(train_df), spend_target(train_df)
    filters = filter_scores(X, y_churn, y_spend)
    table = selection_table(filters, rfe_logistic(X, y_churn), embedded_scores(X, y_churn, y_spend))
    churn_features = choose_features(table, filters, "votes_churn", "mi_churn")
    spend_features = choose_features(table, filters, "votes_spend", "mi_spend")
    return save_selected_features(churn_features, spend_features)
