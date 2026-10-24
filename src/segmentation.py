"""K-Means customer segmentation on recency, frequency and monetary (RFM)."""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
import config
from eda import save_matplotlib


def rfm_matrix(features_df):
    """Return the RFM columns with log1p applied to the skewed ones (frequency, monetary)."""
    rfm = features_df[config.RFM_COLUMNS].copy()
    for column in config.RFM_LOG_COLUMNS:
        rfm[column] = np.log1p(rfm[column])
    return rfm


def evaluate_k(train_df):
    """Fit K-Means for each k in K_RANGE on TRAIN and return inertia and silhouette score per k."""
    # Scaler fitted on TRAIN only, so test customers never influence the segments.
    scaled = StandardScaler().fit_transform(rfm_matrix(train_df))
    rows = []
    for k in config.K_RANGE:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(scaled)
        rows.append({"k": k, "inertia": kmeans.inertia_,
                     "silhouette": silhouette_score(scaled, kmeans.labels_)})
    return pd.DataFrame(rows)


def plot_k_selection(k_scores):
    """Save the elbow plot (inertia) and the silhouette plot for each k."""
    fig_elbow, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_scores["k"], k_scores["inertia"], marker="o")
    ax.set_title("Elbow method")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.grid(alpha=0.3)
    save_matplotlib(fig_elbow, "kmeans_elbow")

    fig_sil, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_scores["k"], k_scores["silhouette"], marker="o", color="darkorange")
    ax.set_title("Silhouette score")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Silhouette score (higher = better separated)")
    ax.grid(alpha=0.3)
    save_matplotlib(fig_sil, "kmeans_silhouette")
    return fig_elbow, fig_sil


def name_clusters(train_df, cluster_ids):
    """Give each cluster a name by ranking the clusters' average RFM values.

    Rule (chosen after looking at the real k=4 averages, see notebook 02):
      - the cluster with the highest average recency (longest since last order) -> "Lost"
      - of the rest, the highest average frequency -> "Champions"
      - of the rest, the lowest average frequency  -> "One-off buyers"
      - any cluster in between                      -> "Loyal" (numbered if more than one)
    Ranking is used instead of fixed thresholds because cluster averages can sit
    right on a median and flip names by accident.
    """
    means = train_df[config.RFM_COLUMNS].groupby(cluster_ids).mean()
    names = {}
    lost = means["recency"].idxmax()
    names[int(lost)] = "Lost"

    rest = means.drop(index=lost).sort_values("frequency", ascending=False)
    names[int(rest.index[0])] = "Champions"
    if len(rest) > 1:
        names[int(rest.index[-1])] = "One-off buyers"
    middle = rest.index[1:-1]
    for number, cluster in enumerate(middle, start=1):
        names[int(cluster)] = "Loyal" if len(middle) == 1 else f"Loyal {number}"
    return names


def fit_segmentation(train_df, k):
    """Fit scaler + K-Means on TRAIN, name the clusters, save everything to models/ and return it."""
    scaler = StandardScaler().fit(rfm_matrix(train_df))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE)
    cluster_ids = kmeans.fit_predict(scaler.transform(rfm_matrix(train_df)))
    names = name_clusters(train_df, pd.Series(cluster_ids, index=train_df.index))

    bundle = {"scaler": scaler, "kmeans": kmeans, "segment_names": names}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.SEGMENTATION_MODEL)
    return bundle


def load_segmentation():
    """Load the saved scaler, K-Means model and segment names."""
    return joblib.load(config.SEGMENTATION_MODEL)


def assign_segment(features_df, bundle=None):
    """Return a DataFrame with segment id and segment name for each customer, using the SAVED models.

    The app calls this too, so new customers are segmented exactly like training customers.
    """
    if bundle is None:
        bundle = load_segmentation()
    scaled = bundle["scaler"].transform(rfm_matrix(features_df))
    segment = bundle["kmeans"].predict(scaled)
    result = pd.DataFrame({"segment": segment}, index=features_df.index)
    result["segment_name"] = result["segment"].map(bundle["segment_names"])
    return result


def segment_summary(table):
    """Return per segment: customers, average RFM, churn rate and average future spend."""
    summary = table.groupby("segment_name").agg(
        customers=("churn", "size"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        churn_rate=("churn", "mean"),
        avg_future_spend=("future_spend", "mean"),
    )
    summary["share_of_customers"] = summary["customers"] / summary["customers"].sum()
    return summary.sort_values("churn_rate").round(3)


def plot_pca_segments(table, bundle, name):
    """Save a 2D PCA scatter of customers coloured by segment (PCA fitted on the scaled RFM)."""
    scaled = bundle["scaler"].transform(rfm_matrix(table))
    points = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(scaled)
    plot_df = pd.DataFrame({"PC1": points[:, 0], "PC2": points[:, 1],
                            "segment": table["segment_name"].values})
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="segment", s=12, alpha=0.6, ax=ax)
    ax.set_title(f"Customer segments in 2D (PCA of scaled RFM) - {name}")
    save_matplotlib(fig, f"segments_pca_{name.lower()}")
    return fig


def add_segments_to_tables(train_df, test_df, bundle):
    """Add segment and segment_name columns to the train and test tables and save them again."""
    train_df = train_df.drop(columns=["segment", "segment_name"], errors="ignore")
    test_df = test_df.drop(columns=["segment", "segment_name"], errors="ignore")
    train_df = train_df.join(assign_segment(train_df, bundle))
    test_df = test_df.join(assign_segment(test_df, bundle))
    train_df.to_parquet(config.TRAIN_TABLE_PARQUET)
    test_df.to_parquet(config.TEST_TABLE_PARQUET)
    return train_df, test_df
