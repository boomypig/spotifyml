"""
visualization.py — All EDA and results plots for the recommendation system.

Design rules:
  - Every plot is SAVED to outputs/graphs/ (never just displayed).
  - Every plot prints a one-paragraph interpretation for the learner.
  - Labels, titles, and legends are always present.
  - matplotlib backend is set to 'Agg' (non-interactive) for script use.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # must be before pyplot import; prevents display attempts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

GRAPH_DIR = "outputs/graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

# ---------- shared style -----------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})
PALETTE = "husl"


def _save(filename: str, interpretation: str = "") -> str:
    """Tight-layout, save, close, print path and interpretation."""
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")
    if interpretation:
        print(f"  Interpretation: {interpretation}")
    return path


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_popularity_distribution(df: pd.DataFrame) -> str:
    print("\n[Viz] Popularity distribution")
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(df["track_popularity"], bins=50, color="steelblue",
            edgecolor="white", alpha=0.85)
    ax.axvline(df["track_popularity"].median(), color="tomato",
               linestyle="--", label=f"Median = {df['track_popularity'].median():.0f}")
    ax.axvline(df["track_popularity"].mean(), color="orange",
               linestyle="--", label=f"Mean = {df['track_popularity'].mean():.1f}")
    ax.set_xlabel("Track Popularity (0–100)")
    ax.set_ylabel("Song Count")
    ax.set_title("Distribution of Track Popularity")
    ax.legend()

    return _save(
        "popularity_distribution.png",
        f"Mean={df['track_popularity'].mean():.1f}, "
        f"Median={df['track_popularity'].median():.0f}. "
        "A left skew (many low-popularity songs) is common — most Spotify tracks "
        "are niche; only a few achieve high play counts. "
        "This means a popularity baseline heavily favours a tiny fraction of tracks.",
    )


def plot_genre_distribution(df: pd.DataFrame) -> str:
    print("\n[Viz] Genre distribution")
    counts = df["playlist_genre"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))

    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette(PALETTE, len(counts)))
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Song Count")
    ax.set_title("Song Count per Genre")
    ax.tick_params(axis="x", rotation=15)

    return _save(
        "genre_distribution.png",
        f"Most common genre: '{counts.index[0]}' ({counts.iloc[0]:,} songs). "
        "Class imbalance in genres means a genre-based recommender is biased "
        "toward larger genres — less common genres receive fewer diverse candidates.",
    )


def plot_correlation_heatmap(df: pd.DataFrame, features: list) -> str:
    print("\n[Viz] Correlation heatmap")
    available = [f for f in features if f in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, linewidths=0.4, ax=ax,
    )
    ax.set_title("Audio Feature Correlation Matrix")
    plt.xticks(rotation=45, ha="right")

    # find top correlations for interpretation text
    pairs = [
        (corr.columns[i], corr.columns[j], corr.iloc[i, j])
        for i in range(len(corr.columns))
        for j in range(i + 1, len(corr.columns))
    ]
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top3 = ", ".join(f"{a}↔{b}={c:.2f}" for a, b, c in pairs[:3])

    return _save(
        "correlation_heatmap.png",
        f"Top correlations: {top3}. "
        "Highly correlated features carry redundant information; "
        "including both slightly inflates their combined weight in similarity. "
        "energy↔loudness is typically strong because loud = high-energy production.",
    )


def plot_energy_vs_danceability(df: pd.DataFrame) -> str:
    print("\n[Viz] Energy vs Danceability scatter")
    fig, ax = plt.subplots(figsize=(10, 6))

    genres = df["playlist_genre"].unique()
    palette = sns.color_palette(PALETTE, len(genres))
    for genre, color in zip(genres, palette):
        sub = df[df["playlist_genre"] == genre]
        ax.scatter(sub["energy"], sub["danceability"],
                   label=genre, alpha=0.35, s=12, color=color)

    ax.set_xlabel("Energy")
    ax.set_ylabel("Danceability")
    ax.set_title("Energy vs Danceability by Genre")
    ax.legend(markerscale=2, loc="lower right")

    return _save(
        "energy_vs_danceability.png",
        "EDM and pop cluster toward the top-right (high energy + high danceability). "
        "Rock spans the energy axis but stays moderate-to-low danceability. "
        "Genre separation here validates that audio features carry genre signal — "
        "a good sign for content-based filtering.",
    )


def plot_valence_vs_acousticness(df: pd.DataFrame) -> str:
    print("\n[Viz] Valence vs Acousticness scatter (coloured by popularity)")
    fig, ax = plt.subplots(figsize=(10, 6))

    sc = ax.scatter(
        df["acousticness"], df["valence"],
        c=df["track_popularity"], cmap="YlOrRd", alpha=0.4, s=10,
    )
    plt.colorbar(sc, ax=ax, label="Track Popularity")
    ax.set_xlabel("Acousticness")
    ax.set_ylabel("Valence (Happiness)")
    ax.set_title("Valence vs Acousticness  (colour = popularity)")

    return _save(
        "valence_vs_acousticness.png",
        "Acoustic songs span the full valence range — from happy folk to sad singer-songwriter. "
        "Electronic songs also vary widely in valence. "
        "Popularity appears scattered throughout: mood alone does not predict popularity.",
    )


def plot_boxplots_by_genre(df: pd.DataFrame, feature: str = "energy") -> str:
    print(f"\n[Viz] Boxplot of '{feature}' by genre")
    order = (
        df.groupby("playlist_genre")[feature]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data_by_genre = [df[df["playlist_genre"] == g][feature].values for g in order]

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(data_by_genre, labels=order, patch_artist=True, notch=False)

    palette = sns.color_palette(PALETTE, len(order))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    ax.set_xlabel("Genre")
    ax.set_ylabel(feature.replace("_", " ").title())
    ax.set_title(f"{feature.replace('_', ' ').title()} Distribution by Genre")
    ax.tick_params(axis="x", rotation=15)

    return _save(
        f"boxplot_{feature}_by_genre.png",
        f"Wide boxes = high within-genre variance (harder to use '{feature}' as a filter). "
        "Well-separated medians across genres = useful signal for content-based filtering. "
        "Large within-genre spread warns that genre label alone cannot predict this feature.",
    )


def plot_tempo_distribution(df: pd.DataFrame) -> str:
    print("\n[Viz] Tempo distribution")
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(df["tempo"], bins=60, color="mediumpurple", edgecolor="white", alpha=0.85)
    ax.axvline(120, color="tomato", linestyle="--", alpha=0.7, label="120 BPM (dance standard)")
    ax.axvline(df["tempo"].mean(), color="orange", linestyle="--",
               label=f"Mean = {df['tempo'].mean():.0f} BPM")
    ax.set_xlabel("Tempo (BPM)")
    ax.set_ylabel("Song Count")
    ax.set_title("Tempo Distribution")
    ax.legend()

    return _save(
        "tempo_distribution.png",
        f"Peaks near 90–100 and 120–130 BPM reflect common music conventions "
        "(hip-hop, pop, dance). Multi-modal bumps suggest genre sub-clusters. "
        f"Mean = {df['tempo'].mean():.0f} BPM.",
    )


# ---------------------------------------------------------------------------
# Clustering plots
# ---------------------------------------------------------------------------

def plot_elbow_curve(k_values: list, inertias: list) -> str:
    print("\n[Viz] KMeans elbow curve")
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(k_values, inertias, "b-o", markersize=6)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow Method — Optimal K for KMeans")
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)

    return _save(
        "elbow_curve.png",
        "Look for the 'elbow' — where inertia starts decreasing linearly. "
        "Adding clusters beyond that point yields diminishing returns. "
        "The chosen K should balance cluster granularity with interpretability.",
    )


def plot_pca_clusters(
    df: pd.DataFrame,
    scaled_features: pd.DataFrame,
    cluster_col: str = "cluster",
) -> str:
    print("\n[Viz] PCA cluster visualisation")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled_features.values)
    var = pca.explained_variance_ratio_

    n_clusters = df[cluster_col].nunique()
    palette = sns.color_palette(PALETTE, n_clusters)

    fig, ax = plt.subplots(figsize=(12, 7))
    for k in sorted(df[cluster_col].unique()):
        mask = (df[cluster_col] == k).values
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=f"Cluster {k}", alpha=0.45, s=12,
            color=palette[int(k) % len(palette)],
        )

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title("PCA 2-D Projection of Song Clusters")
    ax.legend(markerscale=2, ncol=2, loc="upper right")

    return _save(
        "pca_clusters.png",
        f"PCA reduces {scaled_features.shape[1]} features to 2D. "
        f"PC1={var[0]*100:.1f}%, PC2={var[1]*100:.1f}% of total variance explained. "
        "Overlapping clusters indicate that audio features alone cannot fully separate "
        "musical groups — which is expected (genre/era/lyrics also matter).",
    )


# ---------------------------------------------------------------------------
# Result plots
# ---------------------------------------------------------------------------

def plot_recommendation_scores(
    recommendations: pd.DataFrame, model_name: str = "Model"
) -> str:
    print(f"\n[Viz] Recommendation scores — {model_name}")
    if "similarity_score" not in recommendations.columns:
        print("  No similarity_score column — skipping.")
        return ""

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [
        (r["track_name"][:22] + "…" if len(r["track_name"]) > 22 else r["track_name"])
        for _, r in recommendations.iterrows()
    ]
    scores = recommendations["similarity_score"].values
    colors = sns.color_palette("RdYlGn", len(labels))

    ax.barh(range(len(labels)), scores, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_title(f"Recommendation Scores — {model_name}")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.7, color="red", linestyle="--", alpha=0.5, label="0.7 threshold")
    ax.legend()
    ax.invert_yaxis()

    safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
    return _save(
        f"rec_scores_{safe_name}.png",
        f"Mean similarity = {scores.mean():.3f}. "
        "Scores above 0.7 indicate strong feature-level similarity. "
        "If all scores are > 0.95, features may be too coarse to discriminate well.",
    )


def plot_recommendation_genre_distribution(
    recommendations: pd.DataFrame,
    seed_genre: str,
    model_name: str = "Model",
) -> str:
    print(f"\n[Viz] Genre distribution of recommendations — {model_name}")
    if "playlist_genre" not in recommendations.columns:
        print("  No playlist_genre column — skipping.")
        return ""

    counts = recommendations["playlist_genre"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.0f%%",
        colors=sns.color_palette(PALETTE, len(counts)),
        startangle=90,
    )
    ax.set_title(
        f"Genre Distribution of Recommendations\n"
        f"Seed genre: {seed_genre}  |  Model: {model_name}"
    )

    same_pct = counts.get(seed_genre, 0) / len(recommendations) * 100
    safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
    return _save(
        f"genre_dist_{safe_name}.png",
        f"Same-genre fraction: {same_pct:.0f}%. "
        "High same-genre = genre-accurate but less discovery. "
        "Cross-genre recs may surface genuinely similar-feeling songs from other genres.",
    )


def plot_model_comparison(comparison_df: pd.DataFrame) -> str:
    print("\n[Viz] Model comparison bar chart")
    if comparison_df.empty:
        print("  No data — skipping.")
        return ""

    metrics = [c for c in
               ["avg_genre_consistency", "avg_artist_diversity", "avg_genre_diversity"]
               if c in comparison_df.columns]
    if not metrics:
        return ""

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    palette = sns.color_palette(PALETTE, len(comparison_df))
    for ax, metric in zip(axes, metrics):
        vals = pd.to_numeric(comparison_df[metric], errors="coerce").fillna(0)
        bars = ax.bar(comparison_df["model"], vals, color=palette)
        ax.set_title(metric.replace("avg_", "").replace("_", " ").title())
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=20)
        ax.bar_label(bars, fmt="%.2f", padding=3)

    plt.suptitle("Recommender Model Comparison", fontsize=14, fontweight="bold")

    return _save(
        "model_comparison.png",
        "Higher genre consistency = more genre-focused results. "
        "Higher diversity = broader artist/genre spread. "
        "Content-based models should land between the baseline (high consistency, low diversity) "
        "and a random recommender (high diversity, low consistency).",
    )


def plot_user_profile(user_profile: pd.Series, model_name: str = "User") -> str:
    """Bar chart of the user taste profile vector (scaled feature values)."""
    print(f"\n[Viz] User taste profile — {model_name}")
    show = [f for f in user_profile.index if f in [
        "danceability", "energy", "loudness", "valence",
        "acousticness", "instrumentalness", "liveness",
    ]]
    vals = user_profile[show].values

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["steelblue" if v >= 0 else "tomato" for v in vals]
    ax.bar(show, vals, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Scaled Feature Value  (0 = dataset average)")
    ax.set_title(f"User Taste Profile — {model_name}")
    ax.tick_params(axis="x", rotation=20)

    safe_name = model_name.lower().replace(" ", "_")
    return _save(
        f"user_profile_{safe_name}.png",
        "Blue bars = above-average for this feature; red bars = below-average. "
        "The profile is the mean of all liked songs' scaled feature vectors. "
        "Recommendations should have a similar feature shape to match user preference.",
    )
