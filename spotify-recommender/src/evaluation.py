"""
evaluation.py — Proxy evaluation metrics for an offline recommendation system.

THE FUNDAMENTAL CHALLENGE
==========================
We have no ground-truth labels.  We do not know whether a real user
would enjoy any given recommendation.  This is the defining difficulty
of offline recommender evaluation.

Instead, we use PROXY METRICS — measurable quantities that *correlate*
with recommendation quality but are not perfect substitutes for real feedback.

Implemented metrics
-------------------
1. genre_consistency   — fraction of recs in the same genre as the seed
2. similarity_stats    — descriptive stats on similarity scores
3. diversity_score     — artist and genre spread in the recommendation set
4. holdout_test        — leave-one-out: can the model recover a hidden liked song?
5. compare_recommenders — aggregate the above across multiple test cases

IMPORTANT CAVEAT
================
A model can game every proxy metric without actually being good:
  - Genre consistency → just return same-genre songs always.
  - High similarity   → return nearly identical songs (boring).
  - High diversity    → return random songs.

Good evaluation balances multiple metrics and honest self-reflection.
The metrics here are starting points, not definitive benchmarks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Genre Consistency
# ---------------------------------------------------------------------------

def genre_consistency(recommendations: pd.DataFrame, seed_genre: str) -> float | None:
    """
    What fraction of recommendations share the seed song's genre?

    Interpretation
    --------------
      > 0.8   High — genre-focused (may be too narrow, low discovery value)
      0.5–0.8 Medium — good balance of consistency and variety
      < 0.5   Low — cross-genre (could indicate poor fit OR beneficial discovery)

    Note: genre_consistency near 1.0 is NOT always desirable.  Discovery
    of similar-sounding songs from adjacent genres is often what users want.
    """
    if "playlist_genre" not in recommendations.columns or len(recommendations) == 0:
        return None
    same = (recommendations["playlist_genre"] == seed_genre).sum()
    return round(same / len(recommendations), 4)


# ---------------------------------------------------------------------------
# 2. Similarity Score Statistics
# ---------------------------------------------------------------------------

def similarity_stats(recommendations: pd.DataFrame) -> dict:
    """
    Descriptive statistics on the 'similarity_score' column.

    What we look for
    ----------------
      mean > 0.70 → recommendations are meaningfully close
      std  < 0.10 → scores are consistent (model is confident)
      min  > 0.30 → even worst recommendation has some relevance

    A very high mean (> 0.97) might indicate the dataset is too small
    or that features are too coarse to discriminate between songs.
    """
    if "similarity_score" not in recommendations.columns:
        return {}
    s = recommendations["similarity_score"]
    return {
        "mean":   round(float(s.mean()), 4),
        "std":    round(float(s.std()), 4),
        "min":    round(float(s.min()), 4),
        "max":    round(float(s.max()), 4),
        "median": round(float(s.median()), 4),
    }


# ---------------------------------------------------------------------------
# 3. Diversity Score
# ---------------------------------------------------------------------------

def diversity_score(recommendations: pd.DataFrame) -> dict:
    """
    Measure artist and genre spread in the recommendation set.

    artist_diversity = unique_artists / total_recommendations
    genre_diversity  = unique_genres  / total_recommendations

    Both range from 0 (all same) to 1 (all unique).

    Low artist_diversity (e.g., 0.1) means the model leans heavily on
    one artist — a known bias called "same-artist recommendation collapse".
    This is common when a prolific artist has many similar-sounding tracks.
    """
    if len(recommendations) == 0:
        return {"artist_diversity": 0.0, "genre_diversity": 0.0,
                "unique_artists": 0, "unique_genres": 0}

    n = len(recommendations)
    unique_artists = recommendations["track_artist"].nunique()
    unique_genres  = recommendations["playlist_genre"].nunique() \
        if "playlist_genre" in recommendations.columns else 0

    return {
        "artist_diversity": round(unique_artists / n, 4),
        "genre_diversity":  round(unique_genres  / n, 4),
        "unique_artists":   unique_artists,
        "unique_genres":    unique_genres,
    }


# ---------------------------------------------------------------------------
# 4. Holdout Test (Leave-One-Out)
# ---------------------------------------------------------------------------

def holdout_test(
    user_recommender,  # must have .recommend(liked_song_names, n)
    liked_songs: list[str],
    n: int = 20,
) -> dict:
    """
    Hold out one liked song, build a profile from the rest, and check whether
    the held-out song appears in the top-N recommendations.

    This is the standard offline evaluation technique for user-profile models:
    "leave-one-out" evaluation.

    Procedure
    ---------
    1. Set aside the last song in liked_songs as the held-out item.
    2. Build the user profile from the remaining songs.
    3. Generate N recommendations.
    4. Check if the held-out song appears; record its rank.

    Interpretation
    --------------
      Found at rank 1–5    → excellent recall
      Found at rank 6–20   → good recall
      Not found in top 20  → poor recall (but not necessarily wrong —
                             the liked songs may simply be eclectic)
    """
    if len(liked_songs) < 2:
        print("  [holdout_test] Requires at least 2 liked songs — skipping.")
        return {}

    held_out   = liked_songs[-1]
    train_songs = liked_songs[:-1]

    print(f"\n  Holdout test:")
    print(f"    Training likes : {train_songs}")
    print(f"    Held-out song  : '{held_out}'")

    recs = user_recommender.recommend(train_songs, n=n)
    if len(recs) == 0:
        return {"found": False, "rank": None, "total": n}

    held_lower = held_out.strip().lower()
    rec_names  = recs["track_name"].str.lower().tolist()

    found = any(held_lower in name for name in rec_names)
    rank  = None
    if found:
        rank = next(
            (i + 1 for i, name in enumerate(rec_names) if held_lower in name),
            None,
        )

    status = f"FOUND at rank {rank}" if found else "NOT FOUND"
    print(f"    Result         : {status}  (searched top {n})")

    return {"found": found, "rank": rank, "total": n}


# ---------------------------------------------------------------------------
# 5. Compare Recommenders
# ---------------------------------------------------------------------------

def compare_recommenders(
    recommenders: dict,   # {name: recommender_object}
    test_seeds: list[tuple[str, str]],  # [(song_name, genre), …]
    n: int = 10,
) -> pd.DataFrame:
    """
    Evaluate multiple recommenders on the same test seeds and aggregate metrics.

    Returns a DataFrame with one row per model showing averages of:
      - genre_consistency
      - artist_diversity
      - genre_diversity
      - mean_similarity (if available)

    The test_seeds list should cover multiple genres to give a fair picture.
    """
    rows = []

    for model_name, rec in recommenders.items():
        model_rows = []

        for song_name, genre in test_seeds:
            try:
                if model_name == "GenrePopularity":
                    recs = rec.recommend(genre=genre, n=n)
                elif model_name == "Popularity":
                    recs = rec.recommend(n=n)
                else:
                    recs = rec.recommend(song_name, n=n)

                if len(recs) == 0:
                    continue

                row = {
                    "seed": song_name,
                    "genre_consistency": genre_consistency(recs, genre),
                    **diversity_score(recs),
                }
                stats = similarity_stats(recs)
                row["mean_similarity"] = stats.get("mean")
                model_rows.append(row)

            except Exception as exc:
                print(f"  [compare] {model_name} on '{song_name}': {exc}")

        if model_rows:
            agg = pd.DataFrame(model_rows).mean(numeric_only=True)
            rows.append({
                "model":                model_name,
                "avg_genre_consistency": round(agg.get("genre_consistency", 0), 3),
                "avg_artist_diversity":  round(agg.get("artist_diversity", 0), 3),
                "avg_genre_diversity":   round(agg.get("genre_diversity", 0), 3),
                "avg_similarity":
                    round(agg["mean_similarity"], 3)
                    if pd.notna(agg.get("mean_similarity")) else "N/A",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Edge-Case Evaluation
# ---------------------------------------------------------------------------

def evaluate_edge_cases(
    recommender,  # ContentBasedRecommender
    df: pd.DataFrame,
) -> None:
    """
    Run the recommender against documented edge cases and record behaviour.

    This mirrors what a proper QA pass looks like before shipping a model.
    Knowing how the model fails is as important as knowing when it succeeds.
    """
    sep = "-" * 55
    print(f"\n{sep}")
    print("  EDGE-CASE EVALUATION")
    print(sep)

    # --- EC1: Song not in dataset ---
    print("\n[EC1] Non-existent song:")
    print("  Query: 'This Song Absolutely Does Not Exist XYZ'")
    result = recommender.recommend("This Song Absolutely Does Not Exist XYZ", n=5)
    outcome = "Empty DataFrame (correct)" if len(result) == 0 else f"{len(result)} results (unexpected)"
    print(f"  Result: {outcome}")

    # --- EC2: Niche/low-popularity song ---
    print("\n[EC2] Niche song (popularity < 5):")
    niche = df[df["track_popularity"] < 5]
    if len(niche) > 0:
        niche_name = niche.iloc[0]["track_name"]
        niche_artist = niche.iloc[0]["track_artist"]
        print(f"  Query: '{niche_name}' by {niche_artist}")
        result = recommender.recommend(niche_name, n=5)
        if len(result) > 0:
            mean_pop = result["track_popularity"].mean()
            print(f"  Got {len(result)} recommendations. Mean popularity of recs: {mean_pop:.1f}")
            print(
                "  Insight: content-based model is popularity-blind — "
                "niche songs can still get quality recommendations."
            )
    else:
        print("  No songs with popularity < 5 in dataset — skipping.")

    # --- EC3: Partial name match ---
    print("\n[EC3] Partial name ('Shape' → expects 'Shape of You' or similar):")
    result = recommender.recommend("Shape", n=5)
    if len(result) > 0:
        print(f"  Got {len(result)} results — partial matching works.")
    else:
        print("  No results — partial match fallback did not find a song.")

    # --- EC4: Same-artist bias ---
    print("\n[EC4] Same-artist bias — top 2 prolific artists:")
    top_artists = df["track_artist"].value_counts().head(2)
    for artist, count in top_artists.items():
        seed_row = df[df["track_artist"] == artist].iloc[0]
        seed_name = seed_row["track_name"]
        result = recommender.recommend(seed_name, n=10)
        if len(result) > 0:
            same_pct = (result["track_artist"] == artist).sum() / len(result)
            flag = " ⚠ HIGH SAME-ARTIST BIAS" if same_pct > 0.5 else ""
            print(
                f"  Artist: '{artist}' ({count} songs). "
                f"Same-artist in top 10: {same_pct:.0%}{flag}"
            )
