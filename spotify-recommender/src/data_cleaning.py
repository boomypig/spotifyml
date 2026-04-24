"""
data_cleaning.py — Data loading, inspection, and cleaning.

Design philosophy:
  Every decision is explained, not just executed.
  We print what we find so the learner can see the reasoning live.
  Loading and cleaning are separate functions so each can be tested alone.
"""

import os
import requests
import pandas as pd
import numpy as np

# TidyTuesday Spotify dataset (2020) — public, no auth required.
# 32k tracks with full Spotify audio features + genre labels.
DATASET_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday"
    "/master/data/2020/2020-01-21/spotify_songs.csv"
)
DATA_PATH = "data/spotify_songs.csv"

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms",
]


def download_dataset(path: str = DATA_PATH) -> None:
    """Download the Spotify dataset if not already on disk."""
    if os.path.exists(path):
        print(f"[Data] Dataset already present at '{path}' — skipping download.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[Data] Downloading dataset from TidyTuesday GitHub …")
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()
    with open(path, "wb") as fh:
        fh.write(response.content)
    size_kb = os.path.getsize(path) / 1024
    print(f"[Data] Saved to '{path}' ({size_kb:.0f} KB).")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV and return a raw DataFrame."""
    if not os.path.exists(path):
        download_dataset(path)
    df = pd.read_csv(path)
    print(f"[Data] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from '{path}'.")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a thorough first-look at the dataset.

    This step is mandatory in any real ML project — you must understand
    what you are working with before making any cleaning or modelling
    decisions.  Skipping inspection is how silent bugs are born.
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print("  STEP 1 — DATA INSPECTION")
    print(sep)

    print(f"\nShape : {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\nColumn names and dtypes:")
    print("-" * 45)
    for col in df.columns:
        print(f"  {col:<35} {str(df[col].dtype)}")

    print("\nFirst 3 rows (transposed for readability):")
    print(df.head(3).T.to_string())

    # --- Missing values -------------------------------------------------
    print("\nMissing values:")
    print("-" * 45)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_summary = pd.DataFrame(
        {"count": missing, "pct": missing_pct}
    )
    has_missing = missing_summary[missing_summary["count"] > 0]
    if has_missing.empty:
        print("  None — clean dataset!")
    else:
        print(has_missing.to_string())

    # --- Duplicates -----------------------------------------------------
    print(f"\nDuplicate rows       : {df.duplicated().sum():,}")
    if "track_id" in df.columns:
        print(f"Duplicate track_ids  : {df['track_id'].duplicated().sum():,}")
        print(
            "  NOTE: A song can appear in multiple playlists (same track_id, "
            "different playlist_genre). We deduplicate by track_id in cleaning."
        )

    # --- Numeric summary ------------------------------------------------
    available_features = [f for f in AUDIO_FEATURES if f in df.columns]
    if "track_popularity" in df.columns:
        available_features.append("track_popularity")
    print("\nNumeric feature summary:")
    print(df[available_features].describe().round(3).to_string())

    # --- Genre distribution ---------------------------------------------
    if "playlist_genre" in df.columns:
        print("\nGenre distribution:")
        print(df["playlist_genre"].value_counts().to_string())

    # --- Audio feature explanations ------------------------------------
    print("\nAudio feature explanations (Spotify API definitions):")
    print("-" * 65)
    explanations = {
        "danceability":
            "0–1. Rhythm stability + beat strength. 1 = most danceable.",
        "energy":
            "0–1. Perceptual intensity. Fast/loud/noisy = high energy.",
        "loudness":
            "dB. Overall loudness, typically –60 to 0. Near 0 = loud.",
        "speechiness":
            "0–1. Spoken words. >0.66 = speech-only; <0.33 = music.",
        "acousticness":
            "0–1. Confidence track is acoustic (non-electronic).",
        "instrumentalness":
            "0–1. Predicts no vocals. >0.5 = likely instrumental.",
        "liveness":
            "0–1. Live audience presence. >0.8 = likely live recording.",
        "valence":
            "0–1. Musical positiveness. High = happy; Low = sad/tense.",
        "tempo":
            "BPM. Estimated tempo of the track.",
        "duration_ms":
            "Duration in milliseconds (1000 ms = 1 second).",
        "track_popularity":
            "0–100. Spotify's proprietary metric based on recent plays.",
    }
    for feat, explanation in explanations.items():
        print(f"  {feat:<22}: {explanation}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset with a documented reason for every decision.

    Returns the cleaned DataFrame with a reset index.
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print("  STEP 2 — DATA CLEANING")
    print(sep)

    original_rows = len(df)
    print(f"\nStarting rows: {original_rows:,}")

    # ------------------------------------------------------------------
    # 2.1  Drop rows where we cannot display a recommendation
    # ------------------------------------------------------------------
    print("\n[2.1] Dropping rows missing critical text fields …")
    critical_cols = [c for c in ["track_name", "track_artist", "playlist_genre"]
                     if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    print(
        f"  Dropped {before - len(df):,} rows with missing "
        f"{critical_cols}. "
        "  WHY: A recommendation without a name/artist is useless to display."
    )

    # ------------------------------------------------------------------
    # 2.2  Fill missing numeric audio features with the column median
    # ------------------------------------------------------------------
    print("\n[2.2] Filling missing numeric features with column median …")
    available_audio = [f for f in AUDIO_FEATURES if f in df.columns]
    for col in available_audio:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(
                f"  {col:<22}: filled {n_null:,} nulls with median={median_val:.4f}. "
                "  WHY median: audio features can be skewed; median is robust to outliers."
            )

    # ------------------------------------------------------------------
    # 2.3  Deduplicate by track_id (keep first occurrence)
    # ------------------------------------------------------------------
    print("\n[2.3] Removing duplicate track_ids …")
    if "track_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["track_id"], keep="first")
        removed = before - len(df)
        print(
            f"  Removed {removed:,} duplicate track_id rows. "
            "  WHY: Same song in N playlists would appear N× in similarity lookups, "
            "biasing results toward frequently-played songs."
        )
    else:
        print("  No 'track_id' column found — skipping.")

    # ------------------------------------------------------------------
    # 2.4  Validate and clamp bounded features to [0, 1]
    # ------------------------------------------------------------------
    print("\n[2.4] Validating bounded feature ranges …")
    bounded = [
        "danceability", "energy", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence",
    ]
    for col in [c for c in bounded if c in df.columns]:
        out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
        if out_of_range > 0:
            print(f"  WARNING: {out_of_range:,} out-of-range values in '{col}' — clamping.")
            df[col] = df[col].clip(0, 1)
        else:
            lo, hi = df[col].min(), df[col].max()
            print(f"  {col:<22}: OK  [{lo:.3f}, {hi:.3f}]")

    # ------------------------------------------------------------------
    # 2.5  Reset index so iloc and positional indexing stay consistent
    # ------------------------------------------------------------------
    df = df.reset_index(drop=True)

    removed_total = original_rows - len(df)
    pct = removed_total / original_rows * 100
    print(f"\nFinal rows after cleaning : {len(df):,}")
    print(f"Rows removed              : {removed_total:,}  ({pct:.1f}%)")

    return df
