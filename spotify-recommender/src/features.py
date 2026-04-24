"""
features.py — Feature selection, scaling, and encoding.

Key design decisions documented here:
  - We use only audio features (not text metadata) for content-based similarity.
  - StandardScaler is required because raw feature scales differ wildly.
  - Genre encoding is optional — only used for cluster analysis, NOT for
    primary content similarity (categorical ordinal encoding is misleading).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# All Spotify-provided audio features.
# We exclude 'key' (musical key — categorical, 0–11 with no ordering)
# and 'mode' (binary: major/minor — too coarse to be useful alone).
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# Subset emphasising the perceptual "feel" of a song.
# Tempo and duration are less about emotional character and more about structure,
# so they are downweighted implicitly by being excluded from CORE_FEATURES.
CORE_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]


def select_features(df: pd.DataFrame, feature_list: list = None) -> pd.DataFrame:
    """
    Extract a sub-DataFrame containing only the selected audio features.

    Raises ValueError if any feature is absent so failures are loud and obvious.
    """
    if feature_list is None:
        feature_list = AUDIO_FEATURES

    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise ValueError(f"Features not found in DataFrame: {missing}")

    print(f"\n[Features] Selecting {len(feature_list)} audio features:")
    for f in feature_list:
        lo, hi = df[f].min(), df[f].max()
        print(f"  {f:<22} range=[{lo:.3f}, {hi:.3f}]  mean={df[f].mean():.3f}")

    return df[feature_list].copy()


def scale_features(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler (zero mean, unit variance) to all features.

    WHY SCALING IS MANDATORY FOR COSINE / KNN SIMILARITY
    =====================================================
    Raw scales vary enormously:

      loudness   → roughly –60 to 0  (range ≈ 60 units)
      tempo      → roughly  60 to 210 (range ≈ 150 units)
      danceability → 0 to 1          (range =  1 unit)

    Without scaling, a 10-unit difference in loudness completely dwarfs a
    0.3-unit difference in danceability — even though both may carry equal
    musical meaning.  StandardScaler fixes this by giving every feature
    mean=0, std=1.

    Alternative considered: MinMaxScaler (maps to [0,1]).
    We prefer StandardScaler because it is less sensitive to extreme outliers
    and is the standard for ML similarity pipelines.

    Returns
    -------
    scaled_df  : DataFrame with same columns/index, values normalised
    scaler     : fitted StandardScaler (needed to transform query vectors)
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(feature_df.values)
    scaled_df = pd.DataFrame(
        scaled_array,
        columns=feature_df.columns,
        index=feature_df.index,
    )

    print(f"\n[Features] StandardScaler applied to {len(feature_df.columns)} features.")
    print("  Raw feature stats (before scaling):")
    for col in feature_df.columns:
        print(
            f"    {col:<22}  mean={feature_df[col].mean():+8.3f}  "
            f"std={feature_df[col].std():7.3f}"
        )
    print("  After scaling: all features → mean≈0, std≈1")

    return scaled_df, scaler


def encode_genre(
    df: pd.DataFrame, col: str = "playlist_genre"
) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Integer-encode a genre column using LabelEncoder.

    IMPORTANT CAVEAT
    ----------------
    This assigns integers like pop=3, rock=4, edm=1.  That ordering is
    completely arbitrary — it does NOT mean edm is between pop and rock on
    any musical axis.  Therefore this encoding is ONLY used as an extra
    dimension for KMeans cluster analysis, never for cosine similarity
    (where the false ordinality would distort distances).

    For proper categorical use you would want one-hot encoding, but that
    adds N_genres dimensions and inflates the feature space significantly.
    """
    encoder = LabelEncoder()
    df = df.copy()
    df[f"{col}_encoded"] = encoder.fit_transform(df[col])

    print(f"\n[Features] Encoded '{col}' → '{col}_encoded':")
    for idx, cls in enumerate(encoder.classes_):
        count = (df[col] == cls).sum()
        print(f"  {cls:<15} → {idx}  ({count:,} songs)")

    return df, encoder


def get_feature_weights() -> dict:
    """
    Optional per-feature weights for weighted cosine similarity.

    Currently returns uniform weights.  This is where you can inject
    domain knowledge — e.g., downweight liveness (a recording artifact)
    or upweight valence + energy (the strongest emotional signals).
    The weights below are starting points, not tuned values.
    """
    return {
        "danceability":     1.0,
        "energy":           1.0,
        "loudness":         0.8,  # correlated with energy; slight downweight
        "speechiness":      0.7,  # about recording type more than song character
        "acousticness":     1.0,
        "instrumentalness": 1.0,
        "liveness":         0.6,  # recording artefact, least informative
        "valence":          1.0,
        "tempo":            0.9,
        "duration_ms":      0.5,  # weakest musical signal
    }
