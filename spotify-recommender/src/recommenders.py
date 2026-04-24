"""
recommenders.py — Five recommendation strategies in increasing sophistication.

Progression (intentional):
  1. PopularityRecommender      — global top songs.  No user signal at all.
  2. GenrePopularityRecommender — top songs within a genre.  Coarse preference.
  3. ContentBasedRecommender    — cosine similarity on audio features.
  4. NearestNeighborRecommender — sklearn NearestNeighbors (memory-efficient).
  5. UserProfileRecommender     — average liked-song vector → rank all songs.
  6. ClusteringRecommender      — KMeans cluster membership.

Every class exposes the same pattern:
    rec.fit(df, ...)         — learn from the dataset
    rec.recommend(query, n)  — return a DataFrame of top-N recommendations

This uniform interface makes swapping and comparing models straightforward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_index(df: pd.DataFrame, song_name: str, artist: str | None = None) -> int | None:
    """
    Return the DataFrame integer-index of a song by name (+ optional artist).

    Behaviour:
      - Exact case-insensitive match preferred.
      - Falls back to substring match if no exact hit.
      - Prints a warning if multiple matches are found.
      - Returns None if nothing matches at all.
    """
    name_lower = song_name.strip().lower()

    exact_mask = df["track_name"].str.lower() == name_lower
    if artist:
        exact_mask &= df["track_artist"].str.lower() == artist.strip().lower()
    exact_hits = df[exact_mask]

    if len(exact_hits) > 0:
        if len(exact_hits) > 1:
            print(f"  [WARN] {len(exact_hits)} exact matches for '{song_name}' — using first:")
            for _, r in exact_hits.head(3).iterrows():
                print(f"    • {r['track_name']} by {r['track_artist']} ({r['playlist_genre']})")
        return exact_hits.index[0]

    # Substring fallback
    partial_mask = df["track_name"].str.lower().str.contains(name_lower, regex=False)
    if artist:
        partial_mask &= df["track_artist"].str.lower().str.contains(
            artist.strip().lower(), regex=False
        )
    partial_hits = df[partial_mask]

    if len(partial_hits) == 0:
        print(f"  [ERROR] '{song_name}' not found in dataset.")
        return None

    print(f"  [INFO] No exact match for '{song_name}'. "
          f"Using closest: '{partial_hits.iloc[0]['track_name']}' "
          f"by {partial_hits.iloc[0]['track_artist']}")
    return partial_hits.index[0]


# ---------------------------------------------------------------------------
# 1. Popularity Recommender (global baseline)
# ---------------------------------------------------------------------------

class PopularityRecommender:
    """
    Baseline: return the most-popular songs globally.

    Strengths  : trivially simple; popular songs are often decent.
    Weaknesses : zero personalization; rich-get-richer effect;
                 will recommend the same list to everyone always.

    Use when   : no user signal is available at all (cold start).
    """

    def fit(self, df: pd.DataFrame) -> "PopularityRecommender":
        self._df = df.copy()
        print(f"[PopularityRecommender] Fitted on {len(df):,} songs.")
        return self

    def recommend(
        self,
        n: int = 10,
        exclude_ids: list | None = None,
    ) -> pd.DataFrame:
        results = self._df.copy()
        if exclude_ids:
            results = results[~results["track_id"].isin(exclude_ids)]
        results = results.sort_values("track_popularity", ascending=False)
        cols = ["track_name", "track_artist", "playlist_genre", "track_popularity"]
        return results[cols].head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Genre Popularity Recommender
# ---------------------------------------------------------------------------

class GenrePopularityRecommender:
    """
    Baseline: most-popular songs within the user-specified genre.

    Strengths  : respects one dimension of preference (genre).
    Weaknesses : still ignores audio character; genre labels are noisy
                 (based on playlist curatorship, not music taxonomy).

    Use when   : user's genre preference is known but nothing else is.
    """

    def fit(self, df: pd.DataFrame) -> "GenrePopularityRecommender":
        self._df = df.copy()
        self.available_genres = sorted(df["playlist_genre"].unique())
        print(f"[GenrePopularityRecommender] Fitted. Genres: {self.available_genres}")
        return self

    def recommend(
        self,
        genre: str,
        n: int = 10,
        exclude_ids: list | None = None,
    ) -> pd.DataFrame:
        if genre not in self.available_genres:
            print(f"  [ERROR] Genre '{genre}' not recognised. "
                  f"Available: {self.available_genres}")
            return pd.DataFrame()

        results = self._df[self._df["playlist_genre"] == genre].copy()
        if exclude_ids:
            results = results[~results["track_id"].isin(exclude_ids)]
        results = results.sort_values("track_popularity", ascending=False)
        cols = ["track_name", "track_artist", "playlist_genre", "track_popularity"]
        return results[cols].head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Content-Based Recommender (precomputed cosine similarity)
# ---------------------------------------------------------------------------

class ContentBasedRecommender:
    """
    Find the N songs whose audio-feature vectors are closest in angle
    to the seed song (cosine similarity).

    HOW IT WORKS
    ============
    1. Each song is represented as a point in 10-dimensional feature space
       (danceability, energy, loudness, …).
    2. We precompute the full N×N cosine similarity matrix once at fit time.
    3. At query time, we look up the seed song's row and sort by similarity.

    WHY COSINE (not Euclidean distance)?
    =====================================
    Cosine similarity measures the *angle* between vectors, ignoring their
    absolute magnitude.  Two songs that have the same *shape* of features
    are considered similar even if one is consistently louder or faster.
    This is preferable for audio where relative feature ratios matter more
    than absolute values.

    MEMORY NOTE
    ===========
    A 30k × 30k float64 matrix ≈ 7 GB.  For this ~30k-row dataset we use
    float32 (≈ 3.5 GB) or accept the full size.  For datasets larger than
    ~50k songs, switch to NearestNeighborRecommender (no stored matrix).
    """

    def fit(
        self, df: pd.DataFrame, scaled_features: pd.DataFrame
    ) -> "ContentBasedRecommender":
        self._df = df.reset_index(drop=True).copy()
        self._feats = scaled_features.reset_index(drop=True)

        print(
            f"[ContentBasedRecommender] Computing {len(df):,}×{len(df):,} "
            "similarity matrix …"
        )
        mat = self._feats.values.astype(np.float32)
        self._sim = cosine_similarity(mat)

        mb = self._sim.nbytes / 1e6
        print(f"  Matrix shape : {self._sim.shape}   Memory : {mb:.0f} MB")
        return self

    def recommend(
        self,
        song_name: str,
        artist: str | None = None,
        n: int = 10,
        explain: bool = True,
    ) -> pd.DataFrame:
        idx = _find_index(self._df, song_name, artist)
        if idx is None:
            return pd.DataFrame()

        seed = self._df.iloc[idx]
        print(
            f"\n  Seed: '{seed['track_name']}' by {seed['track_artist']}"
            f"  (genre={seed['playlist_genre']}, popularity={seed['track_popularity']})"
        )

        scores = self._sim[idx].copy()
        scores[idx] = -1  # exclude the seed song itself

        top_idx = np.argsort(scores)[::-1][:n]
        recs = self._df.iloc[top_idx].copy()
        recs["similarity_score"] = scores[top_idx].round(4)

        if explain:
            self._explain(seed, self._df.iloc[top_idx])

        cols = ["track_name", "track_artist", "playlist_genre",
                "track_popularity", "similarity_score"]
        return recs[cols].reset_index(drop=True)

    def _explain(self, seed: pd.Series, recs: pd.DataFrame) -> None:
        """Print a feature-level explanation of why the top recommendation is similar."""
        show = ["danceability", "energy", "valence", "acousticness", "tempo"]
        show = [f for f in show if f in seed.index]
        top = recs.iloc[0]

        print("\n  --- WHY THESE SONGS ARE SIMILAR ---")
        print(f"  {'Feature':<22}  {'Seed':>8}  {'Top rec':>8}  {'|diff|':>8}")
        print("  " + "-" * 52)
        for f in show:
            diff = abs(seed[f] - top[f])
            print(f"  {f:<22}  {seed[f]:>8.3f}  {top[f]:>8.3f}  {diff:>8.3f}")
        print(
            "\n  High similarity means these songs sit close together in "
            "audio-feature space — similar energy, mood, and acoustic character."
        )


# ---------------------------------------------------------------------------
# 4. Nearest Neighbor Recommender (memory-efficient alternative)
# ---------------------------------------------------------------------------

class NearestNeighborRecommender:
    """
    Same goal as ContentBasedRecommender but uses sklearn's NearestNeighbors.

    Key difference
    ==============
    - Does NOT store the full N×N matrix.  Computes neighbours on demand.
    - Uses cosine metric via brute-force search (tree methods don't support cosine).
    - Slightly higher per-query cost but dramatically lower memory footprint.

    Use when
    ========
    - Dataset > 100k songs where the N×N matrix would exceed available RAM.
    - You can afford a slower query in exchange for lower startup memory cost.
    """

    def __init__(self, n_neighbors: int = 20):
        # +1 because the query song itself is always returned as neighbour #0
        self._model = NearestNeighbors(
            n_neighbors=n_neighbors + 1,
            metric="cosine",
            algorithm="brute",
        )
        self._n = n_neighbors

    def fit(
        self, df: pd.DataFrame, scaled_features: pd.DataFrame
    ) -> "NearestNeighborRecommender":
        self._df = df.reset_index(drop=True).copy()
        self._feats = scaled_features.reset_index(drop=True)
        print(f"[NearestNeighborRecommender] Fitting on {len(df):,} songs …")
        self._model.fit(self._feats.values)
        print("  Ready.")
        return self

    def recommend(
        self,
        song_name: str,
        artist: str | None = None,
        n: int = 10,
    ) -> pd.DataFrame:
        idx = _find_index(self._df, song_name, artist)
        if idx is None:
            return pd.DataFrame()

        seed = self._df.iloc[idx]
        print(
            f"\n  Seed (NN): '{seed['track_name']}' by {seed['track_artist']}"
        )

        query = self._feats.iloc[idx].values.reshape(1, -1)
        distances, indices = self._model.kneighbors(query, n_neighbors=n + 1)
        distances, indices = distances[0], indices[0]

        # cosine distance = 1 − cosine_similarity  →  similarity = 1 − distance
        mask = indices != idx
        indices = indices[mask][:n]
        sim_scores = 1 - distances[mask][:n]

        recs = self._df.iloc[indices].copy()
        recs["similarity_score"] = sim_scores.round(4)

        cols = ["track_name", "track_artist", "playlist_genre",
                "track_popularity", "similarity_score"]
        return recs[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. User Profile Recommender
# ---------------------------------------------------------------------------

class UserProfileRecommender:
    """
    Simulate a user by averaging the feature vectors of their liked songs,
    then rank all songs by cosine similarity to that average vector.

    HOW IT WORKS
    ============
    1. For each liked song, extract its scaled feature vector.
    2. Average all those vectors → "user taste centroid".
    3. Compute cosine similarity of every song to the centroid.
    4. Return top-N, excluding already-liked songs.

    This is also called "item-average profile" or "implicit user KNN" in
    academic literature.

    WHEN IT BREAKS
    ==============
    If a user likes both death metal and ambient classical, the centroid
    falls somewhere in the middle of feature space — close to neither.
    Solution: cluster the liked songs and build sub-profiles per cluster
    (out of scope here; flagged in FUTURE WORK).
    """

    def fit(
        self, df: pd.DataFrame, scaled_features: pd.DataFrame
    ) -> "UserProfileRecommender":
        self._df = df.reset_index(drop=True).copy()
        self._feats = scaled_features.reset_index(drop=True)
        print(f"[UserProfileRecommender] Fitted on {len(df):,} songs.")
        return self

    def build_profile(
        self, liked_song_names: list[str]
    ) -> tuple[pd.Series | None, list[int]]:
        """
        Average feature vector of the liked songs.

        Returns (user_profile_Series, list_of_found_indices).
        """
        found_indices, not_found = [], []

        for name in liked_song_names:
            mask = self._df["track_name"].str.lower().str.contains(
                name.strip().lower(), regex=False
            )
            hits = self._df[mask]
            if len(hits) > 0:
                found_indices.append(hits.index[0])
            else:
                not_found.append(name)

        if not_found:
            print(f"  [WARN] Songs not found in dataset: {not_found}")
        if not found_indices:
            print("  [ERROR] No liked songs could be located.")
            return None, []

        print(f"  Located {len(found_indices)}/{len(liked_song_names)} liked songs.")

        liked_vecs = self._feats.iloc[found_indices]
        user_profile = liked_vecs.mean(axis=0)

        print("\n  User taste profile (mean scaled audio features):")
        for feat, val in user_profile.items():
            bar_len = int(abs(val) * 6)
            bar = ("▲" if val >= 0 else "▼") * min(bar_len, 12)
            print(f"    {feat:<22} {val:+.3f}  {bar}")

        return user_profile, found_indices

    def recommend(
        self, liked_song_names: list[str], n: int = 10
    ) -> pd.DataFrame:
        user_profile, liked_indices = self.build_profile(liked_song_names)
        if user_profile is None:
            return pd.DataFrame()

        profile_vec = user_profile.values.reshape(1, -1)
        similarities = cosine_similarity(profile_vec, self._feats.values)[0]

        for idx in liked_indices:
            similarities[idx] = -1  # exclude already-liked songs

        top_idx = np.argsort(similarities)[::-1][:n]
        recs = self._df.iloc[top_idx].copy()
        recs["similarity_score"] = similarities[top_idx].round(4)

        cols = ["track_name", "track_artist", "playlist_genre",
                "track_popularity", "similarity_score"]
        return recs[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Clustering Recommender (KMeans)
# ---------------------------------------------------------------------------

class ClusteringRecommender:
    """
    Cluster all songs with KMeans, then recommend within the seed's cluster.

    HOW IT WORKS
    ============
    1. Fit KMeans on scaled feature vectors → each song gets a cluster label.
    2. For a seed song, identify its cluster.
    3. Return songs from the same cluster, ranked by Euclidean distance to
       the cluster centroid (closest to centroid = most representative).

    WHY CLUSTERING?
    ===============
    - Discovers natural musical neighbourhoods without supervision.
    - Fast at inference (no similarity matrix, just cluster lookup).
    - Interpretable: you can describe each cluster's audio profile.

    LIMITATION: cluster boundaries are hard; a song just outside a boundary
    might be more similar to the seed than the nearest in-cluster song.
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self._kmeans = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        )
        self.cluster_labels: np.ndarray | None = None

    def fit(
        self, df: pd.DataFrame, scaled_features: pd.DataFrame
    ) -> "ClusteringRecommender":
        self._df = df.reset_index(drop=True).copy()
        self._feats = scaled_features.reset_index(drop=True)

        print(
            f"[ClusteringRecommender] KMeans with K={self.n_clusters} "
            f"on {len(df):,} songs …"
        )
        self.cluster_labels = self._kmeans.fit_predict(self._feats.values)
        self._df["cluster"] = self.cluster_labels

        sizes = pd.Series(self.cluster_labels).value_counts().sort_index()
        print("  Cluster sizes:")
        for k, cnt in sizes.items():
            print(f"    Cluster {k:2d}: {cnt:5,} songs")

        return self

    def find_optimal_k(
        self,
        scaled_features: pd.DataFrame,
        k_range: range = range(2, 16),
    ) -> tuple[list, list]:
        """Return (k_values, inertias) for elbow-method plot."""
        print("[ClusteringRecommender] Elbow search …")
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(scaled_features.values)
            inertias.append(km.inertia_)
            print(f"  K={k:2d}  inertia={km.inertia_:,.0f}")
        return list(k_range), inertias

    def recommend(self, song_name: str, n: int = 10) -> pd.DataFrame:
        idx = _find_index(self._df, song_name)
        if idx is None:
            return pd.DataFrame()

        seed_cluster = int(self._df.iloc[idx]["cluster"])
        seed_name = self._df.iloc[idx]["track_name"]
        print(f"\n  '{seed_name}' is in cluster {seed_cluster}.")

        in_cluster = self._df[
            (self._df["cluster"] == seed_cluster) & (self._df.index != idx)
        ]

        centroid = self._kmeans.cluster_centers_[seed_cluster]
        cluster_feats = self._feats.iloc[in_cluster.index].values
        distances = np.linalg.norm(cluster_feats - centroid, axis=1)

        in_cluster = in_cluster.copy()
        in_cluster["distance_to_centroid"] = distances
        in_cluster = in_cluster.sort_values("distance_to_centroid")

        cols = ["track_name", "track_artist", "playlist_genre",
                "track_popularity", "distance_to_centroid"]
        return in_cluster[cols].head(n).reset_index(drop=True)

    def get_cluster_profiles(self) -> pd.DataFrame:
        """
        Describe each cluster by mean audio features and a plain-English label.
        This is the interpretability step — what does each cluster mean musically?
        """
        print("\n  Cluster Audio Profiles:")
        rows = []
        for k in range(self.n_clusters):
            mask = self._df["cluster"] == k
            subset = self._df[mask]
            feat_means = self._feats.iloc[subset.index].mean()
            top_genre = subset["playlist_genre"].mode().iloc[0]
            desc = _describe_cluster(feat_means)
            rows.append({
                "cluster": k,
                "size": len(subset),
                "top_genre": top_genre,
                "description": desc,
            })
        profiles = pd.DataFrame(rows)
        print(profiles.to_string(index=False))
        return profiles


def _describe_cluster(profile: pd.Series) -> str:
    traits = []
    if "energy" in profile.index:
        traits.append("high-energy" if profile["energy"] > 0.3 else "low-energy")
    if "valence" in profile.index:
        traits.append("upbeat" if profile["valence"] > 0.3 else "melancholic")
    if "acousticness" in profile.index:
        traits.append("acoustic" if profile["acousticness"] > 0.3 else "electronic")
    if "danceability" in profile.index and profile["danceability"] > 0.3:
        traits.append("danceable")
    if "instrumentalness" in profile.index and profile["instrumentalness"] > 0.5:
        traits.append("instrumental")
    return ", ".join(traits) if traits else "mixed"
