# Spotify Song Recommendation System

A full-scale machine learning project implementing a content-based music
recommendation system using Spotify's audio features — built to teach ML
through reasoning, not just results.

---

## Overview

This project builds a Spotify recommendation engine from scratch, progressing
from trivial baselines to a full audio-feature similarity model.  Every design
decision is explained in code comments and printed output so that the learner
understands *why*, not just *what*.

---

## Problem

Given a seed song (or a list of liked songs), return the N most sonically
similar tracks from a 30k-song corpus.  The system must:

- Explain why each song was recommended
- Evaluate its own quality honestly (with no ground-truth labels)
- Be comparable against simpler baselines

---

## Dataset

**Source:** TidyTuesday Spotify Songs dataset (Jan 2020)  
**Auto-downloaded** on first run — no manual steps needed.  
**Size:** ~32,000 tracks across 6 genres: `edm`, `latin`, `pop`, `r&b`, `rap`, `rock`

| Feature | Description |
|---|---|
| `danceability` | 0–1, suitability for dancing |
| `energy` | 0–1, perceptual intensity |
| `loudness` | dB, typically –60 to 0 |
| `speechiness` | 0–1, presence of spoken words |
| `acousticness` | 0–1, confidence of being acoustic |
| `instrumentalness` | 0–1, likelihood of no vocals |
| `liveness` | 0–1, live audience presence |
| `valence` | 0–1, musical positiveness |
| `tempo` | BPM |
| `duration_ms` | milliseconds |
| `track_popularity` | 0–100 (Spotify's metric) |

---

## Methods

| Model | Description |
|---|---|
| **Popularity Baseline** | Top-N globally popular songs |
| **Genre Popularity** | Top-N popular songs within a genre |
| **Content-Based (Cosine)** | Cosine similarity on scaled audio features |
| **NearestNeighbors** | sklearn KNN with cosine metric (memory-efficient) |
| **User Profile** | Average liked-song feature vector → rank all songs |
| **KMeans Clustering** | Cluster-then-recommend within the seed's cluster |

---

## Project Structure

```
spotify-recommender/
├── data/
│   └── spotify_songs.csv        ← auto-downloaded on first run
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py         ← load, inspect, clean
│   ├── features.py              ← feature selection + StandardScaler
│   ├── recommenders.py          ← all 5 recommender classes
│   ├── evaluation.py            ← proxy metrics + edge-case tests
│   └── visualization.py        ← all EDA + result plots
├── outputs/
│   ├── graphs/                  ← all saved PNG charts
│   └── recommendation_results.csv
├── main.py                      ← full pipeline orchestration
├── pyproject.toml
└── README.md
```

---

## How to Run

**Prerequisites:** [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# 1. Create virtual environment
uv venv

# 2. Install dependencies
uv sync

# 3. Run the full pipeline
uv run python main.py
```

The dataset downloads automatically (~5 MB).  Full run takes 2–5 minutes
depending on hardware (the N×N similarity matrix is the bottleneck).

---

## Evaluation

Since no user feedback exists, we use **proxy metrics**:

| Metric | What it measures | Caveat |
|---|---|---|
| Genre consistency | Fraction of recs in same genre as seed | High ≠ always good (less discovery) |
| Similarity score stats | Mean/std of cosine similarity | High mean might mean dataset is too small |
| Diversity score | Unique artists / unique genres in recs | High ≠ always good (may be random) |
| Holdout test | Rank of hidden liked song in recs | Leave-one-out offline evaluation |
| Edge-case tests | Behaviour on missing, niche, and biased inputs | Correctness, not quality |

---

## Results (typical run)

- **Cosine model**: mean similarity ≈ 0.88–0.95 on popular seeds
- **Genre consistency**: 60–90% same-genre recommendations depending on genre
- **Holdout test**: held-out song found in top 20 for ~50–70% of profiles
- **Same-artist bias**: 10–30% of recs for prolific artists come from same artist

---

## Limitations

1. **Audio features ≠ full taste** — lyrics, timbre, era, and cultural context are absent
2. **No real user feedback** — liked-song lists are simulated
3. **Similarity ≠ preference** — users sometimes want adjacent, not identical
4. **Cold-start** — new users and new songs cannot be personalised
5. **Popularity feedback loop** — Spotify's popularity metric favours blockbusters
6. **Genre label noise** — playlist-derived labels are not musicological ground truth

---

## Future Work

- Spotify API integration for real user listening history
- Collaborative filtering (matrix factorisation / ALS)
- Hybrid model (content + collaborative)
- Deep audio embeddings from spectrograms (CNN)
- A/B testing infrastructure to measure real engagement
- FastAPI + React web app for interactive demo
- Better evaluation via real click/save data (Precision@K, NDCG)

---

## Learning Goals

This project is intentionally transparent:

- Every data decision is printed with a `WHY:` explanation
- Feature scaling rationale is documented at the function level
- Model limitations are surfaced in the output, not hidden
- Edge cases are tested and results are printed — not assumed to work
