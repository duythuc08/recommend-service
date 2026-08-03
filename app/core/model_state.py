"""
Model state - giu Surprise trainset + KNNWithMeans model trong memory.

Mot singleton duoc train luc app startup va refresh khi goi /train.
"""
import threading
from datetime import datetime

import pandas as pd

from app.core.cf_engine import (
    build_surprise_trainset,
    build_utility_matrix,
    predict_ratings_for_user,
    train_knn_model,
)
from app.core.cold_start import compute_popularity_scores
from app.core.config import settings
from app.db.queries import (
    delete_stale_user_preferences,
    load_all_excluded_movie_ids_bulk,
    load_all_reviews,
    load_candidate_movies,
    save_utility_matrix,
    upsert_user_preferences,
)


CF_SOURCE = "cf_pure"


class ModelState:
    def __init__(self):
        self._lock = threading.Lock()
        self.algo = None
        self.trainset = None
        self.utility_long: pd.DataFrame | None = None
        self.candidate_movies: pd.DataFrame | None = None
        self.last_trained_at: datetime | None = None
        self.is_ready: bool = False

    def train(self, db_session) -> dict:
        """
        Train User-Based CF chi tu explicit rating trong bang review.
        """
        t0 = datetime.utcnow()

        review_df = load_all_reviews(db_session)
        candidate_df = load_candidate_movies(db_session)

        utility_long = build_utility_matrix(review_df)
        trainset = build_surprise_trainset(utility_long)
        algo = train_knn_model(trainset)

        save_utility_matrix(db_session, utility_long)

        with self._lock:
            self.algo = algo
            self.trainset = trainset
            self.utility_long = utility_long
            self.candidate_movies = candidate_df
            self.last_trained_at = t0
            self.is_ready = True

        elapsed = (datetime.utcnow() - t0).total_seconds()
        batch_stats = self.predict_all_users(db_session)

        return {
            "trained_at": t0.isoformat(),
            "elapsed_seconds": elapsed,
            "n_users": utility_long["user_id"].nunique() if not utility_long.empty else 0,
            "n_movies_in_matrix": utility_long["movie_id"].nunique() if not utility_long.empty else 0,
            "n_candidate_movies": len(candidate_df) if candidate_df is not None else 0,
            "n_explicit_ratings": len(review_df),
            **batch_stats,
        }

    def predict_all_users(self, db_session) -> dict:
        """
        Sau khi train() xong, tinh prediction cho toan bo user va upsert vao user_preference.
        """
        t0 = datetime.utcnow()
        algo, trainset, utility_long, candidate_movies, _ = self.get_snapshot()

        if utility_long is None or utility_long.empty or candidate_movies is None:
            return {
                "n_users_processed": 0,
                "n_predictions_written": 0,
                "n_stale_predictions_deleted": 0,
                "batch_elapsed_seconds": 0.0,
            }

        all_user_ids = utility_long["user_id"].unique().tolist()
        all_candidate_ids = candidate_movies["movie_id"].tolist()

        excluded_map = load_all_excluded_movie_ids_bulk(db_session)

        all_predictions: list[dict] = []
        n_users_processed = 0
        popularity_cache: dict[int, float] | None = None

        for user_id in all_user_ids:
            excluded = excluded_map.get(str(user_id), set())
            candidate_ids = [m for m in all_candidate_ids if m not in excluded]
            if not candidate_ids:
                continue

            k_u = int((utility_long["user_id"] == user_id).sum())
            if k_u < settings.cold_start_min_interactions:
                if popularity_cache is None:
                    popularity_cache = compute_popularity_scores(db_session, all_candidate_ids)
                for movie_id in candidate_ids:
                    score = popularity_cache.get(movie_id, 0.0)
                    all_predictions.append({
                        "user_id": user_id,
                        "movie_id": movie_id,
                        "predicted_score": score,
                        "neighbor_count": 0,
                        "source": "cold_start_popularity",
                    })
                n_users_processed += 1
                continue

            preds = predict_ratings_for_user(user_id, algo, trainset, candidate_ids)
            if not preds:
                if popularity_cache is None:
                    popularity_cache = compute_popularity_scores(db_session, all_candidate_ids)
                for movie_id in candidate_ids:
                    score = popularity_cache.get(movie_id, 0.0)
                    all_predictions.append({
                        "user_id": user_id,
                        "movie_id": movie_id,
                        "predicted_score": score,
                        "neighbor_count": 0,
                        "source": "cold_start_popularity",
                    })
                n_users_processed += 1
                continue

            for movie_id, (predicted_score, neighbor_count) in preds.items():
                all_predictions.append({
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "predicted_score": predicted_score,
                    "neighbor_count": neighbor_count,
                    "source": CF_SOURCE,
                })
            n_users_processed += 1

        n_written = upsert_user_preferences(db_session, all_predictions)
        n_deleted_stale = delete_stale_user_preferences(db_session, all_candidate_ids)
        elapsed = (datetime.utcnow() - t0).total_seconds()

        return {
            "n_users_processed": n_users_processed,
            "n_predictions_written": n_written,
            "n_stale_predictions_deleted": n_deleted_stale,
            "batch_elapsed_seconds": elapsed,
        }

    def get_snapshot(self):
        with self._lock:
            return (self.algo, self.trainset, self.utility_long, self.candidate_movies, self.last_trained_at)


model_state = ModelState()
