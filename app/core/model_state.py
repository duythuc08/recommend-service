"""
Model state - giữ Surprise trainset + KNNWithMeans model trong memory.

Thiết kế: 1 instance singleton được load lúc app startup và refresh khi
gọi /train. Với quy mô 1010 user x 102 movie, RAM trong process là đủ,
không cần thêm hạ tầng Redis.
"""
import threading
from datetime import datetime

import pandas as pd

from app.core.cf_engine import build_utility_matrix, build_surprise_trainset, train_knn_model, predict_ratings_for_user
from app.core.cold_start import compute_popularity_scores
from app.core.config import settings
from app.core.implicit_scoring import build_implicit_scores, convert_to_rating_scale
from app.db.queries import (
    load_all_reviews, load_all_activity_logs, load_candidate_movies,
    load_scoring_params, load_all_excluded_movie_ids_bulk, upsert_user_preferences,
)


class ModelState:
    def __init__(self):
        self._lock = threading.Lock()
        self.algo = None
        self.trainset = None
        self.utility_long: pd.DataFrame | None = None
        self.candidate_movies: pd.DataFrame | None = None
        self.last_trained_at: datetime | None = None
        self.is_ready: bool = False
        self.last_use_implicit: bool = settings.cf_use_implicit  # mode của lần train gần nhất

    def train(self, db_session, use_implicit: bool | None = None) -> dict:
        """
        use_implicit: None -> dùng default từ config (settings.cf_use_implicit).
        Truyền riêng True/False để chạy 1 lần dưới mode khác, phục vụ
        so sánh benchmark CF Pure vs CF+Implicit ngay trên cùng 1 service
        mà không cần đổi config/restart.
        """
        if use_implicit is None:
            use_implicit = settings.cf_use_implicit

        t0 = datetime.utcnow()

        review_df = load_all_reviews(db_session)
        candidate_df = load_candidate_movies(db_session)

        if use_implicit:
            scoring_params = load_scoring_params(db_session)
            alpha = scoring_params.get("ALPHA")
            s0 = scoring_params.get("S0")

            activity_df = load_all_activity_logs(db_session)
            explicit_pairs = set(zip(review_df["user_id"], review_df["movie_id"]))
            implicit_raw = build_implicit_scores(activity_df, explicit_pairs=explicit_pairs, now=t0, alpha=alpha)
            implicit_scored = convert_to_rating_scale(implicit_raw, s0=s0)
        else:
            activity_df = pd.DataFrame()
            implicit_scored = pd.DataFrame(columns=["user_id", "movie_id", "y"])

        utility_long = build_utility_matrix(review_df, implicit_scored, use_implicit=use_implicit)
        trainset = build_surprise_trainset(utility_long)
        algo = train_knn_model(trainset)

        with self._lock:
            self.algo = algo
            self.trainset = trainset
            self.utility_long = utility_long
            self.candidate_movies = candidate_df
            self.last_trained_at = t0
            self.is_ready = True
            self.last_use_implicit = use_implicit

        elapsed = (datetime.utcnow() - t0).total_seconds()
        batch_stats = self.predict_all_users(db_session)

        return {
            "trained_at": t0.isoformat(),
            "elapsed_seconds": elapsed,
            "use_implicit": use_implicit,
            "n_users": utility_long["user_id"].nunique() if not utility_long.empty else 0,
            "n_movies_in_matrix": utility_long["movie_id"].nunique() if not utility_long.empty else 0,
            "n_candidate_movies": len(candidate_df) if candidate_df is not None else 0,
            "n_explicit_ratings": len(review_df),
            "n_activity_logs": len(activity_df),
            **batch_stats,
        }

    def predict_all_users(self, db_session) -> dict:
        """
        Sau khi train() xong, tinh prediction cho TOAN BO user va UPSERT vao user_preference.

        source duoc ghi theo dung mode dang chay:
          - "cf_implicit"            : CF voi implicit feedback
          - "cf_pure"                : CF chi dung explicit rating
          - "cold_start_popularity"  : user it tuong tac, fallback popularity
        """
        t0 = datetime.utcnow()
        algo, trainset, utility_long, candidate_movies, _ = self.get_snapshot()

        if utility_long is None or utility_long.empty or candidate_movies is None:
            return {"n_users_processed": 0, "n_predictions_written": 0, "batch_elapsed_seconds": 0.0}

        cf_source = "cf_implicit" if self.last_use_implicit else "cf_pure"

        all_user_ids = utility_long["user_id"].unique().tolist()
        all_candidate_ids = candidate_movies["movie_id"].tolist()

        excluded_map = load_all_excluded_movie_ids_bulk(db_session)

        all_predictions: list[dict] = []
        n_users_processed = 0

        # Cache popularity scores de khong tinh lai nhieu lan cho nhieu cold-start user
        _popularity_cache: dict[int, float] | None = None

        for user_id in all_user_ids:
            excluded = excluded_map.get(str(user_id), set())
            candidate_ids = [m for m in all_candidate_ids if m not in excluded]
            if not candidate_ids:
                continue

            k_u = int((utility_long["user_id"] == user_id).sum())
            if k_u < settings.cold_start_min_interactions:
                # Cold-start: tinh popularity va ghi vao DB luon
                if _popularity_cache is None:
                    _popularity_cache = compute_popularity_scores(db_session, all_candidate_ids)
                for movie_id in candidate_ids:
                    score = _popularity_cache.get(movie_id, 0.0)
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
                # Surprise khong du neighbor hop le -> fallback popularity
                if _popularity_cache is None:
                    _popularity_cache = compute_popularity_scores(db_session, all_candidate_ids)
                for movie_id in candidate_ids:
                    score = _popularity_cache.get(movie_id, 0.0)
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
                    "source": cf_source,
                })
            n_users_processed += 1

        n_written = upsert_user_preferences(db_session, all_predictions)
        elapsed = (datetime.utcnow() - t0).total_seconds()

        return {
            "n_users_processed": n_users_processed,
            "n_predictions_written": n_written,
            "batch_elapsed_seconds": elapsed,
        }

    def get_snapshot(self):
        with self._lock:
            return (self.algo, self.trainset, self.utility_long, self.candidate_movies, self.last_trained_at)


model_state = ModelState()
