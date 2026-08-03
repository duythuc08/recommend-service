"""
User-Based Memory CF using Surprise (scikit-surprise, Hug 2020).

Tai lieu: https://surprise.readthedocs.io/en/stable/knn_inspired.html
Paper: Hug, N. (2020). Surprise: A Python library for recommender
systems. Journal of Open Source Software, 5(52), 2174.
"""

import pandas as pd
from surprise import Dataset, KNNWithMeans, Reader
from surprise.trainset import Trainset

from app.core.config import settings


def build_utility_matrix(review_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tao utility long-format DataFrame (user_id, movie_id, rating)
    tu explicit rating trong bang review.
    """
    review_df = review_df.copy()
    review_df["has_explicit"] = True
    return review_df[["user_id", "movie_id", "rating", "has_explicit"]].reset_index(drop=True)


def build_surprise_trainset(utility_long: pd.DataFrame) -> Trainset:
    """
    Surprise can input DataFrame 3 cot (user, item, rating) va Reader
    khai bao rating_scale. Rating cua he thong nam trong [1, 5].
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(utility_long[["user_id", "movie_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    return trainset


def train_knn_model(trainset: Trainset) -> KNNWithMeans:
    """
    KNNWithMeans: prediction = mean_u + weighted_avg(sim(u,v) * (r_vi - mean_v)).
    Day la cong thuc User-Based Memory CF da duoc Surprise implement san.
    """
    sim_options = {
        "name": "cosine",
        "user_based": True,
        "min_support": settings.cf_min_co_rated_items,
    }
    algo = KNNWithMeans(
        k=settings.cf_top_k,
        min_k=2,
        sim_options=sim_options,
    )
    algo.fit(trainset)
    return algo


def predict_ratings_for_user(
    user_id: str,
    algo: KNNWithMeans,
    trainset: Trainset,
    candidate_movie_ids: list[int],
) -> dict[int, tuple[float, int]]:
    """
    Goi .predict() cho tung candidate movie.
    Tra ve dict[movie_id -> (predicted_score, neighbor_count)].
    """
    try:
        trainset.to_inner_uid(user_id)
    except ValueError:
        return {}

    predictions = {}
    for movie_id in candidate_movie_ids:
        try:
            trainset.to_inner_iid(movie_id)
        except ValueError:
            continue

        pred = algo.predict(user_id, movie_id)
        if pred.details.get("was_impossible", False):
            continue
        neighbor_count = int(pred.details.get("actual_k", 0))
        predictions[movie_id] = (float(pred.est), neighbor_count)

    return predictions
