"""
User-Based Memory CF dùng thư viện Surprise (scikit-surprise, Hug 2020).
Không tự viết công thức cosine similarity / prediction tay - dùng đúng
KNNWithMeans có sẵn, theo định hướng đề tài ứng dụng (không phải nghiên cứu).

Tài liệu: https://surprise.readthedocs.io/en/stable/knn_inspired.html
Paper: Hug, N. (2020). Surprise: A Python library for recommender
systems. Journal of Open Source Software, 5(52), 2174.
"""

import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans
from surprise.trainset import Trainset

from app.core.config import settings

def build_utility_matrix(review_df: pd.DataFrame, implicit_df: pd.DataFrame, use_implicit: bool = True,) -> pd.DataFrame:
    """
    Gộp explicit rating và implicit-converted rating thành 1 utility
    long-format DataFrame (user_id, movie_id, rating).

    use_implicit=False  -> CF PURE: chỉ dùng explicit rating (review),
        implicit_df bị bỏ qua hoàn toàn. Dùng để chạy baseline trước
        khi thêm implicit, đúng lộ trình "chạy thuần trước, cải thiện dần"
        đã chốt.
    use_implicit=True   -> CF + IMPLICIT: gộp cả 2 nguồn. Quy tắc: cặp
        (user, movie) có ở cả 2 nguồn -> CHỈ dùng explicit (implicit
        chỉ bù sparsity cho cặp CHƯA có rating thực)
    """
    review_df = review_df.copy()
    review_df["has_explicit"] = True

    if not use_implicit or implicit_df.empty:
        utility_long = review_df[["user_id","movie_id","rating","has_explicit"]].reset_index(drop=True)
        return utility_long

    implicit_part = implicit_df.rename(columns={"y":"rating"})[["user_id","movie_id","rating"]].copy()
    implicit_part["has_explicit"] = False

    combined = pd.concat([review_df[["user_id", "movie_id", "rating", "has_explicit"]], implicit_part])
    combined = combined.sort_values("has_explicit", ascending=False)
    utility_long = combined.drop_duplicates(subset=["user_id", "movie_id"], keep="first").reset_index(drop=True)
    return utility_long


def build_surprise_trainset(utility_long: pd.DataFrame) -> Trainset:
    """
    Surprise yêu cầu input dạng DataFrame 3 cột (user, item, rating) và
    1 Reader khai báo rating_scale. Rating của hệ thống này nằm trong
    [1, 5] cho explicit và [1.0, 4.0] cho implicit-converted (theo tanh
    conversion) - dùng chung scale (1,5) cho Reader là an toàn.
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(utility_long[["user_id", "movie_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    return trainset


def train_knn_model(trainset: Trainset) -> KNNWithMeans:
    """
    KNNWithMeans: prediction = mean_u + weighted_avg(sim(u,v) * (r_vi - mean_v))
    Đây chính là công thức User-Based Memory CF của đề tài, đã được
    implement sẵn, có test, có paper - không tự viết công thức.
    """
    sim_options = {
        "name": "cosine",
        "user_based": True,  # User-Based CF (không phải Item-Based)
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
    Gọi .predict() cho từng candidate movie. Surprise tự xử lý trường hợp
    user/item chưa biết (cold-start) bằng cách trả về global mean - vì
    hệ thống đã có cơ chế cold-start riêng (popularity-based) ở tầng
    router, nên ở đây CHỈ predict cho user đã có trong trainset; nếu
    không có (rải user mới hoàn toàn), trả về {} để router fallback
    sang cold-start.

    Trả về dict[movie_id -> (predicted_score, neighbor_count)].
    neighbor_count = pred.details['actual_k'] (số neighbor thực sự đóng góp).
    """
    try:
        trainset.to_inner_uid(user_id)
    except ValueError:
        return {}  # user chưa từng xuất hiện trong utility matrix -> cold start

    predictions = {}
    for movie_id in candidate_movie_ids:
        try:
            trainset.to_inner_iid(movie_id)
        except ValueError:
            continue  # movie chưa từng được rate bởi ai -> Surprise sẽ trả global mean, không đáng tin, bỏ qua

        pred = algo.predict(user_id, movie_id)
        # pred.details có thể chứa {'was_impossible': True, 'reason': ...}
        # khi không đủ neighbor hợp lệ (vd toàn bộ neighbor có sim <= min_similarity)
        if pred.details.get("was_impossible", False):
            continue
        neighbor_count = int(pred.details.get("actual_k", 0))
        predictions[movie_id] = (float(pred.est), neighbor_count)

    return predictions
