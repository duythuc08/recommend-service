"""
evaluate.py -- Đánh giá chất lượng mô hình CF (User-Based KNNWithMeans).

Chạy độc lập: python evaluate.py

Leave-One-Out (LOO) evaluation:
  - Test item  = 1 rating cao nhất của mỗi user (tie-break ngẫu nhiên seed=42)
  - Train set  = tất cả ratings còn lại + toàn bộ implicit logs
  - User có đúng 1 rating: bỏ qua

Chạy 2 lần:
  Lần 1: use_implicit=False (CF thuần / baseline)
  Lần 2: use_implicit=True  (CF + Implicit)
"""

import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

# ---------------------------------------------------------------------------
# Bootstrap path -- chạy từ thư mục recommend-service/
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")

from app.core.config import settings
from app.core.cf_engine import build_utility_matrix, build_surprise_trainset, train_knn_model
from app.core.implicit_scoring import build_implicit_scores, convert_to_rating_scale
from app.db.queries import (
    load_all_activity_logs,
    load_all_reviews,
    load_candidate_movies,
)
from app.db.session import SessionLocal

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helper: load movie genres từ DB (graceful nếu không có bảng)
# ---------------------------------------------------------------------------
def load_movie_genres(db) -> dict[int, set[str]]:
    """
    Trả về dict[movie_id -> set(genre_name)].
    Thử các tên bảng/cột phổ biến, trả về {} nếu không tìm thấy.
    """
    candidates = [
        # (query, movie_col, genre_col)
        (
            "SELECT movie_id, genre_name FROM movie_genres WHERE 1=1",
            "movie_id", "genre_name",
        ),
        (
            "SELECT movie_id, name FROM movie_genre mg JOIN genre g ON mg.genre_id = g.id",
            "movie_id", "name",
        ),
        (
            "SELECT movie_id, genre FROM movie_genre",
            "movie_id", "genre",
        ),
    ]
    for sql, mc, gc in candidates:
        try:
            rows = db.execute(text(sql)).fetchall()
            result: dict[int, set[str]] = defaultdict(set)
            for row in rows:
                mid = int(row[0])
                gname = str(row[1])
                result[mid].add(gname)
            print(f"  [genre] Tải {len(result)} phim có genre từ DB.")
            return dict(result)
        except Exception:
            continue
    print("  [genre] Không tìm thấy bảng genre -- bỏ qua GMR.")
    return {}


# ---------------------------------------------------------------------------
# LOO split
# ---------------------------------------------------------------------------
def loo_split(review_df: pd.DataFrame, seed: int = RANDOM_SEED):
    """
    Trả về:
      train_df  -- tất cả ratings trừ test item
      test_dict -- dict[user_id -> (movie_id, true_rating)]
      skipped   -- số user bị bỏ do < 2 ratings
    """
    rng = random.Random(seed)
    test_dict = {}
    skipped = 0
    train_rows = []

    for user_id, grp in review_df.groupby("user_id"):
        if len(grp) < 2:
            skipped += 1
            # vẫn đưa vào train để không mất dữ liệu cho implicit users
            train_rows.append(grp)
            continue

        max_rating = grp["rating"].max()
        top_rows = grp[grp["rating"] == max_rating]

        # Tie-break ngẫu nhiên, seed cố định
        test_row = top_rows.sample(n=1, random_state=seed).iloc[0]
        test_dict[user_id] = (int(test_row["movie_id"]), float(test_row["rating"]))

        rest = grp[grp.index != test_row.name]
        train_rows.append(rest)

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame(
        columns=["user_id", "movie_id", "rating"]
    )
    return train_df, test_dict, skipped


# ---------------------------------------------------------------------------
# Build implicit với explicit_pairs từ TRAIN set (không test item)
# ---------------------------------------------------------------------------
def build_implicit_for_train(activity_df: pd.DataFrame, train_df: pd.DataFrame):
    explicit_pairs = set(
        zip(train_df["user_id"].astype(str), train_df["movie_id"].astype(int))
    )
    s_df = build_implicit_scores(activity_df, explicit_pairs=explicit_pairs)
    if s_df.empty:
        return pd.DataFrame(columns=["user_id", "movie_id", "y"])
    y_df = convert_to_rating_scale(s_df)
    return y_df


# ---------------------------------------------------------------------------
# Evaluation cho 1 chế độ (use_implicit=True/False)
# ---------------------------------------------------------------------------
def run_evaluation(
    review_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    movie_genres: dict[int, set[str]],
    use_implicit: bool,
) -> dict:
    label = "CF + Implicit" if use_implicit else "CF thuần"
    print(f"\n{'='*60}")
    print(f"  Chạy: {label}")
    print(f"{'='*60}")

    # --- LOO split ---
    train_df, test_dict, skipped_users = loo_split(review_df)
    print(f"  LOO: {len(test_dict)} user test, {skipped_users} user bị bỏ (< 2 ratings)")

    # --- Build implicit (chỉ dùng explicit_pairs từ TRAIN) ---
    if use_implicit:
        print("  Đang tính implicit scores...")
        implicit_y_df = build_implicit_for_train(activity_df, train_df)
        print(f"  Implicit: {len(implicit_y_df)} cặp (user, movie)")
    else:
        implicit_y_df = pd.DataFrame(columns=["user_id", "movie_id", "y"])

    # --- Build utility matrix + train model ---
    print("  Đang build utility matrix và train model...")
    t0 = time.time()
    utility_long = build_utility_matrix(train_df, implicit_y_df, use_implicit=use_implicit)
    trainset = build_surprise_trainset(utility_long)
    algo = train_knn_model(trainset)
    train_time = time.time() - t0
    print(f"  Train xong trong {train_time:.1f}s. Utility matrix: {len(utility_long)} dòng")

    # --- Candidate movie ids ---
    all_candidate_ids = set(candidate_df["movie_id"].astype(int).tolist())

    # --- Train set lookup: set[movie_id] mỗi user trong train ---
    user_train_movies: dict = defaultdict(set)
    for _, row in train_df.iterrows():
        user_train_movies[row["user_id"]].add(int(row["movie_id"]))

    # --- Genre yêu thích từ train (dùng cho GMR) ---
    has_genre = bool(movie_genres)
    user_fav_genres: dict = {}
    if has_genre:
        for user_id, grp in train_df.groupby("user_id"):
            high_rating_movies = grp[grp["rating"] >= 4]["movie_id"].tolist()
            if not high_rating_movies:
                high_rating_movies = grp[grp["rating"] >= 3]["movie_id"].tolist()
            genres = set()
            for mid in high_rating_movies:
                genres |= movie_genres.get(int(mid), set())
            user_fav_genres[user_id] = genres

    # --- Metrics accumulators ---
    sq_errors = []
    abs_errors = []
    impossible_count = 0
    not_in_candidate_count = 0
    hits_at_5 = []
    hits_at_10 = []
    gmr_scores = []

    # Coverage: user có đủ interactions trong utility matrix
    cf_covered_users = 0
    total_utility_users = len(utility_long["user_id"].unique())
    for uid_str in utility_long["user_id"].unique():
        try:
            iuid = trainset.to_inner_uid(uid_str)
            # Đếm số interaction của user trong trainset
            if len(trainset.ur[iuid]) >= settings.cold_start_min_interactions:
                cf_covered_users += 1
        except ValueError:
            pass

    total_test = len(test_dict)
    done = 0

    for user_id, (test_movie_id, true_rating) in test_dict.items():
        done += 1
        if done % 100 == 0:
            print(f"  Progress: {done}/{total_test} user ({done*100//total_test}%)")

        uid_str = str(user_id)

        # ---- RMSE / MAE: predict test item ----
        try:
            trainset.to_inner_uid(uid_str)
            user_in_trainset = True
        except ValueError:
            user_in_trainset = False

        if user_in_trainset:
            pred = algo.predict(uid_str, test_movie_id)
            if pred.details.get("was_impossible", False):
                impossible_count += 1
            else:
                err = pred.est - true_rating
                sq_errors.append(err ** 2)
                abs_errors.append(abs(err))

        # ---- Hit Rate@K ----
        # Candidate = (NOW_SHOWING + COMING_SOON) trừ phim user đã có trong train
        train_movies = user_train_movies.get(user_id, set())
        candidates_for_user = all_candidate_ids - train_movies

        # Test item không có trong candidate set -> bỏ qua user này khi tính Hit Rate
        if test_movie_id not in candidates_for_user:
            not_in_candidate_count += 1
            continue

        # Predict scores cho candidate set
        if not user_in_trainset:
            # Cold start user -> không có prediction -> hit = 0
            hits_at_5.append(0)
            hits_at_10.append(0)
            if has_genre:
                gmr_scores.append(0.0)
            continue

        # Chỉ predict các movie đã có trong trainset (tránh global mean nhiễu)
        scored = {}
        for cid in candidates_for_user:
            try:
                trainset.to_inner_iid(cid)
            except ValueError:
                continue  # phim chưa ai rate trong train -> bỏ qua (như cf_engine.py)
            pred = algo.predict(uid_str, cid)
            if not pred.details.get("was_impossible", False):
                scored[cid] = float(pred.est)

        if not scored:
            hits_at_5.append(0)
            hits_at_10.append(0)
            if has_genre:
                gmr_scores.append(0.0)
            continue

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        top5_ids = [mid for mid, _ in ranked[:5]]
        top10_ids = [mid for mid, _ in ranked[:10]]

        hits_at_5.append(1 if test_movie_id in top5_ids else 0)
        hits_at_10.append(1 if test_movie_id in top10_ids else 0)

        # ---- GMR ----
        if has_genre:
            fav_genres = user_fav_genres.get(user_id, set())
            if fav_genres:
                match_count = sum(
                    1 for mid in top5_ids
                    if movie_genres.get(mid, set()) & fav_genres
                )
                gmr_scores.append(match_count / 5)
            else:
                gmr_scores.append(0.0)

    # --- Tổng hợp ---
    rmse = math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else float("nan")
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")
    hr5 = sum(hits_at_5) / len(hits_at_5) if hits_at_5 else float("nan")
    hr10 = sum(hits_at_10) / len(hits_at_10) if hits_at_10 else float("nan")
    gmr = sum(gmr_scores) / len(gmr_scores) if gmr_scores else float("nan")
    coverage = cf_covered_users / total_utility_users * 100 if total_utility_users else 0.0
    impossible_pct = impossible_count / total_test * 100 if total_test else 0.0

    return {
        "label": label,
        "rmse": rmse,
        "mae": mae,
        "hr5": hr5,
        "hr10": hr10,
        "gmr": gmr,
        "coverage": coverage,
        "train_time": train_time,
        "impossible_pct": impossible_pct,
        "impossible_count": impossible_count,
        "skipped_users": skipped_users,
        "not_in_candidate": not_in_candidate_count,
        "total_test": total_test,
        "total_utility_users": total_utility_users,
        "cf_covered_users": cf_covered_users,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def fmt(v, decimals=4) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def main():
    print("=== Evaluation Script: CF Model Quality ===")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    db = SessionLocal()
    try:
        print("\n[1/5] Loading explicit reviews...")
        review_df = load_all_reviews(db)
        print(f"  {len(review_df)} ratings, {review_df['user_id'].nunique()} users")

        print("[2/5] Loading activity logs...")
        activity_df = load_all_activity_logs(db)

        # parse metadata JSON nếu cần
        def _parse_meta(x):
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except Exception:
                    return {}
            return x or {}

        activity_df["metadata"] = activity_df["metadata"].apply(_parse_meta)

        # parse datetime columns
        for col in ["created_at", "updated_at", "best_value_at"]:
            if col in activity_df.columns:
                activity_df[col] = pd.to_datetime(activity_df[col], errors="coerce")

        print(f"  {len(activity_df)} logs, {activity_df['user_id'].nunique()} users")

        print("[3/5] Loading candidate movies...")
        candidate_df = load_candidate_movies(db)
        print(f"  {len(candidate_df)} candidate movies (NOW_SHOWING + COMING_SOON)")

        print("[4/5] Loading movie genres...")
        movie_genres = load_movie_genres(db)

    finally:
        db.close()

    total_ratings = len(review_df)
    total_logs = len(activity_df)

    # ---- Run 2 lần ----
    results = []
    for use_implicit in [False, True]:
        res = run_evaluation(
            review_df=review_df,
            activity_df=activity_df,
            candidate_df=candidate_df,
            movie_genres=movie_genres,
            use_implicit=use_implicit,
        )
        results.append(res)

    r_base, r_impl = results[0], results[1]

    # ---- Print bảng so sánh ----
    header = f"\n{'='*70}\n  KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH CF\n{'='*70}"
    col_w = 30

    table_lines = [
        header,
        "",
        f"{'Chỉ số':<{col_w}} {'CF thuần':>15} {'CF + Implicit':>15}",
        f"{'-'*col_w} {'-'*15} {'-'*15}",
        f"{'RMSE':<{col_w}} {fmt(r_base['rmse']):>15} {fmt(r_impl['rmse']):>15}",
        f"{'MAE':<{col_w}} {fmt(r_base['mae']):>15} {fmt(r_impl['mae']):>15}",
        f"{'Hit Rate@5':<{col_w}} {fmt(r_base['hr5']):>15} {fmt(r_impl['hr5']):>15}",
        f"{'Hit Rate@10':<{col_w}} {fmt(r_base['hr10']):>15} {fmt(r_impl['hr10']):>15}",
        f"{'Genre Match Rate':<{col_w}} {fmt(r_base['gmr']):>15} {fmt(r_impl['gmr']):>15}",
        f"{'Coverage (%)':<{col_w}} {fmt(r_base['coverage'], 2):>15} {fmt(r_impl['coverage'], 2):>15}",
        f"{'Thời gian train (s)':<{col_w}} {fmt(r_base['train_time'], 2):>15} {fmt(r_impl['train_time'], 2):>15}",
        f"{'Impossible predictions (%)':<{col_w}} {fmt(r_base['impossible_pct'], 2):>15} {fmt(r_impl['impossible_pct'], 2):>15}",
        f"{'User bỏ (< 2 ratings)':<{col_w}} {str(r_base['skipped_users']):>15} {str(r_impl['skipped_users']):>15}",
        f"{'-'*col_w} {'-'*15} {'-'*15}",
        "",
    ]

    extra_lines = [
        f"Tổng explicit ratings trong DB       : {total_ratings}",
        f"Tổng activity logs trong DB          : {total_logs}",
        f"Số user 'test item not in candidate' : {r_impl['not_in_candidate']} "
        f"({r_impl['not_in_candidate']*100/r_impl['total_test']:.1f}% tổng user test)"
        if r_impl["total_test"] else "",
        f"  (CF thuần: {r_base['not_in_candidate']} user)",
        "",
        f"Chi tiết Coverage:",
        f"  CF thuần   -- {r_base['cf_covered_users']}/{r_base['total_utility_users']} user trong utility matrix có >= {settings.cold_start_min_interactions} interactions",
        f"  CF+Implicit -- {r_impl['cf_covered_users']}/{r_impl['total_utility_users']} user trong utility matrix có >= {settings.cold_start_min_interactions} interactions",
        "",
        f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    all_output = "\n".join(table_lines + extra_lines)
    print(all_output)

    # ---- Ghi ra file ----
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write(all_output)
    print("\n>> Kết quả đã ghi vào evaluation_results.txt")


if __name__ == "__main__":
    main()
