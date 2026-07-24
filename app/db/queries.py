"""
Data access layer - query trực tiếp từ DB MySQL.

Nguyên tắc tránh N+1 (bài học từ UtilityMatrixBuilderTasklet cũ):
- Load TOÀN BỘ review và activity_logs trong 2 query duy nhất (không loop
  từng user/movie để query riêng).
- Pandas sẽ xử lý join/group ở tầng application sau khi load xong.
"""
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session


def load_all_reviews(db: Session) -> pd.DataFrame:
    """
    Load toàn bộ explicit rating (bảng review).
    Trả về DataFrame: user_id, movie_id, rating (1-5, float)
    """
    query = text("""
        SELECT user_id, movie_id, rating
        FROM review
        WHERE entity_status = 'ACTIVE'
          AND review_status = 'APPROVED'
    """)
    rows = db.execute(query).fetchall()
    df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])
    df["rating"] = df["rating"].astype(float)
    df["movie_id"] = df["movie_id"].astype(int)
    return df


def load_all_activity_logs(db: Session) -> pd.DataFrame:
    """
    Load toàn bộ activity log (bảng user_activity_logs).
    Trả về DataFrame: user_id, movie_id, action_type, created_at,
    updated_at, best_value_at, occurrence_count, metadata (dict hoặc None)
    """
    query = text("""
        SELECT user_id, movie_id, action_type, created_at, updated_at,
               best_value_at, occurrence_count, metadata
        FROM user_activity_logs
        WHERE entity_status = 'ACTIVE'
    """)
    rows = db.execute(query).fetchall()
    df = pd.DataFrame(rows, columns=[
        "user_id", "movie_id", "action_type", "created_at", "updated_at",
        "best_value_at", "occurrence_count", "metadata",
    ])
    df["movie_id"] = df["movie_id"].astype(int)
    df["occurrence_count"] = df["occurrence_count"].fillna(1).astype(int)
    return df


def load_candidate_movies(db: Session) -> pd.DataFrame:
    """
    Candidate set = phim đang chiếu hoặc sắp chiếu (NOW_SHOWING / COMING_SOON).
    Đây là phạm vi duy nhất được xét để gợi ý - không gợi ý phim đã STOPPED.
    """
    query = text("""
        SELECT movie_id, title, movie_status, release_date
        FROM movie
        WHERE entity_status = 'ACTIVE'
          AND movie_status IN ('NOW_SHOWING', 'COMING_SOON')
    """)
    rows = db.execute(query).fetchall()
    df = pd.DataFrame(rows, columns=["movie_id", "title", "movie_status", "release_date"])
    df["movie_id"] = df["movie_id"].astype(int)
    return df


def load_scoring_params(db: Session) -> dict[str, float]:
    """
    Đọc ALPHA và S0 từ bảng scoring_params (do Spring Boot tính mỗi thứ 2 3AM).
    Trả về dict rỗng nếu chưa có (lần đầu chạy hệ thống).
    """
    query = text("SELECT param_name, param_value FROM scoring_params")
    rows = db.execute(query).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def load_all_excluded_movie_ids_bulk(db: Session) -> dict[str, set[int]]:
    """
    Load toàn bộ excluded movies cho TẤT CẢ user trong 1 query duy nhất.
    Dùng cho batch predict_all_users() - tránh N query riêng lẻ cho từng user.
    excluded = phim có explicit review HOẶC đã BOOK_TICKET thành công.
    """
    query = text("""
        SELECT user_id, movie_id FROM review
        WHERE entity_status = 'ACTIVE' AND review_status = 'APPROVED'
        UNION
        SELECT user_id, movie_id FROM user_activity_logs
        WHERE action_type = 'BOOK_TICKET' AND entity_status = 'ACTIVE'
    """)
    rows = db.execute(query).fetchall()
    result: dict[str, set[int]] = {}
    for user_id, movie_id in rows:
        result.setdefault(str(user_id), set()).add(int(movie_id))
    return result


def upsert_user_preferences(
    db: Session,
    predictions: list[dict],
    batch_size: int = 500,
) -> int:
    """
    Batch UPSERT list predictions vào bảng user_preference.
    predictions: list of {user_id, movie_id, predicted_score, neighbor_count}
    Trả về số dòng đã upsert.
    """
    if not predictions:
        return 0

    upsert_sql = text("""
        INSERT INTO user_preference (user_id, movie_id, predicted_score, neighbor_count, source)
        VALUES (:user_id, :movie_id, :predicted_score, :neighbor_count, :source)
        ON DUPLICATE KEY UPDATE
            predicted_score = VALUES(predicted_score),
            neighbor_count = VALUES(neighbor_count),
            source = VALUES(source)
    """)

    total = 0
    for i in range(0, len(predictions), batch_size):
        batch = predictions[i : i + batch_size]
        params = [
            {
                "user_id": p["user_id"],
                "movie_id": p["movie_id"],
                "predicted_score": p["predicted_score"],
                "neighbor_count": p["neighbor_count"],
                "source": p["source"],
            }
            for p in batch
        ]
        db.execute(upsert_sql, params)
        total += len(batch)

    db.commit()
    return total


def load_excluded_movie_ids(db: Session, user_id: str) -> set[int]:
    """
    excluded_movies(u) theo đúng định nghĩa đã chốt:
    chỉ loại những phim có rating THẬT (explicit) HOẶC có log BOOK_TICKET
    (đã trả tiền thành công) - KHÔNG loại theo các implicit signal nhẹ
    như WATCH_TRAILER, VIEW_DETAILS...
    """
    query = text("""
        SELECT movie_id FROM review
        WHERE user_id = :uid AND entity_status = 'ACTIVE' AND review_status = 'APPROVED'
        UNION
        SELECT movie_id FROM user_activity_logs
        WHERE user_id = :uid AND action_type = 'BOOK_TICKET' AND entity_status = 'ACTIVE'
    """)
    rows = db.execute(query, {"uid": user_id}).fetchall()
    return {int(r[0]) for r in rows}


def save_utility_matrix(
    db: Session,
    utility_df: pd.DataFrame,
    batch_size: int = 1000,
) -> int:
    """
    TRUNCATE bảng utility_matrix và INSERT toàn bộ dữ liệu mới từ ma trận đã gộp.
    Bảng có các cột: created_at, entity_status, updated_at, has_implicit, y_score, movie_id, user_id, has_explicit.
    """
    if utility_df.empty:
        # Nếu empty, vẫn TRUNCATE bảng
        db.execute(text("TRUNCATE TABLE utility_matrix"))
        db.commit()
        return 0

    # 1. Truncate bảng cũ
    db.execute(text("TRUNCATE TABLE utility_matrix"))
    
    # 2. Chuẩn bị dữ liệu insert
    insert_sql = text("""
        INSERT INTO utility_matrix (
            user_id, movie_id, y_score, has_explicit, has_implicit,
            created_at, updated_at, entity_status
        ) VALUES (
            :user_id, :movie_id, :y_score, :has_explicit, :has_implicit,
            :created_at, :updated_at, :entity_status
        )
    """)
    
    now = datetime.utcnow()
    records = []
    
    for _, row in utility_df.iterrows():
        has_exp = bool(row["has_explicit"])
        has_imp = not has_exp # Vì logic gộp là: nếu có explicit thì lấy explicit, ko thì lấy implicit
        
        records.append({
            "user_id": str(row["user_id"]),
            "movie_id": int(row["movie_id"]),
            "y_score": float(row["rating"]),
            "has_explicit": 1 if has_exp else 0,
            "has_implicit": 1 if has_imp else 0,
            "created_at": now,
            "updated_at": now,
            "entity_status": "ACTIVE"
        })

    # 3. Batch insert
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        db.execute(insert_sql, batch)
        total += len(batch)

    db.commit()
    return total
