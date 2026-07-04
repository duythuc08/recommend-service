"""
Cold-start fallback - Popularity Score (Mục 5.2, v11).

Kích hoạt khi K_u < min-interactions-threshold (user quá mới hoặc quá ít
dữ liệu để CF cho kết quả tin cậy).

Score_Popularity(i) = alpha * Norm_Rating(i) + (1-alpha) * Norm_Tickets(i)
"""
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings


def compute_popularity_scores(db: Session, candidate_movie_ids: list[int]) -> dict[int, float]:
    """
    Norm_Rating(i): rating trung bình của phim i, normalize về [0,1] qua min-max.
    Norm_Tickets(i): số lượt BOOK_TICKET của phim i, normalize về [0,1] qua min-max.
    """
    if not candidate_movie_ids:
        return {}

    ids_str = ",".join(str(i) for i in candidate_movie_ids)

    avg_rating_query = text(f"""
        SELECT movie_id, AVG(rating) as avg_rating
        FROM review
        WHERE movie_id IN ({ids_str}) AND entity_status='ACTIVE' AND review_status='APPROVED'
        GROUP BY movie_id
    """)
    rating_rows = db.execute(avg_rating_query).fetchall()
    rating_map = {int(r[0]): float(r[1]) for r in rating_rows}

    ticket_count_query = text(f"""
        SELECT movie_id, COUNT(*) as ticket_count
        FROM user_activity_logs
        WHERE movie_id IN ({ids_str}) AND action_type = 'BOOK_TICKET' AND entity_status='ACTIVE'
        GROUP BY movie_id
    """)
    ticket_rows = db.execute(ticket_count_query).fetchall()
    ticket_map = {int(r[0]): int(r[1]) for r in ticket_rows}

    all_ratings = list(rating_map.values())
    all_tickets = list(ticket_map.values())

    def normalize(value, values_list):
        if not values_list:
            return 0.0
        lo, hi = min(values_list), max(values_list)
        if hi == lo:
            return 0.5  # tất cả bằng nhau -> trung lập
        return (value - lo) / (hi - lo)

    alpha = settings.cold_start_popularity_alpha
    scores = {}
    for movie_id in candidate_movie_ids:
        norm_rating = normalize(rating_map.get(movie_id, 0.0), all_ratings) if all_ratings else 0.0
        norm_tickets = normalize(ticket_map.get(movie_id, 0), all_tickets) if all_tickets else 0.0
        scores[movie_id] = alpha * norm_rating + (1 - alpha) * norm_tickets

    return scores


def count_user_interactions(utility_long: pd.DataFrame, user_id: str) -> int:
    """K_u = số lượng item user đã tương tác (explicit + implicit, theo utility_long dạng long-format)."""
    if utility_long is None or utility_long.empty:
        return 0
    return int((utility_long["user_id"] == user_id).sum())
