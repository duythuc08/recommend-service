# ===== TAM NGUNG: Implicit Feedback (comment theo yeu cau, khong xoa) =====
# Toan bo logic tinh implicit score (decay theo thoi gian, tanh conversion,
# frequency alpha, chain bonus) trong file nay bi COMMENT tam thoi de he thong
# chi train tren explicit rating (bang review) qua surprise.KNNWithMeans thuan.
# Xem model_state.py (ham train()) - nhanh goi build_implicit_scores/
# convert_to_rating_scale cung da bi comment tuong ung.
# Neu can bat lai: bo comment toan bo file nay + khoi phuc nhanh use_implicit
# trong model_state.py + doi cf_use_implicit=True trong config.py.
#
# """
# Tính implicit score S_{u,i} từ activity log, theo đúng công thức trong
# spec (Mục 2.4 - 2.5, v11):
#
#   S_{u,i} = sum_a [ w_hiệu_dụng(a) * decay(now - best_value_at_hoặc_updated_at) ]
#
#   - Nhóm 1 (w(a) cố định): VIEW_SHOWTIMES, SEARCH, SKIP_RECOMMENDATION,
#     CANCEL_PAYMENT, ABANDON_SEAT_SELECTION, TIMEOUT_HOLD_SEAT
#   - Nhóm 2 (trục độ sâu, dùng best_value_at để decay): WATCH_TRAILER
#     (theo watch_pct), VIEW_DETAILS (theo duration_sec)
#   - Nhóm 3 (trục tần suất, dùng updated_at để decay):
#     w_hiệu_dụng(a) = w_base(a) * c(a),  c(a) = 1 + alpha * occurrence_count
#     (BOOK_TICKET, SHARE_MOVIE - Hu, Koren, Volinsky 2008)
#
#   decay(delta_t) = e^(-lambda * delta_t), delta_t tính theo ngày
#
#   y_{u,i} = neutral_point + amplitude * tanh(S_{u,i} / S0)
#   S0 = median(|S_{u,i}| thực tế, chỉ tính trên cặp có implicit signal)
# """
# from datetime import datetime
# import math
#
# import numpy as np
# import pandas as pd
#
# from app.core.config import settings
#
#
# def _decay(delta_days: float, lam: float) -> float:
#     return math.exp(-lam * max(delta_days, 0))
#
#
# def _watch_trailer_weight(watch_pct: float) -> float:
#     cfg = settings.watch_trailer
#     if watch_pct > cfg.high_threshold:
#         return cfg.high
#     if watch_pct >= cfg.medium_threshold:
#         return cfg.medium
#     if watch_pct > cfg.low_threshold:
#         return cfg.low
#     return cfg.bad
#
#
# def _view_detail_weight(duration_sec: float) -> float:
#     cfg = settings.view_detail
#     if duration_sec > cfg.high_threshold:
#         return cfg.high
#     if duration_sec >= cfg.low_threshold:
#         return cfg.mid
#     return cfg.low
#
#
# # Trọng số cố định cho nhóm 1 - tra theo action_type
# _FIXED_WEIGHTS = {
#     "VIEW_SHOWTIMES": settings.w_view_showtime,
#     "SEARCH": settings.w_search,
#     "SKIP_RECOMMENDATION": settings.w_skip_recommendation,
#     "CANCEL_PAYMENT": settings.w_cancel_payment,
#     "ABANDON_SEAT_SELECTION": settings.w_abandon_seat_selection,
#     "TIMEOUT_HOLD_SEATS": settings.w_timeout_hold_seat,
# }
#
#
# def compute_frequency_alpha(activity_df: pd.DataFrame) -> float:
#     """
#     alpha = 1 / median(occurrence_count thực tế, count >= 2)
#     """
#     freq_actions = activity_df[
#         activity_df["action_type"].isin(["BOOK_TICKET", "SHARE_MOVIE"])
#     ]
#     counts = freq_actions.loc[freq_actions["occurrence_count"] >= 2, "occurrence_count"]
#     if len(counts) == 0:
#         # không có dữ liệu lặp lại -> alpha mặc định nhỏ, ảnh hưởng tối thiểu
#         return 0.1
#     return 1.0 / counts.median()
#
#
# def compute_s0(s_values: pd.Series) -> float:
#     """
#     S0 = median(|S_u_i|), CHỈ trên cặp (user,movie) implicit-only (không
#     có explicit) - dùng đúng logic Java gốc:
#     ParameterEstimationService.estimateS0() gọi
#     userActivityLogRepository.findUserMoviePairsWithoutExplicitRating()
#     trước khi tính median. Hàm này nhận s_values đã được lọc đúng phạm vi
#     đó từ build_implicit_scores() (qua tham số explicit_pairs), nên chỉ
#     cần tính median |S| trên chính series được truyền vào, không lọc thêm
#     ở đây.
#
#     Nếu không có cặp implicit-only nào (list rỗng) - giống đúng Java
#     (pairs.isEmpty()) - hàm gọi trả về None thay vì fallback 1.0, để nơi
#     gọi (convert_to_rating_scale) tự quyết định giữ S0 cũ hay dùng giá
#     trị tối thiểu an toàn, khớp đúng hành vi "không ghi đè S0 cũ" của
#     Java.
#     """
#     abs_vals = s_values.abs()
#     abs_vals = abs_vals[abs_vals > 0]
#     if len(abs_vals) == 0:
#         return None  # khớp Java: trả về None ("giữ S0 cũ"), không tự fallback ở đây
#     return abs_vals.median()
#
#
# def build_implicit_scores(
#     activity_df: pd.DataFrame,
#     explicit_pairs: set[tuple] | None = None,
#     now: datetime | None = None,
#     alpha: float | None = None,
# ) -> pd.DataFrame:
#     """
#     Input: activity_df (output của load_all_activity_logs)
#            explicit_pairs: set các (user_id, movie_id) ĐÃ CÓ rating thực.
#                Theo đúng Java gốc (ParameterEstimationService.estimateS0()
#                gọi findUserMoviePairsWithoutExplicitRating()) - CHỈ tính S
#                cho cặp CHƯA có explicit. Truyền None = không lọc gì (chỉ
#                dùng khi gọi riêng lẻ ngoài pipeline chính, vd debug).
#     Output: DataFrame với cột user_id, movie_id, S (implicit score thô,
#             CHƯA qua tanh conversion) - CHỈ gồm cặp không có explicit.
#     """
#     if now is None:
#         now = datetime.utcnow()
#
#     if explicit_pairs is not None:
#         mask = activity_df.apply(
#             lambda r: (r["user_id"], r["movie_id"]) not in explicit_pairs, axis=1
#         )
#         activity_df = activity_df[mask]
#
#     if alpha is None:
#         alpha = compute_frequency_alpha(activity_df)
#     lam = settings.decay_lambda
#
#     rows = []
#     for _, row in activity_df.iterrows():
#         action = row["action_type"]
#         meta = row["metadata"] or {}
#
#         if action in _FIXED_WEIGHTS:
#             base_w = _FIXED_WEIGHTS[action]
#             ref_time = row["updated_at"] or row["created_at"]
#             delta_days = (now - ref_time).total_seconds() / 86400
#             score = base_w * _decay(delta_days, lam)
#
#         elif action == "WATCH_TRAILER":
#             watch_pct = meta.get("watch_pct", 0) if isinstance(meta, dict) else 0
#             base_w = _watch_trailer_weight(watch_pct)
#             ref_time = row["best_value_at"] or row["updated_at"] or row["created_at"]
#             delta_days = (now - ref_time).total_seconds() / 86400
#             score = base_w * _decay(delta_days, lam)
#
#         elif action == "VIEW_DETAILS":
#             duration_sec = meta.get("duration_sec", 0) if isinstance(meta, dict) else 0
#             base_w = _view_detail_weight(duration_sec)
#             ref_time = row["best_value_at"] or row["updated_at"] or row["created_at"]
#             delta_days = (now - ref_time).total_seconds() / 86400
#             score = base_w * _decay(delta_days, lam)
#
#         elif action in ("BOOK_TICKET", "SHARE_MOVIE"):
#             base = settings.w_book_ticket_base if action == "BOOK_TICKET" else settings.w_share_movie_base
#             c = 1 + alpha * row["occurrence_count"]
#             ref_time = row["updated_at"] or row["created_at"]
#             delta_days = (now - ref_time).total_seconds() / 86400
#             score = base * c * _decay(delta_days, lam)
#
#         else:
#             continue  # WRITE_REVIEW xử lý ở nhánh explicit, không cộng vào đây
#
#         rows.append({"user_id": row["user_id"], "movie_id": row["movie_id"], "S_component": score})
#
#     if not rows:
#         return pd.DataFrame(columns=["user_id", "movie_id", "S"])
#
#     component_df = pd.DataFrame(rows)
#     s_df = component_df.groupby(["user_id", "movie_id"], as_index=False)["S_component"].sum()
#     s_df = s_df.rename(columns={"S_component": "S"})
#
#     chain_bonus_df = _compute_chain_bonus(activity_df, now)
#     if not chain_bonus_df.empty:
#         s_df = s_df.merge(chain_bonus_df, on=["user_id", "movie_id"], how="outer")
#         s_df["S"] = s_df["S"].fillna(0) + s_df["chain_bonus"].fillna(0)
#         s_df = s_df.drop(columns=["chain_bonus"])
#
#     return s_df
#
#
# def _compute_chain_bonus(activity_df: pd.DataFrame, now: datetime) -> pd.DataFrame:
#     """
#     Phát hiện chuỗi hành vi VIEW_SHOWTIMES -> BOOK_TICKET trong vòng
#     chain_window_minutes, cộng thêm w_chain_view_then_book.
#     """
#     showtime_df = activity_df[activity_df["action_type"] == "VIEW_SHOWTIMES"][
#         ["user_id", "movie_id", "updated_at"]
#     ].rename(columns={"updated_at": "showtime_at"})
#
#     book_df = activity_df[activity_df["action_type"] == "BOOK_TICKET"][
#         ["user_id", "movie_id", "updated_at"]
#     ].rename(columns={"updated_at": "book_at"})
#
#     if showtime_df.empty or book_df.empty:
#         return pd.DataFrame(columns=["user_id", "movie_id", "chain_bonus"])
#
#     merged = showtime_df.merge(book_df, on=["user_id", "movie_id"], how="inner")
#     if merged.empty:
#         return pd.DataFrame(columns=["user_id", "movie_id", "chain_bonus"])
#
#     delta_minutes = (merged["book_at"] - merged["showtime_at"]).dt.total_seconds() / 60
#     window = settings.chain_window_minutes
#     is_chain = (delta_minutes >= 0) & (delta_minutes <= window)
#
#     chain_rows = merged[is_chain].copy()
#     if chain_rows.empty:
#         return pd.DataFrame(columns=["user_id", "movie_id", "chain_bonus"])
#
#     lam = settings.decay_lambda
#     chain_rows["delta_days"] = (now - chain_rows["book_at"]).dt.total_seconds() / 86400
#     chain_rows["chain_bonus"] = chain_rows["delta_days"].apply(
#         lambda d: settings.w_chain_view_then_book * _decay(d, lam)
#     )
#     return chain_rows[["user_id", "movie_id", "chain_bonus"]]
#
#
# def convert_to_rating_scale(s_df: pd.DataFrame, s0: float | None = None, previous_s0: float | None = None) -> pd.DataFrame:
#     """
#     y_{u,i} = neutral_point + amplitude * tanh(S / S0)
#
#     s_df PHẢI là output của build_implicit_scores() với explicit_pairs
#     đã được truyền đúng (chỉ gồm cặp implicit-only) - S0 tính từ chính
#     s_df này (median |S|), khớp đúng phạm vi Java gốc.
#
#     previous_s0: S0 của lần train trước (nếu có) - dùng làm fallback khi
#     s_df rỗng (không có cặp implicit-only nào), khớp đúng hành vi Java
#     "không ghi đè S0 cũ nếu pairs.isEmpty()". Nếu cả s_df rỗng VÀ
#     previous_s0=None, dùng 0.0001 (giống Math.max(medianAbsScore, 0.0001)
#     trong Java, áp dụng khi hệ thống hoàn toàn chưa có dữ liệu).
#     """
#     amplitude = settings.tanh_amplitude
#     neutral = settings.tanh_neutral_point
#
#     if s_df.empty:
#         return s_df.assign(y=[])
#
#     if s0 is None:
#         s0 = compute_s0(s_df["S"])
#     if s0 is None:
#         s0 = previous_s0 if previous_s0 is not None else 0.0001
#
#     s_df = s_df.copy()
#     s_df["y"] = neutral + amplitude * np.tanh(s_df["S"] / s0)
#     return s_df
#