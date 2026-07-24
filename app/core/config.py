from pydantic_settings import BaseSettings
from pydantic import BaseModel


class WatchTrailerThresholds(BaseModel):
    high: float = 4
    medium: float = 2
    low: float = 1
    bad: float = 0
    high_threshold: float = 80
    medium_threshold: float = 30
    low_threshold: float = 5


class ViewDetailThresholds(BaseModel):
    high: float = 4
    mid: float = 2
    low: float = 0
    high_threshold: float = 30
    low_threshold: float = 5


class Settings(BaseSettings):
    # ===== Database =====
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = ""
    db_name: str = "movie_ticket"

    # ===== Weights nhóm 1 - cố định =====
    w_view_showtime: float = 3
    w_search: float = 2
    w_skip_recommendation: float = -1
    w_cancel_payment: float = -3
    w_abandon_seat_selection: float = -1
    w_timeout_hold_seat: float = -2
    w_chain_view_then_book: float = 8
    chain_window_minutes: int = 30

    # ===== Weights nhóm 2 - trục độ sâu (MAX value qua best_value_at) =====
    watch_trailer: WatchTrailerThresholds = WatchTrailerThresholds()
    view_detail: ViewDetailThresholds = ViewDetailThresholds()

    # ===== Weights nhóm 3 - trục tần suất (occurrence_count) =====
    w_book_ticket_base: float = 4
    w_share_movie_base: float = 2

    # alpha cho c(a) = 1 + alpha * occurrence_count (Hu et al. 2008, hiệu chỉnh)
    # KHÔNG hard-code - tính động từ median(occurrence_count thực tế, count>=2)
    # Nếu cần override thủ công, set giá trị này; None = tự tính.
    frequency_alpha_override: float | None = None

    # ===== Time decay =====
    decay_lambda: float = 0.01

    # ===== Tanh conversion =====
    tanh_amplitude: float = 1.5
    tanh_neutral_point: float = 2.5
    # S0 KHÔNG hard-code - tính động từ median(|S_u_i|) thực tế mỗi lần build matrix

    # ===== CF params =====
    cf_top_k: int = 20
    cf_min_co_rated_items: int = 2
    cf_min_similarity: float = 0.0

    # ===== Cold start =====
    cold_start_min_interactions: int = 5
    cold_start_popularity_alpha: float = 0.5

    # ===== Prediction =====
    prediction_top_n: int = 5

    # ===== CF mode - công tác chạy thuần trước, cải thiện dần sau =====
    # False -> CF Pure (chỉ explicit rating, dùng để có baseline RMSE/MAE)
    # True  -> CF + Implicit (gộp cả implicit signal, dùng để so sánh
    #          cải thiện so với baseline)
    # TAM NGUNG: implicit feedback bi comment (xem implicit_scoring.py va
    # model_state.py) - doi mac dinh ve False. model_state.train() cung da
    # ep cung use_implicit=False bat ke gia tri truyen vao, gia tri o day
    # chi con y nghia tham chieu/hien thi.
    cf_use_implicit: bool = False

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )

    class Config:
        env_prefix = "REC_"
        env_file = ".env"


settings = Settings()
