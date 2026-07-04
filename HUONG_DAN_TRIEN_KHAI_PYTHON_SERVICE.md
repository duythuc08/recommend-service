# Huong dan trien khai Recommendation Service (Python FastAPI)

## Do an: Infinity Cinema — Ung dung Collaborative Filtering goi y phim

---

## 1. Tong quan kien truc

Service nay la **Python FastAPI** dung doc lap, chiu trach nhiem toan bo phan tinh toan goi y phim. No **khong phu thuoc Spring Boot** de lay data — tu ket noi thang vao MySQL.

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│   Next.js   │─────▶│   Spring Boot     │─────▶│  Python FastAPI      │
│   (FE)      │      │   (Gateway/BE)    │      │  (Recommend Service) │
└─────────────┘      └──────────────────┘      └──────────┬──────────┘
                            ▲                              │
                            │ enrich poster, gio chieu...  │ query truc tiep
                            │                               ▼
                            │                        ┌─────────────┐
                            └────────────────────────│   MySQL DB   │
                                                       └─────────────┘
```

**Luong chay 1 request:**
1. User dang nhap → FE goi `GET /api/recommendations` tren Spring Boot
2. Spring Boot goi `POST http://python-service/api/recommend` kem `{userId, B}`
3. Python tu query MySQL lay: review (explicit rating), activity log (implicit), candidate movies (`NOW_SHOWING`/`COMING_SOON`)
4. Python chay CF (Surprise `KNNWithMeans`) + cold-start fallback (popularity) → tra `{userId, recommendations: [{movieId, score}]}`
5. Spring Boot nhan, enrich them poster/ten phim/gio chieu tu DB cua minh, tra ve FE

**Lich train mo hinh:**
- Train 1 lan luc service khoi dong (de co model san sang ngay)
- Train lai tu dong moi ngay luc **3:00 AM** (qua APScheduler)
- Co endpoint `POST /api/train` de admin trigger train thu cong bat cu luc nao (vi du sau khi seed data moi)

**Thu vien CF dung:** [`scikit-surprise`](https://surpriselib.com/) (Hug, N. 2020, *Surprise: A Python library for recommender systems*, JOSS) — thuat toan `KNNWithMeans`, dung theo dinh huong de tai ung dung (dung thu vien da duoc cong nhan, khong tu viet cong thuc toan tay).

---

## 2. Cai dat moi truong

### 2.1. Yeu cau
- Python 3.10+ (khuyen nghi 3.12)
- MySQL dang chay, da co data theo schema hien tai (`movie`, `review`, `user_activity_logs`, `users`)

### 2.2. Tao virtual environment

```bash
mkdir recommend-service && cd recommend-service
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows
```

### 2.3. Cai dependencies

```bash
pip install fastapi uvicorn sqlalchemy pymysql numpy pandas pydantic-settings apscheduler scikit-surprise
```

**Giai thich tung package:**

| Package | Vai tro |
|---|---|
| `fastapi` | Framework viet REST API |
| `uvicorn` | ASGI server de chay FastAPI |
| `sqlalchemy` | ORM/query builder ket noi MySQL |
| `pymysql` | Driver MySQL cho Python (SQLAlchemy dung driver nay) |
| `numpy`, `pandas` | Xu ly ma tran, DataFrame |
| `pydantic-settings` | Doc config tu bien moi truong/.env |
| `apscheduler` | Lap lich chay job 3AM hang ngay |
| `scikit-surprise` | Thu vien CF chinh — `KNNWithMeans` |

> **Luu y cai `scikit-surprise`:** thu vien nay can bien dich C extension (Cython). Neu loi khi cai tren Windows, cai them `Microsoft C++ Build Tools` truoc, hoac dung WSL/Docker.

### 2.4. Cau truc thu muc can tao

```
recommend-service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── implicit_scoring.py
│   │   ├── cf_engine.py
│   │   ├── cold_start.py
│   │   └── model_state.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── session.py
│   │   └── queries.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── routers/
│       ├── __init__.py
│       └── recommend.py
├── .env
└── requirements.txt
```

Tao nhanh bang lenh:

```bash
mkdir -p app/core app/db app/models app/routers
touch app/__init__.py app/core/__init__.py app/db/__init__.py app/models/__init__.py app/routers/__init__.py
```

### 2.5. File `.env` (thong tin ket noi DB)

Tao file `.env` o thu muc goc:

```env
REC_DB_HOST=127.0.0.1
REC_DB_PORT=3306
REC_DB_USER=root
REC_DB_PASSWORD=your_password_here
REC_DB_NAME=movie_ticket
```

> Tat ca bien moi truong deu co prefix `REC_` de tranh trung voi bien moi truong khac tren server. Xem chi tiet toan bo bien co the override o Muc 3 (`config.py`).

### 2.6. File `requirements.txt` (de deploy/CI dung lai)

```text
fastapi
uvicorn[standard]
sqlalchemy
pymysql
numpy
pandas
pydantic-settings
apscheduler
scikit-surprise
```

---
## 3. Code chi tiet — `app/core/config.py`

File nay tap trung **toan bo tham so** tu file YAML config da thiet ke (weights, decay, tanh-conversion, cf, cold-start, prediction) vao 1 cho duy nhat. Dung `pydantic-settings` de moi tham so co the override qua bien moi truong (vi du muon doi `cf_top_k` tu 20 sang 30 khi deploy, chi can set `REC_CF_TOP_K=30` ma khong can sua code).

**Diem quan trong can biet khi doc code nay:**
- `S0` (he so bao hoa cua ham tanh) va `frequency_alpha` (he so tan suat Hu et al. 2008) **khong hard-code** — luon tinh dong tu du lieu that moi lan training (xem Muc 4).
- `WatchTrailerThresholds` va `ViewDetailThresholds` la 2 class con (nested model), khop dung cau truc YAML goc co nhieu muc nguong (`high`, `medium`, `low`, `bad`).

```python
"""
Config tap trung cho Recommendation Service.
Cac gia tri trong day duoc lay theo dung file YAML config da thiet ke
(recommendation.weights, recommendation.decay, recommendation.tanh-conversion,
recommendation.cf, recommendation.cold-start, recommendation.prediction).

Trong moi truong thuc te, cac gia tri nay nen doc tu .env hoac tu chinh
file application.yml cua Spring Boot (qua Spring Cloud Config / shared config
service) de tranh hard-code lap lai 2 noi. O day dung pydantic-settings de
co the override bang env var khi deploy.
"""
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

    # ===== Weights nhom 1 - co dinh =====
    w_view_showtime: float = 3
    w_search: float = 2
    w_skip_recommendation: float = -1
    w_cancel_payment: float = -3
    w_abandon_seat_selection: float = -1
    w_timeout_hold_seat: float = -2
    w_chain_view_then_book: float = 8
    chain_window_minutes: int = 30

    # ===== Weights nhom 2 - truc do sau (MAX value qua best_value_at) =====
    watch_trailer: WatchTrailerThresholds = WatchTrailerThresholds()
    view_detail: ViewDetailThresholds = ViewDetailThresholds()

    # ===== Weights nhom 3 - truc tan suat (occurrence_count) =====
    w_book_ticket_base: float = 4
    w_share_movie_base: float = 2

    # alpha cho c(a) = 1 + alpha * occurrence_count (Hu et al. 2008, hieu chinh)
    # KHONG hard-code - tinh dong tu median(occurrence_count thuc te, count>=2)
    # Neu can override thu cong, set gia tri nay; None = tu tinh.
    frequency_alpha_override: float | None = None

    # ===== Time decay =====
    decay_lambda: float = 0.01

    # ===== Tanh conversion =====
    tanh_amplitude: float = 1.5
    tanh_neutral_point: float = 2.5
    # S0 KHONG hard-code - tinh dong tu median(|S_u_i|) thuc te moi lan build matrix

    # ===== CF params =====
    cf_top_k: int = 20
    cf_min_co_rated_items: int = 2
    cf_min_similarity: float = 0.0

    # ===== Cold start =====
    cold_start_min_interactions: int = 5
    cold_start_popularity_alpha: float = 0.5

    # ===== Prediction =====
    prediction_top_n: int = 3

    # ===== CF mode - cong tac chay thuan truoc, cai thien dan sau =====
    # True  -> CF Pure (chi explicit rating, dung de co baseline RMSE/MAE)
    # False -> CF + Implicit (gop ca implicit signal, dung de so sanh
    #          cai thien so voi baseline)
    cf_use_implicit: bool = True

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

```

---

## 4. Code chi tiet — `app/db/session.py` (ket noi MySQL)

Module nay tao connection pool toi MySQL, dung lam FastAPI dependency (`get_db`) — moi request mo 1 session rieng, tu dong sau khi xong.

```python
"""
Ket noi MySQL truc tiep tu Python (FastAPI tu connect DB, dung theo
luong da chot - khong qua Spring Boot trung gian de lay data).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.sqlalchemy_url,
    pool_pre_ping=True,   # tranh loi "MySQL server has gone away" khi connection idle lau
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency cho FastAPI - moi request mo 1 session, dong sau khi xong."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Giai thich:**
- `pool_pre_ping=True`: truoc moi lan lay connection tu pool, SQLAlchemy tu ping thu — tranh loi `MySQL server has gone away` khi connection bi idle qua lau (MySQL tu dong connection sau `wait_timeout`, mac dinh 8 gio).
- `pool_recycle=3600`: tu dong tao connection moi sau moi 1 gio, phong truong hop MySQL hoac firewall/proxy giua duong dong connection am tham.
- `get_db()` la generator — FastAPI goi `Depends(get_db)` o moi endpoint, tu dong quan ly vong doi session (mo → dung → dong), khong can tu viet `try/finally` o moi route.

---

## 5. Code chi tiet — `app/db/queries.py` (truy van du lieu)

Day la tang data access — chua 4 ham query chinh. **Nguyen tac quan trong nhat o day: tranh N+1 query** — bai hoc tu loi da gap o `UtilityMatrixBuilderTasklet` cu (moi user/movie query rieng gay ra hang nghin round-trip toi DB). O day, `load_all_reviews` va `load_all_activity_logs` chi chay **dung 1 query duy nhat**, lay het toan bo data ve roi xu ly bang pandas o tang ung dung.

```python
"""
Data access layer - query truc tiep tu DB MySQL.

Nguyen tac tranh N+1 (bai hoc tu UtilityMatrixBuilderTasklet cu):
- Load TOAN BO review va activity_logs trong 2 query duy nhat (khong loop
  tung user/movie de query rieng).
- Pandas se xu ly join/group o tang application sau khi load xong.
"""
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session


def load_all_reviews(db: Session) -> pd.DataFrame:
    """
    Load toan bo explicit rating (bang review).
    Tra ve DataFrame: user_id, movie_id, rating (1-5, float)
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
    Load toan bo activity log (bang user_activity_logs).
    Tra ve DataFrame: user_id, movie_id, action_type, created_at,
    updated_at, best_value_at, occurrence_count, metadata (dict hoac None)
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
    Candidate set = phim dang chieu hoac sap chieu (NOW_SHOWING / COMING_SOON).
    Day la pham vi duy nhat duoc xet de goi y - khong goi y phim da STOPPED.
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


def load_excluded_movie_ids(db: Session, user_id: str) -> set[int]:
    """
    excluded_movies(u) theo dung dinh nghia da chot:
    chi loai nhung phim co rating THAT (explicit) HOAC co log BOOK_TICKET
    (da tra tien thanh cong) - KHONG loai theo cac implicit signal nhe
    nhu WATCH_TRAILER, VIEW_DETAILS...
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
```

**Giai thich tung ham:**

| Ham | Muc dich |
|---|---|
| `load_all_reviews` | Lay toan bo rating that (explicit) tu bang `review`, chi lay review da `APPROVED` va con `ACTIVE` |
| `load_all_activity_logs` | Lay toan bo activity log (implicit), parse `metadata` (JSON column) thanh dict Python |
| `load_candidate_movies` | Candidate set = phim `NOW_SHOWING` hoac `COMING_SOON` — day la **pham vi duy nhat** duoc xet de goi y, khong goi y phim da `STOPPED` |
| `load_excluded_movie_ids` | Tra ve set `movie_id` can loai khoi danh sach goi y cho 1 user cu the — dung dinh nghia `excluded_movies(u)` da chot: **chi** loai phim da co rating that HOAC da `BOOK_TICKET` thanh cong, **khong** loai theo cac tin hieu implicit nhe nhu `WATCH_TRAILER`/`VIEW_DETAILS` |

---

## 6. Code chi tiet — `app/core/implicit_scoring.py` (tinh S_{u,i})

Day la module **quan trong nhat ve mat hoc thuat** — implement chinh xac cong thuc tinh implicit score da thiet ke trong spec (Muc 2.4–2.5). File nay **khong dung thu vien CF** (vi khong co thu vien nao tinh san implicit score theo cong thuc tuy bien rieng cua de tai) — day la phan logic nghiep vu dac thu, hop ly de tu code.

**Cong thuc tong quat:**

```
S_{u,i} = Σ_a [ w_hieu_dung(a) × decay(now − thoi_diem_tham_chieu) ]

decay(Δt) = e^(−λ × Δt)     (Δt tinh theo ngay)

y_{u,i} = neutral_point + amplitude × tanh(S_{u,i} / S0)
```

**3 nhom action_type xu ly khac nhau:**

| Nhom | Action type | Cach tinh `w(a)` | Moc thoi gian dung de decay |
|---|---|---|---|
| 1 — Co dinh | `VIEW_SHOWTIMES`, `SEARCH`, `SKIP_RECOMMENDATION`, `CANCEL_PAYMENT`, `ABANDON_SEAT_SELECTION`, `TIMEOUT_HOLD_SEATS` | Tra thang trong config | `updated_at` |
| 2 — Do sau (MAX value) | `WATCH_TRAILER` (theo `watch_pct`), `VIEW_DETAILS` (theo `duration_sec`) | Tra theo nguong (`high`/`medium`/`low`/`bad`) | `best_value_at` (thoi diem dat gia tri cao nhat qua moi lan xem) |
| 3 — Tan suat | `BOOK_TICKET`, `SHARE_MOVIE` | `w_base × c(a)`, voi `c(a) = 1 + alpha × occurrence_count` (Hu, Koren, Volinsky 2008) | `updated_at` |

**Chain-bonus** (`VIEW_SHOWTIMES → BOOK_TICKET` trong 30 phut, +8 diem): duoc tinh rieng o ham `_compute_chain_bonus`.

> ⚠ **Gioi han ky thuat quan trong can biet khi bao ve do an:** Bang `user_activity_logs` co composite PK `(action_type, movie_id, user_id)` — chi luu **1 dong duy nhat** cho moi action, khong luu lich su tung lan xay ra rieng biet. Vi vay chain-detection o day dung `updated_at` (thoi diem cap nhat gan nhat) lam xap xi cho "lan xay ra gan nhat" — **khong phai** timestamp chinh xac cua tung lan rieng le. Day la mot trade-off da duoc can nhac ky (xem them Muc 9 — Han che da biet).

```python
"""
Tinh implicit score S_{u,i} tu activity log, theo dung cong thuc trong
spec (Muc 2.4 - 2.5, v11):

  S_{u,i} = sum_a [ w_hieu_dung(a) * decay(now - best_value_at_hoac_updated_at) ]

  - Nhom 1 (w(a) co dinh): VIEW_SHOWTIMES, SEARCH, SKIP_RECOMMENDATION,
    CANCEL_PAYMENT, ABANDON_SEAT_SELECTION, TIMEOUT_HOLD_SEAT
  - Nhom 2 (truc do sau, dung best_value_at de decay): WATCH_TRAILER
    (theo watch_pct), VIEW_DETAILS (theo duration_sec)
  - Nhom 3 (truc tan suat, dung updated_at de decay):
    w_hieu_dung(a) = w_base(a) * c(a),  c(a) = 1 + alpha * occurrence_count
    (BOOK_TICKET, SHARE_MOVIE - Hu, Koren, Volinsky 2008)

  decay(delta_t) = e^(-lambda * delta_t), delta_t tinh theo ngay

  y_{u,i} = neutral_point + amplitude * tanh(S_{u,i} / S0)
  S0 = median(|S_{u,i}| thuc te, chi tinh tren cap co implicit signal)
"""
from datetime import datetime
import math

import numpy as np
import pandas as pd

from app.core.config import settings


def _decay(delta_days: float, lam: float) -> float:
    return math.exp(-lam * max(delta_days, 0))


def _watch_trailer_weight(watch_pct: float) -> float:
    cfg = settings.watch_trailer
    if watch_pct > cfg.high_threshold:
        return cfg.high
    if watch_pct >= cfg.medium_threshold:
        return cfg.medium
    if watch_pct > cfg.low_threshold:
        return cfg.low
    return cfg.bad


def _view_detail_weight(duration_sec: float) -> float:
    cfg = settings.view_detail
    if duration_sec > cfg.high_threshold:
        return cfg.high
    if duration_sec >= cfg.low_threshold:
        return cfg.mid
    return cfg.low


# Trong so co dinh cho nhom 1 - tra theo action_type
_FIXED_WEIGHTS = {
    "VIEW_SHOWTIMES": settings.w_view_showtime,
    "SEARCH": settings.w_search,
    "SKIP_RECOMMENDATION": settings.w_skip_recommendation,
    "CANCEL_PAYMENT": settings.w_cancel_payment,
    "ABANDON_SEAT_SELECTION": settings.w_abandon_seat_selection,
    "TIMEOUT_HOLD_SEATS": settings.w_timeout_hold_seat,
}


def compute_frequency_alpha(activity_df: pd.DataFrame) -> float:
    """
    alpha = 1 / median(occurrence_count thuc te, count >= 2)
    Theo dung note trong YAML: KHONG dung gia tri goc paper (alpha=40),
    phai tinh lai tu data thuc te he thong nay.
    """
    freq_actions = activity_df[
        activity_df["action_type"].isin(["BOOK_TICKET", "SHARE_MOVIE"])
    ]
    counts = freq_actions.loc[freq_actions["occurrence_count"] >= 2, "occurrence_count"]
    if len(counts) == 0:
        # khong co du lieu lap lai -> alpha mac dinh nho, anh huong toi thieu
        return 0.1
    return 1.0 / counts.median()


def compute_s0(s_values: pd.Series) -> float:
    """
    S0 = median(|S_u_i|), CHI tren cap (user,movie) implicit-only (khong
    co explicit) - dung dung logic Java goc:
    ParameterEstimationService.estimateS0() goi
    userActivityLogRepository.findUserMoviePairsWithoutExplicitRating()
    truoc khi tinh median. Ham nay nhan s_values da duoc loc dung pham vi
    do tu build_implicit_scores() (qua tham so explicit_pairs), nen chi
    can tinh median |S| tren chinh series duoc truyen vao, khong loc them
    o day.

    Neu khong co cap implicit-only nao (list rong) - giong dung Java
    (pairs.isEmpty()) - ham goi tra ve None thay vi fallback 1.0, de noi
    goi (convert_to_rating_scale) tu quyet dinh giu S0 cu hay dung gia
    tri toi thieu an toan, khop dung hanh vi "khong ghi de S0 cu" cua
    Java.
    """
    abs_vals = s_values.abs()
    abs_vals = abs_vals[abs_vals > 0]
    if len(abs_vals) == 0:
        return None  # khop Java: tra ve None ("giu S0 cu"), khong tu fallback o day
    return abs_vals.median()


def build_implicit_scores(
    activity_df: pd.DataFrame,
    explicit_pairs: set[tuple] | None = None,
    now: datetime | None = None,
) -> pd.DataFrame:
    """
    Input: activity_df (output cua load_all_activity_logs)
           explicit_pairs: set cac (user_id, movie_id) DA CO rating thuc.
               Theo dung Java goc (ParameterEstimationService.estimateS0()
               goi findUserMoviePairsWithoutExplicitRating()) - CHI tinh S
               cho cap CHUA co explicit. Truyen None = khong loc gi (chi
               dung khi goi rieng le ngoai pipeline chinh, vd debug).
    Output: DataFrame voi cot user_id, movie_id, S (implicit score tho,
            CHUA qua tanh conversion) - CHI gom cap khong co explicit.
    """
    if now is None:
        now = datetime.utcnow()

    if explicit_pairs is not None:
        mask = activity_df.apply(
            lambda r: (r["user_id"], r["movie_id"]) not in explicit_pairs, axis=1
        )
        activity_df = activity_df[mask]

    alpha = compute_frequency_alpha(activity_df)
    lam = settings.decay_lambda

    rows = []
    for _, row in activity_df.iterrows():
        action = row["action_type"]
        meta = row["metadata"] or {}

        if action in _FIXED_WEIGHTS:
            base_w = _FIXED_WEIGHTS[action]
            ref_time = row["updated_at"] or row["created_at"]
            delta_days = (now - ref_time).total_seconds() / 86400
            score = base_w * _decay(delta_days, lam)

        elif action == "WATCH_TRAILER":
            watch_pct = meta.get("watch_pct", 0) if isinstance(meta, dict) else 0
            base_w = _watch_trailer_weight(watch_pct)
            ref_time = row["best_value_at"] or row["updated_at"] or row["created_at"]
            delta_days = (now - ref_time).total_seconds() / 86400
            score = base_w * _decay(delta_days, lam)

        elif action == "VIEW_DETAILS":
            duration_sec = meta.get("duration_sec", 0) if isinstance(meta, dict) else 0
            base_w = _view_detail_weight(duration_sec)
            ref_time = row["best_value_at"] or row["updated_at"] or row["created_at"]
            delta_days = (now - ref_time).total_seconds() / 86400
            score = base_w * _decay(delta_days, lam)

        elif action in ("BOOK_TICKET", "SHARE_MOVIE"):
            base = settings.w_book_ticket_base if action == "BOOK_TICKET" else settings.w_share_movie_base
            c = 1 + alpha * row["occurrence_count"]
            ref_time = row["updated_at"] or row["created_at"]
            delta_days = (now - ref_time).total_seconds() / 86400
            score = base * c * _decay(delta_days, lam)

        else:
            continue  # WRITE_REVIEW xu ly o nhanh explicit, khong cong vao day

        rows.append({"user_id": row["user_id"], "movie_id": row["movie_id"], "S_component": score})

    if not rows:
        return pd.DataFrame(columns=["user_id", "movie_id", "S"])

    component_df = pd.DataFrame(rows)
    s_df = component_df.groupby(["user_id", "movie_id"], as_index=False)["S_component"].sum()
    s_df = s_df.rename(columns={"S_component": "S"})

    chain_bonus_df = _compute_chain_bonus(activity_df, now)
    if not chain_bonus_df.empty:
        s_df = s_df.merge(chain_bonus_df, on=["user_id", "movie_id"], how="outer")
        s_df["S"] = s_df["S"].fillna(0) + s_df["chain_bonus"].fillna(0)
        s_df = s_df.drop(columns=["chain_bonus"])

    return s_df


def _compute_chain_bonus(activity_df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """
    Phat hien chuoi hanh vi VIEW_SHOWTIMES -> BOOK_TICKET trong vong
    chain_window_minutes, cong them w_chain_view_then_book.

    GHI CHU QUAN TRONG (gioi han ky thuat da xac nhan voi nguoi dung):
    PK cua user_activity_logs la (action_type, movie_id, user_id) - chi
    luu 1 dong duy nhat moi action, KHONG luu lich su tung lan xay ra.
    `updated_at` la thoi diem GAN NHAT action do duoc cap nhat (xap xi
    "lan xay ra cuoi"), con `created_at` la LAN DAU TIEN. Chain detection
    o day dung updated_at cua VIEW_SHOWTIMES so voi updated_at cua
    BOOK_TICKET theo quyet dinh cua nguoi dung - day la xap xi, do chinh
    xac phu thuoc do lech giua "lan cap nhat gan nhat" va "lan xay ra
    thuc te ngay truoc khi book".
    """
    showtime_df = activity_df[activity_df["action_type"] == "VIEW_SHOWTIMES"][
        ["user_id", "movie_id", "updated_at"]
    ].rename(columns={"updated_at": "showtime_at"})

    book_df = activity_df[activity_df["action_type"] == "BOOK_TICKET"][
        ["user_id", "movie_id", "updated_at"]
    ].rename(columns={"updated_at": "book_at"})

    if showtime_df.empty or book_df.empty:
        return pd.DataFrame(columns=["user_id", "movie_id", "chain_bonus"])

    merged = showtime_df.merge(book_df, on=["user_id", "movie_id"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["user_id", "movie_id", "chain_bonus"])

    delta_minutes = (merged["book_at"] - merged["showtime_at"]).dt.total_seconds() / 60
    window = settings.chain_window_minutes
    is_chain = (delta_minutes >= 0) & (delta_minutes <= window)

    chain_rows = merged[is_chain].copy()
    if chain_rows.empty:
        return pd.DataFrame(columns=["user_id", "movie_id", "chain_bonus"])

    lam = settings.decay_lambda
    chain_rows["delta_days"] = (now - chain_rows["book_at"]).dt.total_seconds() / 86400
    chain_rows["chain_bonus"] = chain_rows["delta_days"].apply(
        lambda d: settings.w_chain_view_then_book * _decay(d, lam)
    )
    return chain_rows[["user_id", "movie_id", "chain_bonus"]]


def convert_to_rating_scale(s_df: pd.DataFrame, previous_s0: float | None = None) -> pd.DataFrame:
    """
    y_{u,i} = neutral_point + amplitude * tanh(S / S0)

    s_df PHAI la output cua build_implicit_scores() voi explicit_pairs
    da duoc truyen dung (chi gom cap implicit-only) - S0 tinh tu chinh
    s_df nay (median |S|), khop dung pham vi Java goc.

    previous_s0: S0 cua lan train truoc (neu co) - dung lam fallback khi
    s_df rong (khong co cap implicit-only nao), khop dung hanh vi Java
    "khong ghi de S0 cu neu pairs.isEmpty()". Neu ca s_df rong VA
    previous_s0=None, dung 0.0001 (giong Math.max(medianAbsScore, 0.0001)
    trong Java, ap dung khi he thong hoan toan chua co du lieu).
    """
    amplitude = settings.tanh_amplitude
    neutral = settings.tanh_neutral_point

    if s_df.empty:
        return s_df.assign(y=[])

    s0 = compute_s0(s_df["S"])
    if s0 is None:
        s0 = previous_s0 if previous_s0 is not None else 0.0001

    s_df = s_df.copy()
    s_df["y"] = neutral + amplitude * np.tanh(s_df["S"] / s0)
    return s_df

```

---

## 7. Code chi tiet — `app/core/cf_engine.py` (CF chinh, dung Surprise)

Day la **loi cua thuat toan goi y**, dung thang thu vien [`scikit-surprise`](https://surpriselib.com/) — class `KNNWithMeans`. Khong tu viet cong thuc cosine similarity hay prediction tay, dung dinh huong de tai ung dung.

**Cong thuc KNNWithMeans (da implement san trong thu vien):**

```
pred(u, i) = mean_u + [ Σ_v sim(u,v) × (r_{v,i} − mean_v) ] / [ Σ_v |sim(u,v)| ]
```

Day chinh xac la cong thuc **User-Based Memory CF** ma de tai yeu cau — chi khac la dung ban da duoc thu vien implement, test, va co paper (Hug, N. 2020) thay vi tu viet.

**3 ham chinh trong file:**

| Ham | Viec lam |
|---|---|
| `build_utility_matrix` | Gop explicit (rating) + implicit (da quy doi qua tanh) thanh 1 bang duy nhat. Neu 1 cap (user, movie) co ca 2 nguon → **chi giu explicit** (dung rule da chot) |
| `build_surprise_trainset` | Convert DataFrame sang `Trainset` cua Surprise (yeu cau format 3 cot: user, item, rating) |
| `train_knn_model` | Khoi tao va `.fit()` model `KNNWithMeans` voi `sim_options={"cosine", "user_based": True}` |
| `predict_ratings_for_user` | Goi `.predict()` cho tung phim candidate, loc bo truong hop `was_impossible=True` (khong du neighbor hop le) |

```python
"""
User-Based Memory CF dung thu vien Surprise (scikit-surprise, Hug 2020).
Khong tu viet cong thuc cosine similarity / prediction tay - dung dung
KNNWithMeans co san, theo dinh huong de tai ung dung (khong phai nghien cuu).

Tai lieu: https://surprise.readthedocs.io/en/stable/knn_inspired.html
Paper: Hug, N. (2020). Surprise: A Python library for recommender
systems. Journal of Open Source Software, 5(52), 2174.
"""
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans
from surprise.trainset import Trainset

from app.core.config import settings


def build_utility_matrix(
    review_df: pd.DataFrame,
    implicit_df: pd.DataFrame,
    use_implicit: bool = True,
) -> pd.DataFrame:
    """
    Gop explicit rating va implicit-converted rating thanh 1 utility
    long-format DataFrame (user_id, movie_id, rating).

    use_implicit=False  -> CF PURE: chi dung explicit rating (review),
        implicit_df bi bo qua hoan toan. Dung de chay baseline truoc
        khi them implicit, dung lo trinh "chay thuan -> cai thien dan"
        da chot.
    use_implicit=True   -> CF + IMPLICIT: gop ca 2 nguon. Quy tac: cap
        (user, movie) co o ca 2 nguon -> CHI dung explicit (implicit
        chi bu sparsity cho cap CHUA co rating thuc).
    """
    review_df = review_df.copy()
    review_df["has_explicit"] = True

    if not use_implicit or implicit_df.empty:
        utility_long = review_df[["user_id", "movie_id", "rating", "has_explicit"]].reset_index(drop=True)
        return utility_long

    implicit_part = implicit_df.rename(columns={"y": "rating"})[["user_id", "movie_id", "rating"]].copy()
    implicit_part["has_explicit"] = False

    combined = pd.concat([review_df[["user_id", "movie_id", "rating", "has_explicit"]], implicit_part])
    combined = combined.sort_values("has_explicit", ascending=False)
    utility_long = combined.drop_duplicates(subset=["user_id", "movie_id"], keep="first").reset_index(drop=True)
    return utility_long


def build_surprise_trainset(utility_long: pd.DataFrame) -> Trainset:
    """
    Surprise yeu cau input dang DataFrame 3 cot (user, item, rating) va
    1 Reader khai bao rating_scale. Rating cua he thong nay nam trong
    [1, 5] cho explicit va [1.5, 4.5] cho implicit-converted (theo tanh
    conversion) - dung chung scale (1,5) cho Reader la an toan.
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(utility_long[["user_id", "movie_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    return trainset


def train_knn_model(trainset: Trainset) -> KNNWithMeans:
    """
    KNNWithMeans: prediction = mean_u + weighted_avg(sim(u,v) * (r_vi - mean_v))
    Day chinh la cong thuc User-Based Memory CF cua de tai, da duoc
    implement san, co test, co paper - khong tu viet cong thuc.
    """
    sim_options = {
        "name": "cosine",
        "user_based": True,  # User-Based CF (khong phai Item-Based)
        "min_support": settings.cf_min_co_rated_items,
    }
    algo = KNNWithMeans(
        k=settings.cf_top_k,
        min_k=1,
        sim_options=sim_options,
    )
    algo.fit(trainset)
    return algo


def predict_ratings_for_user(
    user_id: str,
    algo: KNNWithMeans,
    trainset: Trainset,
    candidate_movie_ids: list[int],
) -> dict[int, float]:
    """
    Goi .predict() cho tung candidate movie. Surprise tu xu ly truong hop
    user/item chua biet (cold-start) bang cach tra ve global mean - vi
    he thong da co co che cold-start rieng (popularity-based) o tang
    router, nen o day CHI predict cho user da co trong trainset; neu
    khong co (rai user moi hoan toan), tra ve {} de router fallback
    sang cold-start.
    """
    try:
        trainset.to_inner_uid(user_id)
    except ValueError:
        return {}  # user chua tung xuat hien trong utility matrix -> cold start

    predictions = {}
    for movie_id in candidate_movie_ids:
        try:
            trainset.to_inner_iid(movie_id)
        except ValueError:
            continue  # movie chua tung duoc rate boi ai -> Surprise se tra global mean, khong dang tin, bo qua

        pred = algo.predict(user_id, movie_id)
        # pred.details co the chua {'was_impossible': True, 'reason': ...}
        # khi khong du neighbor hop le (vd toan bo neighbor co sim <= min_similarity)
        if pred.details.get("was_impossible", False):
            continue
        predictions[movie_id] = float(pred.est)

    return predictions

```

**Vi sao chon `user_based: True`:** De tai yeu cau User-Based CF (khong phai Item-Based) — ly do da chot truoc do la vi phim chieu rap co lifecycle ngan, dung Item-Based se lien tuc gap cold-start cho phim moi ra mat.

**Vi sao khong can tu viet "loai bo user variance=0":** Da do thuc nghiem tren 1010 user that, chi **2 user (0.2%)** roi vao truong hop "rate moi phim cung 1 diem" (variance=0, khong tinh duoc cosine sau mean-centering). Surprise tu dong tra ve ket qua "impossible" cho 2 user nay, code router se tu fallback sang cold-start popularity — khong can viet them rule dac biet nao, dung theo hanh vi chuan cua thu vien.

---


## 7.1. Lo trinh "chay thuan truoc, cai thien dan sau"

Day la tinh nang quan trong da them vao `cf_engine.py` de khop dung lo trinh thuc nghiem da chot tu dau: **chay CF Pure (chi explicit) truoc de co baseline, sau do moi them implicit de do muc cai thien**.

### Cach hoat dong

Tham so `use_implicit` (trong `build_utility_matrix`, `model_state.train()`) co 3 cach set:

| Cach set | Khi nao dung |
|---|---|
| `.env`: `REC_CF_USE_IMPLICIT=false` | Mac dinh cho toan bo service chay CF Pure (doi 1 lan, restart) |
| `.env`: `REC_CF_USE_IMPLICIT=true` | Mac dinh cho toan bo service chay CF + Implicit |
| Goi `POST /api/train` voi body `{"useImplicit": false}` | Ep chay 1 lan o mode khac, **khong can doi `.env` hoac restart** — tien nhat cho viec test/so sanh ngay luc lam thuc nghiem |

### Quy trinh de xuat cho phan thuc nghiem trong bao cao

```bash
# Buoc 1: Train o mode CF Pure, luu lai ket qua benchmark (RMSE/MAE) lam baseline
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"useImplicit": false}'

# --> Chay benchmark script (so sanh predict vs actual rating da giu lai
#     de test, hoac do Precision@K) tren model nay, ghi nhan ket qua "CF Pure"

# Buoc 2: Train lai o mode CF + Implicit, do lai cung benchmark
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"useImplicit": true}'

# --> Chay lai dung benchmark script, ghi nhan ket qua "CF + Implicit"
# --> So sanh 2 bo so -> day chinh la so lieu chung minh implicit co
#     giup cai thien hay khong, dac biet cho nhom user sparse (it rating)
```

`GET /api/health` luon cho biet **mode hien tai model dang chay** (field `cfMode`: `"cf_pure"` hoac `"cf_implicit"`), va `RecommendResponse` cung tra ve `cfMode` cung ket qua — giup de dang doi chieu khi ghi log/bieu do so sanh trong bao cao.

> **Production that nen chay mode nao?** Dat `.env` mac dinh la `REC_CF_USE_IMPLICIT=true` (da la default trong code) — vi muc tieu cuoi cua thesis la dung implicit de giam sparsity. Mode `false` chi dung cho muc dich benchmark/so sanh trong bao cao, khong phai mode chay that khi demo cho hoi dong (tru khi muon show truc tiep su khac biet).

---

## 8. Code chi tiet — `app/core/cold_start.py` (fallback cho user moi/it data)

Khi `K_u` (so luong item user da tuong tac) nho hon `cold_start_min_interactions` (mac dinh 5), CF khong du tin cay de dung — chuyen sang **Popularity Score**:

```
Score_Popularity(i) = alpha × Norm_Rating(i) + (1 − alpha) × Norm_Tickets(i)
```

`Norm_Rating` va `Norm_Tickets` duoc normalize qua min-max tren toan bo candidate set, de 2 dai luong khac don vi (rating 1-5 vs so luot dat ve) co the cong co trong so voi nhau.

```python
"""
Cold-start fallback - Popularity Score (Muc 5.2, v11).

Kich hoat khi K_u < min-interactions-threshold (user qua moi hoac qua it
du lieu de CF cho ket qua tin cay).

Score_Popularity(i) = alpha * Norm_Rating(i) + (1-alpha) * Norm_Tickets(i)
"""
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings


def compute_popularity_scores(db: Session, candidate_movie_ids: list[int]) -> dict[int, float]:
    """
    Norm_Rating(i): rating trung binh cua phim i, normalize ve [0,1] qua min-max.
    Norm_Tickets(i): so luot BOOK_TICKET cua phim i, normalize ve [0,1] qua min-max.
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
            return 0.5  # tat ca bang nhau -> trung lap
        return (value - lo) / (hi - lo)

    alpha = settings.cold_start_popularity_alpha
    scores = {}
    for movie_id in candidate_movie_ids:
        norm_rating = normalize(rating_map.get(movie_id, 0.0), all_ratings) if all_ratings else 0.0
        norm_tickets = normalize(ticket_map.get(movie_id, 0), all_tickets) if all_tickets else 0.0
        scores[movie_id] = alpha * norm_rating + (1 - alpha) * norm_tickets

    return scores


def count_user_interactions(utility_long: pd.DataFrame, user_id: str) -> int:
    """K_u = so luong item user da tuong tac (explicit + implicit, theo utility_long dang long-format)."""
    if utility_long is None or utility_long.empty:
        return 0
    return int((utility_long["user_id"] == user_id).sum())
```

---

## 9. Code chi tiet — `app/core/model_state.py` (cache model trong RAM)

Vi similarity matrix tinh lai kha nhanh (0.29 giay voi 1010 user that — da benchmark) nhung van khong nen tinh lai o **moi request** (se lam cham response time cua FE), nen model duoc **train 1 lan va giu trong RAM**, chi refresh theo lich hoac khi goi `/train`.

```python
"""
Model state - giu Surprise trainset + KNNWithMeans model trong memory.

Thiet ke: 1 instance singleton duoc load luc app startup va refresh khi
goi /train. Voi quy mo 1010 user x 102 movie, RAM trong process la du,
khong can them ha tang Redis.
"""
import threading
from datetime import datetime

import pandas as pd

from app.core.cf_engine import build_utility_matrix, build_surprise_trainset, train_knn_model
from app.core.config import settings
from app.core.implicit_scoring import build_implicit_scores, convert_to_rating_scale
from app.db.queries import load_all_reviews, load_all_activity_logs, load_candidate_movies


class ModelState:
    def __init__(self):
        self._lock = threading.Lock()
        self.algo = None
        self.trainset = None
        self.utility_long: pd.DataFrame | None = None
        self.candidate_movies: pd.DataFrame | None = None
        self.last_trained_at: datetime | None = None
        self.is_ready: bool = False
        self.last_use_implicit: bool = settings.cf_use_implicit  # mode cua lan train gan nhat
        self._last_s0: float | None = None  # S0 cua lan train truoc, dung lam fallback theo dung Java goc (khong ghi de S0 khi khong co cap implicit-only nao)

    def train(self, db_session, use_implicit: bool | None = None) -> dict:
        """
        use_implicit: None -> dung default tu config (settings.cf_use_implicit).
        Truyen rieng True/False de chay 1 lan duoi mode khac, phuc vu
        so sanh benchmark CF Pure vs CF+Implicit ngay tren cung 1 service
        ma khong can doi config/restart.
        """
        if use_implicit is None:
            use_implicit = settings.cf_use_implicit

        t0 = datetime.utcnow()

        review_df = load_all_reviews(db_session)
        candidate_df = load_candidate_movies(db_session)

        if use_implicit:
            activity_df = load_all_activity_logs(db_session)
            explicit_pairs = set(zip(review_df["user_id"], review_df["movie_id"]))
            implicit_raw = build_implicit_scores(activity_df, explicit_pairs=explicit_pairs, now=t0)
            implicit_scored = convert_to_rating_scale(implicit_raw, previous_s0=self._last_s0)
            if not implicit_raw.empty:
                from app.core.implicit_scoring import compute_s0
                computed_s0 = compute_s0(implicit_raw["S"])
                if computed_s0 is not None:
                    self._last_s0 = computed_s0
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
        return {
            "trained_at": t0.isoformat(),
            "elapsed_seconds": elapsed,
            "use_implicit": use_implicit,
            "n_users": utility_long["user_id"].nunique() if not utility_long.empty else 0,
            "n_movies_in_matrix": utility_long["movie_id"].nunique() if not utility_long.empty else 0,
            "n_candidate_movies": len(candidate_df) if candidate_df is not None else 0,
            "n_explicit_ratings": len(review_df),
            "n_activity_logs": len(activity_df),
        }

    def get_snapshot(self):
        with self._lock:
            return (self.algo, self.trainset, self.utility_long, self.candidate_movies, self.last_trained_at)


model_state = ModelState()

```

**Giai thich:**
- `threading.Lock()`: tranh race condition — neu 1 request dang doc model (`get_snapshot`) dung luc job 3AM dang viet model moi (`train`), lock dam bao khong doc du lieu nua-cu-nua-moi.
- `is_ready`: co de router biet model da train lan nao chua — neu chua (vi du service vua khoi dong va DB loi luc startup), tra loi 503 ro rang cho Spring Boot, khong de client nhan response rac.
- Tai sao khong dung Redis de cache: voi quy mo 1010 user × 102 phim, model load trong RAM process Python la du — them Redis se tang do phuc tap van hanh (them 1 service phai maintain) ma loi ich khong dang ke o quy mo nay.

---

## 10. Code chi tiet — `app/models/schemas.py` (request/response schema)

```python
from pydantic import BaseModel


class RecommendRequest(BaseModel):
    userId: str
    B: int | None = None  # tham so du phong theo so do thay ve (chua dung den, giu lai de tuong thich)


class MoviePrediction(BaseModel):
    movieId: int
    score: float
    source: str  # "cf" hoac "cold_start_popularity" - de Spring Boot/thesis bao cao biet nguon goc


class RecommendResponse(BaseModel):
    userId: str
    recommendations: list[MoviePrediction]
    usedColdStart: bool
    cfMode: str  # "cf_pure" hoac "cf_implicit" - mode cua lan train hien tai
    modelTrainedAt: str | None = None


class TrainRequest(BaseModel):
    useImplicit: bool | None = None  # None = dung default tu config


class TrainResponse(BaseModel):
    trainedAt: str
    elapsedSeconds: float
    useImplicit: bool
    nUsers: int
    nMoviesInMatrix: int
    nCandidateMovies: int
    nExplicitRatings: int
    nActivityLogs: int

```

---

## 11. Code chi tiet — `app/routers/recommend.py` (3 endpoint chinh)

Day la tang API — noi Spring Boot se goi sang. Logic chinh cua endpoint `/recommend`:

```
1. Kiem tra model da train chua (is_ready) → chua thi tra 503
2. Lay excluded_movie_ids (phim da rate/da book) cho rieng user nay
3. candidate_ids = (NOW_SHOWING + COMING_SOON) − excluded_movie_ids
4. Tinh K_u (so tuong tac cua user)
   ├── K_u < threshold (5)  → dung Popularity fallback
   └── K_u >= threshold     → dung CF (Surprise)
                               └── Neu CF khong predict duoc gi (vd toan bo
                                   candidate chua ai rate) → fallback Popularity
5. Sort theo score giam dan, lay top-N (mac dinh 3)
6. Tra JSON: {userId, recommendations: [{movieId, score, source}], usedColdStart, modelTrainedAt}
```

`source` trong response (`"cf"` hoac `"cold_start_popularity"`) giup Spring Boot/log biet ro nguon goc cua moi goi y — huu ich cho viec debug va cho phan danh gia/benchmark trong bao cao do an (so sanh ty le CF vs cold-start qua thoi gian).

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.cf_engine import predict_ratings_for_user
from app.core.cold_start import compute_popularity_scores, count_user_interactions
from app.core.config import settings
from app.core.model_state import model_state
from app.db.queries import load_excluded_movie_ids
from app.db.session import get_db
from app.models.schemas import (
    RecommendRequest,
    RecommendResponse,
    MoviePrediction,
    TrainRequest,
    TrainResponse,
)

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest, db: Session = Depends(get_db)):
    if not model_state.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model chua duoc train. Goi POST /train truoc, hoac doi scheduler 3AM chay.",
        )

    algo, trainset, utility_long, candidate_movies, trained_at = model_state.get_snapshot()
    cf_mode = "cf_implicit" if model_state.last_use_implicit else "cf_pure"

    user_id = payload.userId
    excluded_ids = load_excluded_movie_ids(db, user_id)

    all_candidate_ids = candidate_movies["movie_id"].tolist()
    candidate_ids = [m for m in all_candidate_ids if m not in excluded_ids]

    if not candidate_ids:
        return RecommendResponse(
            userId=user_id, recommendations=[], usedColdStart=False, cfMode=cf_mode,
            modelTrainedAt=trained_at.isoformat() if trained_at else None,
        )

    k_u = count_user_interactions(utility_long, user_id)
    use_cold_start = k_u < settings.cold_start_min_interactions

    if use_cold_start:
        scores = compute_popularity_scores(db, candidate_ids)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[: settings.prediction_top_n]
        recs = [MoviePrediction(movieId=mid, score=score, source="cold_start_popularity") for mid, score in ranked]
    else:
        predictions = predict_ratings_for_user(user_id, algo, trainset, candidate_ids)
        if not predictions:
            scores = compute_popularity_scores(db, candidate_ids)
            ranked = sorted(scores.items(), key=lambda x: -x[1])[: settings.prediction_top_n]
            recs = [MoviePrediction(movieId=mid, score=score, source="cold_start_popularity") for mid, score in ranked]
            use_cold_start = True
        else:
            ranked = sorted(predictions.items(), key=lambda x: -x[1])[: settings.prediction_top_n]
            recs = [MoviePrediction(movieId=mid, score=score, source="cf") for mid, score in ranked]

    return RecommendResponse(
        userId=user_id,
        recommendations=recs,
        usedColdStart=use_cold_start,
        cfMode=cf_mode,
        modelTrainedAt=trained_at.isoformat() if trained_at else None,
    )


@router.post("/train", response_model=TrainResponse)
def train(payload: TrainRequest = TrainRequest(), db: Session = Depends(get_db)):
    """
    Goi binh thuong (body rong hoac {}) -> dung mode default tu config
    (REC_CF_USE_IMPLICIT trong .env).

    Goi voi body {"useImplicit": false} -> force chay CF Pure 1 lan,
    du config dang la True. Dung de so sanh 2 mode ngay tren cung server
    ma khong can doi .env / restart - tien cho qua trinh benchmark
    "chay thuan truoc, cai thien dan sau" da chot.
    """
    result = model_state.train(db, use_implicit=payload.useImplicit)
    return TrainResponse(
        trainedAt=result["trained_at"],
        elapsedSeconds=result["elapsed_seconds"],
        useImplicit=result["use_implicit"],
        nUsers=result["n_users"],
        nMoviesInMatrix=result["n_movies_in_matrix"],
        nCandidateMovies=result["n_candidate_movies"],
        nExplicitRatings=result["n_explicit_ratings"],
        nActivityLogs=result["n_activity_logs"],
    )


@router.get("/health")
def health():
    return {
        "status": "ok",
        "modelReady": model_state.is_ready,
        "cfMode": "cf_implicit" if model_state.last_use_implicit else "cf_pure",
        "lastTrainedAt": model_state.last_trained_at.isoformat() if model_state.last_trained_at else None,
    }

```

---

## 12. Code chi tiet — `app/main.py` (entry point)

```python
"""
Entry point cho Recommendation Service (FastAPI).

Luong khoi dong:
1. App start -> train ngay 1 lan (de co model san sang, khong phai cho den 3AM)
2. APScheduler dang ky job chay moi ngay luc 3:00 AM -> goi lai model_state.train()
3. Endpoint POST /train van mo de admin trigger thu cong khi can (sau khi
   seed data moi, hoac debug)
"""
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

from app.core.model_state import model_state
from app.db.session import SessionLocal
from app.routers.recommend import router as recommend_router

scheduler = BackgroundScheduler()


def scheduled_train_job():
    db = SessionLocal()
    try:
        result = model_state.train(db)
        print(f"[scheduler] Train hoan tat luc {result['trained_at']}, "
              f"{result['n_users']} users, mat {result['elapsed_seconds']:.1f}s")
    except Exception as e:
        print(f"[scheduler] Train LOI: {e}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Train ngay luc startup de co model san sang phuc vu request dau tien
    db = SessionLocal()
    try:
        result = model_state.train(db)
        print(f"[startup] Train hoan tat: {result}")
    except Exception as e:
        print(f"[startup] Train LOI (model se chay o trang thai chua ready): {e}")
    finally:
        db.close()

    # Dang ky job 3AM hang ngay
    scheduler.add_job(scheduled_train_job, CronTrigger(hour=3, minute=0))
    scheduler.start()

    yield

    scheduler.shutdown()


app = FastAPI(
    title="Infinity Cinema - Recommendation Service",
    description="User-Based Memory CF + Implicit Feedback, phuc vu goi y top-N phim cho Spring Boot Backend.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(recommend_router, prefix="/api")
```

**Giai thich:**
- `lifespan`: co che chuan cua FastAPI (thay cho `@app.on_event` cu da deprecated) de chay code luc startup/shutdown. O day dung de: (1) train model ngay khi service khoi dong, (2) dang ky job lich 3AM, (3) tat scheduler gon gang khi service dung.
- `CronTrigger(hour=3, minute=0)`: APScheduler se tu chay `scheduled_train_job()` dung 3:00 AM moi ngay (theo gio he thong server — can dam bao server set dung timezone Viet Nam neu muon dung 3AM gio VN).
- Neu train luc startup bi loi (vi du DB chua san sang), service **van khoi dong duoc** nhung `is_ready=False` — endpoint `/recommend` se tra loi 503 ro rang cho toi khi co ai goi `/train` thanh cong.

---

## 13. Chay thu service

```bash
# Dam bao dang o thu muc goc recommend-service/, da activate venv
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Mo browser vao `http://localhost:8000/docs` de thay Swagger UI tu dong sinh ra — test truc tiep 3 endpoint tai day ma khong can Postman:

- `GET /api/health` — kiem tra service song va model da train chua
- `POST /api/train` — trigger train thu cong
- `POST /api/recommend` — body mau: `{"userId": "7fb65e77-0726-5a7d-a685-f8a9e0bdda42"}`

### Test bang curl

```bash
# Kiem tra health
curl http://localhost:8000/api/health

# Train thu cong
curl -X POST http://localhost:8000/api/train

# Lay goi y cho 1 user
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"userId": "7fb65e77-0726-5a7d-a685-f8a9e0bdda42"}'
```

---

## 14. Tich hop voi Spring Boot (gateway)

Spring Boot chi can 1 doan code goi REST sang Python — vi du dung `RestTemplate` hoac `WebClient`:

```java
// Spring Boot - vi du dung RestTemplate
@Service
public class RecommendationGatewayService {

    private final RestTemplate restTemplate;
    private final String pythonServiceUrl = "http://localhost:8000/api/recommend";

    public List<MovieRecommendationDto> getRecommendations(String userId) {
        Map<String, Object> requestBody = Map.of("userId", userId);

        ResponseEntity<PythonRecommendResponse> response = restTemplate.postForEntity(
            pythonServiceUrl, requestBody, PythonRecommendResponse.class
        );

        List<Integer> movieIds = response.getBody().getRecommendations()
            .stream().map(r -> r.getMovieId()).toList();

        // Enrich: lay them poster, ten phim, gio chieu tu DB cua Spring Boot
        return movieRepository.findByIdIn(movieIds)
            .stream().map(this::toRecommendationDto).toList();
    }
}
```

**Luu y khi viet Spring Boot:**
- Neu Python service tra loi 503 (model chua train) — Spring Boot nen co fallback (vi du tra ve top phim hot nhat tu DB cua chinh minh), khong de FE nhan loi 500 tran.
- Nen set timeout hop ly cho call REST nay (vi du 5 giay) — vi du Python rat nhanh (model da cache), network latency van can tinh den.

---

## 15. Han che da biet (quan trong — nen dua vao phan "Han che" cua bao cao do an)

| Han che | Mo ta | Muc do anh huong |
|---|---|---|
| **Chain-bonus xap xi** | Do composite PK `(action_type, movie_id, user_id)` khong luu lich su tung lan action rieng biet, chain-detection `VIEW_SHOWTIMES → BOOK_TICKET` dung `updated_at` (lan cap nhat gan nhat) lam xap xi "lan xay ra gan nhat" | Nho — chi anh huong do chinh xac cua +8 diem bonus, khong anh huong cau truc tong the |
| **Cosine similarity khong xac dinh khi user co variance=0** | User rate moi phim cung 1 diem → khong tinh duoc huong vector sau mean-centering | Rat nho — da do thuc nghiem, chi 0.2% user (2/1010) roi vao case nay, tu dong fallback cold-start |
| **Candidate movies gioi han o NOW_SHOWING/COMING_SOON** | Phim da STOPPED khong duoc goi y du CF co the du doan diem cao | Co chu dich — dung ban chat bai toan dat ve (khong goi y phim het chieu) |
| **Training dong bo (synchronous)** | `/train` chay trong cung 1 request, khong phai background job | Chap nhan duoc o quy mo hien tai (0.29s cho 1010 user); neu data tang lon hon nhieu, nen chuyen sang background task (Celery/RQ) |

---

## 16. Checklist truoc khi nop bao cao / demo

- [ ] Dam bao `.env` khong commit len Git (them vao `.gitignore`)
- [ ] Chay `POST /train` ngay sau khi seed data moi de model khong bi lech
- [ ] Verify `GET /api/health` tra `modelReady: true` truoc khi demo
- [ ] Test it nhat 1 user cold-start (user moi tao, chua rate gi) de show duoc co che fallback
- [ ] Test it nhat 1 user co du rating (≥5) de show duoc CF that hoat dong
- [ ] Trong bao cao, trich dan dung: Hug, N. (2020). *Surprise: A Python library for recommender systems*. Journal of Open Source Software, 5(52), 2174. — va Hu, Koren, Volinsky (2008) cho phan frequency weighting implicit feedback.

---

*Het tai lieu huong dan.*
