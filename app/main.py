"""
Entry point cho Recommendation Service (FastAPI).

Luồng khởi động:
1. App start -> train ngay 1 lần (để có model sẵn sàng, không phải chờ đến 3AM)
2. Spring Boot scheduler gọi POST /api/train mỗi ngày lúc 3:00 AM
3. Endpoint POST /api/train vẫn mở để admin trigger thủ công khi cần
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.model_state import model_state
from app.db.session import SessionLocal
from app.routers.recommend import router as recommend_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Train ngay lúc startup để có model sẵn sàng phục vụ request đầu tiên
    db = SessionLocal()
    try:
        result = model_state.train(db)
        print(f"[startup] Train hoàn tất: {result}")
    except Exception as e:
        print(f"[startup] Train LỖI (model sẽ chạy ở trạng thái chưa ready): {e}")
    finally:
        db.close()

    yield


app = FastAPI(
    title="Infinity Cinema - Recommendation Service",
    description="User-Based Memory CF + Implicit Feedback, phục vụ gợi ý top-N phim cho Spring Boot Backend.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(recommend_router, prefix="/api")
