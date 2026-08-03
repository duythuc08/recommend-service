"""
Entry point cho Recommendation Service (FastAPI).

Luong khoi dong:
1. App start -> train ngay 1 lan de co model san sang
2. Spring Boot scheduler goi POST /api/train moi ngay luc 3:00 AM
3. Endpoint POST /api/train van mo de admin trigger thu cong khi can
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.model_state import model_state
from app.db.session import SessionLocal
from app.routers.recommend import router as recommend_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = SessionLocal()
    try:
        result = model_state.train(db)
        print(f"[startup] Train hoan tat: {result}")
    except Exception as e:
        print(f"[startup] Train loi, model chua ready: {e}")
    finally:
        db.close()

    yield


app = FastAPI(
    title="Infinity Cinema - Recommendation Service",
    description="User-Based Memory CF phuc vu goi y top-N phim cho Spring Boot Backend.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(recommend_router, prefix="/api")
