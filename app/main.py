"""
Entry point cho Recommendation Service (FastAPI).

Luồng khởi động:
1. App start -> train ngay 1 lần (để có model sẵn sàng, không phải chờ đến 3AM)
2. APScheduler đăng ký job chạy mỗi ngày lúc 3:00 AM -> gọi lại model_state.train()
3. Endpoint POST /train vẫn mở để admin trigger thủ công khi cần (sau khi
   seed data mới, hoặc debug)
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
        print(f"[scheduler] Train hoàn tất lúc {result['trained_at']}, "
              f"{result['n_users']} users, mất {result['elapsed_seconds']:.1f}s")
    except Exception as e:
        print(f"[scheduler] Train LỖI: {e}")
    finally:
        db.close()


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

    # Đăng ký job 3AM hàng ngày
    scheduler.add_job(scheduled_train_job, CronTrigger(hour=3, minute=0))
    scheduler.start()

    yield

    scheduler.shutdown()


app = FastAPI(
    title="Infinity Cinema - Recommendation Service",
    description="User-Based Memory CF + Implicit Feedback, phục vụ gợi ý top-N phim cho Spring Boot Backend.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(recommend_router, prefix="/api")
