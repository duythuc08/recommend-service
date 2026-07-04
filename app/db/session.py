from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.sqlalchemy_url,
    pool_pre_ping=True,   # tránh lỗi "MySQL server has gone away" khi connection idle lâu
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency cho FastAPI - mỗi request mở 1 session, đóng sau khi xong."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
