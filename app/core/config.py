from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = ""
    db_name: str = "movie_ticket"

    # CF params
    cf_top_k: int = 40
    cf_min_co_rated_items: int = 10
    cf_min_similarity: float = 0.0

    # Cold start
    cold_start_min_interactions: int = 10
    cold_start_popularity_alpha: float = 0.5

    # Prediction
    prediction_top_n: int = 5

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
