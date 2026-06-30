from pydantic import BaseModel


class RecommendRequest(BaseModel):
    userId: str
    B: int | None = None  # tham số dự phòng theo sơ đồ thầy về (chưa dùng đến, giữ lại để tương thích)


class MoviePrediction(BaseModel):
    movieId: int
    score: float
    source: str  # "cf" hoặc "cold_start_popularity" - để Spring Boot/thesis báo cáo biết nguồn gốc


class RecommendResponse(BaseModel):
    userId: str
    recommendations: list[MoviePrediction]
    usedColdStart: bool
    cfMode: str  # "cf_pure" hoặc "cf_implicit" - mode của lần train hiện tại
    modelTrainedAt: str | None = None


class TrainRequest(BaseModel):
    useImplicit: bool | None = None  # None = dùng default từ config


class TrainResponse(BaseModel):
    trainedAt: str
    elapsedSeconds: float
    useImplicit: bool
    nUsers: int
    nMoviesInMatrix: int
    nCandidateMovies: int
    nExplicitRatings: int
    nActivityLogs: int
