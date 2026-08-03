from pydantic import BaseModel


class RecommendRequest(BaseModel):
    userId: str
    B: int | None = None


class MoviePrediction(BaseModel):
    movieId: int
    predictedScore: float
    neighborCount: int
    source: str


class RecommendResponse(BaseModel):
    userId: str
    recommendations: list[MoviePrediction]
    usedColdStart: bool
    cfMode: str
    modelTrainedAt: str | None = None


class TrainResponse(BaseModel):
    trainedAt: str
    elapsedSeconds: float
    nUsers: int
    nMoviesInMatrix: int
    nCandidateMovies: int
    nExplicitRatings: int
    nUsersProcessed: int
    nPredictionsWritten: int
    nStalePredictionsDeleted: int
    batchElapsedSeconds: float
