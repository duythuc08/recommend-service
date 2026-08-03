from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.cf_engine import predict_ratings_for_user
from app.core.cold_start import compute_popularity_scores, count_user_interactions
from app.core.config import settings
from app.core.model_state import CF_SOURCE, model_state
from app.db.queries import load_excluded_movie_ids
from app.db.session import get_db
from app.models.schemas import (
    MoviePrediction,
    RecommendRequest,
    RecommendResponse,
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

    user_id = payload.userId
    excluded_ids = load_excluded_movie_ids(db, user_id)

    all_candidate_ids = candidate_movies["movie_id"].tolist()
    candidate_ids = [m for m in all_candidate_ids if m not in excluded_ids]

    if not candidate_ids:
        return RecommendResponse(
            userId=user_id,
            recommendations=[],
            usedColdStart=False,
            cfMode=CF_SOURCE,
            modelTrainedAt=trained_at.isoformat() if trained_at else None,
        )

    k_u = count_user_interactions(utility_long, user_id)
    use_cold_start = k_u < settings.cold_start_min_interactions

    if use_cold_start:
        scores = compute_popularity_scores(db, candidate_ids)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[: settings.prediction_top_n]
        recs = [
            MoviePrediction(movieId=mid, predictedScore=score, neighborCount=0, source="cold_start_popularity")
            for mid, score in ranked
        ]
    else:
        predictions = predict_ratings_for_user(user_id, algo, trainset, candidate_ids)
        if not predictions:
            scores = compute_popularity_scores(db, candidate_ids)
            ranked = sorted(scores.items(), key=lambda x: -x[1])[: settings.prediction_top_n]
            recs = [
                MoviePrediction(movieId=mid, predictedScore=score, neighborCount=0, source="cold_start_popularity")
                for mid, score in ranked
            ]
            use_cold_start = True
        else:
            ranked = sorted(predictions.items(), key=lambda x: -x[1][0])[: settings.prediction_top_n]
            recs = [
                MoviePrediction(movieId=mid, predictedScore=score, neighborCount=nc, source=CF_SOURCE)
                for mid, (score, nc) in ranked
            ]

    return RecommendResponse(
        userId=user_id,
        recommendations=recs,
        usedColdStart=use_cold_start,
        cfMode=CF_SOURCE,
        modelTrainedAt=trained_at.isoformat() if trained_at else None,
    )


@router.post("/train", response_model=TrainResponse)
def train(db: Session = Depends(get_db)):
    result = model_state.train(db)
    return TrainResponse(
        trainedAt=result["trained_at"],
        elapsedSeconds=result["elapsed_seconds"],
        nUsers=result["n_users"],
        nMoviesInMatrix=result["n_movies_in_matrix"],
        nCandidateMovies=result["n_candidate_movies"],
        nExplicitRatings=result["n_explicit_ratings"],
        nUsersProcessed=result["n_users_processed"],
        nPredictionsWritten=result["n_predictions_written"],
        nStalePredictionsDeleted=result["n_stale_predictions_deleted"],
        batchElapsedSeconds=result["batch_elapsed_seconds"],
    )


@router.get("/health")
def health():
    return {
        "status": "ok",
        "modelReady": model_state.is_ready,
        "cfMode": CF_SOURCE,
        "lastTrainedAt": model_state.last_trained_at.isoformat() if model_state.last_trained_at else None,
    }
