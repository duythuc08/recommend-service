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
            detail="Model chưa được train. Gọi POST /train trước, hoặc đợi scheduler 3AM chạy.",
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
    Gọi bình thường (body rỗng hoặc {}) -> dùng mode default từ config
    (REC_CF_USE_IMPLICIT trong .env).

    Gọi với body {"useImplicit": false} -> force chạy CF Pure 1 lần,
    dù config đang là True. Dùng để so sánh 2 mode ngay trên cùng server
    mà không cần đổi .env / restart - tiện cho quá trình benchmark
    "chạy thuần trước, cải thiện dần sau" đã chốt.
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
