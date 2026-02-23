import os
import pickle

import joblib
import pandas as pd
from fastapi import APIRouter, Query
from lol_ml_services.analytics import (
    query1_top_kda,
    query2_winrate_by_position,
    query3_champion_stats,
    query4_total_participants,
    query5_avg_damage,
    query5_match_duration,
    query5_most_banned,
    get_ml_dataset
)

from lol_ml_services.automl_pycarret import train_automl
from lol_ml_services.ml_baseline import train_baseline_model
from pydantic import BaseModel

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/kda")
def kda_chart():
    return query1_top_kda()


@router.get("/winrate-by-position")
def winrate_by_position():
    return query2_winrate_by_position()


@router.get("/champion/{name}")
def champion_stats(name: str):
    return query3_champion_stats(name)


@router.get("/participants/count")
def total_participants():
    return {"total": query4_total_participants()}


@router.get("/damage")
def damage_chart():
    return query5_avg_damage()


@router.get("/match-duration")
def match_duration_chart():
    return query5_match_duration()


@router.get("/bans")
def bans_chart():
    return query5_most_banned()

class PlayerStats(BaseModel):
    avg_kills: float
    avg_deaths: float
    avg_assists: float
    avg_gold: float
    model_type: str = "baseline"   # baseline или automl


def load_model(model_type: str):
    if model_type == "automl":
        path = "models/best_automl_damage_model.pkl"
    else:
        path = "models/baseline_rf_damage.pkl"

    if not os.path.exists(path):
        return None
    return joblib.load(path)


@router.post("/predict")
def predict(stats: PlayerStats):

    model = load_model(stats.model_type)

    if model is None:
        return {"error": "Model not trained yet"}

    X = pd.DataFrame([{
        "avg_kills": stats.avg_kills,
        "avg_deaths": stats.avg_deaths,
        "avg_assists": stats.avg_assists,
        "avg_gold": stats.avg_gold
    }])

    prediction = model.predict(X)[0]

    return {
        "model_used": stats.model_type,
        "predicted_avg_damage": round(float(prediction), 4)
    }


@router.post("/ml/train")
def train_ml(method: str = Query("baseline", enum=["baseline", "automl"])):

    df = get_ml_dataset()

    if method == "automl":
        return train_automl(df)

    return train_baseline_model(df)