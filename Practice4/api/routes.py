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

@router.post("/ml/train")
def train_ml(method: str = "baseline"):
    df = get_ml_dataset()

    if method == "automl":
        return train_automl(df)

    return train_baseline_model(df)
