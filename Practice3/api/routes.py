import pickle

from fastapi import APIRouter, Query
from lol_services.analytics import *
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
    kills: int
    deaths: int
    assists: int
    gold: int

with open("model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    metrics = data["metrics"]

@router.post("/predict")
def predict(stats: PlayerStats):
    X = [[stats.kills, stats.deaths, stats.assists, stats.gold]]
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return {
        "prediction": int(pred),
        "prob_win": float(prob[1]),
        "prob_lose": float(prob[0])
    }

@router.get("/metrics")
def get_metrics():
    return metrics