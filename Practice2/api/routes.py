from fastapi import APIRouter, Query
from lol_services.analytics import *


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
