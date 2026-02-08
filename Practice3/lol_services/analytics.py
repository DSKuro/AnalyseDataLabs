from pyspark.sql.functions import (
    col, count, avg, round, when
)
from spark_session import load_data

data = load_data()

champs = data["champs"]
matches = data["matches"]
participants = data["participants"]
stats1 = data["stats1"]
teambans = data["teambans"]

stats1 = stats1.withColumn(
    "win_num",
    when(col("win") == 1, 1).otherwise(0)
)

def query1_top_kda(min_games=100, limit=10):
    df = (
        participants
        .join(stats1, "id")
        .join(champs, participants.championid == champs.id)
        .groupBy(champs.name.alias("champion"))
        .agg(
            count("*").alias("games"),
            round(
                avg((col("kills") + col("assists")) /
                    when(col("deaths") != 0, col("deaths"))),
                2
            ).alias("avg_kda")
        )
        .filter(col("games") > min_games)
        .orderBy(col("avg_kda").desc())
        .limit(limit)
    )
    return df.toPandas().to_dict("records")

def query2_winrate_by_position(min_games=50):
    df = (
        participants
        .join(stats1, "id")
        .join(champs, participants.championid == champs.id)
        .groupBy(
            champs.name.alias("champion"),
            participants.position
        )
        .agg(
            count("*").alias("games"),
            round(avg(col("win_num")) * 100, 2).alias("winrate")
        )
        .filter(col("games") > min_games)
        .orderBy(col("winrate").desc())
    )
    return df.toPandas().to_dict("records")

def query3_champion_stats(champion_name: str):
    df = (
        champs
        .filter(col("name") == champion_name)
        .join(participants, champs.id == participants.championid)
        .join(stats1, "id")
        .groupBy(champs.name.alias("champion"))
        .agg(
            count("*").alias("games"),
            round(avg(col("win_num")) * 100, 2).alias("winrate"),
            round(avg(col("kills")), 2).alias("avg_kills"),
            round(avg(col("deaths")), 2).alias("avg_deaths"),
            round(avg(col("assists")), 2).alias("avg_assists"),
            round(avg(col("goldearned")), 0).alias("avg_gold"),
            round(avg(col("totdmgtochamp")), 0).alias("avg_damage")
        )
    )
    return df.toPandas().to_dict("records")

def query4_total_participants():
    return participants.join(matches, participants.matchid == matches.id).count()

def query5_avg_damage(min_games=100, limit=10):
    df = (
        participants
        .join(stats1, "id")
        .join(champs, participants.championid == champs.id)
        .groupBy(champs.name.alias("champion"))
        .agg(
            round(avg(col("totdmgtochamp")), 0).alias("avg_damage"),
            count("*").alias("games")
        )
        .filter(col("games") > min_games)
        .orderBy(col("avg_damage").desc())
        .limit(limit)
    )
    return df.toPandas().to_dict("records")

def query5_match_duration():
    df = (
        matches
        .groupBy("version")
        .agg(
            round(avg(col("duration")) / 60, 2).alias("avg_duration_minutes"),
            count("*").alias("matches")
        )
        .orderBy(col("avg_duration_minutes").desc())
    )
    return df.toPandas().to_dict("records")

def query5_most_banned(limit=10):
    df = (
        teambans
        .join(champs, teambans.championid == champs.id)
        .groupBy(champs.name.alias("champion"))
        .agg(count("*").alias("bans"))
        .orderBy(col("bans").desc())
        .limit(limit)
    )
    return df.toPandas().to_dict("records")
