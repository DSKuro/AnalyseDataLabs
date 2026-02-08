from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, round, when

def get_spark():
    return SparkSession.builder \
        .appName("LoL DataFrame API") \
        .getOrCreate()

def load_data(spark):
    champs = spark.read.csv("../data/champs.csv", header=True, inferSchema=True)
    matches = spark.read.csv("../data/matches.csv", header=True, inferSchema=True)
    participants = spark.read.csv("../data/participants.csv", header=True, inferSchema=True)
    stats1 = spark.read.csv("../data/stats1.csv", header=True, inferSchema=True)
    teambans = spark.read.csv("../data/teambans.csv", header=True, inferSchema=True)

    stats1 = stats1.withColumn(
        "win_num",
        when(col("win") == 1, 1).otherwise(0)
    )

    return champs, matches, participants, stats1, teambans

def run_queries(champs, matches, participants, stats1, teambans):

    query1 = (
        participants
        .join(stats1, "id")
        .join(champs, participants.championid == champs.id)
        .groupBy(champs.name.alias("champion"))
        .agg(
            count("*").alias("games"),
            round(
                avg(
                    (col("kills") + col("assists")) /
                    when(col("deaths") != 0, col("deaths")).otherwise(1)
                ), 2
            ).alias("avg_kda")
        )
        .filter(col("games") > 100)
        .orderBy(col("avg_kda").desc())
    )

    query2 = (
        participants
        .join(stats1, "id")
        .join(champs, participants.championid == champs.id)
        .groupBy(champs.name.alias("champion"), participants.position)
        .agg(
            count("*").alias("games"),
            round(avg(col("win_num")) * 100, 2).alias("winrate")
        )
        .filter(col("games") > 50)
        .orderBy(col("winrate").desc())
    )

    query3 = (
        champs
        .filter(col("name") == "Jax")
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

    query5_2 = (
        matches
        .groupBy("version")
        .agg(
            round(avg(col("duration")) / 60, 2).alias("avg_duration_minutes"),
            count("*").alias("matches")
        )
        .orderBy(col("avg_duration_minutes").desc())
    )

    return query1, query2, query3, query5_2
