from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, round, when, sum as spark_sum
)

spark = SparkSession.builder \
    .appName("LoL DataFrame API") \
    .getOrCreate()

champs = spark.read.csv("../data/champs.csv", header=True, inferSchema=True)
matches = spark.read.csv("../data/matches.csv", header=True, inferSchema=True)
participants = spark.read.csv("../data/participants.csv", header=True, inferSchema=True)
stats1 = spark.read.csv("../data/stats1.csv", header=True, inferSchema=True)
teamstats = spark.read.csv("../data/teamstats.csv", header=True, inferSchema=True)
teambans = spark.read.csv("../data/teambans.csv", header=True, inferSchema=True)

print("=== Schemas ===")
champs.printSchema()
participants.printSchema()
stats1.printSchema()
teamstats.printSchema()


stats1 = stats1.withColumn(
    "win_num",
    when(stats1.win == 1, 1).otherwise(0)
)

query1 = (
    participants
    .join(stats1, participants.id == stats1.id)
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
    .filter(col("games") > 100)
    .orderBy(col("avg_kda").desc())
)

query1.show(10)

query2 = (
    participants
    .join(stats1, participants.id == stats1.id)
    .join(champs, participants.championid == champs.id)
    .groupBy(
        champs.name.alias("champion"),
        participants.position
    )
    .agg(
        count("*").alias("games"),
        round(avg(col("win_num")) * 100, 2).alias("winrate")
    )
    .filter(col("games") > 50)
    .orderBy(col("winrate").desc())
)

query2.show(10)

query3 = (
    champs
    .filter(col("name") == "Jax")
    .join(participants, champs.id == participants.championid)
    .join(stats1, participants.id == stats1.id)
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

query3.show()

query4 = (
    participants
    .join(matches, participants.matchid == matches.id)
    .count()
)

print("Total participants in matches:", query4)

query5_1 = (
    participants
    .join(stats1, participants.id == stats1.id)
    .join(champs, participants.championid == champs.id)
    .groupBy(champs.name.alias("champion"))
    .agg(
        round(avg(col("totdmgtochamp")), 0).alias("avg_damage"),
        count("*").alias("games")
    )
    .filter(col("games") > 100)
    .orderBy(col("avg_damage").desc())
)

query5_1.show(10)

query5_2 = (
    matches
    .groupBy("version")
    .agg(
        round(avg(col("duration")) / 60, 2).alias("avg_duration_minutes"),
        count("*").alias("matches")
    )
    .orderBy(col("avg_duration_minutes").desc())
)

query5_2.show()

query5_3 = (
    teambans
    .join(champs, teambans.championid == champs.id)
    .groupBy(champs.name.alias("champion"))
    .agg(count("*").alias("bans"))
    .orderBy(col("bans").desc())
)

query5_3.show(10)

spark.stop()
