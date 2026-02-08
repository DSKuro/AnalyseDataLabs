from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("LoL Analytics API")
    .getOrCreate()
)

def load_data():
    base = "../data/"
    return {
        "champs": spark.read.csv(base + "champs.csv", header=True, inferSchema=True),
        "matches": spark.read.csv(base + "matches.csv", header=True, inferSchema=True),
        "participants": spark.read.csv(base + "participants.csv", header=True, inferSchema=True),
        "stats1": spark.read.csv(base + "stats1.csv", header=True, inferSchema=True),
        "teamstats": spark.read.csv(base + "teamstats.csv", header=True, inferSchema=True),
        "teambans": spark.read.csv(base + "teambans.csv", header=True, inferSchema=True),
    }
