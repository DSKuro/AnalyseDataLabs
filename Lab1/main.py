import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

DB_USER = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "lol_db"

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

stats1 = pd.read_sql("SELECT * FROM stats1", engine)
stats2 = pd.read_sql("SELECT * FROM stats2", engine)
stats = pd.concat([stats1, stats2], ignore_index=True)

participants = pd.read_sql("SELECT * FROM participants", engine)
df = stats.merge(participants, left_on="id", right_on="id")

plt.figure(figsize=(8,5))
sns.histplot(df['kills'], bins=30, kde=True, color='skyblue')
plt.title("Распределение kills")
plt.xlabel("Kills")
plt.ylabel("Количество игроков")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['goldearned'], bins=30, kde=True, color='salmon')
plt.title("Распределение goldearned")
plt.xlabel("Gold earned")
plt.ylabel("Количество игроков")
plt.show()