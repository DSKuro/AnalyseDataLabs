import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DB_USER = "postgres"
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "lol_db"

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

csv_files = {
    "champs": "./champs.csv",
    "matches": "./matches.csv",
    "participants": "./participants.csv",
    "stats1": "./stats1.csv",
    "stats2": "./stats2.csv",
    "teamstats": "./teamstats.csv",
    "teambans": "./teambans.csv"
}

for table_name, file_path in csv_files.items():
    print(f"Загружаем {table_name}...")
    df = pd.read_csv(file_path, encoding='utf-8', na_values='\\N')
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Таблица {table_name} загружена, {df.shape[0]} строк.")

print("Все таблицы успешно загружены в PostgreSQL!")
