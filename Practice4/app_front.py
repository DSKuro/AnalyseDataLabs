import streamlit as st
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/analytics"

st.set_page_config(page_title="LoL Analytics", layout="wide")
st.title("Анализ матчей League of Legends")

st.header("Основные результаты EDA")

if st.checkbox("Топ чемпионов по KDA"):
    r = requests.get(f"{API_URL}/kda")
    df = pd.DataFrame(r.json())
    st.dataframe(df.head(10))

if st.checkbox("Winrate чемпионов по позициям"):
    r = requests.get(f"{API_URL}/winrate-by-position")
    df = pd.DataFrame(r.json())
    st.dataframe(df.head(10))

champ = st.text_input("Введите имя чемпиона для статистики")
if champ:
    r = requests.get(f"{API_URL}/champion/{champ}")
    df = pd.DataFrame(r.json())
    st.dataframe(df)

r = requests.get(f"{API_URL}/participants/count")
st.metric("Всего участников", r.json()["total"])

if st.checkbox("Средний урон по матчам"):
    r = requests.get(f"{API_URL}/damage")
    df = pd.DataFrame(r.json())
    df = df.sort_values("avg_damage", ascending=False)
    st.bar_chart(df.set_index("champion")["avg_damage"])

if st.checkbox("Длительность матчей"):
    r = requests.get(f"{API_URL}/match-duration")
    df = pd.DataFrame(r.json())
    st.line_chart(df.set_index("version")["avg_duration_minutes"])

if st.checkbox("Наиболее забаненные чемпионы"):
    r = requests.get(f"{API_URL}/bans")
    df = pd.DataFrame(r.json())
    df = df.sort_values("bans", ascending=False)
    st.bar_chart(df.set_index("champion")["bans"])


st.header("Прогноз среднего урона (avg_damage)")

model_type = st.selectbox("Выберите модель", ["baseline", "automl"], key="predict_model_type")

avg_kills = st.number_input("Avg Kills", 0.0, 20.0, 5.0)
avg_deaths = st.number_input("Avg Deaths", 0.0, 20.0, 5.0)
avg_assists = st.number_input("Avg Assists", 0.0, 20.0, 5.0)
avg_gold = st.number_input("Avg Gold", 0.0, 30000.0, 12000.0)

if st.button("Обучить модель"):
    r = requests.post(f"{API_URL}/ml/train?method={model_type}")
    result = r.json()
    st.success(f"Модель обучена! Тип: {result.get('model', result.get('type'))}")
    st.write(result)

if st.button("Предсказать damage"):
    payload = {
        "avg_kills": avg_kills,
        "avg_deaths": avg_deaths,
        "avg_assists": avg_assists,
        "avg_gold": avg_gold,
        "model_type": model_type
    }
    r = requests.post(f"{API_URL}/predict", json=payload)
    data = r.json()

    if "error" in data:
        st.error(data["error"])
    elif "predicted_avg_damage" in data:
        st.success(f"Прогнозируемый средний урон: {data['predicted_avg_damage']:.2f}")
        st.info(f"Использована модель: {data['model_used']}")
    else:
        st.warning("Что-то пошло не так с предсказанием")