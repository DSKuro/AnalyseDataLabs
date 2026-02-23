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
    df = df.set_index("champion")
    st.bar_chart(df["bans"])

st.header("Прогноз победы игрока")

kills = st.number_input("Kills", 0, 30, 5)
deaths = st.number_input("Deaths", 0, 30, 5)
assists = st.number_input("Assists", 0, 30, 5)
gold = st.number_input("Gold Earned", 0, 30000, 10000)

if st.button("Предсказать победу"):
    payload = {
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "gold": gold
    }
    r = requests.post(f"{API_URL}/predict", json=payload)
    data = r.json()

    win_prob = data["prob_win"] * 100
    lose_prob = data["prob_lose"] * 100
    pred = data["prediction"]

    if pred == 1:
        st.success(f"Победа (вероятность {win_prob:.2f}%)")
    else:
        st.error(f"Поражение (вероятность {lose_prob:.2f}%)")

    prob_df = pd.DataFrame({
        "Исход": ["Поражение", "Победа"],
        "Вероятность (%)": [lose_prob, win_prob]
    }).set_index("Исход")
    st.bar_chart(prob_df)

if st.checkbox("Показать метрики модели"):
    r = requests.get(f"{API_URL}/metrics")
    metrics = r.json()
    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    st.metric("Precision", f"{metrics['precision']:.3f}")
    st.metric("Recall", f"{metrics['recall']:.3f}")
    st.metric("F1-score", f"{metrics['f1']:.3f}")
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")