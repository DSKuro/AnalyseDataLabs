import pandas as pd
import streamlit as st
import pickle
from spark_queries import get_spark, load_data, run_queries

st.set_page_config(page_title="LoL Data Analysis", layout="wide")

st.title("Анализ матчей League of Legends")

st.markdown("""
Данное веб-приложение представляет витрину данных,
полученных в ходе исследовательского анализа матчей League of Legends
с использованием Apache Spark.
""")

spark = get_spark()
champs, matches, participants, stats1, teambans = load_data(spark)

q1, q2, q3, q5_2 = run_queries(
    champs, matches, participants, stats1, teambans
)

st.header("Основные результаты EDA")

st.subheader("Топ чемпионов по KDA")
st.dataframe(q1.toPandas().head(10))

st.subheader("Winrate чемпионов по позициям")
st.dataframe(q2.toPandas().head(10))

st.subheader("Статистика чемпиона Jax")
st.dataframe(q3.toPandas())

st.subheader("Средняя длительность матчей по версиям")
df_duration = q5_2.toPandas()
st.line_chart(df_duration.set_index("version")["avg_duration_minutes"])

st.header("Машинное обучение")

with open("model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    metrics = data["metrics"]

st.markdown("Пример модели: предсказание победы по статистике игрока")

kills = st.number_input("Kills", 0, 30, 5)
deaths = st.number_input("Deaths", 0, 30, 5)
assists = st.number_input("Assists", 0, 30, 5)
gold = st.number_input("Gold Earned", 0, 30000, 10000)

if st.button("Предсказать победу"):
    X = [[kills, deaths, assists, gold]]

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    win_prob = prob[1] * 100
    lose_prob = prob[0] * 100

    st.subheader("Результат предсказания")

    if pred == 1:
        st.success(f"Победа (вероятность {win_prob:.2f}%)")
    else:
        st.error(f"Поражение (вероятность {lose_prob:.2f}%)")

    prob_df = pd.DataFrame({
        "Исход": ["Поражение", "Победа"],
        "Вероятность (%)": [lose_prob, win_prob]
    }).set_index("Исход")

    st.bar_chart(prob_df)

    st.markdown("### Интерпретация модели")

    explanation = []

    if kills + assists > 10:
        explanation.append("Высокое количество убийств и ассистов повышает шанс победы.")

    if deaths > 7:
        explanation.append("Большое число смертей негативно влияет на вероятность победы.")

    if gold > 12000:
        explanation.append("Хороший показатель золота — положительный фактор.")

    if not explanation:
        explanation.append("Показатели находятся в среднем диапазоне.")

    for e in explanation:
        st.write("•", e)

st.subheader("Метрики качества модели")

st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
st.metric("Precision", f"{metrics['precision']:.3f}")
st.metric("Recall", f"{metrics['recall']:.3f}")
st.metric("F1-score", f"{metrics['f1']:.3f}")
st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
st.metric("R² (pseudo)", f"{metrics['r2_pseudo']:.3f}")

st.caption(
    "R² приведён как псевдо-метрика. "
    "Для задачи классификации основными являются Accuracy, F1 и ROC-AUC."
)


spark.stop()
