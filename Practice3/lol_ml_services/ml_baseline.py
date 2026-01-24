import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


def train_baseline_model(df: pd.DataFrame):
    features = [
        "avg_kills",
        "avg_deaths",
        "avg_assists",
        "avg_damage",
        "avg_gold"
    ]

    X = df[features]
    y = df["winrate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/baseline_random_forest.pkl")

    return {
        "model": "RandomForestRegressor",
        "r2": round(r2_score(y_test, y_pred), 4),
        "rmse": round(mean_squared_error(y_test, y_pred, squared=False), 4),
        "model_path": "models/baseline_random_forest.pkl"
    }
