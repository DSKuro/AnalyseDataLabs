import os

import pandas as pd
from pycaret.regression import setup, compare_models, pull, finalize_model, save_model

def train_automl(df: pd.DataFrame):
    # 🔹 Выбираем те же признаки, что в baseline
    try:
        features = ["avg_kills", "avg_deaths", "avg_assists", "avg_gold", "avg_damage"]
        df = df[features]

        exp = setup(
            data=df,
            target="avg_damage",
            session_id=42,
            normalize=False,    # деревья не требуют нормализации
            fold=5,
            verbose=False
        )

        # 🔹 Ограничиваем модели деревьями
        best_model = compare_models(include=["rf", "et", "lightgbm"])
        results = pull()
        final_model = finalize_model(best_model)

        os.makedirs("models", exist_ok=True)
        model_path = "models/best_automl_damage_model.pkl"
        save_model(final_model, model_path)

        viz_data = results[["Model", "R2 Score", "RMSE", "MAE"]]

        return {
            "type": "AutoML (PyCaret)",
            "best_model": str(best_model),
            "leaderboard": viz_data.to_dict("records"),
            "visualization": {
                "models": viz_data["Model"].tolist(),
                "r2": viz_data["R2 Score"].tolist(),
                "rmse": viz_data["RMSE"].tolist()
            },
            "model_path": model_path
        }
    except Exception as e:
        return {"error": f"Ошибка при обучении AutoML: {str(e)}"}