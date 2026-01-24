import os

import pandas as pd
from pycaret.regression import (
    setup,
    compare_models,
    pull,
    finalize_model, save_model
)


def train_automl(df: pd.DataFrame):
    setup(
        data=df,
        target="winrate",
        session_id=42,
        fold=5,
        normalize=True,
        verbose=False
    )

    best_model = compare_models()
    results = pull()
    final_model = finalize_model(best_model)

    # 🔹 данные для визуализации
    viz_data = results[["Model", "R2", "RMSE", "MAE"]].head(10)

    os.makedirs("models", exist_ok=True)
    model_path = "models/best_automl_winrate_model"
    save_model(final_model, model_path)

    return {
        "type": "AutoML (PyCaret)",
        "best_model": str(best_model),
        "leaderboard": viz_data.to_dict("records"),
        "visualization": {
            "x": viz_data["Model"].tolist(),
            "r2": viz_data["R2"].tolist(),
            "rmse": viz_data["RMSE"].tolist()
        },
        "model_path": model_path,
    }
