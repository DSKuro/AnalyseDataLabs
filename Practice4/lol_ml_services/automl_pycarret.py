import os
import pandas as pd
from pycaret.regression import setup, compare_models, pull, finalize_model, save_model

def train_automl(df: pd.DataFrame):
    try:
        features = ["avg_kills", "avg_deaths", "avg_assists", "avg_gold"]
        target = "avg_damage"

        for col in features + [target]:
            if col not in df.columns:
                df[col] = 0

        X = df[features].copy()
        y = df[target].copy()

        df_model = pd.concat([X, y], axis=1)

        exp = setup(
            data=df_model,
            target=target,
            numeric_features=features,
            session_id=42,
            normalize=False,
            fold=3,
            verbose=False,
            feature_selection=False,
            remove_multicollinearity=False,
            polynomial_features=False,
            log_experiment=False
        )

        best_model = compare_models(include=["rf", "et", "lightgbm"])
        results = pull()
        final_model = finalize_model(best_model)

        os.makedirs("models", exist_ok=True)
        model_path = "models/best_automl_damage_model.pkl"
        save_model(final_model, model_path)

        r2_col = "R2 Score" if "R2 Score" in results.columns else "R2"
        viz_data = results[["Model", r2_col, "RMSE", "MAE"]]

        return {
            "type": "AutoML (PyCaret)",
            "best_model": str(best_model),
            "leaderboard": viz_data.to_dict("records"),
            "visualization": {
                "models": viz_data["Model"].tolist(),
                "r2": viz_data[r2_col].tolist(),
                "rmse": viz_data["RMSE"].tolist()
            },
            "model_path": model_path
        }

    except Exception as e:
        return {"error": f"Ошибка при обучении AutoML: {str(e)}"}