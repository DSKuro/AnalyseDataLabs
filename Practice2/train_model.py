import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

df = pd.read_csv("../data/stats1.csv")

features = ["kills", "deaths", "assists", "goldearned"]
df = df.dropna(subset=features + ["win"])

X = df[features]
y = df["win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_prob),
    "r2_pseudo": model.score(X_test, y_test)
}

with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "metrics": metrics}, f)

print("Model and metrics saved")
