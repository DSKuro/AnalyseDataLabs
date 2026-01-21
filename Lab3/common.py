# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

data = datasets.load_breast_cancer()
X = pd.DataFrame(data["data"], columns=data["feature_names"])
y = pd.Series(data["target"], name="target")

Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
X_out = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
y_out = y[X_out.index]

X_train, X_test, y_train, y_test = train_test_split(
    X_out, y_out, test_size=0.2, random_state=42, stratify=y_out
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_scaled, y_train)

models = {'KNN': knn, 'LogReg': logreg, 'SVM': svm}
roc_data = {}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

    print(f"\n{name}:\nAccuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, ROC_AUC={roc}")
    print("Confusion matrix:\n", cm)

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr)

plt.figure(figsize=(8, 6))
for name, (fpr, tpr) in roc_data.items():
    plt.plot(fpr, tpr,
             label=f'{name} (AUC = {roc_auc_score(y_test, models[name].predict_proba(X_test_scaled)[:, 1]):.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые моделей')
plt.legend()
plt.grid(True)
plt.show()
