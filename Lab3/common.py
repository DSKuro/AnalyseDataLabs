import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings("ignore")

def load_data():
    data = datasets.load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

def remove_outliers_iqr(X, y):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    return X[mask], y[mask]

def split_and_scale(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def get_models():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogReg": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="linear", probability=True)
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n{name}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("F1:", f1_score(y_test, y_pred))
        print("ROC_AUC:", roc_auc_score(y_test, y_proba))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

        roc_data[name] = roc_curve(y_test, y_proba)

    return roc_data


def plot_roc_curves(roc_data):
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, _) in roc_data.items():
        plt.plot(fpr, tpr, label=name)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые моделей")
    plt.legend()
    plt.grid()
    plt.show()

def plot_decision_boundaries(X, y, models,
                             features=('mean radius', 'mean texture')):
    X_vis = X[list(features)]
    X_vis_scaled = StandardScaler().fit_transform(X_vis)

    xx, yy = np.meshgrid(
        np.arange(X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1, 0.02),
        np.arange(X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1, 0.02)
    )

    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#0000FF"])

    plt.figure(figsize=(18, 5))

    for i, (name, model) in enumerate(models.items(), 1):
        model.fit(X_vis_scaled, y)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        plt.subplot(1, 3, i)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
        plt.scatter(X_vis_scaled[:, 0], X_vis_scaled[:, 1],
                    c=y, cmap=cmap_bold, edgecolor='k', s=20)
        plt.title(f"Decision Boundary: {name}")
        plt.xlabel(features[0])
        plt.ylabel(features[1])

    plt.tight_layout()
    plt.show()


def main():
    X, y = load_data()
    X, y = remove_outliers_iqr(X, y)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    models = get_models()

    roc_data = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    plot_roc_curves(roc_data)
    plot_decision_boundaries(X, y, models)


if __name__ == "__main__":
    main()
