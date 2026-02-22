import sys
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import umap

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QComboBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE


class PenguinApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dimensionality Reduction App")
        self.resize(1100, 900)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Загрузите CSV файл")
        self.layout.addWidget(self.label)

        self.load_btn = QPushButton("Загрузить CSV")
        self.load_btn.clicked.connect(self.load_csv)
        self.layout.addWidget(self.load_btn)

        self.eda_btn = QPushButton("Выполнить EDA")
        self.eda_btn.clicked.connect(self.perform_eda)
        self.layout.addWidget(self.eda_btn)

        self.layout.addWidget(QLabel("Метод снижения размерности:"))

        self.dim_combo = QComboBox()
        self.dim_combo.addItems(["Kernel PCA", "t-SNE", "UMAP"])
        self.layout.addWidget(self.dim_combo)

        self.dim_btn = QPushButton("Применить метод")
        self.dim_btn.clicked.connect(self.apply_dim_reduction)
        self.layout.addWidget(self.dim_btn)

        self.load_model_btn = QPushButton("Загрузить модель")
        self.load_model_btn.clicked.connect(self.load_model)
        self.layout.addWidget(self.load_model_btn)

        self.canvas = FigureCanvas(plt.Figure(figsize=(8, 6)))
        self.layout.addWidget(self.canvas)

        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

        self.df = None
        self.X_scaled = None
        self.scaler = None
        self.kpca_models = {}

    # ===========================
    # Загрузка данных
    # ===========================
    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать CSV", "", "CSV Files (*.csv)"
        )

        if not file_path:
            return

        self.df = pd.read_csv(file_path).dropna()
        self.label.setText(f"Файл загружен: {file_path}")

        num_cols = self.df.select_dtypes(include=np.number).columns
        X = self.df[num_cols]

        # ===== EDA базовые характеристики =====
        print("\n===== DESCRIBE =====")
        print(X.describe())

        print("\n===== CORRELATION =====")
        print(X.corr())

        # ===== Удаление выбросов (IQR) =====
        for col in num_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            X = X[(X[col] >= Q1 - 1.5 * IQR) &
                  (X[col] <= Q3 + 1.5 * IQR)]

        self.df = self.df.loc[X.index]

        # ===== Нормализация =====
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        joblib.dump(self.scaler, "scaler.pkl")

        self.status_label.setText("Данные загружены и обработаны")

    # ===========================
    # EDA
    # ===========================
    def perform_eda(self):
        if self.df is None:
            self.status_label.setText("Сначала загрузите данные")
            return

        num_cols = self.df.select_dtypes(include=np.number).columns

        fig = self.canvas.figure
        fig.clear()

        # Корреляционная матрица
        ax1 = fig.add_subplot(221)
        sns.heatmap(self.df[num_cols].corr(),
                    annot=True, cmap="coolwarm", ax=ax1)
        ax1.set_title("Correlation Matrix")

        # Boxplot
        ax2 = fig.add_subplot(222)
        self.df[num_cols].boxplot(ax=ax2)
        ax2.set_title("Boxplot")

        # Гистограмма первого признака
        ax3 = fig.add_subplot(223)
        self.df[num_cols[0]].hist(ax=ax3)
        ax3.set_title(f"Distribution: {num_cols[0]}")

        self.canvas.draw()
        self.status_label.setText("EDA выполнен")

    # ===========================
    # Снижение размерности
    # ===========================
    def apply_dim_reduction(self):
        if self.X_scaled is None:
            self.status_label.setText("Сначала загрузите данные")
            return

        method = self.dim_combo.currentText()
        fig = self.canvas.figure
        fig.clear()

        if method == "Kernel PCA":
            kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']

            for i, kernel in enumerate(kernels):
                kpca = KernelPCA(n_components=2, kernel=kernel)
                X_kpca = kpca.fit_transform(self.X_scaled)
                self.kpca_models[kernel] = kpca

                ax = fig.add_subplot(231 + i)
                ax.scatter(X_kpca[:, 0], X_kpca[:, 1], alpha=0.6)
                ax.set_title(kernel)

                joblib.dump(kpca, f"kpca_{kernel}.pkl")

                # Linear → считаем дисперсию
                if kernel == "linear":
                    pca = PCA()
                    X_pca = pca.fit_transform(self.X_scaled)

                    explained = pca.explained_variance_ratio_
                    cumulative = np.cumsum(explained)
                    lost_variance = 1 - cumulative[1]

                    print("\nExplained variance:")
                    print(explained[:5])
                    print("Lost variance (2 components):",
                          round(lost_variance, 4))

                    ax_var = fig.add_subplot(236)
                    ax_var.plot(cumulative, marker='o')
                    ax_var.set_title("Cumulative Variance")

            self.status_label.setText(
                "Kernel PCA выполнен для всех ядер"
            )

        elif method == "t-SNE":
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(self.X_scaled)

            ax = fig.add_subplot(111)
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1])
            ax.set_title("t-SNE")

            joblib.dump(tsne, "tsne.pkl")
            self.status_label.setText("t-SNE выполнен")

        elif method == "UMAP":
            reducer = umap.UMAP(n_components=2)
            X_umap = reducer.fit_transform(self.X_scaled)

            ax = fig.add_subplot(111)
            ax.scatter(X_umap[:, 0], X_umap[:, 1])
            ax.set_title("UMAP")

            joblib.dump(reducer, "umap.pkl")
            self.status_label.setText("UMAP выполнен")

        self.canvas.draw()

    # ===========================
    # Загрузка модели
    # ===========================
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить модель", "", "PKL Files (*.pkl)"
        )

        if not file_path:
            return

        model = joblib.load(file_path)
        self.status_label.setText(f"Модель загружена: {file_path}")
        print("Загружена модель:", model)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PenguinApp()
    window.show()
    sys.exit(app.exec())