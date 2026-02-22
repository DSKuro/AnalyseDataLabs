import sys
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import umap

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QTextEdit, QMessageBox, QHBoxLayout
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

class DimReductionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dimensionality Reduction App")
        self.resize(1200, 900)

        self.df = None
        self.X_scaled = None
        self.scaler = None
        self.kpca_models = {}
        self.model = None

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.btn_load = QPushButton("Загрузить CSV")
        self.btn_eda = QPushButton("EDA")
        self.btn_kernel = QPushButton("Kernel PCA")
        self.btn_tsne = QPushButton("t-SNE")
        self.btn_umap = QPushButton("UMAP")
        self.btn_save_model = QPushButton("Сохранить модель")
        self.btn_load_model = QPushButton("Загрузить модель")

        top = QHBoxLayout()
        top.addWidget(self.btn_load)
        top.addWidget(self.btn_eda)
        top.addWidget(self.btn_kernel)
        top.addWidget(self.btn_tsne)
        top.addWidget(self.btn_umap)
        top.addWidget(self.btn_save_model)
        top.addWidget(self.btn_load_model)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.output)

        self.figure = plt.Figure(figsize=(8,6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.btn_load.clicked.connect(self.load_csv)
        self.btn_eda.clicked.connect(self.eda)
        self.btn_kernel.clicked.connect(self.run_kernel_pca)
        self.btn_tsne.clicked.connect(self.run_tsne)
        self.btn_umap.clicked.connect(self.run_umap)
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_load_model.clicked.connect(self.load_model)

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return

        self.df = pd.read_csv(file_path).dropna()
        self.output.setText(f"Файл загружен: {file_path}\nРазмер: {self.df.shape}")

        num_cols = self.df.select_dtypes(include=np.number).columns
        X = self.df[num_cols]

        for col in num_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            X = X[(X[col] >= Q1 - 1.5*IQR) & (X[col] <= Q3 + 1.5*IQR)]

        self.df = self.df.loc[X.index]

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, "scaler.pkl")
        self.output.append("Данные обработаны и нормализованы")

    def eda(self):
        if self.df is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return

        n_rows, n_cols = self.df.shape
        mem = self.df.memory_usage(deep=True).sum() / 1024 ** 2

        num_cols = self.df.select_dtypes(np.number)
        num_stats = num_cols.agg({
            col: ['min', 'median', 'mean', 'max',
                  lambda x: x.quantile(0.25),
                  lambda x: x.quantile(0.75)]
            for col in num_cols.columns
        })
        num_stats.index = ['min','median','mean','max','25%','75%']

        # категориальные
        cat_cols = self.df.select_dtypes("object").columns
        cat_info = ""
        for col in cat_cols:
            mode_val = self.df[col].mode()[0]
            mode_count = (self.df[col]==mode_val).sum()
            cat_info += f"{col}: мода={mode_val}, встречается={mode_count} раз\n"

        self.output.setText(
            f"EDA\nСтрок: {n_rows}\nСтолбцов: {n_cols}\nПамять: {mem:.2f} MB\n\n"
            f"Числовые переменные:\n{num_stats}\n\n"
            f"Категориальные переменные:\n{cat_info}"
        )

    def compute_metrics(self, X_proj, method_name):
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X_proj)
        sil_score = silhouette_score(X_proj, labels)

        self.output.append(f"\nМетрики для {method_name}:")
        self.output.append(f"Silhouette Score: {sil_score:.3f}")

        return sil_score

    def run_kernel_pca(self):
        if self.X_scaled is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return

        kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
        fig = self.figure
        fig.clear()

        metrics_summary = ""
        for i, kernel in enumerate(kernels):
            kpca = KernelPCA(n_components=2, kernel=kernel)
            X_kpca = kpca.fit_transform(self.X_scaled)
            self.kpca_models[kernel] = kpca
            joblib.dump(kpca, f"kpca_{kernel}.pkl")

            ax = fig.add_subplot(231+i)
            ax.scatter(X_kpca[:,0], X_kpca[:,1], alpha=0.6)
            ax.set_title(kernel)

            sil_score = self.compute_metrics(X_kpca, f"Kernel PCA ({kernel})")
            metrics_summary += f"{kernel}: Silhouette={sil_score:.3f}\n"

            if kernel=="linear":
                pca = PCA()
                X_pca = pca.fit_transform(self.X_scaled)
                explained = pca.explained_variance_ratio_
                cumulative = np.cumsum(explained)
                lost_variance = 1 - cumulative[1]
                self.output.append(f"Linear Kernel PCA: lost_variance={lost_variance:.3f}")
                ax_var = fig.add_subplot(236)
                ax_var.plot(cumulative, marker='o')
                ax_var.set_title("Cumulative Variance")

        self.canvas.draw()
        self.output.append("\nСравнение Silhouette Score для всех ядер Kernel PCA:\n" + metrics_summary)

    def run_tsne(self):
        if self.X_scaled is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(self.X_scaled)
        joblib.dump(tsne, "tsne.pkl")

        fig = self.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(X_tsne[:,0], X_tsne[:,1], alpha=0.6)
        ax.set_title("t-SNE")
        self.canvas.draw()

        sil_score = self.compute_metrics(X_tsne, "t-SNE")
        self.output.append(f"t-SNE выполнен. Silhouette Score={sil_score:.3f}")

    def run_umap(self):
        if self.X_scaled is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return
        reducer = umap.UMAP(n_components=2)
        X_umap = reducer.fit_transform(self.X_scaled)
        joblib.dump(reducer, "umap.pkl")

        fig = self.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(X_umap[:,0], X_umap[:,1], alpha=0.6)
        ax.set_title("UMAP")
        self.canvas.draw()

        sil_score = self.compute_metrics(X_umap, "UMAP")
        self.output.append(f"UMAP выполнен. Silhouette Score={sil_score:.3f}")

    def save_model(self):
        if self.model is None:
            QMessageBox.warning(self, "Ошибка", "Модель не обучена")
            return
        joblib.dump(self.model, "model.pkl")
        joblib.dump(self.scaler, "scaler.pkl")
        QMessageBox.information(self, "Сохранение", "Модель и scaler сохранены")

    def load_model(self):
        try:
            self.model = joblib.load("model.pkl")
            self.scaler = joblib.load("scaler.pkl")
            QMessageBox.information(self, "Загрузка", "Модель и scaler загружены")
        except:
            QMessageBox.warning(self, "Ошибка", "Файл модели не найден")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DimReductionApp()
    window.show()
    sys.exit(app.exec())