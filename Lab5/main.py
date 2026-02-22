import sys
import pandas as pd
import numpy as np
import joblib
import umap

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QTextEdit, QMessageBox, QHBoxLayout, QComboBox
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
        self.resize(1300, 900)

        self.df = None
        self.X_scaled = None
        self.X_sample = None
        self.scaler = None
        self.kpca_models = {}
        self.model = None
        self.dataset = None

        # ====== Widgets ======
        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Penguins", "LoL"])
        self.btn_load = QPushButton("Загрузить датасет")
        self.btn_eda = QPushButton("EDA")
        self.btn_kernel = QPushButton("Kernel PCA")
        self.btn_tsne = QPushButton("t-SNE")
        self.btn_umap = QPushButton("UMAP")
        self.btn_save_model = QPushButton("Сохранить модель")
        self.btn_load_model = QPushButton("Загрузить модель")

        top = QHBoxLayout()
        top.addWidget(QLabel("Выберите датасет:"))
        top.addWidget(self.dataset_combo)
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

        self.figure = plt.Figure(figsize=(12,6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        # ====== Signals ======
        self.btn_load.clicked.connect(self.load_dataset)
        self.btn_eda.clicked.connect(self.eda)
        self.btn_kernel.clicked.connect(self.run_kernel_pca)
        self.btn_tsne.clicked.connect(self.run_tsne)
        self.btn_umap.clicked.connect(self.run_umap)
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_load_model.clicked.connect(self.load_model)

    # ===========================
    # Загрузка данных
    # ===========================
    def load_dataset(self):
        self.dataset = self.dataset_combo.currentText()
        try:
            if self.dataset == "Penguins":
                self.df = pd.read_csv("../data/penguins.csv").dropna()
                self.output.setText(f"Penguins Dataset загружен. Размер: {self.df.shape}")
            elif self.dataset == "LoL":
                champs = pd.read_csv("../data/champs.csv")
                matches = pd.read_csv("../data/matches.csv")
                participants = pd.read_csv("../data/participants.csv")
                stats = pd.read_csv("../data/stats1.csv")

                df = participants.merge(stats, on="id")
                df = df.merge(matches[["id","duration"]], left_on="matchid", right_on="id", suffixes=("","_match"))
                df = df.merge(champs, left_on="championid", right_on="id", suffixes=("","_champ"))

                self.df = df.dropna()
                self.output.setText(f"LoL Dataset загружен. Размер: {self.df.shape}")
            else:
                QMessageBox.warning(self, "Ошибка", "Выберите датасет")
                return

            # числовые признаки
            num_cols = self.df.select_dtypes(include=np.number).columns
            X = self.df[num_cols]

            # удаление выбросов IQR
            for col in num_cols:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                X = X[(X[col] >= Q1 - 1.5*IQR) & (X[col] <= Q3 + 1.5*IQR)]

            self.df = self.df.loc[X.index]

            # нормализация
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(X)
            joblib.dump(self.scaler, "scaler.pkl")
            self.output.append("Данные нормализованы и выбросы обработаны")

            # подвыборка для LoL, чтобы ускорить визуализацию
            if self.dataset == "LoL" and self.X_scaled.shape[0] > 5000:
                np.random.seed(42)
                idx = np.random.choice(self.X_scaled.shape[0], 5000, replace=False)
                self.X_sample = self.X_scaled[idx]
                self.output.append("Используется подвыборка 5000 строк для ускорения алгоритмов")
            else:
                self.X_sample = self.X_scaled

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    # ===========================
    # EDA
    # ===========================
    def eda(self):
        if self.df is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return

        n_rows, n_cols = self.df.shape
        mem = self.df.memory_usage(deep=True).sum() / 1024**2

        num_cols = self.df.select_dtypes(np.number)
        num_stats = num_cols.agg({
            col: ['min','median','mean','max',
                  lambda x: x.quantile(0.25),
                  lambda x: x.quantile(0.75)]
            for col in num_cols.columns
        })
        num_stats.index = ['min','median','mean','max','25%','75%']

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

    # ===========================
    # Метрики
    # ===========================
    def compute_metrics(self, X_proj, method_name):
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X_proj)
        sil_score = silhouette_score(X_proj, labels)
        self.output.append(f"\nМетрики для {method_name}:\nSilhouette Score: {sil_score:.3f}")
        return sil_score

    # ===========================
    # Kernel PCA
    # ===========================
    def run_kernel_pca(self):
        if self.X_sample is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return

        kernels = ['linear','poly','rbf','sigmoid','cosine']
        fig = self.figure
        fig.clear()

        metrics_summary = ""
        for i, kernel in enumerate(kernels):
            kpca = KernelPCA(n_components=2, kernel=kernel)
            X_kpca = kpca.fit_transform(self.X_sample)
            self.kpca_models[kernel] = kpca
            joblib.dump(kpca, f"kpca_{kernel}.pkl")

            ax = fig.add_subplot(1,len(kernels),i+1)
            ax.scatter(X_kpca[:,0], X_kpca[:,1], alpha=0.6)
            ax.set_title(kernel)

            sil_score = self.compute_metrics(X_kpca, f"Kernel PCA ({kernel})")
            metrics_summary += f"{kernel}: Silhouette={sil_score:.3f}\n"

            if kernel=="linear":
                pca = PCA()
                X_pca = pca.fit_transform(self.X_sample)
                explained = pca.explained_variance_ratio_
                cumulative = np.cumsum(explained)
                lost_variance = 1 - cumulative[1]
                self.output.append(f"Linear Kernel PCA: lost_variance={lost_variance:.3f}")

        self.canvas.draw()
        self.output.append("\nСравнение Silhouette Score для Kernel PCA:\n" + metrics_summary)

    # ===========================
    # t-SNE
    # ===========================
    def run_tsne(self):
        if self.X_sample is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return
        tsne = TSNE(n_components=2, random_state=42, method='barnes_hut', perplexity=30)
        X_tsne = tsne.fit_transform(self.X_sample)
        joblib.dump(tsne, "tsne.pkl")

        fig = self.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(X_tsne[:,0], X_tsne[:,1], alpha=0.6)
        ax.set_title("t-SNE")
        self.canvas.draw()

        sil_score = self.compute_metrics(X_tsne, "t-SNE")
        self.output.append(f"t-SNE выполнен. Silhouette Score={sil_score:.3f}")

    # ===========================
    # UMAP
    # ===========================
    def run_umap(self):
        if self.X_sample is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(self.X_sample)
        joblib.dump(reducer, "umap.pkl")

        fig = self.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(X_umap[:,0], X_umap[:,1], alpha=0.6)
        ax.set_title("UMAP")
        self.canvas.draw()

        sil_score = self.compute_metrics(X_umap, "UMAP")
        self.output.append(f"UMAP выполнен. Silhouette Score={sil_score:.3f}")

    # ===========================
    # Сохранение и загрузка модели
    # ===========================
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