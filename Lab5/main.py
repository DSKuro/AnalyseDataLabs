import sys
import pandas as pd
import numpy as np
import joblib
import umap
from scipy.cluster.hierarchy import dendrogram, linkage

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QTextEdit, QMessageBox, QHBoxLayout, QComboBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, AgglomerativeClustering

class DimReductionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dimensionality Reduction App")
        self.resize(1400, 950)

        self.df = None
        self.X_scaled = None
        self.X_sample = None
        self.scaler = None
        self.kpca_models = {}
        self.model = None
        self.dataset = None
        self.projections = {}
        self.optimal_k = 3

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Penguins", "LoL"])
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["Исходные данные"])

        self.btn_load = QPushButton("Загрузить датасет")
        self.btn_eda = QPushButton("EDA")
        self.btn_kernel = QPushButton("Kernel PCA")
        self.btn_tsne = QPushButton("t-SNE")
        self.btn_umap = QPushButton("UMAP")
        self.btn_kmeans_opt = QPushButton("Оптимальное k (локоть/силуэт)")
        self.btn_kmeans_run = QPushButton("K-Means кластеризация")
        self.btn_hierarchical = QPushButton("Иерархическая кластеризация")
        self.btn_save_model = QPushButton("Сохранить модель")
        self.btn_load_model = QPushButton("Загрузить модель")

        top1 = QHBoxLayout()
        top1.addWidget(QLabel("Выберите датасет:"))
        top1.addWidget(self.dataset_combo)
        top1.addWidget(self.btn_load)
        top1.addWidget(self.btn_eda)
        top1.addWidget(self.btn_kernel)
        top1.addWidget(self.btn_tsne)
        top1.addWidget(self.btn_umap)

        top2 = QHBoxLayout()
        top2.addWidget(QLabel("Выберите проекцию:"))
        top2.addWidget(self.projection_combo)
        top2.addWidget(self.btn_kmeans_opt)
        top2.addWidget(self.btn_kmeans_run)
        top2.addWidget(self.btn_hierarchical)
        top2.addWidget(self.btn_save_model)
        top2.addWidget(self.btn_load_model)

        layout = QVBoxLayout()
        layout.addLayout(top1)
        layout.addLayout(top2)
        layout.addWidget(self.output)

        self.figure = plt.Figure(figsize=(12,6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.btn_load.clicked.connect(self.load_dataset)
        self.btn_eda.clicked.connect(self.eda)
        self.btn_kernel.clicked.connect(self.run_kernel_pca)
        self.btn_tsne.clicked.connect(self.run_tsne)
        self.btn_umap.clicked.connect(self.run_umap)
        self.btn_kmeans_opt.clicked.connect(self.find_optimal_k)
        self.btn_kmeans_run.clicked.connect(self.run_kmeans)
        self.btn_hierarchical.clicked.connect(self.run_hierarchical)
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_load_model.clicked.connect(self.load_model)

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
            self.output.append("Данные нормализованы и выбросы обработаны")

            if self.dataset == "LoL" and self.X_scaled.shape[0] > 5000:
                np.random.seed(42)
                idx = np.random.choice(self.X_scaled.shape[0], 5000, replace=False)
                self.X_sample = self.X_scaled[idx]
                self.output.append("Используется подвыборка 5000 строк для ускорения алгоритмов")
            else:
                self.X_sample = self.X_scaled

            self.projections = {"Исходные данные": self.X_sample}
            self.projection_combo.clear()
            self.projection_combo.addItems(list(self.projections.keys()))

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", str(e))

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

    def compute_metrics(self, X_proj, method_name):
        k = self.optimal_k
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_proj)
        sil_score = silhouette_score(X_proj, labels)
        self.output.append(f"\nМетрики для {method_name}:\nSilhouette Score: {sil_score:.3f}")
        return sil_score

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
            self.projections[f"KernelPCA_{kernel}"] = X_kpca
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

        self.projection_combo.clear()
        self.projection_combo.addItems(list(self.projections.keys()))

        self.canvas.draw()
        self.output.append("\nСравнение Silhouette Score для Kernel PCA:\n" + metrics_summary)

    def run_tsne(self):
        if self.X_sample is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return
        tsne = TSNE(n_components=2, random_state=42, method='barnes_hut', perplexity=30)
        X_tsne = tsne.fit_transform(self.X_sample)
        self.projections["t-SNE"] = X_tsne
        joblib.dump(tsne, "tsne.pkl")

        self.projection_combo.clear()
        self.projection_combo.addItems(list(self.projections.keys()))

        fig = self.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(X_tsne[:,0], X_tsne[:,1], alpha=0.6)
        ax.set_title("t-SNE")
        self.canvas.draw()

        sil_score = self.compute_metrics(X_tsne, "t-SNE")
        self.output.append(f"t-SNE выполнен. Silhouette Score={sil_score:.3f}")

    def run_umap(self):
        if self.X_sample is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(self.X_sample)
        self.projections["UMAP"] = X_umap
        joblib.dump(reducer, "umap.pkl")

        self.projection_combo.clear()
        self.projection_combo.addItems(list(self.projections.keys()))

        fig = self.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(X_umap[:,0], X_umap[:,1], alpha=0.6)
        ax.set_title("UMAP")
        self.canvas.draw()

        sil_score = self.compute_metrics(X_umap, "UMAP")
        self.output.append(f"UMAP выполнен. Silhouette Score={sil_score:.3f}")

    def find_optimal_k(self):
        if self.X_sample is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return

        distortions = []
        silhouettes = []
        K_range = range(2, 11)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.X_sample)
            distortions.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.X_sample, labels))

        self.optimal_k = K_range[np.argmax(silhouettes)]
        self.output.append(f"Оптимальное k по силуэту: {self.optimal_k}")

        fig = self.figure
        fig.clear()

        ax1 = fig.add_subplot(211)
        ax1.plot(K_range, distortions, 'bx-')
        ax1.set_xlabel('k')
        ax1.set_ylabel('Inertia (локоть)')
        ax1.set_title("Метод локтя")

        ax2 = fig.add_subplot(212)
        ax2.bar(K_range, silhouettes, color='orange')
        ax2.set_xlabel('k')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title("Метод силуэта")

        fig.tight_layout()
        self.canvas.draw()
        self.output.append("Определение оптимального числа кластеров завершено")

    def run_kmeans(self):
        if self.X_sample is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return

        projection = self.projection_combo.currentText()
        if projection not in self.projections:
            QMessageBox.warning(self, "Ошибка", f"Проекция {projection} не найдена")
            return

        X_input = self.projections[projection]
        k = self.optimal_k
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_input)
        self.model = kmeans
        joblib.dump(kmeans, "kmeans_model.pkl")

        sil_score = silhouette_score(X_input, labels)
        ch_score = calinski_harabasz_score(X_input, labels)
        db_score = davies_bouldin_score(X_input, labels)

        fig = self.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(X_input[:,0], X_input[:,1], c=labels, cmap='viridis', alpha=0.6)
        ax.set_title(f"K-Means Clustering (k={k}) on {projection}")
        self.canvas.draw()

        self.output.append(
            f"K-Means выполнен на {projection}.\n"
            f"Метрики:\n"
            f"Silhouette Score: {sil_score:.3f}\n"
            f"Calinski-Harabasz Index: {ch_score:.3f}\n"
            f"Davies-Bouldin Index: {db_score:.3f}"
        )

    def run_hierarchical(self):
        if self.X_sample is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return

        projection = self.projection_combo.currentText()
        if projection not in self.projections:
            QMessageBox.warning(self, "Ошибка", f"Проекция {projection} не найдена")
            return

        X_plot = self.projections[projection]
        #if X_plot.shape[0] > 200:

        #    np.random.seed(42)
         #   idx = np.random.choice(X_plot.shape[0], 200, replace=False)
          #  X_plot = X_plot[idx]

        linked = linkage(X_plot, method='ward')
        joblib.dump(linked, "hierarchical_model.pkl")

        clusterer = AgglomerativeClustering(n_clusters=self.optimal_k, linkage='ward')
        labels = clusterer.fit_predict(X_plot)

        sil_score = silhouette_score(X_plot, labels)
        ch_score = calinski_harabasz_score(X_plot, labels)
        db_score = davies_bouldin_score(X_plot, labels)

        fig = self.figure
        fig.clear()
        ax = fig.add_subplot(111)
        dendrogram(linked, truncate_mode='level', p=5, ax=ax)
        ax.set_title(f"Hierarchical Clustering Dendrogram ({projection})")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Distance")
        self.canvas.draw()

        self.output.append(
            f"Иерархическая кластеризация выполнена на {X_plot.shape[0]} точках ({projection}).\n"
            f"Метрики:\n"
            f"Silhouette Score: {sil_score:.3f}\n"
            f"Calinski-Harabasz Index: {ch_score:.3f}\n"
            f"Davies-Bouldin Index: {db_score:.3f}"
        )

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