import sys
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import joblib

class PenguinApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Penguin Analysis & Clustering")
        self.resize(1000, 900)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Загрузите CSV")
        self.layout.addWidget(self.label)

        self.load_btn = QPushButton("Загрузить CSV")
        self.load_btn.clicked.connect(self.load_csv)
        self.layout.addWidget(self.load_btn)

        # EDA кнопка
        self.eda_btn = QPushButton("Выполнить EDA")
        self.eda_btn.clicked.connect(self.perform_eda)
        self.layout.addWidget(self.eda_btn)

        # Метод снижения размерности
        self.dim_combo = QComboBox()
        self.dim_combo.addItems(["Kernel PCA", "t-SNE"])
        self.layout.addWidget(QLabel("Метод снижения размерности:"))
        self.layout.addWidget(self.dim_combo)

        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
        self.layout.addWidget(QLabel("Ядро для Kernel PCA:"))
        self.layout.addWidget(self.kernel_combo)

        self.dim_btn = QPushButton("Применить снижение размерности")
        self.dim_btn.clicked.connect(self.apply_dim_reduction)
        self.layout.addWidget(self.dim_btn)

        # Кластеризация
        self.cluster_btn = QPushButton("Определить оптимальное число кластеров и кластеризовать (K-Means)")
        self.cluster_btn.clicked.connect(self.kmeans_clustering)
        self.layout.addWidget(self.cluster_btn)

        self.hier_btn = QPushButton("Иерархическая кластеризация")
        self.hier_btn.clicked.connect(self.hierarchical_clustering)
        self.layout.addWidget(self.hier_btn)

        # Canvas для графиков
        self.canvas = FigureCanvas(plt.Figure(figsize=(8,6)))
        self.layout.addWidget(self.canvas)

        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

        self.df = None
        self.X_scaled = None
        self.dim_data = None
        self.kpca_models = {}
        self.kmeans_model = None

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV", "", "CSV Files (*.csv)")
        if file_path:
            self.df = pd.read_csv(file_path).dropna()
            self.label.setText(f"Файл загружен: {file_path}, {len(self.df)} строк")
            num_features = self.df.select_dtypes(include=np.number).columns
            X = self.df[num_features]
            # Удаление выбросов через IQR
            for col in num_features:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                X = X[(X[col] >= Q1 - 1.5*IQR) & (X[col] <= Q3 + 1.5*IQR)]
            self.X_scaled = StandardScaler().fit_transform(X)
            self.df = self.df.iloc[X.index]
        else:
            self.label.setText("Файл не выбран")

    def perform_eda(self):
        if self.df is None:
            self.status_label.setText("Сначала загрузите CSV")
            return
        num_features = self.df.select_dtypes(include=np.number).columns
        for col in num_features:
            plt.figure()
            self.df[col].hist(bins=20)
            plt.title(f"Распределение {col}")
            plt.show()
        self.status_label.setText("EDA выполнено")

    def apply_dim_reduction(self):
        if self.X_scaled is None:
            self.status_label.setText("Сначала загрузите CSV")
            return

        method = self.dim_combo.currentText()
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        if method == "Kernel PCA":
            kernel = self.kernel_combo.currentText()
            kpca = KernelPCA(n_components=2, kernel=kernel)
            self.dim_data = kpca.fit_transform(self.X_scaled)
            self.kpca_models[kernel] = kpca

            ax.scatter(self.dim_data[:,0], self.dim_data[:,1], alpha=0.6)
            ax.set_title(f"Kernel PCA ({kernel})")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

            if kernel == "linear":
                # дисперсия и lost_variance
                pca_linear = PCA()
                X_pca_linear = pca_linear.fit_transform(self.X_scaled)
                explained_var = pca_linear.explained_variance_ratio_
                cumulative = np.cumsum(explained_var)
                lost_variance = 1 - cumulative[1]  # потеря для 2 компонент
                self.status_label.setText(f"Linear Kernel PCA: lost_variance={lost_variance:.3f}")

        elif method == "t-SNE":
            tsne = TSNE(n_components=2, random_state=42)
            self.dim_data = tsne.fit_transform(self.X_scaled)
            ax.scatter(self.dim_data[:,0], self.dim_data[:,1], alpha=0.6)
            ax.set_title("t-SNE")
            ax.set_xlabel("Dim1")
            ax.set_ylabel("Dim2")
            self.status_label.setText("t-SNE применён")

        self.canvas.draw()
        joblib.dump(self.kpca_models.get(self.kernel_combo.currentText(), None), "kpca_model.pkl")
        joblib.dump(self.X_scaled, "scaler.pkl")

    def kmeans_clustering(self):
        if self.dim_data is None:
            self.status_label.setText("Сначала выполните снижение размерности")
            return

        # Определение оптимального числа кластеров методом локтя
        inertias = []
        sil_scores = []
        for k in range(2,7):
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(self.X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(self.X_scaled, labels))

        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(range(2,7), sil_scores, marker='o')
        ax.set_title("Силуэт для разных K")
        ax.set_xlabel("Количество кластеров")
        ax.set_ylabel("Silhouette Score")
        self.canvas.draw()

        # Фиксируем K=3 для penguins
        k_opt = 3
        kmeans = KMeans(n_clusters=k_opt, random_state=42)
        labels = kmeans.fit_predict(self.X_scaled)
        score = silhouette_score(self.X_scaled, labels)
        self.df['kmeans_cluster'] = labels
        self.kmeans_model = kmeans

        self.status_label.setText(f"K-Means выполнен: Silhouette Score={score:.3f}")
        joblib.dump(kmeans, "kmeans_model.pkl")

        # Визуализация кластеров
        fig2 = plt.figure(figsize=(6,4))
        plt.scatter(self.dim_data[:,0], self.dim_data[:,1], c=labels, cmap='Set1', s=80)
        plt.title(f"K-Means Clusters (K={k_opt})")
        plt.xlabel("Dim1")
        plt.ylabel("Dim2")
        plt.show()

    def hierarchical_clustering(self):
        if self.dim_data is None:
            self.status_label.setText("Сначала выполните снижение размерности")
            return

        Z = linkage(self.X_scaled, method='ward')
        labels = fcluster(Z, t=3, criterion='maxclust')
        score = silhouette_score(self.X_scaled, labels)
        self.df['hier_cluster'] = labels
        self.hier_model = Z
        joblib.dump(Z, "hierarchical_model.pkl")

        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        dendrogram(Z, truncate_mode='level', p=3, ax=ax)
        ax.set_title(f"Hierarchical Clustering (Silhouette={score:.3f})")
        self.canvas.draw()
        self.status_label.setText(f"Иерархическая кластеризация выполнена: Silhouette Score={score:.3f}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PenguinApp()
    window.show()
    sys.exit(app.exec())
