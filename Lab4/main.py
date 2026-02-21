import sys
import pandas as pd
import numpy as np
import joblib

from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QTextEdit, QMessageBox, QLabel, QHBoxLayout
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class MLApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ML анализ: Diabetes + LoL")
        self.resize(1200, 850)

        self.dataset = None
        self.df = None
        self.model = None

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.btn_diabetes = QPushButton("Выбрать Diabetes")
        self.btn_lol = QPushButton("Выбрать LoL")

        self.btn_load = QPushButton("Загрузить датасет")
        self.btn_eda = QPushButton("EDA")
        self.btn_prepare = QPushButton("3️Подготовка данных")

        self.btn_lr = QPushButton("Linear Regression")
        self.btn_lasso = QPushButton("LASSO")
        self.btn_knn = QPushButton("KNN")

        self.btn_save = QPushButton("Сохранить модель")
        self.btn_load_model = QPushButton("Загрузить модель")

        top = QHBoxLayout()
        top.addWidget(self.btn_diabetes)
        top.addWidget(self.btn_lol)

        self.btn_hypotheses = QPushButton("Проверка гипотез")

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_eda)
        layout.addWidget(self.btn_prepare)
        layout.addWidget(self.btn_hypotheses)
        layout.addWidget(self.btn_lr)
        layout.addWidget(self.btn_lasso)
        layout.addWidget(self.btn_knn)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_load_model)
        layout.addWidget(self.output)

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.btn_hypotheses.clicked.connect(self.check_hypotheses)

        self.btn_diabetes.clicked.connect(self.set_diabetes)
        self.btn_lol.clicked.connect(self.set_lol)
        self.btn_load.clicked.connect(self.load_data)
        self.btn_eda.clicked.connect(self.eda)
        self.btn_prepare.clicked.connect(self.prepare)
        self.btn_lr.clicked.connect(self.train_lr)
        self.btn_lasso.clicked.connect(self.train_lasso)
        self.btn_knn.clicked.connect(self.train_knn)
        self.btn_save.clicked.connect(self.save_model)
        self.btn_load_model.clicked.connect(self.load_model)

    def set_diabetes(self):
        self.dataset = "diabetes"
        self.output.setText("Выбран датасет: Diabetes")

    def set_lol(self):
        self.dataset = "lol"
        self.output.setText("Выбран датасет: League of Legends")

    def load_data(self):
        if self.dataset == "diabetes":
            self.df = pd.read_csv("../data/diabetes.csv")

        elif self.dataset == "lol":
            champs = pd.read_csv("../data/champs.csv")
            matches = pd.read_csv("../data/matches.csv")
            participants = pd.read_csv("../data/participants.csv")
            stats = pd.read_csv("../data/stats1.csv")

            self.df = (
                participants
                .merge(stats, on="id")
                .merge(matches[["id", "duration"]], left_on="matchid", right_on="id")
                .merge(champs, left_on="championid", right_on="id")
            )

        else:
            QMessageBox.warning(self, "Ошибка", "Выберите датасет")
            return

        self.output.setText(f"Данные загружены\nРазмер: {self.df.shape}")

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

        num_stats.index = ['min', 'median', 'mean', 'max', '25%', '75%']

        cat_cols = self.df.select_dtypes("object").columns
        cat_info = ""
        for col in cat_cols:
            mode_val = self.df[col].mode()[0]
            mode_count = (self.df[col] == mode_val).sum()
            cat_info += f"{col}: мода={mode_val}, встречается={mode_count} раз\n"

        self.output.setText(
            f"Обзор данных (EDA)\n"
            f"Строк: {n_rows}\n"
            f"Столбцов: {n_cols}\n"
            f"Память: {mem:.2f} MB\n\n"
            f"Числовые переменные:\n{num_stats}\n\n"
            f"Категориальные переменные:\n{cat_info}"
        )

    def check_hypotheses(self):
        if self.df is None:
            QMessageBox.warning(self, "Ошибка", "Данные не загружены")
            return

        result_text = "Проверка гипотез:\n\n"

        if self.dataset == "diabetes":
            target = "Outcome"
            features = self.df.columns.drop(target)
            corr = self.df[features].corrwith(self.df[target]).sort_values(ascending=False)
            result_text += "Корреляция признаков с Outcome:\n"
            result_text += corr.to_string(float_format="{:.3f}".format)
            result_text += "\n\nВыводы:\n"
            result_text += (
                "Наибольшая положительная корреляция с диабетом у Glucose и BMI.\n"
                "Возраст оказывает умеренное влияние.\n"
                "Второстепенные признаки имеют минимальное влияние.\n"
                "Наследственная предрасположенность слабо влияет.\n"
                "Количество беременностей показывает умеренную корреляцию.\n"
            )

        elif self.dataset == "lol":
            target = "totdmgtochamp"
            features = ["kills", "deaths", "assists", "goldearned", "duration"]
            corr = self.df[features + [target]].corr()[target].drop(target).sort_values(ascending=False)
            result_text += "Корреляция ключевых показателей с totdmgtochamp:\n"
            result_text += corr.to_string(float_format="{:.3f}".format)
            result_text += "\n\nВыводы:\n"
            result_text += (
                "Наибольшая положительная корреляция с общим уроном по чемпионам у kills и goldearned.\n"
                "assists показывает умеренное влияние.\n"
                "deaths отрицательно влияет.\n"
                "Продолжительность матча имеет слабую положительную корреляцию.\n"
            )

        else:
            QMessageBox.warning(self, "Ошибка", "Выберите датасет")
            return

        self.output.setText(result_text)

    def prepare(self):
        df = self.df.copy()

        df.fillna(df.median(numeric_only=True), inplace=True)

        for col in df.select_dtypes(np.number):
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        if self.dataset == "lol":
            df = pd.get_dummies(df, columns=["role", "position"], drop_first=True)
            features = ["kills", "deaths", "assists", "goldearned", "duration"]
            self.y = df["totdmgtochamp"]

        else:
            features = df.columns.drop("Glucose")
            self.y = df["Glucose"]

        self.X = df[features]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        self.output.setText(
            "Данные подготовлены\n"
            "Гипотезы:\n"
            "1) Чем больше ключевых ресурсов, тем выше целевая переменная\n"
            "2) Основные боевые показатели влияют сильнее вспомогательных"
        )

    def metrics(self, y_true, y_pred):
        return (
            f"MAE: {mean_absolute_error(y_true, y_pred):.3f}\n"
            f"MSE: {mean_squared_error(y_true, y_pred):.3f}\n"
            f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}\n"
            f"MAPE: {np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100:.2f}%\n"
            f"R2: {r2_score(y_true, y_pred):.3f}"
        )

    def plot_predictions(self, y_true, y_pred, title="Реальные vs Предсказанные"):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(y_true, y_pred, color='blue', alpha=0.6, edgecolors='k')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
        ax.set_xlabel("Реальные значения")
        ax.set_ylabel("Предсказанные значения")
        ax.set_title(title)
        self.canvas.draw()

    def train_lr(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        preds = self.model.predict(self.X_test)
        self.output.setText("Linear Regression\n" + self.metrics(self.y_test, preds))
        self.plot_predictions(self.y_test, preds, title="Linear Regression")

    def train_lasso(self):
        self.model = Lasso(alpha=0.01)
        self.model.fit(self.X_train, self.y_train)
        preds = self.model.predict(self.X_test)
        self.output.setText("LASSO\n" + self.metrics(self.y_test, preds))
        self.plot_predictions(self.y_test, preds, title="LASSO Regression")

    def train_knn(self):
        self.model = KNeighborsRegressor(n_neighbors=7)
        self.model.fit(self.X_train, self.y_train)
        preds = self.model.predict(self.X_test)
        self.output.setText("KNN\n" + self.metrics(self.y_test, preds))
        self.plot_predictions(self.y_test, preds, title="KNN Regression")

    def save_model(self):
        joblib.dump(self.model, "model.pkl")
        joblib.dump(self.scaler, "scaler.pkl")
        QMessageBox.information(self, "OK", "Модель сохранена")

    def load_model(self):
        self.model = joblib.load("model.pkl")
        self.scaler = joblib.load("scaler.pkl")
        QMessageBox.information(self, "OK", "Модель загружена")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLApp()
    window.show()
    sys.exit(app.exec())
