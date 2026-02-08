import sys
import pandas as pd
from urllib.parse import quote_plus

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QComboBox, QPushButton, QLineEdit, QFormLayout,
    QMessageBox, QTextEdit
)

from sqlalchemy import create_engine

import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DataVizApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("League of Legends Data Analyzer")
        self.setGeometry(100, 100, 700, 800)

        self.layout = QVBoxLayout(self)

        form = QFormLayout()
        self.host = QLineEdit("localhost")
        self.port = QLineEdit("5432")
        self.user = QLineEdit("postgres")
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.db = QLineEdit("lol_db")

        form.addRow("Host:", self.host)
        form.addRow("Port:", self.port)
        form.addRow("User:", self.user)
        form.addRow("Password:", self.password)
        form.addRow("Database:", self.db)
        self.layout.addLayout(form)

        self.connect_btn = QPushButton("Подключиться")
        self.connect_btn.clicked.connect(self.connect_db)
        self.layout.addWidget(self.connect_btn)

        self.layout.addWidget(QLabel("Просмотр таблиц matches / participants / teamstats"))
        self.table_selector = QComboBox()
        self.table_selector.addItems(["matches", "participants", "teamstats"])
        self.layout.addWidget(self.table_selector)

        self.preview_btn = QPushButton("Показать таблицу")
        self.preview_btn.clicked.connect(self.preview_table)
        self.preview_btn.setEnabled(False)
        self.layout.addWidget(self.preview_btn)

        self.table_view = QTextEdit()
        self.table_view.setReadOnly(True)
        self.layout.addWidget(self.table_view)

        self.layout.addWidget(QLabel("X (количественный):"))
        self.combo_x = QComboBox()
        self.layout.addWidget(self.combo_x)

        self.layout.addWidget(QLabel("Y (количественный):"))
        self.combo_y = QComboBox()
        self.layout.addWidget(self.combo_y)

        self.layout.addWidget(QLabel("Категориальный (hue):"))
        self.combo_hue = QComboBox()
        self.layout.addWidget(self.combo_hue)

        self.layout.addWidget(QLabel("Размер (size, опционально):"))
        self.combo_size = QComboBox()
        self.layout.addWidget(self.combo_size)

        self.hist_btn = QPushButton("Гистограмма")
        self.hist_btn.clicked.connect(self.plot_histogram)
        self.hist_btn.setEnabled(False)
        self.layout.addWidget(self.hist_btn)

        self.scatter_btn = QPushButton("Многомерный анализ")
        self.scatter_btn.clicked.connect(self.plot_scatter)
        self.scatter_btn.setEnabled(False)
        self.layout.addWidget(self.scatter_btn)

        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

    def connect_db(self):
        try:
            password = quote_plus(self.password.text())
            self.engine = create_engine(
                f"postgresql://{self.user.text()}:{password}@"
                f"{self.host.text()}:{self.port.text()}/{self.db.text()}"
            )

            # ---- ДАННЫЕ ДЛЯ АНАЛИЗА ----
            stats1 = pd.read_sql("SELECT * FROM stats1 LIMIT 1000", self.engine)
            stats2 = pd.read_sql("SELECT * FROM stats2 LIMIT 1000", self.engine)
            self.df = pd.concat([stats1, stats2], ignore_index=True)

            # Типы
            for col in ["win", "firstblood"]:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype("category")

            # Выгрузка промежуточная
            self.df.to_csv("stats_combined.csv", index=False)

            self.describe_features()
            self.fill_comboboxes()

            self.preview_btn.setEnabled(True)
            self.hist_btn.setEnabled(True)
            self.scatter_btn.setEnabled(True)

            QMessageBox.information(
                self,
                "Успех",
                "stats1 и stats2 загружены\n"
                "Файл сохранён: stats_combined.csv"
            )

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def preview_table(self):
        table = self.table_selector.currentText()
        df = pd.read_sql(f"SELECT * FROM {table} LIMIT 5", self.engine)

        text = f"Таблица: {table}\n\n"
        text += "Первые строки:\n"
        text += df.to_string()
        text += "\n\nТипы данных:\n"
        text += df.dtypes.to_string()

        self.table_view.setText(text)

    def describe_features(self):
        print("\n=== ОПИСАНИЕ ПРИЗНАКОВ stats ===")

        numeric = self.df.select_dtypes(include="number").columns
        categorical = self.df.select_dtypes(include=["object", "category"]).columns
        binary = [c for c in self.df.columns if self.df[c].nunique() == 2]

        print("Количественные:", list(numeric))
        print("Категориальные:", list(categorical))
        print("Бинарные:", binary)

    def fill_comboboxes(self):
        num = self.df.select_dtypes(include="number").columns.tolist()
        cat = self.df.select_dtypes(include=["category", "object"]).columns.tolist()

        self.combo_x.addItems(num)
        self.combo_y.addItems(num)
        self.combo_size.addItems([""] + num)
        self.combo_hue.addItems([""] + cat)

    def plot_histogram(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        col = self.combo_x.currentText()
        ax.hist(self.df[col].dropna(), bins=30)
        ax.set_title(f"Гистограмма: {col}")

        self.canvas.draw()

    def plot_scatter(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        sns.scatterplot(
            data=self.df,
            x=self.combo_x.currentText(),
            y=self.combo_y.currentText(),
            hue=self.combo_hue.currentText() or None,
            size=self.combo_size.currentText() or None,
            sizes=(20, 200),
            alpha=0.7,
            ax=ax
        )

        ax.set_title("Многомерный анализ (stats)")
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataVizApp()
    window.show()
    sys.exit(app.exec())
