import sys
import pandas as pd
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit, QFormLayout, QMessageBox
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import quote_plus

class DataVizApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("League of Legends Data Analyzer")
        self.setGeometry(100, 100, 450, 500)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        form_layout = QFormLayout()
        self.host_input = QLineEdit("localhost")
        self.port_input = QLineEdit("5432")
        self.user_input = QLineEdit("postgres")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.db_input = QLineEdit("lol_db")

        form_layout.addRow("Host:", self.host_input)
        form_layout.addRow("Port:", self.port_input)
        form_layout.addRow("User:", self.user_input)
        form_layout.addRow("Password:", self.password_input)
        form_layout.addRow("Database:", self.db_input)
        self.layout.addLayout(form_layout)

        self.connect_button = QPushButton("Подключиться к БД")
        self.connect_button.clicked.connect(self.connect_db)
        self.layout.addWidget(self.connect_button)

        self.layout.addWidget(QLabel("Выберите X (количественный):"))
        self.combo_x = QComboBox()
        self.layout.addWidget(self.combo_x)

        self.layout.addWidget(QLabel("Выберите Y (количественный):"))
        self.combo_y = QComboBox()
        self.layout.addWidget(self.combo_y)

        self.layout.addWidget(QLabel("Выберите категориальный признак (hue):"))
        self.combo_hue = QComboBox()
        self.layout.addWidget(self.combo_hue)

        self.layout.addWidget(QLabel("Выберите размер точек (size, количественный, optional):"))
        self.combo_size = QComboBox()
        self.layout.addWidget(self.combo_size)

        self.hist_button = QPushButton("Построить гистограмму")
        self.hist_button.clicked.connect(self.plot_histogram)
        self.hist_button.setEnabled(False)
        self.layout.addWidget(self.hist_button)

        self.scatter_button = QPushButton("Построить многомерный scatter")
        self.scatter_button.clicked.connect(self.plot_multivariate)
        self.scatter_button.setEnabled(False)
        self.layout.addWidget(self.scatter_button)

    def connect_db(self):
        host = self.host_input.text()
        port = self.port_input.text()
        user = self.user_input.text()
        password = self.password_input.text()
        dbname = self.db_input.text()
        password_encoded = quote_plus(password)

        try:
            self.engine = create_engine(f"postgresql://{user}:{password_encoded}@{host}:{port}/{dbname}")

            stats1 = pd.read_sql("SELECT * FROM stats1 LIMIT 1000", self.engine)
            stats2 = pd.read_sql("SELECT * FROM stats2 LIMIT 1000", self.engine)
            self.df = pd.concat([stats1, stats2], ignore_index=True)

            self.df['win'] = self.df['win'].astype('category')
            self.df['firstblood'] = self.df['firstblood'].astype('category')

            numeric_cols = self.df.select_dtypes(include='number').columns.tolist()
            cat_cols = self.df.select_dtypes(include=['category']).columns.tolist()

            self.combo_x.clear()
            self.combo_y.clear()
            self.combo_size.clear()
            self.combo_hue.clear()

            for col in numeric_cols:
                self.combo_x.addItem(col)
                self.combo_y.addItem(col)
                self.combo_size.addItem(col)

            for col in cat_cols:
                self.combo_hue.addItem(col)

            self.hist_button.setEnabled(True)
            self.scatter_button.setEnabled(True)

            QMessageBox.information(self, "Успех", "Подключение выполнено, данные загружены!")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка подключения", str(e))

    def plot_histogram(self):
        col = self.combo_x.currentText()
        plt.figure(figsize=(8,5))
        plt.hist(self.df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Гистограмма: {col}")
        plt.xlabel(col)
        plt.ylabel("Количество")
        plt.show()

    def plot_multivariate(self):
        x_col = self.combo_x.currentText()
        y_col = self.combo_y.currentText()
        hue_col = self.combo_hue.currentText() if self.combo_hue.currentText() else None
        size_col = self.combo_size.currentText() if self.combo_size.currentText() else None

        plt.figure(figsize=(10,6))
        sns.scatterplot(
            data=self.df,
            x=x_col,
            y=y_col,
            hue=hue_col,
            size=size_col,
            sizes=(20,200) if size_col else None,
            alpha=0.7
        )
        plt.title(f"Multivariate plot: {x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend(loc='upper left')
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataVizApp()
    window.show()
    sys.exit(app.exec())
