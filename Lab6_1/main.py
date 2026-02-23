import re
import sys
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymorphy3
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from gensim.models import Word2Vec
from wordcloud import WordCloud

from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QTextEdit, QVBoxLayout
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

nltk.download("stopwords")

morph = pymorphy3.MorphAnalyzer()
stemmer = SnowballStemmer("russian")
stop_words = set(stopwords.words("russian"))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^а-яё\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [t for t in tokens if re.fullmatch(r"[а-яё]+", t)]
    lemmas = [morph.parse(t)[0].normal_form for t in tokens]
    return " ".join(lemmas)

def load_texts(path: str):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    texts = raw.split("===")
    return [preprocess_text(t) for t in texts if len(t.strip()) > 0]

songs = load_texts("songs_raw.txt")
poems = load_texts("poems_raw.txt")
all_texts = songs + poems

print(all_texts)

tfidf_vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                        columns=tfidf_vectorizer.get_feature_names_out())
tfidf_means = tfidf_df.mean().sort_values(ascending=False)

sentences = [t.split() for t in all_texts]
w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4)

df_songs = pd.DataFrame({"text": songs, "label": 0})
df_poems = pd.DataFrame({"text": poems, "label": 1})
df = pd.concat([df_songs, df_poems]).reset_index(drop=True)

X = tfidf_vectorizer.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(),
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

results = {}
trained_models = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    trained_models[name] = model
    predictions[name] = preds

    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    }

    # сохраняем модель
    joblib.dump(model, f"{name}_model.pkl")

class NLPApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NLP анализ песен и стихов")
        self.output = QTextEdit()
        self.output.setReadOnly(True)

        btn_tfidf = QPushButton("Показать TF-IDF")
        btn_wc = QPushButton("WordCloud")
        btn_tsne = QPushButton("t-SNE")
        btn_cls = QPushButton("Классификация")
        btn_similar = QPushButton("Похожие слова для топ-слова")

        btn_tfidf.clicked.connect(self.show_tfidf)
        btn_wc.clicked.connect(self.show_wordcloud)
        btn_tsne.clicked.connect(self.show_tsne)
        btn_cls.clicked.connect(self.show_classification)
        btn_similar.clicked.connect(self.show_similar_word)

        self.canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.ax = self.canvas.figure.subplots()


        layout = QVBoxLayout()
        layout.addWidget(btn_tfidf)
        layout.addWidget(btn_wc)
        layout.addWidget(btn_tsne)
        layout.addWidget(btn_cls)
        layout.addWidget(btn_similar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.output)
        self.setLayout(layout)

    def show_tfidf(self):
        self.output.setText(str(tfidf_means.head(10)))

    def show_classification(self):
        self.ax.clear()

        metrics_text = ""
        model_names = list(results.keys())
        accuracies = []

        for name in model_names:
            metrics = results[name]
            accuracies.append(metrics["accuracy"])

            metrics_text += (
                f"{name}\n"
                f"Accuracy: {metrics['accuracy']:.2f}\n"
                f"Precision: {metrics['precision']:.2f}\n"
                f"Recall: {metrics['recall']:.2f}\n"
                f"F1-score: {metrics['f1']:.2f}\n\n"
            )

        self.ax.bar(model_names, accuracies)
        self.ax.set_title("Accuracy моделей")
        self.ax.set_ylabel("Accuracy")
        self.canvas.draw()

        self.output.setText(metrics_text)

    def show_similar_word(self):
        top_word = tfidf_means.index[0]
        similar = w2v_model.wv.most_similar(top_word, topn=10)
        self.output.setText(f"Похожие слова для '{top_word}':\n" + "\n".join([f"{w[0]} ({w[1]:.2f})" for w in similar]))

    def show_wordcloud(self):
        self.ax.clear()
        wc = WordCloud(width=800, height=400, background_color="white")
        wc.generate_from_frequencies(tfidf_means.to_dict())
        self.ax.imshow(wc, interpolation='bilinear')
        self.ax.axis('off')
        self.canvas.draw()

    def show_tsne(self):
        self.ax.clear()
        words = list(tfidf_means.head(15).index)
        vectors = np.array([w2v_model.wv[w] for w in words])
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        points = tsne.fit_transform(vectors)
        self.ax.scatter(points[:, 0], points[:, 1])
        for i, word in enumerate(words):
            self.ax.annotate(word, (points[i, 0], points[i, 1]))
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NLPApp()
    window.show()
    sys.exit(app.exec())