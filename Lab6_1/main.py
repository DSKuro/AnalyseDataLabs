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
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

from gensim.models import Word2Vec
from wordcloud import WordCloud

from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QTextEdit, QVBoxLayout
)

nltk.download("stopwords")

morph = pymorphy3.MorphAnalyzer()
stemmer = SnowballStemmer("russian")
stop_words = set(stopwords.words("russian"))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^а-яё\s]", " ", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]

    lemmas = [morph.parse(t)[0].normal_form for t in tokens]
    stems = [stemmer.stem(l) for l in lemmas]

    return " ".join(stems)

def load_texts(path: str):
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    texts = raw.split("===")
    texts = [t.strip() for t in texts if len(t.strip()) > 0]
    return [preprocess_text(t) for t in texts]

songs = load_texts("songs_raw.txt")
poems = load_texts("poems_raw.txt")

tfidf_vectorizer = TfidfVectorizer(max_features=50)
tfidf_matrix = tfidf_vectorizer.fit_transform(songs)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

tfidf_means = tfidf_df.mean().sort_values(ascending=False)

def show_wordcloud():
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(tfidf_means.to_dict())

    plt.figure(figsize=(10, 5))
    plt.imshow(wc)
    plt.axis("off")
    plt.show()

sentences = [text.split() for text in songs]

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

def show_tsne():
    words = list(tfidf_means.head(15).index)
    vectors = np.array([w2v_model.wv[w] for w in words])

    tsne = TSNE(
        n_components=2,
        perplexity=5,
        random_state=42
    )
    points = tsne.fit_transform(vectors)

    plt.figure(figsize=(8, 8))
    for i, word in enumerate(words):
        plt.scatter(points[i, 0], points[i, 1])
        plt.annotate(word, (points[i, 0], points[i, 1]))
    plt.show()

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

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = accuracy_score(y_test, preds)

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

        btn_tfidf.clicked.connect(self.show_tfidf)
        btn_wc.clicked.connect(show_wordcloud)
        btn_tsne.clicked.connect(show_tsne)
        btn_cls.clicked.connect(self.show_classification)

        layout = QVBoxLayout()
        layout.addWidget(btn_tfidf)
        layout.addWidget(btn_wc)
        layout.addWidget(btn_tsne)
        layout.addWidget(btn_cls)
        layout.addWidget(self.output)

        self.setLayout(layout)

    def show_tfidf(self):
        self.output.setText(str(tfidf_means.head(10)))

    def show_classification(self):
        text = "\n".join([f"{k}: {v:.2f}" for k, v in results.items()])
        self.output.setText(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NLPApp()
    window.show()
    sys.exit(app.exec())
