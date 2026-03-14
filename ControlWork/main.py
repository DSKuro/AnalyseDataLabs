import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from scipy.stats import ttest_ind, chi2_contingency
import pickle

# 1. Загрузка данных

df = pd.read_csv("advertising.csv")

print("Первые 20 строк:")
print(df.head(20))

print("\nИнформация о датасете:")
print(df.info())

print("\nОписание числовых признаков:")
print(df.describe())

print("\nОписание категориальных признаков:")
print(df.describe(include='object'))

# 2. Визуализация

sns.countplot(x='Clicked on Ad', data=df)
plt.title("Distribution of Target Variable")
plt.show()

sns.boxplot(x=df['Daily Time Spent on Site'])
plt.title("Boxplot: Daily Time Spent on Site")
plt.show()

plt.scatter(df['Daily Time Spent on Site'], df['Clicked on Ad'])
plt.xlabel("Daily Time Spent on Site")
plt.ylabel("Clicked on Ad")
plt.title("Scatter Plot")
plt.show()

# 3. EDA

print("\n===== EDA =====")

missing_ratio = df.isnull().mean()
print("\nДоля пропусков:")
print(missing_ratio)

numeric_cols = df.select_dtypes(include=np.number).columns

eda_numeric = pd.DataFrame(index=numeric_cols)
eda_numeric['Min'] = df[numeric_cols].min()
eda_numeric['Max'] = df[numeric_cols].max()
eda_numeric['Mean'] = df[numeric_cols].mean()
eda_numeric['Median'] = df[numeric_cols].median()
eda_numeric['Variance'] = df[numeric_cols].var()
eda_numeric['Quantile_0.1'] = df[numeric_cols].quantile(0.1)
eda_numeric['Quantile_0.9'] = df[numeric_cols].quantile(0.9)
eda_numeric['Q1'] = df[numeric_cols].quantile(0.25)
eda_numeric['Q3'] = df[numeric_cols].quantile(0.75)

print("\nЧисловой EDA:")
print(eda_numeric)

# 4. Подготовка данных

df_model = df.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1)

df_encoded = pd.get_dummies(df_model, drop_first=True)

print("\nСтолбцы после кодирования:")
print(df_encoded.columns)

print("\nРазмер датасета после кодирования:")
print(df_encoded.shape)

# 5. Корреляция

corr = df_encoded.corr()

target_corr = corr['Clicked on Ad'].sort_values(ascending=False)
print("\nТоп корреляций с целевой переменной:")
print(target_corr)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 6. Разделение данных

X = df_encoded.drop('Clicked on Ad', axis=1)
y = df_encoded['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# 7. Масштабирование

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Подбор лучшего k

errors = []

for k in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    errors.append(1 - accuracy_score(y_test, pred))

plt.figure()
plt.plot(range(1,30), errors)
plt.xlabel("K")
plt.ylabel("Error")
plt.title("Choosing optimal K")
plt.show()

# выбираем k
model = KNeighborsClassifier(n_neighbors=10, weights='distance')

# 9. Обучение модели

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\nTrain Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy:", accuracy_score(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(y_test, test_pred))

# 10. Метрики

cm = confusion_matrix(y_test, test_pred)
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred)
recall = recall_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\nConfusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC-AUC:", roc_auc)

# ROC кривая

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# 11. Гипотеза 1 (t-test)

print("\n===== Гипотеза 1 =====")
print("H0: Среднее время на сайте одинаково")

group0 = df[df['Clicked on Ad']==0]['Daily Time Spent on Site']
group1 = df[df['Clicked on Ad']==1]['Daily Time Spent on Site']

t_stat, p_value = ttest_ind(group0, group1)

print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Отклоняем H0 — средние различаются")
else:
    print("Не отклоняем H0")

# 12. Гипотеза 2 (Chi-square)

print("\n===== Гипотеза 2 =====")
print("H0: Пол и клик независимы")

contingency_table = pd.crosstab(df['Male'], df['Clicked on Ad'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-square:", chi2)
print("P-value:", p)

if p < 0.05:
    print("Отклоняем H0 — есть зависимость")
else:
    print("Не отклоняем H0 — зависимости нет")

# 13. Сохранение модели

with open("advertising_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nМодель сохранена в advertising_model.pkl")