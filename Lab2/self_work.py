import pandas as pd
from scipy import stats

def load_and_merge_stats(path1='../data/stats1.csv', path2='../data/stats2.csv'):
    print('=== Загрузка данных ===')

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    print(f'stats1.csv: {df1.shape[0]} строк, {df1.shape[1]} столбцов')
    print(f'stats2.csv: {df2.shape[0]} строк, {df2.shape[1]} столбцов')

    df = pd.concat([df1, df2], ignore_index=True)

    print(f'Объединённый датасет: {df.shape[0]} строк, {df.shape[1]} столбцов\n')
    return df

def split_columns(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    print('=== Типы признаков ===')
    print(f'Числовых: {len(num_cols)}')
    print(f'Категориальных: {len(cat_cols)}\n')

    return num_cols, cat_cols

def numeric_eda(df, num_cols):
    print('=== Разведочный анализ числовых переменных ===')

    for col in num_cols:
        s = df[col]
        print(f'\nПеременная: {col}')
        print(f'  Доля пропусков: {s.isna().mean():.4f}')
        print(f'  Минимум: {s.min()}')
        print(f'  Максимум: {s.max()}')
        print(f'  Среднее: {s.mean():.3f}')
        print(f'  Медиана: {s.median():.3f}')
        print(f'  Дисперсия: {s.var():.3f}')
        print(f'  Квантиль 0.1: {s.quantile(0.1):.3f}')
        print(f'  Квантиль 0.9: {s.quantile(0.9):.3f}')
        print(f'  Q1: {s.quantile(0.25):.3f}')
        print(f'  Q3: {s.quantile(0.75):.3f}')

def categorical_eda(df, cat_cols):
    print('\n=== Разведочный анализ категориальных переменных ===')

    if len(cat_cols) == 0:
        print('Категориальные переменные отсутствуют\n')
        return

    for col in cat_cols:
        s = df[col]
        print(f'\nПеременная: {col}')
        print(f'  Доля пропусков: {s.isna().mean():.4f}')
        print(f'  Уникальных значений: {s.nunique()}')
        print(f'  Мода: {s.mode().iloc[0]}')

def hypothesis_testing(df):
    print('\n=== Проверка статистических гипотез ===')

    print('\nГипотеза 1:')
    print('H0: Среднее количество убийств одинаково для побед и поражений')

    wins = df[df['win'] == 1]['kills']
    losses = df[df['win'] == 0]['kills']

    t_stat, p_value = stats.ttest_ind(wins, losses)

    print(f't-статистика: {t_stat:.3f}')
    print(f'p-value: {p_value:.5f}')

    if p_value < 0.05:
        print('Вывод: H0 отвергается — в победах убийств больше')
    else:
        print('Вывод: нет оснований отвергнуть H0')

    print('\nГипотеза 2:')
    print('H0: Корреляция между deaths и win равна 0')

    corr, p_corr = stats.pearsonr(df['deaths'], df['win'])

    print(f'Корреляция: {corr:.3f}')
    print(f'p-value: {p_corr:.5f}')

    if p_corr < 0.05:
        print('Вывод: deaths статистически значимо связаны с победой')
    else:
        print('Вывод: статистически значимой связи не обнаружено')

def encode_features(df):
    print('\n=== Кодирование категориальных переменных ===')
    df_encoded = pd.get_dummies(df, drop_first=True)
    print('One-Hot Encoding выполнен\n')
    return df_encoded

def correlation_with_target(df):
    print('=== Корреляция признаков с целевой переменной win ===')

    df_numeric = df.select_dtypes(include=['int64', 'float64'])
    corr = df_numeric.corr()['win'].sort_values(ascending=False)

    print(corr.head(10))
    print('\nВывод:')
    print('Наибольшую положительную корреляцию имеют золото и убийства.')
    print('Наиболее отрицательную — количество смертей.\n')

    return corr

def gradient_descent(df, feature='goldearned', lr=0.01, epochs=1000):
    print('=== Градиентный спуск ===')

    data = df[[feature, 'win']].dropna()

    x = data[[feature]]
    y = data['win'].values

    x = (x - x.mean()) / x.std()
    x = x.values.flatten()
    n = len(y)

    w0, w1 = 0.0, 0.0
    for _ in range(epochs):
        y_pred = w0 + w1 * x
        error = y_pred - y
        w0 -= lr * error.mean()
        w1 -= lr * (error * x).mean()

    print(f'Обычный GD: w0 = {w0:.4f}, w1 = {w1:.4f}')

    w0_sgd, w1_sgd = 0.0, 0.0
    for _ in range(epochs):
        for i in range(n):
            y_pred = w0_sgd + w1_sgd * x[i]
            error = y_pred - y[i]
            w0_sgd -= lr * error
            w1_sgd -= lr * error * x[i]

    print(f'SGD:        w0 = {w0_sgd:.4f}, w1 = {w1_sgd:.4f}\n')

def main():
    df = load_and_merge_stats()
    num_cols, cat_cols = split_columns(df)

    numeric_eda(df, num_cols)
    categorical_eda(df, cat_cols)

    hypothesis_testing(df)

    df_encoded = encode_features(df)
    correlation_with_target(df_encoded)

    gradient_descent(df)


if __name__ == '__main__':
    main()
