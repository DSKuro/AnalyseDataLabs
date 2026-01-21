import seaborn as sns
import pandas as pd
from scipy import stats

def load_data():
    df = sns.load_dataset('mpg')
    print('Данные успешно загружены')
    print(f'Количество строк: {df.shape[0]}')
    print(f'Количество столбцов: {df.shape[1]}\n')
    return df

def split_columns(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    return num_cols, cat_cols

def numeric_summary(df, num_cols):
    print('Анализ числовых переменных')
    summary = []

    for col in num_cols:
        s = df[col]
        stats_dict = {
            'variable': col,
            'missing_share': round(s.isna().mean(), 3),
            'min': s.min(),
            'max': s.max(),
            'mean': round(s.mean(), 3),
            'median': round(s.median(), 3),
            'variance': round(s.var(), 3),
            'q_0.1': s.quantile(0.1),
            'q_0.9': s.quantile(0.9),
            'Q1': s.quantile(0.25),
            'Q3': s.quantile(0.75)
        }
        summary.append(stats_dict)

        print(f"\nПеременная: {col}")
        for k, v in stats_dict.items():
            if k != 'variable':
                print(f"  {k}: {v}")

    print()
    return pd.DataFrame(summary)

def categorical_summary(df, cat_cols):
    print('=== Анализ категориальных переменных ===')
    summary = []

    for col in cat_cols:
        s = df[col]
        stats_dict = {
            'variable': col,
            'missing_share': round(s.isna().mean(), 3),
            'unique_values': s.nunique(),
            'mode': s.mode().iloc[0]
        }
        summary.append(stats_dict)

        print(f"\nПеременная: {col}")
        for k, v in stats_dict.items():
            if k != 'variable':
                print(f"  {k}: {v}")

    print()
    return pd.DataFrame(summary)

def hypothesis_tests(df):
    print('=== Проверка статистических гипотез ===')

    usa = df[df['origin'] == 'usa']['mpg']
    europe = df[df['origin'] == 'europe']['mpg']

    t_stat, p_value = stats.ttest_ind(usa, europe, nan_policy='omit')

    print('\nГипотеза 1:')
    print('H0: Средний mpg не отличается для автомобилей из США и Европы')
    print(f't-статистика = {t_stat:.3f}')
    print(f'p-value = {p_value:.5f}')

    if p_value < 0.05:
        print('Вывод: H0 отвергается — различия статистически значимы')
    else:
        print('Вывод: нет оснований отвергнуть H0')

    corr, p_corr = stats.pearsonr(df['weight'], df['mpg'])

    print('\nГипотеза 2:')
    print('H0: Корреляция между weight и mpg равна 0')
    print(f'Коэффициент корреляции = {corr:.3f}')
    print(f'p-value = {p_corr:.5f}')

    if p_corr < 0.05:
        print('Вывод: существует статистически значимая связь\n')
    else:
        print('Вывод: статистически значимой связи не обнаружено\n')

def correlation_analysis(df):
    print('=== Корреляция с целевой переменной mpg ===')

    df_encoded = pd.get_dummies(df, columns=['origin'], drop_first=True)

    df_encoded = df_encoded.select_dtypes(include=['int64', 'float64'])

    corr = df_encoded.corr()['mpg'].sort_values(ascending=False)

    print(corr)
    print('\nВывод:')
    print('Наибольшую отрицательную корреляцию с mpg имеют weight и horsepower.')
    print('Это означает, что более тяжёлые и мощные автомобили менее экономичны.\n')

    return corr

def gradient_descent(df, feature='weight', lr=0.01, epochs=1000):
    print('=== Линейная регрессия (градиентный спуск) ===')

    data = df[[feature, 'mpg']].dropna()
    x = data[[feature]]
    y = data['mpg'].values

    x = (x - x.mean()) / x.std()
    x = x.values.flatten()
    n = len(y)

    w0, w1 = 0.0, 0.0
    for _ in range(epochs):
        y_pred = w0 + w1 * x
        error = y_pred - y
        w0 -= lr * error.mean()
        w1 -= lr * (error * x).mean()

    print(f'Обычный GD: w0 = {w0:.3f}, w1 = {w1:.3f}')

    w0_sgd, w1_sgd = 0.0, 0.0
    for _ in range(epochs):
        for i in range(n):
            y_pred = w0_sgd + w1_sgd * x[i]
            error = y_pred - y[i]
            w0_sgd -= lr * error
            w1_sgd -= lr * error * x[i]

    print(f'SGD:        w0 = {w0_sgd:.3f}, w1 = {w1_sgd:.3f}\n')

def main():
    df = load_data()
    num_cols, cat_cols = split_columns(df)

    numeric_summary(df, num_cols)
    categorical_summary(df, cat_cols)

    hypothesis_tests(df)
    correlation_analysis(df)
    gradient_descent(df, feature='weight')


main()
