# Практика 3. Работа с библиотекой fastapi

## 1. Описание

В данной лабораторной работе необходимо отделить логику проекта от интерфейса и реализовать серверную часть с использованием FastAPI

Программа работает с датасетом по игре **League of Legends** — **League of Legends Ranked Matches**.

Датасет доступен по следующей ссылке:  
[League of Legends Ranked Matches - Kaggle](https://www.kaggle.com/datasets/paololol/league-of-legends-ranked-matches)

## 2. Запуск

Для установки библиотеки PySpark необходимо установить Java на компьютер и указать путь в системные переменные как JAVA_HOME. Да-лее необходимо либо установить PySpark через официальный сайт (https://spark.apache.org/docs/latest/api/python/index.html), либо через установщик пакетов python. Также стоит учитывать, что для работы с PySpark необходимо установить Python версии не выше 3.9.

Для создания модели используйте файл train_model.py

Для работы с streamlit - app.py

Для работы с fastapi - main.py

Для запуска fastapi используется команда python -m uvicorn main:app.
Для запуска интерфейса используется команда streamlit run app.py.

## 3. Требования

Установите зависимости с помощью команды в корне проекта Practice2:
```text
pip install -r requirements.txt
```