# Практика 1. Работа с библиотекой pySpark

## 1. Описание

В данной лабораторной работе необходимо разработать и реализовать систему работы с реляционными базами данных PostgreSQL, а также применить фреймворк pySpark для обработки данных, выполнением SQL-запросов и работы с таблицами в базе данных.

Программа работает с датасетом по игре **League of Legends** — **League of Legends Ranked Matches**.

Датасет доступен по следующей ссылке:  
[League of Legends Ranked Matches - Kaggle](https://www.kaggle.com/datasets/paololol/league-of-legends-ranked-matches)

## 2. Запуск

Для установки библиотеки PySpark необходимо установить Java на компьютер и указать путь в системные переменные как JAVA_HOME. Да-лее необходимо либо установить PySpark через официальный сайт (https://spark.apache.org/docs/latest/api/python/index.html), либо через установщик пакетов python. Также стоит учитывать, что для работы с PySpark необходимо установить Python версии не выше 3.9.
Для работы с SqlAlchemy необходимо задать переменные в файле .env: DB_NAME и DB_PASSWORD.

Для работы с sqlalchemy используется файл alchemy.py и models.py.

Для работы с pyspark - main.py

## 3. Требования

Установите зависимости с помощью команды в корне всего проекта:
```text
pip install -r requirements.txt
```